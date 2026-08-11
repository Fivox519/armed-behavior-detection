"""
持械人员检测器 V2 - 集成时序平滑与状态机
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from pathlib import Path


class ArmedPersonDetectorV2:
    """
    持械人员检测器 V2
    - 双阶段检测：YOLO-Pose + 武器分类
    - 时序平滑：消除告警闪烁
    - 状态机：跟踪人员持械状态变化
    - 动态裁剪：根据人体大小动态调整裁剪窗口
    """

    WEAPON_CLASSES = ['knife', 'axe', 'hammer', 'stick']
    WEAPON_COLORS = {
        'knife': (0, 165, 255),
        'axe': (0, 69, 255),
        'hammer': (128, 128, 128),
        'stick': (139, 69, 19)
    }

    def __init__(self, pose_model_path, weapon_model_path,
                 weapon_conf_threshold=0.75,
                 temporal_window=15,
                 alert_threshold=0.6,
                 base_crop_size=226,
                 crop_scale=0.4,
                 dynamic_crop=True):
        self.pose_model = YOLO(pose_model_path)
        self.weapon_model = YOLO(weapon_model_path)
        self.weapon_conf_threshold = weapon_conf_threshold
        self.base_crop_size = base_crop_size
        self.crop_scale = crop_scale
        self.dynamic_crop = dynamic_crop
        self.classifier_input_size = 226

        self.track_history = defaultdict(lambda: deque(maxlen=temporal_window))
        self.track_positions = defaultdict(lambda: {'last_seen': 0, 'armed': False})
        self.temporal_window = temporal_window
        self.alert_threshold = alert_threshold
        self.frame_count = 0

        self.colors = {
            'person': (0, 255, 0),
            'left_wrist': (255, 0, 0),
            'right_wrist': (0, 0, 255),
            'armed': (0, 0, 255),
            'normal': (0, 255, 0),
            'tracked': (255, 255, 0)
        }

    def _classify_weapon(self, crop_image):
        """使用Sigmoid阈值判断武器类别"""
        results = self.weapon_model(crop_image, verbose=False)

        if not results or not results[0].probs:
            return {'class': 'none', 'confidence': 0.0, 'is_weapon': False}

        probs = results[0].probs.data[0].cpu().numpy()
        weapon_probs = probs[:4]
        max_weapon_prob = float(np.max(weapon_probs))

        if max_weapon_prob < self.weapon_conf_threshold:
            return {'class': 'none', 'confidence': float(probs[4]), 'is_weapon': False}

        weapon_cls_id = int(np.argmax(weapon_probs))
        return {
            'class': self.WEAPON_CLASSES[weapon_cls_id],
            'confidence': max_weapon_prob,
            'is_weapon': True
        }

    def _get_crop_size(self, person_box):
        """根据人体框大小动态计算裁剪窗口尺寸"""
        x1, y1, x2, y2 = person_box
        person_width = x2 - x1
        person_height = y2 - y1

        dynamic_size = int(max(person_width, person_height) * self.crop_scale)
        crop_size = max(self.base_crop_size, dynamic_size)
        return crop_size

    def _crop_wrist_region(self, frame, x, y, person_box=None, padding=50):
        """
        裁剪手腕区域 - 支持动态裁剪窗口

        Args:
            frame: 输入图像
            x, y: 手腕关键点坐标
            person_box: 人体边界框(用于动态裁剪)
            padding: 额外padding
        """
        h, w = frame.shape[:2]

        if self.dynamic_crop and person_box is not None:
            crop_size = self._get_crop_size(person_box)
        else:
            crop_size = self.base_crop_size

        x1 = max(0, int(x - crop_size // 2 - padding))
        y1 = max(0, int(y - crop_size // 2 - padding))
        x2 = min(w, int(x + crop_size // 2 + padding))
        y2 = min(h, int(y + crop_size // 2 + padding))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros((self.classifier_input_size, self.classifier_input_size, 3), dtype=np.uint8)

        resized = cv2.resize(crop, (self.classifier_input_size, self.classifier_input_size),
                           interpolation=cv2.INTER_LINEAR)
        return resized

    def _get_foot_center(self, box):
        """获取人体脚底中心点 (x_center, y_max)"""
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) // 2
        y_max = int(y2)
        return (x_center, y_max)

    def detect(self, frame, track_ids=None):
        """
        检测单帧图像

        Args:
            frame: 输入图像
            track_ids: 可选的人员跟踪ID列表

        Returns:
            dict: {
                'persons': list of person detections,
                'alerts': list of 当前告警的人员ID,
                'frame': annotated frame
            }
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        annotated = frame.copy()

        pose_results = self.pose_model(frame, verbose=False)
        kpts = pose_results[0].keypoints
        boxes = pose_results[0].boxes

        persons = []
        current_track_ids = set()

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                track_id = track_ids[i] if track_ids and i < len(track_ids) else i
                current_track_ids.add(track_id)

                foot_x, foot_y = self._get_foot_center((x1, y1, x2, y2))

                person_data = {
                    'track_id': track_id,
                    'box': (x1, y1, x2, y2),
                    'foot_center': (foot_x, foot_y),
                    'left_wrist': {'class': 'none', 'confidence': 0.0, 'is_weapon': False},
                    'right_wrist': {'class': 'none', 'confidence': 0.0, 'is_weapon': False},
                    'is_armed': False,
                    'weapon_ratio': 0.0,
                    'armed_frames': 0,
                    'total_frames': 0
                }

                if kpts is not None and i < len(kpts.data):
                    kp_data = kpts.data[i].cpu().numpy()

                    lw_x, lw_y = int(kp_data[9][0]), int(kp_data[9][1])
                    rw_x, rw_y = int(kp_data[10][0]), int(kp_data[10][1])

                    person_box = (x1, y1, x2, y2)
                    lw_crop = self._crop_wrist_region(frame, lw_x, lw_y, person_box)
                    rw_crop = self._crop_wrist_region(frame, rw_x, rw_y, person_box)

                    person_data['left_wrist'] = self._classify_weapon(lw_crop)
                    person_data['right_wrist'] = self._classify_weapon(rw_crop)

                    person_data['is_armed'] = (
                        person_data['left_wrist']['is_weapon'] or
                        person_data['right_wrist']['is_weapon']
                    )

                    cv2.circle(annotated, (lw_x, lw_y), 6, self.colors['left_wrist'], -1)
                    cv2.circle(annotated, (rw_x, rw_y), 6, self.colors['right_wrist'], -1)

                self.track_history[track_id].append(1 if person_data['is_armed'] else 0)

                if len(self.track_history[track_id]) > 0:
                    person_data['total_frames'] = len(self.track_history[track_id])
                    person_data['armed_frames'] = sum(self.track_history[track_id])
                    person_data['weapon_ratio'] = person_data['armed_frames'] / person_data['total_frames']

                self.track_positions[track_id]['last_seen'] = self.frame_count
                self.track_positions[track_id]['armed'] = person_data['is_armed']

                cv2.rectangle(annotated, (x1, y1), (x2, y2), self.colors['person'], 2)

                status_color = self.colors['armed'] if person_data['is_armed'] else self.colors['normal']
                status_text = f"ID:{track_id} {'ARMED!' if person_data['is_armed'] else 'Normal'}"
                cv2.putText(annotated, status_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                if person_data['left_wrist']['is_weapon']:
                    lw_text = f"L:{person_data['left_wrist']['class']}"
                    cv2.putText(annotated, lw_text, (lw_x + 10, lw_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if person_data['right_wrist']['is_weapon']:
                    rw_text = f"R:{person_data['right_wrist']['class']}"
                    cv2.putText(annotated, rw_text, (rw_x + 10, rw_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.circle(annotated, (foot_x, foot_y), 5, (0, 255, 255), -1)

                persons.append(person_data)

        stale_tracks = [tid for tid in self.track_positions
                        if self.frame_count - self.track_positions[tid]['last_seen'] > 30]
        for tid in stale_tracks:
            del self.track_history[tid]
            del self.track_positions[tid]

        alerts = [p['track_id'] for p in persons
                  if p['total_frames'] >= 10 and p['weapon_ratio'] >= self.alert_threshold]

        for alert_id in alerts:
            for p in persons:
                if p['track_id'] == alert_id:
                    x1, y1, x2, y2 = p['box']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, "⚠️ ALERT", (x1, y2 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return {
            'persons': persons,
            'alerts': alerts,
            'frame': annotated
        }


class ROIManager:
    """区域-of-Interest (ROI) 管理器"""

    def __init__(self):
        self.danger_zones = []
        self.safe_zones = []

    def add_polygon(self, points, zone_type='danger'):
        """
        添加多边形区域

        Args:
            points: 多边形顶点列表 [(x1,y1), (x2,y2), ...]
            zone_type: 'danger' 或 'safe'
        """
        if zone_type == 'danger':
            self.danger_zones.append(np.array(points, np.int32))
        else:
            self.safe_zones.append(np.array(points, np.int32))

    def add_rectangle(self, x1, y1, x2, y2, zone_type='danger'):
        """添加矩形区域"""
        self.add_polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], zone_type)

    def point_in_zone(self, point, zone_type='danger'):
        """判断点是否在指定区域内"""
        zones = self.danger_zones if zone_type == 'danger' else self.safe_zones
        for zone in zones:
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                return True
        return False

    def get_zone_status(self, point):
        """获取点的区域状态"""
        in_danger = self.point_in_zone(point, 'danger')
        in_safe = self.point_in_zone(point, 'safe')
        return {
            'in_danger_zone': in_danger,
            'in_safe_zone': in_safe,
            'is_restricted': in_danger and not in_safe
        }

    def draw_zones(self, frame):
        """在图像上绘制区域"""
        annotated = frame.copy()

        for i, zone in enumerate(self.danger_zones):
            cv2.polylines(annotated, [zone], True, (0, 0, 255), 2)
            cv2.putText(annotated, f"DANGER {i+1}", tuple(zone[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for i, zone in enumerate(self.safe_zones):
            cv2.polylines(annotated, [zone], True, (0, 255, 0), 2)
            cv2.putText(annotated, f"SAFE {i+1}", tuple(zone[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated


def main():
    import argparse

    parser = argparse.ArgumentParser(description="持械人员检测器V2")
    parser.add_argument('--pose-model', type=str,
                        default='weights/pose_best.pt')
    parser.add_argument('--weapon-model', type=str,
                        default='weights/cls_best.pt')
    parser.add_argument('--image', type=str, help='测试图片路径')
    parser.add_argument('--video', type=str, help='测试视频路径')
    args = parser.parse_args()

    detector = ArmedPersonDetectorV2(
        args.pose_model,
        args.weapon_model,
        weapon_conf_threshold=0.75,
        temporal_window=15,
        alert_threshold=0.6
    )

    if args.image:
        img = cv2.imread(args.image)
        result = detector.detect(img)
        cv2.imwrite('result_v2.jpg', result['frame'])
        print(f"检测到 {len(result['persons'])} 人, 告警: {result['alerts']}")
        for p in result['persons']:
            print(f"  ID:{p['track_id']} - {'持械' if p['is_armed'] else '正常'} "
                  f"(置信度: L={p['left_wrist']['confidence']:.2f} R={p['right_wrist']['confidence']:.2f})")

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.detect(frame)
            cv2.imshow('Detection', result['frame'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()