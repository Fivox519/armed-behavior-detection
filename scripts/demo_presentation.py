"""
课题组汇报演示脚本 v3.0
- Batch推理：所有手腕裁剪一次性推理，多人场景FPS大幅提升
- 关键点几何约束：高度过滤+运动幅度过滤，物理外挂降误报
- 卡尔曼滤波追踪：简易KF预测，遮挡/交叉时ID不丢失
- Hard Negative Mining：误报裁剪图自动保存，供后续重训
- 时序平滑 + ROI越界时序判定 + 自动加载默认危险区
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import time
import json
from collections import deque


class SimpleKalmanFilter:
    """简易卡尔曼滤波器：4状态 [x, y, vx, vy]"""

    def __init__(self, x, y, dt=1.0):
        self.dt = dt
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 50.0, 50.0])
        self.Q = np.diag([1.0, 1.0, 5.0, 5.0])
        self.R = np.diag([8.0, 8.0])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    def predict(self):
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, z_x, z_y):
        z = np.array([z_x, z_y], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class ROIDrawer:
    def __init__(self, window_name, frame_shape, scale=1.0):
        self.window_name = window_name
        self.scale = scale
        self.points = []
        self.finished = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([int(x / self.scale), int(y / self.scale)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) >= 3:
                self.finished = True

    def draw_preview(self, frame):
        vis = frame.copy()
        if len(self.points) > 0:
            pts = np.array(self.points, np.int32)
            for pt in self.points:
                cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
            if len(self.points) >= 2:
                cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
            if len(self.points) >= 3:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 80))
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        return vis

    def get_polygon(self):
        if self.finished and len(self.points) >= 3:
            return np.array(self.points, np.int32)
        return None


class PresentationDemo:
    def __init__(self, pose_model_path, weapon_model_path, save_fp=False):
        self.pose_model = YOLO(pose_model_path)
        self.weapon_model = YOLO(weapon_model_path)
        self.weapon_classes = [self.weapon_model.names[i] for i in range(len(self.weapon_model.names))]
        # 安全类（不触发告警的硬负样本类别）
        self.safe_classes = {'bottle', 'toy_stick'}
        self.conf_thresholds = {'normal': 0.95, 'armed': 0.40, 'roi': 0.50}

        self.armed_histories = {}
        self.roi_stay_histories = {}
        self.history_len = 15
        self.armed_thresholds = {'normal': 0.95, 'armed': 0.70, 'roi': 0.80}
        self.roi_alert_frame_threshold = 5

        self.next_track_id = 0
        self.kalman_filters = {}
        self.prev_positions = {}
        self.max_lost_frames = 30
        self.lost_counters = {}

        self.wrist_pos_histories = {}

        self.fps_history = deque(maxlen=30)
        self.pose_time_history = deque(maxlen=30)
        self.cls_time_history = deque(maxlen=30)

        self.total_frames = 0
        self.total_persons = 0
        self.total_armed = 0

        self.colors = {
            'person': (0, 255, 0),
            'armed': (0, 0, 255),
            'normal': (0, 255, 0),
            'wrist': (255, 255, 0),
            'filtered': (128, 128, 128),
        }

        self.roi_polygon = None
        self.in_roi_count = 0
        self.roi_alert_count = 0

        self.save_fp = save_fp
        self.fp_save_dir = Path('runs/false_positives/none')
        self.fp_count = 0

        self.height_filtered_count = 0
        self.motion_filtered_count = 0

        # 持物时长记录
        self.weapon_hold_frames = {}  # {track_id: {weapon_class: count}}

        # 分类推理batch统计
        self.cls_batch_sizes = deque(maxlen=30)

    def _assign_track_id(self, cx, cy):
        best_id = None
        best_dist = float('inf')

        for tid, kf in self.kalman_filters.items():
            px, py = kf.predict()
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist and dist < 200:
                best_dist = dist
                best_id = tid

        if best_id is not None:
            self.kalman_filters[best_id].update(cx, cy)
            self.prev_positions[best_id] = (cx, cy)
            self.lost_counters[best_id] = 0
            return best_id

        new_id = self.next_track_id
        self.next_track_id += 1
        self.kalman_filters[new_id] = SimpleKalmanFilter(cx, cy)
        self.prev_positions[new_id] = (cx, cy)
        self.armed_histories[new_id] = deque(maxlen=self.history_len)
        self.roi_stay_histories[new_id] = deque(maxlen=self.history_len)
        self.wrist_pos_histories[new_id] = deque(maxlen=5)
        self.lost_counters[new_id] = 0
        return new_id

    def _cleanup_lost_tracks(self):
        lost_ids = []
        for tid in list(self.lost_counters.keys()):
            self.lost_counters[tid] += 1
            if self.lost_counters[tid] > self.max_lost_frames:
                lost_ids.append(tid)
        for tid in lost_ids:
            self.prev_positions.pop(tid, None)
            self.armed_histories.pop(tid, None)
            self.roi_stay_histories.pop(tid, None)
            self.lost_counters.pop(tid, None)
            self.kalman_filters.pop(tid, None)
            self.wrist_pos_histories.pop(tid, None)
            self.weapon_hold_frames.pop(tid, None)  # 清理持物时长记录

    def _check_wrist_height(self, lw_y, rw_y, kp_data, person_y2):
        hip_y = (kp_data[11][1] + kp_data[12][1]) / 2 if len(kp_data) > 12 else person_y2
        shoulder_y = (kp_data[5][1] + kp_data[6][1]) / 2 if len(kp_data) > 6 else hip_y
        max_valid_y = hip_y + (person_y2 - hip_y) * 0.3
        lw_ok = lw_y < max_valid_y
        rw_ok = rw_y < max_valid_y
        return lw_ok, rw_ok

    def _check_wrist_motion(self, track_id, lw_x, lw_y, rw_x, rw_y):
        if track_id not in self.wrist_pos_histories:
            self.wrist_pos_histories[track_id] = deque(maxlen=5)
            self.wrist_pos_histories[track_id].append((lw_x, lw_y, rw_x, rw_y))
            return 'normal', 'normal'

        hist = self.wrist_pos_histories[track_id]
        hist.append((lw_x, lw_y, rw_x, rw_y))

        if len(hist) < 3:
            return 'normal', 'normal'

        lw_disp = 0
        rw_disp = 0
        for j in range(1, len(hist)):
            lw_disp += ((hist[j][0] - hist[j - 1][0]) ** 2 + (hist[j][1] - hist[j - 1][1]) ** 2) ** 0.5
            rw_disp += ((hist[j][2] - hist[j - 1][2]) ** 2 + (hist[j][3] - hist[j - 1][3]) ** 2) ** 0.5

        avg_lw = lw_disp / (len(hist) - 1)
        avg_rw = rw_disp / (len(hist) - 1)

        def get_motion_state(disp):
            if disp < 3.0:
                return 'still'      # 静止，跳过
            elif disp < 30.0:
                return 'normal'     # 正常持握，进入分类
            else:
                return 'swinging'   # 挥动动作，直接升级告警

        lw_state = get_motion_state(avg_lw)
        rw_state = get_motion_state(avg_rw)

        return lw_state, rw_state

    def _compute_arm_features(self, kp_data, side='right'):
        """
        计算手臂几何特征
        
        Args:
            kp_data: 关键点数据
            side: 'right' 或 'left'
            
        Returns:
            dict: elbow_angle, wrist_height_ratio, arm_extension
        """
        if side == 'right':
            shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10
        else:
            shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9

        if len(kp_data) <= max(shoulder_idx, elbow_idx, wrist_idx):
            return {'elbow_angle': 180, 'wrist_height_ratio': 0.5, 'arm_extension': 0.5}

        # 获取关键点坐标
        shoulder = kp_data[shoulder_idx]
        elbow = kp_data[elbow_idx]
        wrist = kp_data[wrist_idx]

        # 计算人体高度（头顶到脚底）
        head_y = kp_data[0][1] if len(kp_data) > 0 else 0
        foot_y = max(kp_data[15][1], kp_data[16][1]) if len(kp_data) > 16 else 1
        body_height = max(1, foot_y - head_y)

        # 计算肘部角度
        v1 = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
        v2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0]**2 + v1[1]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2)**0.5
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
            elbow_angle = np.degrees(np.arccos(cos_angle))
        else:
            elbow_angle = 180

        # 计算手腕高度比例（越小=手举得越高）
        wrist_height_ratio = (wrist[1] - head_y) / body_height if body_height > 0 else 0.5
        wrist_height_ratio = max(0, min(1, wrist_height_ratio))

        # 计算手臂伸展程度
        shoulder_to_wrist_dist = ((wrist[0] - shoulder[0])**2 + (wrist[1] - shoulder[1])**2)**0.5
        arm_extension = shoulder_to_wrist_dist / body_height if body_height > 0 else 0.5
        arm_extension = max(0, min(1, arm_extension))

        return {
            'elbow_angle': elbow_angle,
            'wrist_height_ratio': wrist_height_ratio,
            'arm_extension': arm_extension
        }

    def _check_behavior_sequence(self, track_id, side='right'):
        """
        基于轨迹历史判断行为类型
        
        Args:
            track_id: 追踪ID
            side: 'right' 或 'left'
            
        Returns:
            str: 'swing'（挥动）/ 'carry'（携带）/ 'stationary'（静止）
        """
        if track_id not in self.wrist_pos_histories:
            return 'stationary'

        hist = self.wrist_pos_histories[track_id]
        if len(hist) < 5:
            return 'stationary'

        # 根据side选择手腕坐标
        if side == 'right':
            coords = [(h[2], h[3]) for h in hist]  # rw_x, rw_y
        else:
            coords = [(h[0], h[1]) for h in hist]  # lw_x, lw_y

        # 计算移动方向一致性（挥舞通常是单向持续运动）
        directions = []
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx)
                directions.append(angle)

        if len(directions) < 2:
            return 'stationary'

        # 计算方向一致性（方差小=方向一致=可能在挥舞）
        dir_std = np.std(directions)
        
        # 计算总位移
        total_disp = 0
        for i in range(1, len(coords)):
            total_disp += ((coords[i][0] - coords[i-1][0])**2 + 
                         (coords[i][1] - coords[i-1][1])**2)**0.5
        avg_disp = total_disp / (len(coords) - 1)

        # 计算高度方差（方差大=上下挥动）
        heights = [c[1] for c in coords]
        height_var = np.var(heights)

        # 判断行为类型
        if avg_disp < 3.0:
            return 'stationary'
        elif dir_std < 0.5 and (avg_disp > 20 or height_var > 100):
            return 'swing'
        else:
            return 'carry'

    def _two_stage_decision(self, weapon_prob, none_prob, behavior_type, elbow_angle, wrist_height_ratio, mode='normal'):
        """
        两阶段决策树判断告警级别
        
        Args:
            weapon_prob: 武器概率
            none_prob: none概率
            behavior_type: 行为类型
            elbow_angle: 肘部角度
            wrist_height_ratio: 手腕高度比例
            mode: 检测模式（影响阈值设置）
            
        Returns:
            str: 'CONFIRMED' / 'SUSPECTED' / 'NORMAL'
        """
        # 根据模式调整阈值
        if mode in ['armed', 'roi']:
            # 持械检测模式：更宽松，提高召回率
            appearance_threshold = 0.3
            diff_threshold = 0.05
            wrist_height_thresh = 0.6
            elbow_angle_thresh = 150
            confirmed_thresh = 0.7
        else:
            # 正常模式：更严格，降低误报率
            appearance_threshold = 0.5
            diff_threshold = 0.25
            wrist_height_thresh = 0.35
            elbow_angle_thresh = 100
            confirmed_thresh = 0.85
        
        # 第一阶段：外观（分类器）
        appearance_suspicious = weapon_prob >= appearance_threshold and (weapon_prob - none_prob) >= diff_threshold
        
        # 第二阶段：行为（至少满足其中一个）
        # 持械模式下行为条件更宽松
        if mode in ['armed', 'roi']:
            behavior_suspicious = (
                behavior_type == 'swing' or          # 正在挥动
                behavior_type == 'carry' or          # 携带（持械模式下也算）
                wrist_height_ratio < wrist_height_thresh or  # 手举得比较高
                elbow_angle < elbow_angle_thresh     # 手臂明显弯曲（握持姿势）
            )
        else:
            behavior_suspicious = (
                behavior_type == 'swing' or          # 正在挥动
                wrist_height_ratio < wrist_height_thresh or  # 手举得比较高
                elbow_angle < elbow_angle_thresh     # 手臂明显弯曲（握持姿势）
            )
        
        # 两阶段决策
        if appearance_suspicious and behavior_suspicious:
            return 'CONFIRMED' if weapon_prob >= confirmed_thresh else 'SUSPECTED'
        elif appearance_suspicious and not behavior_suspicious:
            return 'SUSPECTED'   # 只有外观可疑，降级为疑似
        else:
            return 'NORMAL'

    def _save_false_positive(self, crop, label, conf):
        if not self.save_fp:
            return
        self.fp_save_dir.mkdir(parents=True, exist_ok=True)
        self.fp_count += 1
        fname = f"fp_{self.fp_count:06d}_{label}_{conf:.2f}.jpg"
        cv2.imwrite(str(self.fp_save_dir / fname), crop)

    def is_point_in_roi(self, x, y):
        if self.roi_polygon is None:
            return False
        return cv2.pointPolygonTest(self.roi_polygon, (x, y), False) >= 0

    def draw_roi(self, frame, alert_active=False):
        if self.roi_polygon is not None:
            overlay = frame.copy()
            fill_color = (0, 0, 200) if alert_active else (0, 0, 80)
            cv2.fillPoly(overlay, [self.roi_polygon], fill_color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            border_color = (0, 0, 255) if alert_active else (0, 128, 255)
            border_width = 4 if alert_active else 3
            cv2.polylines(frame, [self.roi_polygon], True, border_color, border_width)
            label = "DANGER ZONE - ALERT!" if alert_active else "DANGER ZONE"
            label_color = (0, 0, 255) if alert_active else (0, 128, 255)
            cv2.putText(frame, label,
                        (int(self.roi_polygon[0][0] + 10), int(self.roi_polygon[0][1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

    def _batch_classify(self, crop_list, current_threshold, mode='normal'):
        if not crop_list:
            return []

        batch_results = self.weapon_model(crop_list, verbose=False)

        outputs = []
        weapon_indices = [i for i, name in enumerate(self.weapon_classes) if name not in ('none',) and name not in self.safe_classes]
        safe_indices = [i for i, name in enumerate(self.weapon_classes) if name in self.safe_classes]
        none_idx = next((i for i, name in enumerate(self.weapon_classes) if name == 'none'), -1)
        
        # 根据模式调整差异阈值
        diff_threshold = 0.1 if mode in ['armed', 'roi'] else 0.3

        for res in batch_results:
            if res.probs is not None:
                raw = res.probs.data.cpu().numpy()
                probs = raw[0] if raw.ndim > 1 else raw
                cls_idx = int(res.probs.top1)
                cls_conf = float(res.probs.top1conf)
                cls_name = self.weapon_classes[cls_idx] if cls_idx < len(self.weapon_classes) else 'none'
                none_prob = probs[none_idx] if none_idx >= 0 and none_idx < probs.shape[0] else 0.0

                # 安全类别（bottle/toy_stick）- 直接返回Safe标记
                if cls_name in self.safe_classes and cls_conf >= 0.3:
                    outputs.append((cls_name, cls_conf, probs, True))  # True = safe
                    continue

                # 检查安全类别概率是否高于武器类（防止bottle/toy_stick被误判为武器）
                max_safe_prob = 0.0
                max_safe_name = None
                for si in safe_indices:
                    if si < probs.shape[0] and probs[si] > max_safe_prob:
                        max_safe_prob = probs[si]
                        max_safe_name = self.weapon_classes[si]
                max_weapon_prob = 0.0
                for wi in weapon_indices:
                    if wi < probs.shape[0] and probs[wi] > max_weapon_prob:
                        max_weapon_prob = probs[wi]
                # 如果safe类概率>0.3且高于武器类概率，标记为safe
                if max_safe_prob >= 0.3 and max_safe_prob > max_weapon_prob and max_safe_name:
                    outputs.append((max_safe_name, max_safe_prob, probs, True))
                    continue

                # 武器类别
                if cls_conf >= current_threshold and cls_idx in weapon_indices:
                    if cls_conf - none_prob >= diff_threshold:
                        outputs.append((self.weapon_classes[cls_idx], cls_conf, probs, False))
                        continue

                max_weapon_idx = -1
                max_weapon_prob = 0.0
                for wi in weapon_indices:
                    if wi < probs.shape[0] and probs[wi] > max_weapon_prob:
                        max_weapon_prob = probs[wi]
                        max_weapon_idx = wi
                if max_weapon_idx >= 0 and max_weapon_prob >= current_threshold:
                    if max_weapon_prob - none_prob >= diff_threshold:
                        outputs.append((self.weapon_classes[max_weapon_idx], max_weapon_prob, probs, False))
                        continue

            outputs.append(('none', 0.0, None, False))

        return outputs

    def detect_frame(self, frame, mode='normal'):
        h, w = frame.shape[:2]
        results = []
        current_threshold = self.conf_thresholds.get(mode, 0.50)

        pose_start = time.time()
        pose_results = self.pose_model(frame, verbose=False)
        pose_time = (time.time() - pose_start) * 1000

        kpts = pose_results[0].keypoints
        boxes = pose_results[0].boxes

        cls_start = time.time()
        any_roi_alert = False

        if boxes is not None:
            self._cleanup_lost_tracks()

            person_data = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['person'], 2)

                if kpts is None or i >= len(kpts):
                    continue
                kp = kpts[i]
                if kp is None or len(kp.data[0]) < 11:
                    continue

                kp_data = kp.data[0].cpu().numpy()
                lw_x, lw_y = int(kp_data[9][0]), int(kp_data[9][1])
                rw_x, rw_y = int(kp_data[10][0]), int(kp_data[10][1])
                lw_kp_conf = kp_data[9][2] if len(kp_data[9]) > 2 else 0
                rw_kp_conf = kp_data[10][2] if len(kp_data[10]) > 2 else 0

                cv2.circle(frame, (lw_x, lw_y), 8, self.colors['wrist'], -1)
                cv2.circle(frame, (rw_x, rw_y), 8, self.colors['wrist'], -1)

                person_cx = (x1 + x2) // 2
                person_cy = (y1 + y2) // 2
                track_id = self._assign_track_id(person_cx, person_cy)

                lw_height_ok, rw_height_ok = self._check_wrist_height(lw_y, rw_y, kp_data, y2)
                lw_motion_state, rw_motion_state = self._check_wrist_motion(track_id, lw_x, lw_y, rw_x, rw_y)

                if not lw_height_ok:
                    self.height_filtered_count += 1
                    cv2.circle(frame, (lw_x, lw_y), 8, self.colors['filtered'], 2)
                if not rw_height_ok:
                    self.height_filtered_count += 1
                    cv2.circle(frame, (rw_x, rw_y), 8, self.colors['filtered'], 2)
                if lw_motion_state == 'still':
                    self.motion_filtered_count += 1
                if rw_motion_state == 'still':
                    self.motion_filtered_count += 1

                lw_is_swinging = lw_motion_state == 'swinging'
                rw_is_swinging = rw_motion_state == 'swinging'

                lw_should_classify = (lw_kp_conf >= 0.5) and lw_height_ok and lw_motion_state != 'still'
                rw_should_classify = (rw_kp_conf >= 0.5) and rw_height_ok and rw_motion_state != 'still'

                person_h = y2 - y1
                person_w = x2 - x1
                crop_size = max(224, int(max(person_w, person_h) * 0.5))
                padding = int(crop_size * 0.35)

                def crop_wrist(x, y, cs=crop_size, pad=padding):
                    cx1 = max(0, int(x - cs // 2 - pad))
                    cy1 = max(0, int(y - cs // 2 - pad))
                    cx2 = min(w, int(x + cs // 2 + pad))
                    cy2 = min(h, int(y + cs // 2 + pad))
                    return frame[cy1:cy2, cx1:cx2]

                lw_crop = crop_wrist(lw_x, lw_y) if lw_should_classify else None
                rw_crop = crop_wrist(rw_x, rw_y) if rw_should_classify else None

                person_data.append({
                    'i': i, 'track_id': track_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'lw_x': lw_x, 'lw_y': lw_y, 'rw_x': rw_x, 'rw_y': rw_y,
                    'lw_should': lw_should_classify, 'rw_should': rw_should_classify,
                    'lw_crop': lw_crop, 'rw_crop': rw_crop,
                    'lw_kp_conf': lw_kp_conf, 'rw_kp_conf': rw_kp_conf,
                    'lw_is_swinging': lw_is_swinging, 'rw_is_swinging': rw_is_swinging,
                    'kp_data': kp_data,  # 保存关键点数据用于后续行为分析
                })

            batch_crops = []
            crop_indices = []
            for pd in person_data:
                if pd['lw_should'] and pd['lw_crop'] is not None and pd['lw_crop'].size > 0:
                    batch_crops.append(cv2.resize(pd['lw_crop'], (224, 224)))
                    crop_indices.append((pd['track_id'], 'lw'))
                if pd['rw_should'] and pd['rw_crop'] is not None and pd['rw_crop'].size > 0:
                    batch_crops.append(cv2.resize(pd['rw_crop'], (224, 224)))
                    crop_indices.append((pd['track_id'], 'rw'))

            batch_outputs = self._batch_classify(batch_crops, current_threshold, mode) if batch_crops else []
            self.cls_batch_sizes.append(len(batch_crops))

            cls_map = {}
            for (tid, side), output in zip(crop_indices, batch_outputs):
                # output = (cls_name, conf, probs, is_safe)
                if len(output) == 4:
                    cls_name, conf, probs, is_safe = output
                else:
                    cls_name, conf, probs = output
                    is_safe = cls_name in self.safe_classes
                cls_map[(tid, side)] = (cls_name, conf, probs, is_safe)

            for pd in person_data:
                track_id = pd['track_id']
                x1, y1, x2, y2 = pd['x1'], pd['y1'], pd['x2'], pd['y2']
                kp_data = pd['kp_data']

                lw_cls, lw_conf, lw_probs, lw_is_safe = cls_map.get((track_id, 'lw'), ('none', 0.0, None, False))
                rw_cls, rw_conf, rw_probs, rw_is_safe = cls_map.get((track_id, 'rw'), ('none', 0.0, None, False))

                if not pd['lw_should']:
                    lw_cls, lw_conf, lw_is_safe = 'none', 0.0, False
                if not pd['rw_should']:
                    rw_cls, rw_conf, rw_is_safe = 'none', 0.0, False

                lw_is_swinging = pd.get('lw_is_swinging', False)
                rw_is_swinging = pd.get('rw_is_swinging', False)

                if lw_cls != 'none' and not lw_is_safe and self.save_fp and mode == 'normal':
                    if pd['lw_crop'] is not None and pd['lw_crop'].size > 0:
                        self._save_false_positive(pd['lw_crop'], lw_cls, lw_conf)
                if rw_cls != 'none' and not rw_is_safe and self.save_fp and mode == 'normal':
                    if pd['rw_crop'] is not None and pd['rw_crop'].size > 0:
                        self._save_false_positive(pd['rw_crop'], rw_cls, rw_conf)

                # 获取none概率
                none_idx = next((i for i, name in enumerate(self.weapon_classes) if name == 'none'), -1)
                lw_none_prob = lw_probs[none_idx] if lw_probs is not None and none_idx >= 0 else 0.0
                rw_none_prob = rw_probs[none_idx] if rw_probs is not None and none_idx >= 0 else 0.0

                # 计算手臂特征
                lw_features = self._compute_arm_features(kp_data, side='left')
                rw_features = self._compute_arm_features(kp_data, side='right')

                # 判断行为序列
                lw_behavior = self._check_behavior_sequence(track_id, side='left')
                rw_behavior = self._check_behavior_sequence(track_id, side='right')

                # 选择置信度更高的手腕进行最终判断
                # 安全类别优先判断
                any_safe = lw_is_safe or rw_is_safe
                safe_cls = lw_cls if lw_is_safe else (rw_cls if rw_is_safe else None)

                if any_safe:
                    # 安全类别（bottle/toy_stick），直接标记为SAFE
                    alert_level = "SAFE"
                    dominant_cls = safe_cls
                    weapon_prob = lw_conf if lw_is_safe else rw_conf
                    hold_count = 0
                elif lw_conf >= rw_conf:
                    weapon_prob, none_prob = lw_conf, lw_none_prob
                    behavior_type = lw_behavior
                    elbow_angle = lw_features['elbow_angle']
                    wrist_height_ratio = lw_features['wrist_height_ratio']
                    dominant_cls = lw_cls
                else:
                    weapon_prob, none_prob = rw_conf, rw_none_prob
                    behavior_type = rw_behavior
                    elbow_angle = rw_features['elbow_angle']
                    wrist_height_ratio = rw_features['wrist_height_ratio']
                    dominant_cls = rw_cls

                # 持物时长过滤
                if track_id not in self.weapon_hold_frames:
                    self.weapon_hold_frames[track_id] = {}

                hold_count = 0
                if dominant_cls != 'none' and dominant_cls not in self.safe_classes and weapon_prob > 0.3:
                    if dominant_cls not in self.weapon_hold_frames[track_id]:
                        self.weapon_hold_frames[track_id][dominant_cls] = 0
                    self.weapon_hold_frames[track_id][dominant_cls] += 1
                    hold_count = self.weapon_hold_frames[track_id][dominant_cls]
                    
                    # 清零其他武器类别的计数
                    for cls in list(self.weapon_hold_frames[track_id].keys()):
                        if cls != dominant_cls:
                            self.weapon_hold_frames[track_id][cls] = 0
                elif not any_safe:
                    # 当前帧判为none，所有武器计数归零
                    self.weapon_hold_frames[track_id] = {}

                # 只有连续多帧以上才进入时序滑动窗口（持械模式更宽松）
                if any_safe:
                    alert_level = "SAFE"
                elif hold_count < (2 if mode in ['armed', 'roi'] else 5):
                    alert_level = "NORMAL"
                else:
                    # 使用两阶段决策
                    alert_level = self._two_stage_decision(weapon_prob, none_prob, behavior_type, elbow_angle, wrist_height_ratio, mode)

                current_armed = alert_level in ["CONFIRMED", "SUSPECTED", "SWINGING"]
                if track_id not in self.armed_histories:
                    self.armed_histories[track_id] = deque(maxlen=self.history_len)
                self.armed_histories[track_id].append(current_armed)

                armed_ratio = sum(self.armed_histories[track_id]) / len(self.armed_histories[track_id]) if self.armed_histories[track_id] else 0
                is_armed = armed_ratio >= self.armed_thresholds.get(mode, 0.85)

                person_cx = (x1 + x2) // 2
                person_cy = (y1 + y2) // 2
                in_roi = self.is_point_in_roi(person_cx, person_cy)

                if track_id not in self.roi_stay_histories:
                    self.roi_stay_histories[track_id] = deque(maxlen=self.history_len)
                self.roi_stay_histories[track_id].append(in_roi)

                roi_stay_count = sum(self.roi_stay_histories[track_id])
                roi_armed_count = sum(
                    1 for a, r in zip(self.armed_histories[track_id], self.roi_stay_histories[track_id])
                    if a and r
                ) if len(self.armed_histories[track_id]) == len(self.roi_stay_histories[track_id]) else 0

                roi_stay_exceeds = roi_stay_count >= self.roi_alert_frame_threshold
                roi_armed_exceeds = roi_armed_count >= self.roi_alert_frame_threshold

                if lw_cls != 'none' and not lw_is_safe:
                    level_str = f" SWING" if lw_is_swinging else ""
                    cv2.putText(frame, f"L:{lw_cls} {lw_conf:.2f}{level_str}", (pd['lw_x'] + 10, pd['lw_y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif lw_is_safe:
                    cv2.putText(frame, f"L:[Safe:{lw_cls}] {lw_conf:.2f}", (pd['lw_x'] + 10, pd['lw_y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                if rw_cls != 'none' and not rw_is_safe:
                    level_str = f" SWING" if rw_is_swinging else ""
                    cv2.putText(frame, f"R:{rw_cls} {rw_conf:.2f}{level_str}", (pd['rw_x'] + 10, pd['rw_y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif rw_is_safe:
                    cv2.putText(frame, f"R:[Safe:{rw_cls}] {rw_conf:.2f}", (pd['rw_x'] + 10, pd['rw_y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                if lw_is_swinging:
                    cv2.putText(frame, "SWING!", (pd['lw_x'] - 20, pd['lw_y'] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if rw_is_swinging:
                    cv2.putText(frame, "SWING!", (pd['rw_x'] - 20, pd['rw_y'] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                # 显示持物计数（调试用）
                if hold_count > 0:
                    cv2.putText(frame, f"hold:{hold_count}/5", (x1, y2 + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if mode == 'roi':
                    if in_roi and is_armed and roi_armed_exceeds:
                        status = "ROI ALERT!"
                        status_color = (0, 0, 255)
                        self.roi_alert_count += 1
                        any_roi_alert = True
                        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 4)
                    elif in_roi and is_armed and not roi_armed_exceeds:
                        status = f"Armed in ROI ({roi_armed_count}/{self.roi_alert_frame_threshold})"
                        status_color = (0, 165, 255)
                    elif in_roi and roi_stay_exceeds:
                        status = "IN ZONE"
                        status_color = (0, 255, 255)
                        self.in_roi_count += 1
                    elif in_roi:
                        status = f"Entering ({roi_stay_count}/{self.roi_alert_frame_threshold})"
                        status_color = (0, 200, 200)
                    elif is_armed:
                        status = f"{alert_level} (Outside)"
                        if alert_level == "SWINGING":
                            status_color = (0, 0, 255)
                        elif alert_level == "CONFIRMED":
                            status_color = (0, 0, 200)
                        elif alert_level == "SUSPECTED":
                            status_color = (0, 165, 255)
                        else:
                            status_color = (0, 128, 255)
                    else:
                        status = "Normal"
                        status_color = self.colors['normal']
                else:
                    if alert_level == "SAFE":
                        safe_label = dominant_cls.upper() if dominant_cls else "SAFE"
                        status = f"Safe [{safe_label}]"
                        status_color = (0, 200, 0)
                        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 200, 0), 2)
                    elif alert_level == "SWINGING":
                        status = "SWINGING!"
                        status_color = (0, 0, 255)
                        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 4)
                    elif alert_level == "CONFIRMED":
                        status = "CONFIRMED"
                        status_color = (0, 0, 200)
                        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 200), 3)
                    elif alert_level == "SUSPECTED":
                        status = "SUSPECTED"
                        status_color = (0, 165, 255)
                    else:
                        status = "Normal"
                        status_color = self.colors['normal']

                cv2.putText(frame, f"ID:{track_id} {status}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                if in_roi:
                    cv2.putText(frame, f"[IN ROI] stay:{roi_stay_count}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                results.append({
                    'person_id': pd['i'],
                    'track_id': track_id,
                    'is_armed': is_armed,
                    'in_roi': in_roi,
                    'roi_stay_count': roi_stay_count,
                    'roi_armed_count': roi_armed_count,
                    'left_wrist': {'class': lw_cls, 'confidence': lw_conf},
                    'right_wrist': {'class': rw_cls, 'confidence': rw_conf}
                })

        cls_time = (time.time() - cls_start) * 1000

        self.pose_time_history.append(pose_time)
        self.cls_time_history.append(cls_time)
        self.total_frames += 1
        self.total_persons += len(results)
        self.total_armed += sum(1 for r in results if r['is_armed'])

        return frame, results, {
            'pose_time': pose_time,
            'cls_time': cls_time,
            'any_roi_alert': any_roi_alert,
        }

    def draw_info_panel(self, frame, fps, mode):
        h, w = frame.shape[:2]

        panel_h = 180 if mode == 'roi' else 155
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, 10), (w - 10, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 35
        cv2.putText(frame, f"Mode: {mode.upper()}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25
        cv2.putText(frame, f"Persons: {self.total_persons}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(frame, f"Armed: {self.total_armed}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 25
        cv2.putText(frame, f"H-Filter: {self.height_filtered_count}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)
        y += 22
        cv2.putText(frame, f"M-Filter: {self.motion_filtered_count}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

        # 显示分类batch信息
        if self.cls_batch_sizes:
            avg_batch = np.mean(list(self.cls_batch_sizes))
            y += 22
            cv2.putText(frame, f"Cls-Batch: {avg_batch:.1f}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

        if mode == 'roi':
            y += 22
            cv2.putText(frame, f"ROI Alerts: {self.roi_alert_count}", (w - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def init_roi_interactive(self, frame, window_name, scale):
        h, w = frame.shape[:2]
        drawer = ROIDrawer(window_name, frame.shape, scale)
        cv2.setMouseCallback(window_name, drawer.mouse_callback)

        print("\n" + "=" * 60)
        print("  ROI Drawing Mode")
        print("  Left-click: Add point | Right-click: Finish | 'c': Clear | 'd': Default")
        print("=" * 60)

        while not drawer.finished:
            preview_frame = drawer.draw_preview(frame.copy())
            if scale != 1.0:
                display_frame = cv2.resize(preview_frame, (int(w * scale), int(h * scale)))
            else:
                display_frame = preview_frame
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('c'):
                drawer.points = []
                drawer.finished = False
            elif key == ord('d'):
                drawer.finished = True
                self.roi_polygon = np.array([
                    [int(w * 0.1), int(h * 0.6)],
                    [int(w * 0.9), int(h * 0.6)],
                    [int(w * 0.9), int(h * 0.95)],
                    [int(w * 0.1), int(h * 0.95)],
                ], np.int32)
                cv2.setMouseCallback(window_name, lambda *a: None)
                return
            elif key == ord('q'):
                cv2.setMouseCallback(window_name, lambda *a: None)
                return

        polygon = drawer.get_polygon()
        if polygon is not None:
            self.roi_polygon = polygon
            self._save_roi_config(polygon)
        else:
            self.roi_polygon = np.array([
                [int(w * 0.1), int(h * 0.6)],
                [int(w * 0.9), int(h * 0.6)],
                [int(w * 0.9), int(h * 0.95)],
                [int(w * 0.1), int(h * 0.95)],
            ], np.int32)
        cv2.setMouseCallback(window_name, lambda *a: None)

    def _save_roi_config(self, polygon):
        config_path = Path('configs/roi_config.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {"rois": []}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass
        config["rois"].append({"roi_points": polygon.tolist()})
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _load_roi_config(self):
        config_path = Path('configs/roi_config.json')
        if not config_path.exists():
            return None
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if "rois" in config and len(config["rois"]) > 0:
                return np.array(config["rois"][-1]["roi_points"], np.int32)
        except Exception:
            pass
        return None

    def run_demo(self, mode='normal', video_path=None, max_width=1280, max_height=720):
        # 判断是否为摄像头输入
        is_camera = video_path is not None and video_path.isdigit()
        if video_path and not is_camera:
            input_path = video_path
        elif is_camera:
            input_path = int(video_path)  # cv2.VideoCapture接受整数
        else:
            demo_dir = Path('demo/videos')
            mode_file = {'normal': 'demo_normal.mp4', 'armed': 'demo_armed_action.mp4', 'roi': 'demo_roi_rules.mp4'}
            candidate = demo_dir / mode_file.get(mode, 'demo_normal.mp4')
            if candidate.exists():
                input_path = str(candidate)
            else:
                # 回退：选取demo/videos下第一个视频文件
                video_files = list(demo_dir.glob('*.mp4')) + list(demo_dir.glob('*.avi'))
                if not video_files:
                    print(f"No video files found in {demo_dir}")
                    return
                video_files.sort(key=lambda x: x.name)
                input_path = str(video_files[0])
            print(f"  Selected video: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open: {input_path}")
            return

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(max_width / orig_width, max_height / orig_height)
        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)

        window_name = 'Presentation Demo'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        print("=" * 60)
        print(f"  Presentation Demo v3.0 - Mode: {mode.upper()}")
        print("=" * 60)
        print(f"  Input: {input_path}")
        print(f"  Original: {orig_width}x{orig_height} -> Display: {display_width}x{display_height}")
        print(f"  Optimizations: Batch Inference | Kalman Tracking | Keypoint Filter | Hard Neg Mining")
        if self.save_fp:
            print(f"  False Positive Save: ON -> {self.fp_save_dir}")
        print(f"  Press 'q' quit | 'r' restart" + (" | 'p' redraw ROI" if mode == 'roi' else ""))
        print("=" * 60)

        if mode == 'roi':
            ret, first_frame = cap.read()
            if ret:
                saved_roi = self._load_roi_config()
                if saved_roi is not None:
                    self.roi_polygon = saved_roi
                    print(f"  ROI: Loaded saved polygon ({len(saved_roi)} points)")
                else:
                    fh, fw = first_frame.shape[:2]
                    self.roi_polygon = np.array([
                        [int(fw * 0.1), int(fh * 0.6)],
                        [int(fw * 0.9), int(fh * 0.6)],
                        [int(fw * 0.9), int(fh * 0.95)],
                        [int(fw * 0.1), int(fh * 0.95)],
                    ], np.int32)
                    self._save_roi_config(self.roi_polygon)
                    print("  ROI: Auto-loaded default danger zone (bottom 60%-95%)")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    print("  Camera disconnected!")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_start = time.time()
            frame, results, perf = self.detect_frame(frame, mode)
            frame_time = (time.time() - frame_start) * 1000
            self.fps_history.append(frame_time)

            fps = 1000.0 / np.mean(self.fps_history) if self.fps_history else 0

            if mode == 'roi':
                self.draw_roi(frame, alert_active=perf.get('any_roi_alert', False))

            frame = self.draw_info_panel(frame, fps, mode)

            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not is_camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.total_frames = 0
                self.total_persons = 0
                self.total_armed = 0
                self.in_roi_count = 0
                self.roi_alert_count = 0
                self.height_filtered_count = 0
                self.motion_filtered_count = 0
                self.armed_histories = {}
                self.roi_stay_histories = {}
                self.wrist_pos_histories = {}
                self.next_track_id = 0
                self.prev_positions = {}
                self.lost_counters = {}
                self.kalman_filters = {}
            elif key == ord('p') and mode == 'roi':
                self.roi_polygon = None
                ret, first_frame = cap.read()
                if ret:
                    self.init_roi_interactive(first_frame, window_name, scale)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("  Demo Statistics")
        print("=" * 60)
        print(f"  Total frames: {self.total_frames}")
        print(f"  Total persons detected: {self.total_persons}")
        print(f"  Armed detections: {self.total_armed}")
        print(f"  Height-filtered wrists: {self.height_filtered_count}")
        print(f"  Motion-filtered wrists: {self.motion_filtered_count}")
        if self.save_fp:
            print(f"  False positives saved: {self.fp_count} -> {self.fp_save_dir}")
        if mode == 'roi':
            print(f"  Persons in ROI: {self.in_roi_count}")
            print(f"  ROI Alerts: {self.roi_alert_count}")
        print(f"  Average FPS: {fps:.1f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Presentation Demo v3.0")
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'armed', 'roi'])
    parser.add_argument('--video', type=str, help='Custom video path')
    parser.add_argument('--camera', type=int, default=None, help='Camera device index (e.g. 0)')
    parser.add_argument('--pose-model', type=str,
                        default='weights/pose_best.pt')
    parser.add_argument('--weapon-model', type=str,
                        default='weights/cls_best.pt',
                        help='8-class weapon classifier (includes bottle/toy_stick safe classes)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--save-fp', action='store_true',
                        help='Save false positive crops for Hard Negative Mining')
    args = parser.parse_args()

    demo = PresentationDemo(args.pose_model, args.weapon_model, save_fp=args.save_fp)
    if args.camera is not None:
        demo.run_demo(args.mode, str(args.camera), args.width, args.height)
    else:
        demo.run_demo(args.mode, args.video, args.width, args.height)


if __name__ == "__main__":
    main()
