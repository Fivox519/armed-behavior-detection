"""
鼠标交互式 ROI 定义工具

用法：
  python scripts/roi_editor.py --video demo/videos/demo_roi_rules.mp4

操作说明：
  1. 视频播放后暂停（按空格键）
  2. 鼠标左键点击画面，逐点定义 ROI 多边形
  3. 右键点击闭合多边形并保存
  4. 按 'r' 重新定义
  5. 按 'q' 退出并保存

保存位置：
  configs/roi_config.json
"""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np


CONFIG_PATH = Path("configs/roi_config.json")


class ROIEditor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open: {video_path}")
            return

        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.max_w = 1280
        self.max_h = 720
        self.scale = min(self.max_w / self.orig_w, self.max_h / self.orig_h)
        self.disp_w = int(self.orig_w * self.scale)
        self.disp_h = int(self.orig_h * self.scale)

        self.points = []
        self.roi_polygon = None
        self.is_drawing = False
        self.is_paused = False
        self.frame = None

        self.window_name = "ROI Editor - Click to define danger zone"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((orig_x, orig_y))
            self.is_drawing = True
            print(f"  Point {len(self.points)}: ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) >= 3:
            self.roi_polygon = np.array(self.points, np.int32)
            self.is_drawing = False
            print(f"  ROI polygon closed with {len(self.points)} points")

    def draw_overlay(self, frame):
        display = frame.copy()

        if len(self.points) > 0:
            scaled_pts = [(int(x * self.scale), int(y * self.scale)) for x, y in self.points]
            for i, pt in enumerate(scaled_pts):
                cv2.circle(display, pt, 8, (0, 255, 0), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(scaled_pts) > 1:
                cv2.polylines(display, [np.array(scaled_pts)], False, (0, 255, 255), 2)

        if self.roi_polygon is not None:
            scaled_roi = np.array([(int(x * self.scale), int(y * self.scale))
                                    for x, y in self.roi_polygon], np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [scaled_roi], (0, 0, 100))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.polylines(display, [scaled_roi], True, (0, 0, 255), 3)
            cv2.putText(display, "DANGER ZONE",
                        (scaled_roi[0][0] + 20, scaled_roi[0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        h, w = display.shape[:2]
        cv2.rectangle(display, (10, h - 80), (500, h - 10), (0, 0, 0), -1)
        cv2.putText(display, "Left-click: Add point | Right-click: Close polygon",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Space: Pause/Play | R: Reset | Q: Save & Quit",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def save_roi(self):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "video_path": str(self.video_path),
            "roi_points": self.points,
            "frame_size": [self.orig_w, self.orig_h],
        }

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                existing = json.load(f)
            if "rois" not in existing:
                existing["rois"] = []
            existing["rois"].append(config)
            config = existing
        else:
            config = {"rois": [config]}

        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  ROI saved to: {CONFIG_PATH}")
        print(f"  Points: {self.points}")

    def run(self):
        if not self.cap.isOpened():
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("=" * 60)
        print("  ROI Editor - Interactive Danger Zone Definition")
        print("=" * 60)
        print(f"  Video: {self.video_path}")
        print(f"  Size: {self.orig_w}x{self.orig_h} -> {self.disp_w}x{self.disp_h}")
        print()
        print("  Controls:")
        print("    Left-click:  Add polygon vertex")
        print("    Right-click: Close polygon (need >= 3 points)")
        print("    Space:       Pause / Resume")
        print("    R:           Reset polygon")
        print("    Q:           Save & Quit")
        print("=" * 60)

        while True:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.frame = frame

            if self.frame is not None:
                display = self.draw_overlay(self.frame)
                display = cv2.resize(display, (self.disp_w, self.disp_h))
                cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30 if not self.is_paused else 100) & 0xFF

            if key == ord(' '):
                self.is_paused = not self.is_paused
                print(f"  {'Paused' if self.is_paused else 'Playing'}")

            elif key == ord('r'):
                self.points = []
                self.roi_polygon = None
                self.is_drawing = False
                print("  ROI reset")

            elif key == ord('q'):
                if self.roi_polygon is not None:
                    self.save_roi()
                else:
                    print("  No ROI defined, skipping save")
                break

        self.cap.release()
        cv2.destroyAllWindows()


def load_roi_config(config_path=None):
    """加载 ROI 配置，供 demo_presentation.py 使用"""
    if config_path is None:
        config_path = CONFIG_PATH

    if not Path(config_path).exists():
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    if "rois" in config and len(config["rois"]) > 0:
        latest = config["rois"][-1]
        return np.array(latest["roi_points"], np.int32)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive ROI Editor")
    parser.add_argument('--video', type=str,
                        default='demo/videos/demo_roi_rules.mp4',
                        help='Video file to define ROI on')
    args = parser.parse_args()

    editor = ROIEditor(args.video)
    editor.run()
