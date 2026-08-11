"""
智能裁剪与分类格式转换脚本（增强版）

功能：
  1. 自动检测数据集格式（分类格式 vs 检测格式）
  2. 如果是检测格式（YOLO labels），按 bounding box 裁剪，外扩 10%~20% 防截断
  3. 如果已经是分类格式（按类别分文件夹），直接复制
  4. 裁剪后自动 resize 到 224x224
  5. 负样本数据集（手机、伞等）自动归类到 none 文件夹
  6. 支持自定义类别映射

用法（数据集需自行下载后配置路径）：
  # 转换单个检测格式数据集
  python src/datasets/convert_to_classification.py --src data/raw_dataset/hand_weapons --dst data/classification

  # 转换合并后的数据集
  python src/datasets/convert_to_classification.py --src data/raw_dataset/merged --dst data/classification

  # 指定外扩比例和目标尺寸
  python src/datasets/convert_to_classification.py --src data/raw_dataset/merged --expand 0.15 --size 224

分类格式目录结构：
  datasets/classification/
  ├── train/
  │   ├── knife/
  │   ├── axe/
  │   ├── hammer/
  │   ├── stick/
  │   └── none/
  └── val/
      ├── knife/
      ├── axe/
      ├── hammer/
      ├── stick/
      └── none/
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


WEAPON_CLASSES = {0: 'knife', 1: 'axe', 2: 'hammer', 3: 'stick', 4: 'steel_pipe', 5: 'bottle', 6: 'toy_stick', 7: 'none'}

NEGATIVE_KEYWORDS = ['phone', 'umbrella', 'bottle', 'cup', 'key', 'bag', 'book', 'daily']

CLASS_MAPPING = {
    'knife': 'knife',
    'knives': 'knife',
    'axe': 'axe',
    'axes': 'axe',
    'hammer': 'hammer',
    'hammers': 'hammer',
    'stick': 'stick',
    'sticks': 'stick',
    'bat': 'stick',
    'club': 'stick',
    'pipe': 'steel_pipe',
    'steel_pipe': 'steel_pipe',
    'gun': 'none',
    'pistol': 'none',
    'rifle': 'none',
    'phone': 'none',
    'umbrella': 'none',
    'cup': 'none',
    'key': 'none',
    'bag': 'none',
    'book': 'none',
    'person': 'none',
    'hand': 'none',
    'bottle': 'bottle',
    'water_bottle': 'bottle',
    'thermos': 'bottle',
    'drink': 'bottle',
    'toy_stick': 'toy_stick',
    'toy': 'toy_stick',
    'bubble': 'toy_stick',
    'glow_stick': 'toy_stick',
    'wand': 'toy_stick',
}


def detect_format(src_dir):
    """检测数据集格式：classification / detection / unknown"""
    src = Path(src_dir)

    for split in ['train', 'val']:
        split_dir = src / split
        if not split_dir.exists():
            continue

        subdirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
        if 'images' in subdirs and 'labels' in subdirs:
            return 'detection'

        class_dirs = [d for d in subdirs if d in WEAPON_CLASSES.values()]
        if class_dirs:
            return 'classification'

    return 'unknown'


def crop_bbox(image, x_center, y_center, width, height, expand=0.15):
    """按 bounding box 裁剪，外扩 expand 比例防止截断"""
    h, w = image.shape[:2]

    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * expand))
    y1 = max(0, int(y1 - bh * expand))
    x2 = min(w, int(x2 + bw * expand))
    y2 = min(h, int(y2 + bh * expand))

    return image[y1:y2, x1:x2]


def map_class(raw_class_name, force_class=None):
    """将原始类别名映射到标准5类"""
    if force_class:
        return force_class

    raw_lower = raw_class_name.lower().strip()
    if raw_lower in CLASS_MAPPING:
        return CLASS_MAPPING[raw_lower]

    for keyword, target in CLASS_MAPPING.items():
        if keyword in raw_lower:
            return target

    return 'none'


def convert_detection_to_classification(src_dir, dst_dir, target_size=224, expand=0.15, force_class=None):
    """将 YOLO 检测格式转换为分类格式（含智能裁剪）"""
    src = Path(src_dir)
    dst = Path(dst_dir)

    stats = {}

    for split in ['train', 'val', 'test']:
        images_dir = src / split / 'images'
        labels_dir = src / split / 'labels'

        if not images_dir.exists():
            images_dir = src / split
            labels_dir = src / (split + '_labels') if (src / (split + '_labels')).exists() else None

        if not images_dir.exists():
            continue

        split_name = 'train' if split == 'test' else split

        for cls_name in WEAPON_CLASSES.values():
            (dst / split_name / cls_name).mkdir(parents=True, exist_ok=True)

        stats[split_name] = {c: 0 for c in WEAPON_CLASSES.values()}

        for img_file in sorted(images_dir.glob('*.*')):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            image = cv2.imread(str(img_file))
            if image is None:
                continue

            label_file = None
            if labels_dir:
                label_file = labels_dir / (img_file.stem + '.txt')

            if label_file and label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    class_id = int(parts[0])

                    raw_class_name = WEAPON_CLASSES.get(class_id, str(class_id))
                    cls_name = map_class(raw_class_name, force_class)

                    if len(parts) >= 5:
                        x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        cropped = crop_bbox(image, x_c, y_c, bw, bh, expand)

                        if cropped.size == 0:
                            continue

                        cropped = cv2.resize(cropped, (target_size, target_size))

                        suffix = f"_{line_idx}" if line_idx > 0 else ""
                        dst_name = f"{img_file.stem}{suffix}.jpg"
                        dst_path = dst / split_name / cls_name / dst_name
                        cv2.imwrite(str(dst_path), cropped)
                        stats[split_name][cls_name] = stats[split_name].get(cls_name, 0) + 1
                    else:
                        cls_name = force_class if force_class else 'none'
                        resized = cv2.resize(image, (target_size, target_size))
                        dst_name = f"{img_file.stem}.jpg"
                        dst_path = dst / split_name / cls_name / dst_name
                        cv2.imwrite(str(dst_path), resized)
                        stats[split_name][cls_name] = stats[split_name].get(cls_name, 0) + 1
            else:
                cls_name = force_class if force_class else 'none'
                resized = cv2.resize(image, (target_size, target_size))
                dst_name = f"{img_file.stem}.jpg"
                dst_path = dst / split_name / cls_name / dst_name
                cv2.imwrite(str(dst_path), resized)
                stats[split_name][cls_name] = stats[split_name].get(cls_name, 0) + 1

    return stats


def convert_classification_to_classification(src_dir, dst_dir, target_size=224, force_class=None):
    """将已有的分类格式数据集复制并标准化"""
    src = Path(src_dir)
    dst = Path(dst_dir)

    stats = {}

    for split in ['train', 'val', 'test']:
        split_dir = src / split
        if not split_dir.exists():
            continue

        split_name = 'train' if split == 'test' else split

        for cls_name in WEAPON_CLASSES.values():
            (dst / split_name / cls_name).mkdir(parents=True, exist_ok=True)

        stats[split_name] = {c: 0 for c in WEAPON_CLASSES.values()}

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            raw_class = class_dir.name
            cls_name = map_class(raw_class, force_class)

            for img_file in class_dir.glob('*.*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue

                image = cv2.imread(str(img_file))
                if image is None:
                    continue

                resized = cv2.resize(image, (target_size, target_size))

                prefix = raw_class if raw_class != cls_name else ""
                dst_name = f"{prefix}_{img_file.name}" if prefix else img_file.name
                dst_path = dst / split_name / cls_name / dst_name

                if dst_path.exists():
                    dst_name = f"{class_dir.name}_{img_file.name}"
                    dst_path = dst / split_name / cls_name / dst_name

                cv2.imwrite(str(dst_path), resized)
                stats[split_name][cls_name] = stats[split_name].get(cls_name, 0) + 1

    return stats


def print_stats(stats, dst_dir):
    print("\n" + "=" * 60)
    print("  Classification Dataset Conversion Complete")
    print("=" * 60)

    grand_total = 0
    for split in ['train', 'val']:
        if split not in stats:
            continue
        print(f"\n  {split.upper()}:")
        total = 0
        for cls_name, count in stats[split].items():
            print(f"    {cls_name}: {count}")
            total += count
        print(f"    TOTAL: {total}")
        grand_total += total

    print(f"\n  GRAND TOTAL: {grand_total}")
    print(f"  Output: {dst_dir}")
    print()
    print("  Next step:")
    print(f"    yolo classify train data={dst_dir} epochs=100 imgsz=224 model=yolov8n-cls.pt")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Smart dataset conversion to classification format")
    parser.add_argument('--src', type=str, default='data/raw_dataset/merged',
                        help='Source dataset directory (download dataset first, then configure path)')
    parser.add_argument('--dst', type=str, default='data/classification',
                        help='Destination classification directory')
    parser.add_argument('--expand', type=float, default=0.15,
                        help='Bounding box expansion ratio (0.15 = 15%%)')
    parser.add_argument('--size', type=int, default=224,
                        help='Target image size (default: 224)')
    parser.add_argument('--force-class', type=str, default=None,
                        help='Force all images to this class (e.g., "none" for negative samples)')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] Source directory not found: {src}")
        print("  Please download datasets first: python scripts/download_and_merge.py")
        return

    fmt = detect_format(src)
    print("=" * 60)
    print(f"  Source: {src}")
    print(f"  Format: {fmt}")
    print(f"  Target size: {args.size}x{args.size}")
    print(f"  BBox expansion: {args.expand * 100:.0f}%")
    if args.force_class:
        print(f"  Force class: {args.force_class}")
    print("=" * 60)

    if fmt == 'detection':
        stats = convert_detection_to_classification(
            src, dst, args.size, args.expand, args.force_class
        )
    elif fmt == 'classification':
        stats = convert_classification_to_classification(
            src, dst, args.size, args.force_class
        )
    else:
        print("[ERROR] Unknown dataset format. Expected detection or classification format.")
        return

    print_stats(stats, dst)


if __name__ == '__main__':
    main()
