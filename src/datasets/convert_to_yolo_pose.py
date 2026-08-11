import json
import shutil
from pathlib import Path
from tqdm import tqdm

COCO_KEYPOINTS_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

WRIST_INDICES = [9, 10]

KEYPOINT_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def convert_coco_to_yolo_pose(coco_annotation_file, coco_images_dir, output_dir, class_id=0):
    output_dir = Path(output_dir)
    images_output = output_dir / "images"
    labels_output = output_dir / "labels"
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO annotations: {coco_annotation_file}")
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    images_dict = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat for cat in coco_data['categories']}

    print(f"Total images: {len(images_dict)}")
    print(f"Total annotations: {len(coco_data['annotations'])}")

    person_count = 0
    wrist_visible_count = 0
    skipped_count = 0

    person_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] == 1]

    print(f"Person annotations: {len(person_annotations)}")

    for ann in tqdm(person_annotations, desc="Converting"):
        img_id = ann['image_id']
        img_info = images_dict[img_id]

        img_width = img_info['width']
        img_height = img_info['height']
        img_file_name = img_info['file_name']

        src_img_path = Path(coco_images_dir) / img_file_name
        if not src_img_path.exists():
            skipped_count += 1
            continue

        keypoints = ann['keypoints']
        num_keypoints = ann['num_keypoints']

        left_wrist_idx = 9 * 3
        right_wrist_idx = 10 * 3

        left_wrist = keypoints[left_wrist_idx:left_wrist_idx+3]
        right_wrist = keypoints[right_wrist_idx:right_wrist_idx+3]

        left_wrist_visible = left_wrist[2] > 0
        right_wrist_visible = right_wrist[2] > 0

        if not (left_wrist_visible or right_wrist_visible):
            skipped_count += 1
            continue

        bbox = ann['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height

        yolo_pose_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        for i, (px, py, pv) in enumerate(zip(
            keypoints[0::3], keypoints[1::3], keypoints[2::3]
        )):
            if px == 0 and py == 0:
                norm_x, norm_y, vis = 0.0, 0.0, 0
            else:
                norm_x = px / img_width
                norm_y = py / img_height
                vis = 1 if pv > 0 else 0
            yolo_pose_line += f" {norm_x:.6f} {norm_y:.6f} {vis}"

        label_file = labels_output / f"{Path(img_file_name).stem}.txt"
        with open(label_file, 'a') as f:
            f.write(yolo_pose_line + '\n')

        dest_img_path = images_output / img_file_name
        if not dest_img_path.exists():
            shutil.copy(src_img_path, dest_img_path)

        person_count += 1
        if left_wrist_visible and right_wrist_visible:
            wrist_visible_count += 1

    print(f"\nConversion complete!")
    print(f"  Person boxes extracted: {person_count}")
    print(f"  Both wrists visible: {wrist_visible_count}")
    print(f"  Skipped (no visible wrist): {skipped_count}")
    print(f"  Output images: {images_output}")
    print(f"  Output labels: {labels_output}")

    return {
        'total': len(person_annotations),
        'extracted': person_count,
        'both_wrists_visible': wrist_visible_count,
        'skipped': skipped_count
    }


def create_dataset_yaml(data_yaml_path, train_path, val_path, nc=1, kpt_shape=17):
    yaml_content = f"""# YOLO-Pose Dataset Configuration
# Person keypoint detection dataset for weapon/wrist detection

path: {Path(data_yaml_path).parent}
train: {train_path}
val: {val_path}

nc: {nc}
kpt_shape: [{kpt_shape}, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

names:
  0: person
"""
    with open(data_yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created dataset YAML: {data_yaml_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert COCO keypoints to YOLO-Pose format")
    parser.add_argument("--annotations", type=str, required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--images", type=str, required=True, help="Path to COCO images folder")
    parser.add_argument("--output", type=str, default="data/processed/coco_keypoints_yolo", help="Output path (download COCO dataset first, then configure path)")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train split ratio")
    args = parser.parse_args()

    convert_coco_to_yolo_pose(
        args.annotations,
        args.images,
        args.output,
        class_id=0
    )
