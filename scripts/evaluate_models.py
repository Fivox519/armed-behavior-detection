"""
模型评估脚本 - 评估阶段1和阶段2模型性能
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse


def evaluate_pose_model(model_path, data_yaml, split='val'):
    """评估阶段1 YOLO-Pose模型"""
    print("\n" + "="*60)
    print("阶段1评估: YOLO-Pose 人体+手腕检测")
    print("="*60)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        split=split,
        verbose=True,
        save=True,
        project='runs/pose/eval',
        name='evaluation'
    )

    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'Pose_mAP50': results.keypoints.map50 if hasattr(results, 'keypoints') else None,
    }

    print("\n📊 阶段1评估结果:")
    print(f"   Box mAP@50: {metrics['mAP50']:.4f}")
    print(f"   Box mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall: {metrics['Recall']:.4f}")

    return metrics


def evaluate_weapon_model(model_path, data_yaml, split='val'):
    """评估阶段2武器分类模型"""
    print("\n" + "="*60)
    print("阶段2评估: 手部危险物品分类")
    print("="*60)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        split=split,
        verbose=True,
        save=True,
        project='runs/detect/eval',
        name='evaluation'
    )

    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
    }

    print("\n📊 阶段2评估结果:")
    print(f"   mAP@50: {metrics['mAP50']:.4f}")
    print(f"   mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall: {metrics['Recall']:.4f}")

    return metrics


def evaluate_armed_detection(pose_model_path, weapon_model_path, test_images_dir, output_dir=None):
    """评估完整持械检测流程"""
    print("\n" + "="*60)
    print("完整流程评估: 持械人员检测")
    print("="*60)

    pose_model = YOLO(pose_model_path)
    weapon_model = YOLO(weapon_model_path)

    weapon_classes = {0: 'knife', 1: 'axe', 2: 'hammer', 3: 'stick', 4: 'none'}
    crop_size = 226

    test_images = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))

    if not test_images:
        print(f"⚠️  测试目录为空: {test_images_dir}")
        return None

    total_persons = 0
    armed_detected = 0
    normal_detected = 0
    detection_failures = 0

    results_summary = []

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        pose_results = pose_model(img, verbose=False)
        kpts = pose_results[0].keypoints

        if kpts is None or len(kpts) == 0:
            detection_failures += 1
            continue

        img_result = {
            'image': img_path.name,
            'persons': []
        }

        for kp in kpts:
            if kp is None or len(kp.data[0]) < 11:
                continue

            total_persons += 1

            lw = kp.data[0][9][:2].cpu().numpy()
            rw = kp.data[0][10][:2].cpu().numpy()

            def crop_wrist(x, y, padding=40):
                h, w = img.shape[:2]
                x1 = max(0, int(x - crop_size//2 - padding))
                y1 = max(0, int(y - crop_size//2 - padding))
                x2 = min(w, int(x + crop_size//2 + padding))
                y2 = min(h, int(y + crop_size//2 + padding))
                return img[y1:y2, x1:x2]

            lw_crop = crop_wrist(lw[0], lw[1])
            rw_crop = crop_wrist(rw[0], rw[1])

            lw_res = weapon(lw_crop, verbose=False)[0]
            rw_res = weapon(rw_crop, verbose=False)[0]

            lw_cls = weapon_classes.get(lw_res.probs.top1 if lw_res.probs else 4, 'none')
            rw_cls = weapon_classes.get(rw_res.probs.top1 if rw_res.probs else 4, 'none')
            lw_conf = lw_res.probs.top1conf.item() if lw_res.probs else 0
            rw_conf = rw_res.probs.top1conf.item() if rw_res.probs else 0

            is_armed = lw_cls != 'none' or rw_cls != 'none'

            if is_armed:
                armed_detected += 1
            else:
                normal_detected += 1

            img_result['persons'].append({
                'left_wrist': {'class': lw_cls, 'confidence': lw_conf},
                'right_wrist': {'class': rw_cls, 'confidence': rw_conf},
                'is_armed': is_armed
            })

        results_summary.append(img_result)

    print("\n📊 完整检测流程评估结果:")
    print(f"   测试图片数: {len(test_images)}")
    print(f"   检测失败图片数: {detection_failures}")
    print(f"   检测到总人数: {total_persons}")
    print(f"   持械人数: {armed_detected}")
    print(f"   正常人数: {normal_detected}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'detection_results.txt', 'w', encoding='utf-8') as f:
            f.write("持械检测结果报告\n")
            f.write("="*50 + "\n\n")
            for result in results_summary:
                f.write(f"图片: {result['image']}\n")
                for i, person in enumerate(result['persons']):
                    status = "⚠️ 持械" if person['is_armed'] else "✅ 正常"
                    f.write(f"  人员{i+1}: {status}\n")
                    f.write(f"    左手腕: {person['left_wrist']['class']} ({person['left_wrist']['confidence']:.2f})\n")
                    f.write(f"    右手腕: {person['right_wrist']['class']} ({person['right_wrist']['confidence']:.2f})\n")
                f.write("\n")

        print(f"\n💾 结果已保存: {output_path / 'detection_results.txt'}")

    return {
        'total_images': len(test_images),
        'detection_failures': detection_failures,
        'total_persons': total_persons,
        'armed': armed_detected,
        'normal': normal_detected
    }


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=3,
                        help='评估阶段: 1=阶段1模型, 2=阶段2模型, 3=完整流程')
    parser.add_argument('--pose-model', type=str,
                        default='weights/pose_best.pt',
                        help='阶段1模型路径')
    parser.add_argument('--weapon-model', type=str,
                        default='weights/cls_best.pt',
                        help='阶段2模型路径')
    parser.add_argument('--pose-data', type=str,
                        default='configs/yolo_pose_stage1.yaml',
                        help='阶段1数据集配置')
    parser.add_argument('--weapon-data', type=str,
                        default='data/hand_weapons/dataset.yaml',
                        help='阶段2数据集配置')
    parser.add_argument('--test-images', type=str,
                        default='data/coco_val/images',
                        help='测试图片目录')
    parser.add_argument('--output', type=str, default='runs/evaluation',
                        help='评估结果输出目录')
    args = parser.parse_args()

    if args.stage == 1:
        evaluate_pose_model(args.pose_model, args.pose_data)
    elif args.stage == 2:
        evaluate_weapon_model(args.weapon_model, args.weapon_data)
    else:
        evaluate_armed_detection(args.pose_model, args.weapon_model,
                                args.test_images, args.output)


if __name__ == "__main__":
    main()