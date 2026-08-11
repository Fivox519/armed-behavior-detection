"""
训练武器分类模型 v4 - 渐进式7分类训练（YOLO框架版）
核心策略：
1. 从旧5分类模型加载，手动扩展分类头（5->8类），保留旧权重
2. 阶段1：冻结backbone，只训练分类头
3. 阶段2：解冻最后几层，微调整个模型
4. 类别权重 + Label Smoothing + MotionBlur增强
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import random


class MotionBlurAugmentation:
    """动态模糊增强：模拟快速挥舞时的画面拉丝"""

    def __init__(self, kernel_size_range=(3, 15), angle_range=(0, 360), prob=0.3):
        self.kernel_size_range = kernel_size_range
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
        if kernel_size % 2 == 0:
            kernel_size += 1

        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = cv2.warpAffine(
            kernel,
            cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1.0),
            (kernel_size, kernel_size)
        )
        kernel = kernel / kernel.sum()

        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            blurred = cv2.filter2D(img_np, -1, kernel)
            img_tensor = torch.from_numpy(blurred.astype(np.float32) / 255.0).permute(2, 0, 1)
            return img_tensor.to(img.device)
        else:
            return cv2.filter2D(img, -1, kernel)


def compute_class_weights(data_dir):
    """计算类别权重 - 逆频率权重"""
    class_counts = {}
    for cls_dir in sorted(Path(data_dir).joinpath('train').iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob('*.*')))
            class_counts[cls_dir.name] = n

    total = sum(class_counts.values())
    n_classes = len(class_counts)
    weights = {}
    for cls_name, count in class_counts.items():
        w = total / (n_classes * max(count, 1))
        weights[cls_name] = min(w, 5.0)

    print(f"  Class distribution (train):")
    for cls_name, count in sorted(class_counts.items()):
        print(f"    {cls_name}: {count} images, weight={weights[cls_name]:.3f}")

    weight_list = [weights[k] for k in sorted(class_counts.keys())]
    return torch.tensor(weight_list, dtype=torch.float32)


def expand_classifier_and_save(old_model_path, new_num_classes=8):
    """
    从旧5分类模型加载，扩展分类头5->8类，保存为新模型
    关键：保留backbone权重，只扩展分类头
    """
    print(f"\n  Loading old 5-class model: {old_model_path}")
    model = YOLO(old_model_path)
    print(f"  Old model classes: {model.names}")

    # 找到分类头: model.model.model[9].linear
    cls_head = model.model.model[9].linear
    old_weight = cls_head.weight.data.clone()  # [5, 1280]
    old_bias = cls_head.bias.data.clone()       # [5]
    in_features = old_weight.shape[1]

    print(f"  Old classifier: weight={old_weight.shape}, bias={old_bias.shape}")

    # 旧类别 -> 新类别映射
    # 旧: {0:'axe', 1:'hammer', 2:'knife', 3:'none', 4:'stick'}
    # 新: {0:'axe', 1:'bottle', 2:'hammer', 3:'knife', 4:'none', 5:'steel_pipe', 6:'stick', 7:'toy_stick'}
    old_to_new = {0: 0, 1: 2, 2: 3, 3: 4, 4: 6}

    # 创建新分类头
    new_head = nn.Linear(in_features, new_num_classes)
    nn.init.xavier_uniform_(new_head.weight)
    nn.init.zeros_(new_head.bias)

    # 迁移旧权重
    for old_idx, new_idx in old_to_new.items():
        new_head.weight.data[new_idx] = old_weight[old_idx]
        new_head.bias.data[new_idx] = old_bias[old_idx]
        old_name = model.names[old_idx]
        new_names = ['axe', 'bottle', 'hammer', 'knife', 'none', 'steel_pipe', 'stick', 'toy_stick']
        print(f"  Migrated: old[{old_idx}]({old_name}) -> new[{new_idx}]({new_names[new_idx]})")

    # 替换分类头
    model.model.model[9].linear = new_head
    print(f"  New classifier: weight={new_head.weight.shape}, bias={new_head.bias.shape}")

    # 更新模型类别信息
    model.model.nc = new_num_classes
    new_names_dict = {i: name for i, name in enumerate(new_names)}
    # YOLO的names是property，需要修改底层_model的names
    if hasattr(model, '_model') and hasattr(model._model, 'names'):
        model._model.names = new_names_dict
    # 直接修改model.model的names
    model.model.names = new_names_dict
    print(f"  New model classes: {model.model.names}")

    # 保存扩展后的模型
    save_path = Path('runs/train_v4_expanded')
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path / 'expanded_8cls.pt'))
    print(f"  Saved expanded model: {save_path / 'expanded_8cls.pt'}")

    return str(save_path / 'expanded_8cls.pt')


def train_phase1(
    expanded_model_path,
    data_dir='data/classification',
    epochs=30,
    batch=32,
    imgsz=224,
    device='0',
):
    """
    阶段1：冻结backbone，只训练分类头
    """
    print("\n" + "=" * 60)
    print("  Phase 1: Freeze Backbone, Train Classifier Head Only")
    print("=" * 60)

    class_weights = compute_class_weights(data_dir)

    # 加载扩展后的模型
    model = YOLO(expanded_model_path)
    print(f"  Model classes: {model.names}")

    # 冻结backbone（只保留分类头可训练）
    # model.model.model[0:9] 是backbone，model.model.model[9] 是分类头
    for i in range(9):  # 冻结前9层（backbone）
        for param in model.model.model[i].parameters():
            param.requires_grad = False

    # 确保分类头可训练
    for param in model.model.model[9].parameters():
        param.requires_grad = True

    frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.model.parameters() if p.requires_grad)
    print(f"  Frozen params: {frozen}, Trainable params: {trainable}")

    # 替换损失函数
    loss_replaced = [False]

    def replace_loss(trainer):
        if loss_replaced[0]:
            return
        dev = next(trainer.model.parameters()).device
        w = class_weights.to(dev)
        trainer.criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.15)
        loss_replaced[0] = True
        print(f"  Replaced criterion with WeightedCrossEntropyLoss (label_smoothing=0.15)")
        print(f"  Weights: {[f'{x:.2f}' for x in w.tolist()]}")

    model.add_callback("on_train_start", replace_loss)

    results = model.train(
        data=data_dir,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project='runs',
        name='train_v4_phase1',
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        patience=15,
        save=True,
        save_period=5,
        cache=False,
        workers=4,
        amp=True,
        warmup_epochs=3,
        close_mosaic=5,
        # 轻度增强
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.05,
        fliplr=0.3,
        mosaic=0.2,
        mixup=0.1,
        erasing=0.2,
        auto_augment='randaugment',
        dropout=0.2,
    )

    phase1_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"\n  Phase 1 complete! Best model: {phase1_path}")
    return str(phase1_path)


def train_phase2(
    phase1_model_path,
    data_dir='data/classification',
    epochs=50,
    batch=32,
    imgsz=224,
    device='0',
):
    """
    阶段2：解冻最后几层，微调整个模型
    """
    print("\n" + "=" * 60)
    print("  Phase 2: Unfreeze Last Layers, Fine-tune Full Model")
    print("=" * 60)

    class_weights = compute_class_weights(data_dir)

    model = YOLO(phase1_model_path)
    print(f"  Model classes: {model.names}")

    # 解冻最后3层backbone + 分类头
    # model.model.model[0:7] 冻结, [7:9] 解冻, [9] 分类头解冻
    for i in range(7):  # 冻结前7层
        for param in model.model.model[i].parameters():
            param.requires_grad = False

    for i in range(7, 10):  # 解冻后3层 + 分类头
        for param in model.model.model[i].parameters():
            param.requires_grad = True

    frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.model.parameters() if p.requires_grad)
    print(f"  Frozen params: {frozen}, Trainable params: {trainable}")

    # 替换损失函数
    loss_replaced = [False]

    def replace_loss(trainer):
        if loss_replaced[0]:
            return
        dev = next(trainer.model.parameters()).device
        w = class_weights.to(dev)
        trainer.criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
        loss_replaced[0] = True
        print(f"  Replaced criterion with WeightedCrossEntropyLoss (label_smoothing=0.1)")
        print(f"  Weights: {[f'{x:.2f}' for x in w.tolist()]}")

    model.add_callback("on_train_start", replace_loss)

    results = model.train(
        data=data_dir,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project='runs',
        name='train_v4_phase2',
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.0003,
        lrf=0.001,
        patience=20,
        save=True,
        save_period=5,
        cache=False,
        workers=4,
        amp=True,
        warmup_epochs=2,
        close_mosaic=10,
        # 更强的增强
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.15,
        scale=0.4,
        shear=5.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.15,
        erasing=0.3,
        auto_augment='randaugment',
        dropout=0.2,
    )

    phase2_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"\n  Phase 2 complete! Best model: {phase2_path}")
    return str(phase2_path)


def validate_model(model_path, data_dir='data/classification'):
    """验证模型在各类别上的表现"""
    print("\n" + "=" * 60)
    print(f"  Validating: {model_path}")
    print("=" * 60)

    model = YOLO(model_path)
    print(f"  Model classes: {model.names}")

    metrics = model.val(data=data_dir)
    print(f"\n  Top1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top5 Accuracy: {metrics.top5:.4f}")

    # 对每个类别进行单独测试
    val_dir = Path(data_dir) / 'val'
    print(f"\n  Per-class validation:")
    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        files = list(cls_dir.glob('*.jpg'))
        if not files:
            continue

        correct = 0
        total = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            results = model(img, verbose=False)
            if results[0].probs is not None:
                pred_cls = model.names[int(results[0].probs.top1)]
                if pred_cls == cls_name:
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0
        print(f"    {cls_name}: {correct}/{total} = {acc:.2%}")

    return metrics


def train_weapon_classifier_v4(
    data_dir='data/classification',
    old_model_path='weights/cls_best.pt',
    phase1_epochs=30,
    phase2_epochs=50,
    batch=32,
    imgsz=224,
    device='0',
):
    """
    v4完整训练流程：渐进式7分类训练
    Step 0: 扩展分类头（5->8类，保留旧权重）
    Step 1: 冻结backbone，只训练分类头
    Step 2: 解冻最后几层，微调整个模型
    """
    print("\n" + "=" * 60)
    print("  Weapon Classifier Training v4 (YOLO Framework)")
    print("  Progressive 7-Class Training with Hard Negatives")
    print("=" * 60)

    # Step 0: 扩展分类头
    expanded_path = expand_classifier_and_save(old_model_path, new_num_classes=8)

    # 验证扩展后的模型是否还能正确推理旧类别
    print("\n  Quick sanity check on expanded model...")
    model = YOLO(expanded_path)
    test_img = cv2.imread('data/classification/train/knife/knife_0000.jpg')
    if test_img is not None:
        test_img = cv2.resize(test_img, (224, 224))
        result = model(test_img, verbose=False)
        if result[0].probs is not None:
            top1 = model.names[int(result[0].probs.top1)]
            top1conf = float(result[0].probs.top1conf)
            print(f"  Test prediction on knife image: {top1} ({top1conf:.3f})")
            probs = result[0].probs.data.cpu().numpy()
            for i, name in model.names.items():
                print(f"    {name}: {probs[i]:.4f}")

    # Phase 1
    phase1_path = train_phase1(
        expanded_model_path=expanded_path,
        data_dir=data_dir,
        epochs=phase1_epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
    )

    # Phase 1 验证
    print("\n  === Phase 1 Validation ===")
    validate_model(phase1_path, data_dir)

    # Phase 2
    phase2_path = train_phase2(
        phase1_model_path=phase1_path,
        data_dir=data_dir,
        epochs=phase2_epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
    )

    # Phase 2 验证
    print("\n  === Phase 2 Final Validation ===")
    validate_model(phase2_path, data_dir)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Expanded model: {expanded_path}")
    print(f"  Phase 1 model: {phase1_path}")
    print(f"  Phase 2 model (FINAL): {phase2_path}")
    print("=" * 60)

    return phase2_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train weapon classifier v4 (YOLO framework)")
    parser.add_argument("--data", type=str, default="data/classification")
    parser.add_argument("--old-model", type=str, default="weights/cls_best.pt")
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--phase", type=str, default="both", choices=["both", "expand", "1", "2"])
    args = parser.parse_args()

    if args.phase == "expand":
        expand_classifier_and_save(args.old_model, new_num_classes=8)
    elif args.phase == "1":
        path = train_phase1(
            expanded_model_path=args.old_model,  # 这里old_model实际是expanded模型
            data_dir=args.data,
            epochs=args.phase1_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
        validate_model(path, args.data)
    elif args.phase == "2":
        path = train_phase2(
            phase1_model_path=args.old_model,  # 这里old_model实际是phase1模型
            data_dir=args.data,
            epochs=args.phase2_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
        validate_model(path, args.data)
    else:
        train_weapon_classifier_v4(
            data_dir=args.data,
            old_model_path=args.old_model,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
