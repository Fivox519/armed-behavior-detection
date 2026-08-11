from pathlib import Path
from ultralytics import YOLO


def train_yolo_pose_stage1(
    data_yaml: str,
    model_config: str = "yolov8n-pose.yaml",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/pose",
    name: str = "weapon_wrist_detection",
    resume: bool = False,
    pretrained: bool = True
):
    print("=" * 60)
    print("YOLO-Pose Stage 1 Training - Person + Wrist Detection")
    print("=" * 60)
    print(f"Data config: {data_yaml}")
    print(f"Model: {model_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")
    print("=" * 60)

    model = YOLO(model_config if not pretrained else 'weights/yolov8n-pose.pt')

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        resume=resume,
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.001,
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        workers=2,
        amp=False,
        warmup_epochs=3,
        close_mosaic=10,
    )

    print("\nTraining complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")

    metrics = model.val(data=data_yaml)
    print(f"\nValidation metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  OKS: {metrics.keypoint_map:.4f}" if hasattr(metrics, 'keypoint_map') else "")

    return results


def export_model(weights_path: str, format: str = "onnx"):
    print(f"Exporting model to {format}...")
    model = YOLO(weights_path)
    model.export(format=format)
    print(f"Export complete: {weights_path.replace('.pt', f'.{format}')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO-Pose model")
    parser.add_argument("--data", type=str, default="configs/yolo_pose_stage1.yaml", help="Dataset YAML path")
    parser.add_argument("--model", type=str, default="yolov8n-pose.yaml", help="Model config or pretrained weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    train_yolo_pose_stage1(
        data_yaml=args.data,
        model_config=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume
    )
