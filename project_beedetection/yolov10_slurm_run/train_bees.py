from ultralytics import YOLO
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Bee YOLOv10 Training")

    parser.add_argument("--data", type=str, default="Worker/Drone-2/data.yaml")
    parser.add_argument("--model", type=str, default="yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--name", type=str, default="yolov10_bees")
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    print("=== Training Configuration ===")
    print(f"Data:      {data_path}")
    print(f"Model:     {args.model}")
    print(f"Epochs:    {args.epochs}")
    print(f"Batch:     {args.batch}")
    print(f"Img size:  {args.imgsz}")
    print(f"Run name:  {args.name}")
    print(f"GPU:       {args.device}")
    print("==============================")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        device=args.device,
        workers=4,
        project="runs/train"
    )

    print("\nTraining completed.")
    print(f"Results saved to: runs/train/{args.name}")


if __name__ == "__main__":
    main()
