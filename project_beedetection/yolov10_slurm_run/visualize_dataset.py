from ultralytics import YOLO
from pathlib import Path
import random

def visualize_random_samples(data_yaml, output_dir="viz_samples", n=20):
    data_yaml = Path(data_yaml).resolve()
    train_imgs = list((data_yaml.parent / "train/images").glob("*.*"))

    sample = random.sample(train_imgs, min(n, len(train_imgs)))

    model = YOLO("yolov10n.pt")

    model.predict(
        source=[str(p) for p in sample],
        save=True,
        project=output_dir,
        name="samples",
        imgsz=1280
    )

    print(f"Saved training-sample visualizations to {output_dir}/samples")


def visualize_predictions(model_path, data_yaml, output_dir="viz_preds"):
    data_yaml = Path(data_yaml).resolve()
    val_imgs = list((data_yaml.parent / "valid/images").glob("*.*"))

    model = YOLO(model_path)

    model.predict(
        source=[str(p) for p in val_imgs[:20]],
        save=True,
        project=output_dir,
        name="preds",
        imgsz=1280
    )

    print(f"Saved prediction visualizations to {output_dir}/preds")


if __name__ == "__main__":
    data = "Worker/Drone-2/data.yaml"
    best = "runs/train/yolov10_bees/weights/best.pt"

    visualize_random_samples(data, n=20)
    visualize_predictions(best, data)
