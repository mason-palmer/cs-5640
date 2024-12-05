from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="data.yaml",
    epochs=3,
    batch_size=8,
    img_size=640,
    device="cuda",
    project="runs/train",
    name="exp",
    exist_ok=True
)