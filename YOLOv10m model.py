from ultralytics import YOLO

model_v10m = YOLO("yolov10m.pt")

def on_fit_epoch_end_v10(trainer):
    epoch = trainer.epoch
    total_epochs = trainer.epochs
    mAP50 = trainer.metrics.get('metrics/mAP50(B)', 0)
    mAP5095 = trainer.metrics.get('metrics/mAP50-95(B)', 0)
    box_loss = trainer.metrics.get('train/box_loss', 0)
    cls_loss = trainer.metrics.get('train/cls_loss', 0)
    dfl_loss = trainer.metrics.get('train/dfl_loss', 0)
    loss = box_loss + cls_loss + dfl_loss
    num_samples = len(trainer.train_loader.dataset)
    elapsed = getattr(trainer, 'epoch_time', 1) or 1
    fps = num_samples / elapsed

    print(
        f"Epoch [{epoch+1}/{total_epochs}] | "
        f"mAP50: {mAP50:.4f}, mAP50-95: {mAP5095:.4f} | "
        f"Train Loss: {loss:.4f} | FPS: {fps:.2f}"
    )

model_v10m.add_callback("on_fit_epoch_end", on_fit_epoch_end_v10)

model_v10m.train(
    data="/kaggle/input/bdd100k-yolo10-yaml/bdd100k_ultralytics.yaml",
    project="YOLOv10_Training",
    name="bdd100k_fixed_stabilized",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    classes=[1, 2, 3, 11, 12, 13, 14, 15, 16, 17],
    lr0=0.0001,
    lrf=0.01,
    optimizer='AdamW',
    warmup_epochs=5,
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=10,
    amp=True,
    plots=True,
    cos_lr=True,
    save=True,
    exist_ok=True

)
