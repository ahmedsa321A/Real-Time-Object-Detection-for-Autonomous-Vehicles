from ultralytics import YOLO

def on_fit_epoch_end(trainer):
    epoch = trainer.epoch
    total_epochs = trainer.epochs
    
    mAP50 = trainer.metrics.get('metrics/mAP50(B)', 0) 
    
    box_loss = trainer.metrics.get('train/box_loss', 0)
    cls_loss = trainer.metrics.get('train/cls_loss', 0)
    dfl_loss = trainer.metrics.get('train/dfl_loss', 0)
    loss = box_loss + cls_loss + dfl_loss
    
    num_samples = len(trainer.train_loader.dataset)
    elapsed = getattr(trainer, 'epoch_time', 1) or 1
    fps = num_samples / elapsed

    # Corrected the epoch counter to use 'epoch' instead of 'epoch+1'
    print(f"✅ Epoch [{epoch+1}/{total_epochs}] | "
          f"🎯 mAP@0.5: {mAP50:.4f} | "
          f"📉 Train Loss: {loss:.4f} | "
          f"⚡ FPS: {fps:.2f}")

model = YOLO('yolov9m.pt')

model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

# STAGE 1
model.train(
    data="/kaggle/input/bdd100k-yolo10-yaml/bdd100k_ultralytics.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    project="YOLOv9_Training",
    name="bdd100k_yolov9m_aug",
    save=True,
    fraction=1.0,
    lr0=0.002,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    degrees=0.5,
    translate=0.1,
    scale=0.5,
    shear=0.2,
    patience=10

)
