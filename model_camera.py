import cv2
import time
import pandas as pd
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "best.onnx"            
# CHANGE: Use 0 for webcam. (Use 1 or 2 if you have external cams)
CAMERA_SOURCE = 0  
CSV_OUTPUT = "live_camera_results.csv"     
DISPLAY_WIDTH = 1024                
IMG_SIZE = 320   # Keep this low for high FPS on CPU

# --- THE BAN LIST ---
# We use this to tell the script: "If you see a Lane, IGNORE IT."
CLASS_RULES = {
    # === ALLOWED CLASSES (Set sensible thresholds) ===
    1: 0.30,   # Bike
    2: 0.50,   # Bus
    3: 0.50,   # Car (Main focus)
    11: 0.30,  # Motor
    12: 0.30,  # Person (Safety critical)
    13: 0.30,  # Rider
    14: 0.40,  # Traffic Light
    15: 0.20,  # Traffic Sign
    16: 0.50,  # Train
    17: 0.50,  # Truck

    # === BANNED CLASSES (Set to 2.0 to hide) ===
    0: 2.0,    # Area/Unknown - HIDDEN
    4: 2.0,    # Lane/Crosswalk - HIDDEN
    5: 2.0,    # Lane/Double Other - HIDDEN
    6: 2.0,    # Lane/Double White - HIDDEN
    7: 2.0,    # Lane/Double Yellow - HIDDEN
    8: 2.0,    # Lane/Single Other - HIDDEN
    9: 2.0,    # Lane/Single White - HIDDEN
    10: 2.0,   # Lane/Single Yellow - HIDDEN
}

DEFAULT_CONF = 0.50 # Fallback confidence
# ==================================================

def test_model():
    print(f"Loading ONNX Model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect') 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # CHANGE: Initialize Camera instead of Video File
    print(f"Opening Camera {CAMERA_SOURCE}...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Cannot access camera {CAMERA_SOURCE}.")
        return

    readings = []
    frame_id = 0
    
    print("Starting Live Inference... Press 'q' to stop.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: 
            print("Failed to grab frame.")
            break
        
        frame_id += 1

        # 1. Run Inference (Capture everything first)
        # device='cpu' ensures stability on your hardware
        results = model(frame, conf=0.1, verbose=False)
        
        # 2. FILTERING LOOP
        final_boxes = []
        if len(results) > 0:
            original_boxes = results[0].boxes
            
            for box in original_boxes:
                cls_id = int(box.cls[0])    # Get Class ID
                conf = float(box.conf[0])   # Get Confidence
                
                # Check the rule. If it's a Lane (4-10), rule is 2.0, so this fails.
                threshold = CLASS_RULES.get(cls_id, DEFAULT_CONF)
                
                if conf >= threshold:
                    final_boxes.append(box)

            # Update the results so only valid boxes are drawn
            results[0].boxes = final_boxes

        # 3. Calculate Stats
        end_time = time.time()
        inference_time = end_time - start_time
        if inference_time > 0:
            fps = 1 / inference_time
        else:
            fps = 0
        latency_ms = inference_time * 1000
        objects_detected = len(final_boxes)

        # 4. Log Data
        readings.append({
            "Frame": frame_id,
            "FPS": round(fps, 1),
            "Latency_ms": round(latency_ms, 1),
            "Objects": objects_detected
        })

        # 5. Display
        annotated_frame = results[0].plot()
        
        h, w = annotated_frame.shape[:2]
        if w > 0: # Prevent divide by zero
            scale = DISPLAY_WIDTH / w
            new_height = int(h * scale)
            display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, new_height))

            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Milestone 3 - Live Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Save Report
    if readings:
        df = pd.DataFrame(readings)
        df.to_csv(CSV_OUTPUT, index=False)
        print("\n" + "="*30)
        print(f"Final Average FPS: {df['FPS'].mean():.2f}")
        print("="*30)

if __name__ == "__main__":
    test_model()