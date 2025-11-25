import cv2
import time
import pandas as pd
from ultralytics import YOLO
import os

# ==================== SETTINGS ====================
MODEL_PATH = "best.onnx"            

# CHANGE: Put your video filename here
CAMERA_SOURCE = "Relaxing Night Drive in Tokyo _ 8K 60fps HDR _ Soft Lofi Beats - Abao Vision (1080p, h264).mp4"

CSV_OUTPUT = "test_results.csv"     
DISPLAY_WIDTH = 1024                
IMG_SIZE = 320   

# --- THE BAN LIST ---
CLASS_RULES = {
    1: 0.30, 2: 0.50, 3: 0.50, 11: 0.30, 12: 0.30, 13: 0.30, 
    14: 0.40, 15: 0.20, 16: 0.50, 17: 0.50,
    0: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 2.0
}
DEFAULT_CONF = 0.50 
# ==================================================

def draw_hud(frame, fps, latency_ms, counts):
    """Draw stats and class breakdown on the frame"""
    total_obj = sum(counts.values())
    
    # Line 1: Technical Stats
    stats_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Total: {total_obj}"
    
    # Line 2: Object Breakdown (e.g., "car: 3 | person: 1")
    # Sort alphabetically so the text doesn't jump around
    sorted_counts = dict(sorted(counts.items()))
    breakdown_text = " | ".join([f"{k}: {v}" for k, v in sorted_counts.items()])
    
    # Draw Black Background
    # Height adjusts if we have a second line of text
    h_bg = 70 if breakdown_text else 40
    cv2.rectangle(frame, (0, 0), (display_width_hud(frame), h_bg), (0, 0, 0), -1)
    
    # Draw Text (Green for stats, Yellow for objects)
    cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if breakdown_text:
        cv2.putText(frame, breakdown_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    return frame

def display_width_hud(frame):
    # Helper to get dynamic width for the black bar
    return frame.shape[1]

def test_model():
    print(f"Loading ONNX Model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect') 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Opening Video Source: {CAMERA_SOURCE}...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file. Check the filename!")
        return

    # Reset the CSV file at the start
    if os.path.exists(CSV_OUTPUT):
        os.remove(CSV_OUTPUT)
    
    # Create empty CSV with headers
    pd.DataFrame(columns=["Frame", "FPS", "Latency_ms", "Objects"]).to_csv(CSV_OUTPUT, index=False)

    readings_buffer = []
    frame_id = 0
    
    print("Starting Video Inference... Press 'q' to stop.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: 
            print("End of video.")
            break
        
        frame_id += 1

        # 1. Inference
        # Using auto-fallback logic for 320 vs 640

        results = model(frame, conf=0.1, verbose=False)
        
        # 2. Filter & Count
        final_boxes = []
        counts = {} # Dictionary to store "Car: 3", "Bus: 1"
        
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                threshold = CLASS_RULES.get(cls_id, DEFAULT_CONF)
                
                if conf >= threshold:
                    final_boxes.append(box)
                    # Count the object
                    class_name = model.names[cls_id]
                    counts[class_name] = counts.get(class_name, 0) + 1
                    
            results[0].boxes = final_boxes

        # 3. Stats
        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        latency_ms = inference_time * 1000
        objects_detected = len(final_boxes)

        # 4. Log Data to Buffer
        readings_buffer.append({
            "Frame": frame_id,
            "FPS": round(fps, 1),
            "Latency_ms": round(latency_ms, 1),
            "Objects": objects_detected
        })

        # === SAVE TO DISK EVERY 10 FRAMES ===
        if frame_id % 10 == 0:
            df_chunk = pd.DataFrame(readings_buffer)
            df_chunk.to_csv(CSV_OUTPUT, mode='a', header=False, index=False)
            readings_buffer = [] # Clear buffer

        # 5. Display
        annotated_frame = results[0].plot()
        
        # Resize for display
        h, w = annotated_frame.shape[:2]
        if w > 0:
            scale = DISPLAY_WIDTH / w
            new_height = int(h * scale)
            display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, new_height))
            
            # === DRAW THE HUD (Stats + Counts) ===
            display_frame = draw_hud(display_frame, fps, latency_ms, counts)
            
            cv2.imshow("Milestone 3 - Video Inference", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()