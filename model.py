import cv2
from ultralytics import YOLO
import math

# --- CONFIGURATION ---

# Path to your custom-trained model
model_path = "C:\\Users\\ahmed\\Downloads\\best.pt" 

# YouTube video URL
video_url = 'https://www.youtube.com/watch?v=naz_RVh48vg&t=103s'

# --- NEW: SET DISPLAY SIZE ---
# You can change these values to make the window bigger or smaller
display_width = 1280
display_height = 720


# --- LOAD THE MODEL ---
try:
    model = YOLO(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- RUN INFERENCE ON THE VIDEO ---
try:
    results = model(video_url, stream=True)

    for r in results:
        frame = r.plot()

        # --- NEW: RESIZE THE FRAME ---
        # Resize the frame to your desired display size
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Display the RESIZED frame
        cv2.imshow('YOLO YouTube Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred during prediction: {e}")

finally:
    # --- CLEANUP ---
    cv2.destroyAllWindows()
    print("Stream stopped.")