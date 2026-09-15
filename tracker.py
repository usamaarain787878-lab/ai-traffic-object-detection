from datetime import datetime
import os
import sqlite3
import time
import cv2
from ultralytics import YOLO

# Initialize YOLOv8 model and default camera stream
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Create directory to store auto-snapshots
SNAPSHOT_DIR = "captures"
if not os.path.exists(SNAPSHOT_DIR):
  os.makedirs(SNAPSHOT_DIR)

# --- SQLITE DATABASE SETUP ---
DB_NAME = "traffic_system.db"
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Create detections table if it doesn't already exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        track_id INTEGER,
        detected_object TEXT,
        status TEXT
    )
""")
conn.commit()

prev_time = 0
frame_count = 0

print(
    "Starting Advanced SQLite-Integrated Traffic Tracking System... Press 'q'"
    " to stop."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Video stream ended or camera disconnected.")
    break

  frame_count += 1

  # --- REAL-TIME FPS CALCULATION ---
  current_time = time.time()
  fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
  prev_time = current_time

  # --- YOLOV8 + BYTETRACK EXECUTION ---
  results = model.track(
      frame, persist=True, tracker="bytetrack.yaml", conf=0.4, show=False
  )
  current_frame = results[0].plot()

  # Process detected objects and track IDs
  if results[0].boxes.id is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
      class_name = model.names[int(cls_id)]
      timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

      print(
          f"[{timestamp}] ID: {int(track_id)} | Object: {class_name} (Saved to"
          " DB)"
      )

      # --- AUTO-SNAPSHOT CAPTURE FEATURE ---
      if frame_count % 30 == 0:
        snapshot_name = os.path.join(
            SNAPSHOT_DIR, f"snapshot_{int(track_id)}_{timestamp_filename}.jpg"
        )
        cv2.imwrite(snapshot_name, current_frame)

      # --- INSERT DATA DIRECTLY INTO SQLITE DATABASE ---
      cursor.execute(
          """
            INSERT INTO detections (timestamp, track_id, detected_object, status)
            VALUES (?, ?, ?, ?)
        """,
          (timestamp, int(track_id), class_name, "Logged to SQLite"),
      )
      conn.commit()

  # Display Real-Time Metrics Overlay on Video Stream
  cv2.putText(
      current_frame,
      f"SQLite Traffic AI | FPS: {int(fps)}",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2,
  )

  # Show live window
  cv2.imshow("Advanced SQLite Traffic Monitor", current_frame)

  # Exit loop when 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Release resources and close database connection
cap.release()
cv2.destroyAllWindows()
conn.close()

print(
    "\n[Success] Session ended. All records successfully saved to SQLite"
    f" database '{DB_NAME}'!"
)
print(f"[Success] Evidence snapshots securely stored in '{SNAPSHOT_DIR}/'.")