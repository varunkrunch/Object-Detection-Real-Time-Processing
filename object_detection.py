import cv2
import time
import csv
import numpy as np
import torch
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog

# ==============================
# Load Pre-trained YOLOv5 Model
# ==============================
model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",       # yolov5n / yolov5s / yolov5m
    pretrained=True
)

model.conf = 0.5   # confidence threshold

# ==============================
# Select Input Source
# ==============================
print("Select input source:")
print("1. Live Webcam Detection")
print("2. Video File")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
    input_type = "Webcam"
    print("Webcam selected.")

elif choice == "2":
    print("\nSelect video input method:")
    print("1. Type file name / full path")
    print("2. Select using file picker")

    method = input("Enter your choice (1 or 2): ").strip()

    if method == "1":
        video_path = input("Enter video file name or full path: ").strip()

    elif method == "2":
        root = tk.Tk()
        root.withdraw()

        video_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )

        if not video_path:
            print("No video selected. Exiting.")
            exit()

    else:
        print("Invalid choice.")
        exit()

    if not os.path.exists(video_path):
        print("Error: Video file not found.")
        exit()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        print("Tip: Convert .MOV to .MP4 if needed.")
        exit()

    input_type = "Video File"
    print(f"Video selected: {video_path}")

else:
    print("Invalid choice.")
    exit()

# ==============================
# RESET CSV FILE (EVERY RUN)
# ==============================
csv_file = "detections.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Object", "Confidence"])

# ==============================
# Performance Tracking
# ==============================
frame_count = 0
start_time = time.time()
confidence_scores = []

print("Detection started. Press 'q' to quit.")

# ==============================
# Detection Loop
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        confidence = float(row["confidence"])
        label = row["name"]

        confidence_scores.append(confidence)

        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {confidence:.2f}"
        cv2.putText(
            frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, label, confidence])

    cv2.imshow("YOLOv5 Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# SAFE FPS & STATISTICS
# ==============================
end_time = time.time()
elapsed_time = end_time - start_time

if frame_count > 0:
    fps = frame_count / elapsed_time
else:
    fps = 0.0

total_detections = len(confidence_scores)

if confidence_scores:
    avg_confidence = float(np.mean(confidence_scores))
else:
    avg_confidence = 0.0

# ==============================
# FORCE PERFORMANCE REPORT WRITE
# ==============================
with open("performance_report.txt", "w") as f:
    f.write("YOLOv5 Real-Time Object Detection Performance Report\n")
    f.write("--------------------------------------------------\n")
    f.write(f"Input Source: {input_type}\n")
    f.write(f"Total Frames Processed: {frame_count}\n")
    f.write(f"Total Detections: {total_detections}\n")
    f.write(f"Average Confidence Score: {avg_confidence:.2f}\n")
    f.write(f"FPS Achieved: {fps:.2f}\n")

print("Performance report generated successfully.")

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()

print("Detection completed successfully.")
print(f"FPS Achieved: {fps:.2f}")
