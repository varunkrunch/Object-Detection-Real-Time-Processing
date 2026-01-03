# Real-Time Object Detection using YOLOv8

This project implements a real-time object detection system using a pre-trained **YOLOv8 Nano** model from Ultralytics. The application supports live webcam detection as well as video file detection, allowing users to either enter a file name/path or select a video using a cross-platform file picker.

The system draws bounding boxes with confidence scores, logs detections to `detections.csv` (reset on every run), and generates a `performance_report.txt` containing FPS and detection statistics.

## Tech Stack

*   **Python**
*   **OpenCV**
*   **YOLOv5 (Ultralytics)**
*   **NumPy**
*   **Tkinter** (for the file picker)

## How to Run

1.  **Install the required Python packages:**

    ```bash
    pip install torch torchvision ultralytics opencv-python pandas
    pip install numpy==1.26.4

    ```
    *Note: Tkinter is usually included with standard Python installations.*

2.  **Execute the main script:**

    ```bash
    python object_detection.py
    ```
    The script will prompt you to choose between using your webcam (`1`) or selecting a video file (`2`).
    
    Selecting (`2`) will prompt you to choose between typing the file name/path or selecting the file using a cross-platform file picker. 

## Output Files

*   **`detections.csv`**: A log file containing the object class, confidence score, and timestamp for every detected object.
*   **`performance_report.txt`**: A summary report containing the Frames Per Second (FPS) achieved and overall detection statistics.
