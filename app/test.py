import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Checkbutton, IntVar, Scale, DoubleVar, messagebox
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from typing import List

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

# Function to dynamically draw polygons on the video

def get_dynamic_polygon(video_path: str) -> List[np.ndarray]:
    """
    Allows the user to define dynamic polygon zones by drawing on the video.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read the video file.")

    polygons = []
    window_name = "Draw Zones"
    points = []

    def draw_polygon(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 2:
            polygons.append(np.array(points, np.int32))
            points = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    while True:
        temp_frame = frame.copy()
        for polygon in polygons:
            cv2.polylines(temp_frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, point in enumerate(points):
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(temp_frame, points[i - 1], point, (255, 0, 0), 2)

        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()
    cap.release()
    return polygons

class VideoProcessor:
    def __init__(self, source_weights_path, source_video_path, target_video_path, confidence_threshold, iou_threshold, with_zones):
        self.source_weights_path = source_weights_path
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.with_zones = with_zones
        self.model = YOLO(source_weights_path)
        self.zones_in = []

    def process_video(self):
        cap = cv2.VideoCapture(self.source_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(self.target_video_path, fourcc, frame_fps, (frame_width, frame_height))
        self.zones_in = get_dynamic_polygon(self.source_video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            detections = sv.Detections.from_ultralytics(results)

            if self.with_zones:
                for zone in self.zones_in:
                    cv2.polylines(frame, [zone], isClosed=True, color=(0, 255, 0), thickness=2)

            out.write(frame)

        cap.release()
        out.release()

if __name__ == "__main__":
    root = Tk()
    root.title("Video Processing with YOLO")
    root.geometry("400x400")

    def select_video():
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        video_label.config(text=f"Video: {file_path}")
        return file_path

    def select_weights():
        file_path = filedialog.askopenfilename(filetypes=[("Weights files", "*.pt")])
        weights_label.config(text=f"Weights: {file_path}")
        return file_path

    def process():
        video_path = select_video()
        weights_path = select_weights()
        target_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 file", "*.mp4")])
        if not (video_path and weights_path and target_path):
            messagebox.showerror("Error", "All inputs are required.")
            return

        processor = VideoProcessor(
            source_weights_path=weights_path,
            source_video_path=video_path,
            target_video_path=target_path,
            confidence_threshold=confidence_var.get(),
            iou_threshold=iou_var.get(),
            with_zones=zones_var.get()
        )
        processor.process_video()
        messagebox.showinfo("Success", f"Processing complete. Saved to {target_path}")

    Label(root, text="Select Video and Weights").pack(pady=10)

    video_label = Label(root, text="No video selected")
    video_label.pack(pady=5)

    weights_label = Label(root, text="No weights selected")
    weights_label.pack(pady=5)

    zones_var = IntVar()
    Checkbutton(root, text="Include Zones", variable=zones_var).pack(pady=10)

    confidence_var = DoubleVar(value=0.3)
    Label(root, text="Confidence Threshold").pack()
    Scale(root, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=confidence_var).pack()

    iou_var = DoubleVar(value=0.5)
    Label(root, text="IOU Threshold").pack()
    Scale(root, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=iou_var).pack()

    Button(root, text="Process Video", command=process).pack(pady=20)

    root.mainloop()
