import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict
import time
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import supervision as sv


class DetectionsManager:
    def __init__(self) -> None:
        self.unique_tracker_ids = set()  # To track unique objects

    def update(self, detections: sv.Detections) -> sv.Detections:
        # Add new tracker IDs to the unique set
        self.unique_tracker_ids.update(detections.tracker_id)
        return detections

    def get_count(self) -> int:
        # Return the count of unique tracker IDs
        return len(self.unique_tracker_ids)


class Processor:
    def __init__(
        self,
        source_weights_path: str,
        source_path: str,
        canvas: tk.Canvas,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_path = source_path
        self.canvas = canvas

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        COLORS = sv.ColorPalette([
            sv.Color.from_hex("#FF0000"),
            sv.Color.from_hex("#00FF00"),
            sv.Color.from_hex("#0000FF"),
            sv.Color.from_hex("#FFFF00")
        ])
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)
        self.detections_manager = DetectionsManager()

    def process(self):
        if self.source_path.lower().endswith((".mp4", ".avi",".mov")):
            self._process_video()
        elif self.source_path.lower().endswith((".jpg", ".jpeg", ".png")):
            self._process_image()
        else:
            messagebox.showerror("Error", "Unsupported file type. Please select a video or image file.")

    def _process_video(self):
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps  # Time delay between frames

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process and annotate the frame
            annotated_frame = self._process_frame(frame)

            # Resize frame to fit within the canvas while maintaining aspect ratio
            frame_height, frame_width = annotated_frame.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

            # Convert the resized frame to ImageTk format
            frame_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))

            # Clear the canvas and display the new frame
            self.canvas.delete("all")
            self.canvas.create_image(
                (canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor=tk.NW, image=frame_image
            )
            self.canvas.image = frame_image

            # Update the canvas
            self.canvas.update()

            # Maintain the frame rate
            time.sleep(delay)

        cap.release()
        messagebox.showinfo("Summary", f"Total Vehicles Detected: {self.detections_manager.get_count()}")

    def _process_image(self):
        frame = cv2.imread(self.source_path)
        if frame is None:
            raise ValueError(f"Could not open image file: {self.source_path}")

        # Process and annotate the image
        annotated_frame = self._process_frame(frame)

        # Resize frame to fit within the canvas while maintaining aspect ratio
        frame_height, frame_width = annotated_frame.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

        # Convert the resized frame to ImageTk format
        frame_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))

        # Clear the canvas and display the new frame
        self.canvas.delete("all")
        self.canvas.create_image(
            (canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor=tk.NW, image=frame_image
            )
        self.canvas.image = frame_image
        messagebox.showinfo("Summary", f"Total Vehicles Detected: {self.detections_manager.get_count()}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        detections = self.detections_manager.update(detections)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.box_annotator.annotate(frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        return annotated_frame


def start_processing(
    weights_path: str, source_path: str, canvas: tk.Canvas, conf: float, iou: float
):
    if not os.path.exists(weights_path) or not os.path.exists(source_path):
        messagebox.showerror("Error", "Please select valid weights and source files!")
        return

    processor = Processor(
        source_weights_path=weights_path,
        source_path=source_path,
        canvas=canvas,
        confidence_threshold=conf,
        iou_threshold=iou,
    )
    processor.process()


def select_file(file_type: str) -> str:
    file_path = filedialog.askopenfilename(
        title=f"Select {file_type}",
        filetypes=[
            ("All files", "*.*"),
            ("Weights files", "*.pt"),
            ("Image and Video files", "*.mp4;*.avi;*.mov;*.jpg;*.jpeg;*.png"),
        ],
    )
    return file_path


def main():
    root = tk.Tk()
    root.title("Traffic Flow Analysis")
    root.geometry("800x600")

    weights_path = tk.StringVar()
    source_path = tk.StringVar()

    tk.Label(root, text="Weights File:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=weights_path, width=50).grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: weights_path.set(select_file("Weights"))).grid(
        row=0, column=2, padx=10, pady=5
    )

    tk.Label(root, text="Input File:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=source_path, width=50).grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: source_path.set(select_file("Source File"))).grid(
        row=1, column=2, padx=10, pady=5
    )

    tk.Label(root, text="Confidence Threshold:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    conf_threshold = tk.DoubleVar(value=0.3)
    tk.Entry(root, textvariable=conf_threshold).grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="IoU Threshold:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    iou_threshold = tk.DoubleVar(value=0.7)
    tk.Entry(root, textvariable=iou_threshold).grid(row=3, column=1, padx=10, pady=5)

    canvas = tk.Canvas(root, width=640, height=360, bg="black")
    canvas.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    tk.Button(
        root,
        text="Start Processing",
        command=lambda: start_processing(
            weights_path.get(),
            source_path.get(),
            canvas,
            conf_threshold.get(),
            iou_threshold.get(),
        ),
    ).grid(row=5, column=0, columnspan=3, pady=20, sticky="ew")

    root.mainloop()


if __name__ == "__main__":
    main()
