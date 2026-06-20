import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict, Iterable, List, Optional, Set
import time
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


def select_file(file_type: str) -> str:
    """Open a file dialog to select a file."""
    file_path = filedialog.askopenfilename(
        title=f"Select {file_type}",
        filetypes=[("All files", "*.*"), ("Weights files", "*.pt"), ("Video files", "*.mp4;*.avi")],
    )
    return file_path

from PIL import Image, ImageTk

class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        canvas: tk.Canvas,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.canvas = canvas

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        cap = cv2.VideoCapture(self.source_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.source_video_path}")

        # Get video frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps  # Time delay between frames

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process and annotate the frame
            annotated_frame = self.process_frame(frame)

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

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


def start_processing(
    weights_path: str, video_path: str, canvas: tk.Canvas, conf: float, iou: float
):
    if not os.path.exists(weights_path) or not os.path.exists(video_path):
        messagebox.showerror("Error", "Please select valid weights and video files!")
        return

    processor = VideoProcessor(
        source_weights_path=weights_path,
        source_video_path=video_path,
        canvas=canvas,
        confidence_threshold=conf,
        iou_threshold=iou,
    )
    processor.process_video()
    messagebox.showinfo("Success", "Video processing completed!")


def main():
    root = tk.Tk()
    root.title("Traffic Flow Analysis")
    root.geometry("800x600")  # Resize the window for better layout.

    weights_path = tk.StringVar()
    video_path = tk.StringVar()

    # Weight file input
    tk.Label(root, text="Weights File:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=weights_path, width=50).grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: weights_path.set(select_file("Weights"))).grid(
        row=0, column=2, padx=10, pady=5
    )

    # Video file input
    tk.Label(root, text="Input Video:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=video_path, width=50).grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: video_path.set(select_file("Video"))).grid(
        row=1, column=2, padx=10, pady=5
    )

    # Confidence threshold input
    tk.Label(root, text="Confidence Threshold:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    conf_threshold = tk.DoubleVar(value=0.3)
    tk.Entry(root, textvariable=conf_threshold).grid(row=2, column=1, padx=10, pady=5)

    # IoU threshold input
    tk.Label(root, text="IoU Threshold:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    iou_threshold = tk.DoubleVar(value=0.7)
    tk.Entry(root, textvariable=iou_threshold).grid(row=3, column=1, padx=10, pady=5)

    # Canvas for video display
    canvas = tk.Canvas(root, width=640, height=360, bg="black")  # Smaller canvas size.
    canvas.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    # Submit button
    tk.Button(
        root,
        text="Start Processing",
        command=lambda: start_processing(
            weights_path.get(),
            video_path.get(),
            canvas,
            conf_threshold.get(),
            iou_threshold.get(),
        ),
    ).grid(row=5, column=0, columnspan=3, pady=20, sticky="ew")  # Ensure it spans all columns.

    # Configure grid row/column weights for resizing
    root.grid_rowconfigure(4, weight=1)  # Allow canvas to expand vertically.
    root.grid_columnconfigure(1, weight=1)  # Center the elements.

    root.mainloop()



if __name__ == "__main__":
    main()
