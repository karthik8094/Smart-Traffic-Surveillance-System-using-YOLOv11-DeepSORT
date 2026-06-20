# 🚦 Smart Traffic Surveillance System using YOLOv11 & ByteTrack

## 📌 Project Overview

The Smart Traffic Surveillance System is a computer vision-based application designed to detect, track, and analyze vehicle movement in images and videos. The system leverages YOLOv11 for real-time vehicle detection and ByteTrack for multi-object tracking, enabling accurate traffic flow monitoring and vehicle counting.

The application provides an interactive desktop interface built with Tkinter, allowing users to process traffic footage, visualize vehicle tracking, and perform zone-based traffic analysis.

---

## 🎯 Objectives

* Detect vehicles in real-time using YOLOv11.
* Track vehicles across frames using ByteTrack.
* Perform traffic flow analysis through region-based counting.
* Visualize vehicle movement and tracking IDs.
* Provide a user-friendly graphical interface for traffic monitoring.

---

## ✨ Key Features

* 🚗 Real-time vehicle detection
* 🎯 Multi-object tracking with unique vehicle IDs
* 📊 Zone-based traffic flow analysis
* 🖼️ Image and video processing support
* 🖥️ Interactive Tkinter desktop application
* 📈 Vehicle counting and analytics
* 🔍 Adjustable confidence and IoU thresholds

---

## 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core Programming Language |
| YOLOv11      | Vehicle Detection         |
| ByteTrack    | Multi-Object Tracking     |
| OpenCV       | Image & Video Processing  |
| Supervision  | Detection Visualization   |
| NumPy        | Numerical Operations      |
| Tkinter      | Desktop GUI Development   |
| PIL (Pillow) | Image Handling            |

---

## 📂 Project Structure

```text
Smart-Traffic-Surveillance-System/
│
├── app/
│   ├── image_video_app.py
│   ├── video_app_with_Deepsort.py
│   └── test.py
│
├── models/
│   └── traffic_analysis.pt
│
├── dataset/
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── notebooks/
│   ├── vehicle-detection-yolo11.ipynb
│   └── yolov9.ipynb
│
├── screenshots/
│
├── demo/
│
├── docs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Smart-Traffic-Surveillance-System.git
cd Smart-Traffic-Surveillance-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Navigate to the app directory:

```bash
cd app
```

Run:

```bash
python image_video_app.py
```

or

```bash
python video_app_with_Deepsort.py
```

---

## 🚦 Workflow

1. Load the trained YOLO model (`traffic_analysis.pt`).
2. Select an image or video file.
3. Detect vehicles using YOLOv11.
4. Track vehicles using ByteTrack.
5. Analyze vehicle movement across defined zones.
6. Display detection results and traffic analytics.

---

## 📊 Results

The system successfully:

* Detects multiple vehicle classes.
* Tracks vehicles with unique IDs.
* Performs traffic flow monitoring.
* Provides real-time visualization of detections.
* Supports both image and video analytics.

---

## 🔮 Future Enhancements

* Flask/Web-based deployment
* Live CCTV integration
* Traffic congestion prediction
* Dashboard-based analytics
* Cloud deployment support
* Real-time alert generation

---

## 👨‍💻 Author

B.Tech Electronics and Communication Engineering Graduate

Interested in Data Science, Machine Learning, Computer Vision, and Artificial Intelligence.

