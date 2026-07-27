# 🤖 SmartVision AI

An AI-powered Object Detection Platform built using **Python, Streamlit, OpenCV, and YOLOv8**.

SmartVision AI allows users to detect objects from uploaded images and live webcam feeds with real-time AI predictions, interactive analytics, and downloadable detection reports.

---

# 📌 Project Overview

SmartVision AI is an intelligent computer vision application designed to detect and recognize everyday objects using the powerful YOLOv8 deep learning model.

The project provides an elegant user interface with image detection, live webcam detection, analytics dashboard, and automatic detection history generation.

The application is built using Streamlit and is designed to provide a professional AI experience.

---

# ✨ Features

✔️ Modern Premium UI Design

✔️ Image Object Detection

✔️ Live Webcam Detection

✔️ YOLOv8 AI Model

✔️ Detection Confidence Scores

✔️ Object Count Summary

✔️ Analytics Dashboard

✔️ Detection History (CSV)

✔️ Download Detection Report

✔️ Responsive Streamlit Interface

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application Framework |
| YOLOv8 | Object Detection Model |
| OpenCV | Image & Video Processing |
| Pandas | Data Processing |
| Pillow (PIL) | Image Handling |
| Ultralytics | YOLO Implementation |

---

# 📂 Project Structure

```text
SmartVision_AI
│
├── __pycache__
│
├── images
│
├── logs
│
├── models
│
├── outputs
│
├── runs
│   └── detect
│       └── predict
│
├── screenshots
│   ├── detection_result.png
│   ├── face_detection.png
│   ├── face_original.png
│   ├── phone_detection.png
│   ├── phone_original.png
│   ├── remote_detection.png
│   └── remote_original.png
│
├── venv
│
├── app.py
│
├── config.py
│
├── detection_history.csv
│
├── detector.py
│
├── README.md
│
├── requirements.txt
│
├── webcam.py
│
└── yolov8s.pt
```

---

# 📸 Screenshots

### 🏠 Home Page

![Home Page](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/home_page.png)

---

### 📷 Image Detection

![Image Detection](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/image_detection.png)

---

### 📹 Live Webcam Detection

![Webcam Detection](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/webcam_detection.png)

---

### 📊 Analytics Dashboard

![Analytics](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/analytics.png)

---

### 📱 Phone Detection

#### Original Image

![Phone Original](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/phone_original.png)

#### Detection Result

![Phone Detection](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/phone_object_detection.png)

---

### 📚 Book Detection

#### Original Image

![Book Original](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/book_original.png)

#### Detection Result

![Book Detection](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/book_object_detection.png)

---

### 😀 Face Detection

#### Original Image

![Face Original](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/face_original.png)

#### Detection Result

![Face Detection](https://github.com/YourGitHubUsername/SmartVision_AI/blob/main/screenshots/face_object_detection.png)

---

# ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/YourGitHubUsername/SmartVision_AI.git
```

---

### 2. Navigate to Project Folder

```bash
cd SmartVision_AI
```

---

### 3. Create Virtual Environment

```bash
python -m venv venv
```

---

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

---

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 6. Run Application

```bash
streamlit run app.py
```

---

### 7. Open Browser

```
http://localhost:8501
```

---

# 📊 Analytics

The application automatically generates:

- Detection History
- Confidence Scores
- Total Objects
- Unique Categories
- Most Detected Object
- Images Processed
- Downloadable CSV Report

---

# 🚀 Future Enhancements

- Multiple Image Detection
- Video File Detection
- Object Tracking
- Custom YOLO Model Support
- Dark/Light Theme Switching
- PDF Detection Reports
- Cloud Deployment
- AI Performance Monitoring

---

# 👨‍💻 Author

**Abhigna Nadupalli**

AI & Data Science Student

Python Developer | Computer Vision Enthusiast

---

# ⭐ Support

If you like this project, please consider giving it a **⭐ Star** on GitHub.    