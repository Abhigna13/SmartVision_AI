import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import cv2
import os
from datetime import datetime
CONFIDENCE_THRESHOLD = 0.25

st.set_page_config(page_title="SmartVision AI", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, rgba(74, 144, 226, 0.18), transparent 26%),
                    linear-gradient(135deg, #040816 0%, #09101f 45%, #030712 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(7, 12, 24, 0.94);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .glass-card {
        background: linear-gradient(135deg, rgba(9, 16, 32, 0.95), rgba(20, 32, 56, 0.9));
        border: 1px solid rgba(132, 153, 255, 0.24);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(18px);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #f8fbff;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.03rem;
        color: #9fb0d4;
        line-height: 1.7;
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(79, 140, 255, 0.2), rgba(139, 92, 246, 0.2));
        border: 1px solid rgba(132, 153, 255, 0.25);
        color: #dfe8ff;
        margin: 0.2rem 0.2rem 0.2rem 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return YOLO("yolov8s.pt")

def save_detection_history(objects, confidence):

    file = "detection_history.csv"

    data = {
        "Date_Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Objects_Detected": [", ".join(objects)],
        "Confidence": [confidence]
    }

    df = pd.DataFrame(data)

    if os.path.exists(file):
        old_data = pd.read_csv(file)
        df = pd.concat([old_data, df], ignore_index=True)

    df.to_csv(file, index=False)

    st.success("History Saved Successfully")

def render_glass_card(title, content, icon="✨"):
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <span style="font-size:1.2rem;">{icon}</span>
                <h4 style="margin:0; color:#f5f7ff;">{title}</h4>
            </div>
            <div style="color:#9fb0d4; line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_home_page():
    st.markdown("""
<h1 style='font-size:55px;
color:white;
font-weight:800;'>

🚀 SmartVision AI

</h1>

<h4 style='color:#74b9ff;'>

AI Vision Intelligence Platform

</h4>

<p style='font-size:18px;color:#b9c7df;'>

Experience next-generation Object Detection powered by Artificial Intelligence.

</p>

""",unsafe_allow_html=True)

    hero_col, side_col = st.columns([1.35, 0.65], gap="large")
    with hero_col:
        render_glass_card(
            "Vision Intelligence",
            "SmartVision AI combines Python, OpenCV, Streamlit, and YOLO to bring a sleek, futuristic object detection workflow to life.",
            icon="🤖",
        )
        render_glass_card(
            "Why it stands out",
            "The experience is designed to feel like a startup-grade AI product with immersive visuals, polished layout, and professional storytelling.",
            icon="🚀",
        )

    with side_col:
        render_glass_card(
            "Project Highlights",
            "<ul style='margin:0; padding-left:1rem; color:#9fb0d4;'><li>Image upload and instant analysis</li><li>Live webcam detection ready</li><li>Premium dashboard presentation</li></ul>",
            icon="✨",
        )

def render_image_detection_page(model):
    st.markdown("### 📷 Image Detection")
    st.markdown("Upload an image and let the AI analyze it with a polished detection workflow.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        left_col, right_col = st.columns([1, 1], gap="large")
        with left_col:
            render_glass_card("Source Image", "", icon="🖼️")
            st.image(image, width=500)

        with right_col:
            render_glass_card("Detection Output", "", icon="🎯")
            with st.spinner("Analyzing your image..."):
                results = model(image, conf = 0.25)

            result_image = results[0].plot(conf = True)
            st.image(result_image, width=500)
            
            boxes = results[0].boxes.data.tolist()

            if boxes:
                names = [results[0].names[int(box[5])] for box in boxes]
                confidence_scores = [round(float(box[4]) * 100, 1) for box in boxes]
                counts = Counter(names)

                if "images_processed" not in st.session_state:

                    st.session_state["images_processed"] = 0

                st.session_state["images_processed"] += 1
                st.session_state["most_detected"] = counts.most_common(1)[0][0]
                st.session_state["total_objects"] = len(boxes)
                st.session_state["unique_objects"] = len(counts)
                st.session_state["avg_confidence"] = round(sum(confidence_scores) / len(confidence_scores), 1)
                save_detection_history(names,round(sum(confidence_scores) / len(confidence_scores),1))
                st.session_state["max_confidence"] = max(confidence_scores)
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric(
                "Total Objects",
                len(boxes)
                )

                summary_col2.metric(
                "Categories",
                len(counts)
                )

                summary_col3.metric(
                "Confidence",
                f"{round(sum(confidence_scores)/len(confidence_scores),1)}%"
                )
                st.markdown("#### Detection Summary")

                for name, count in counts.most_common(4):
                    conf_value = round(
                        sum(conf for detected_name, conf in zip(names, confidence_scores) if detected_name == name) / count,
                        1,
                    )
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-top:0.7rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:#f5f7ff;">{name}</strong>
                                <span style="color:#69e7ff;">{conf_value:.1f}% confidence</span>
                            </div>
                            <div style="margin-top:0.6rem; height:8px; border-radius:999px; overflow:hidden; background:rgba(255,255,255,0.12);">
                                <div style="height:100%; width:{conf_value:.1f}%; background:linear-gradient(90deg, #4f8cff, #8b5cf6);"></div>
                            </div>
                            <div style="margin-top:0.45rem; color:#9fb0d4;">Detected {count} object(s)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No objects detected in this image.")
    else:
        st.info("Upload an image to begin the detection experience.")


def render_webcam_page(model):

    st.markdown("### 🎥 Live Webcam Detection")

    render_glass_card(
        "Real-Time AI Camera",
        "Detect objects live using YOLO deep learning model.",
        icon="📹",
    )

    start_camera = st.toggle("Start Camera")

    FRAME_WINDOW = st.image([])

    if start_camera:

        cap = cv2.VideoCapture(0)

        try:
            stop_button = st.button("⏹ Stop Camera")

            while cap.isOpened() and not stop_button:

                success, frame = cap.read()

                if not success:
                    st.error("Camera not available")
                    break

                results = model(frame, conf=CONFIDENCE_THRESHOLD)

                annotated_frame = results[0].plot()

                annotated_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                FRAME_WINDOW.image(
                    annotated_frame,
                    channels="RGB"
               )

        finally:
            cap.release()

    else:
        st.info("Enable Start Camera to begin live detection.")

def render_analytics_page():
    st.markdown("### 📊 Analytics")
    download_report()
    metrics = st.columns(6)
    metrics[0].metric(
    "Objects Detected",
    st.session_state.get("total_objects", 0)
)

    metrics[1].metric(
    "Unique Objects",
    st.session_state.get("unique_objects", 0)
)

    metrics[2].metric(
    "Average Confidence",
    f"{st.session_state.get('avg_confidence',0)}%"
)

    metrics[3].metric(
    "Highest Confidence",
    f"{st.session_state.get('max_confidence',0)}%"
)

    metrics[4].metric(
    "Images Processed",
    st.session_state.get("images_processed", 0)
)

    metrics[5].metric(
    "Most Detected",
    st.session_state.get("most_detected", "None")
)

    trend_data = pd.DataFrame(
{
    "Confidence": [
        st.session_state.get("avg_confidence",0)
    ],
    "Objects": [
        st.session_state.get("total_objects",0)
    ],
},
    index=["Current Detection"]
    )

    render_glass_card("Performance Trend", "", icon="📈")
    st.bar_chart(trend_data)

def download_report():

    file = "detection_history.csv"

    if os.path.exists(file):

        df = pd.read_csv(file)

        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Detection Report",
            data=csv,
            file_name="SmartVision_AI_Report.csv",
            mime="text/csv"
        )

    else:
        st.info("No detection history available.")

def main():
    model = get_model()

    with st.sidebar:
        try:
            st.image("logo.png", width=120)
        except:
            pass

        st.markdown("""
        <h2 style='color:white;margin-bottom:0px;'>
        🧠 SmartVision AI
        </h2>

        <p style='color:#8fd3ff;font-size:16px;'>
        AI Vision Intelligence Platform
        </p>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Home", "Image Detection", "Live Webcam Detection", "Analytics"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Built with**")
        st.markdown("<span class='pill'>YOLO</span><span class='pill'>OpenCV</span><span class='pill'>Python</span><span class='pill'>Streamlit</span>", unsafe_allow_html=True)

    if page == "Home":
        render_home_page()
    elif page == "Image Detection":
        render_image_detection_page(model)
    elif page == "Live Webcam Detection":
        render_webcam_page(model)
    else:
        render_analytics_page()

    st.markdown(
    """
    <div style='
    margin-top:2rem;
    text-align:center;
    padding:1rem;
    color:#8da0c7;
    border-top:1px solid rgba(255,255,255,0.08);'>

    🚀 SmartVision AI | AI Object Detection Platform

    </div>
    """,
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()