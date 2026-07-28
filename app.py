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
        background:
            radial-gradient(circle at top left, rgba(87, 255, 173, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(7, 100, 57, 0.25), transparent 20%),
            linear-gradient(135deg, #040706 0%, #09120e 45%, #020403 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6, 13, 10, 0.98), rgba(4, 10, 8, 0.96));
        border-right: 1px solid rgba(111, 255, 178, 0.15);
    }
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }
    .hero-panel, .panel-card, .metric-card, .status-card {
        background: linear-gradient(135deg, rgba(8, 17, 13, 0.95), rgba(11, 24, 18, 0.92));
        border: 1px solid rgba(111, 255, 178, 0.16);
        border-radius: 26px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(18px);
        position: relative;
        overflow: hidden;
    }
    .hero-panel::before, .panel-card::before, .metric-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(111, 255, 178, 0.08), transparent 45%, rgba(0, 0, 0, 0));
        pointer-events: none;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        color: #f5fffa;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        max-width:750px;
        color: #8fdeae;
        line-height: 1.7;
    }
    .pill {
        display: inline-block;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        background: rgba(111, 255, 178, 0.10);
        border: 1px solid rgba(111, 255, 178, 0.24);
        color: #dff8e9;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .section-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(111, 255, 178, 0.10);
        color: #98f7b6;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    .nav-card {
        padding: 0.9rem;
        border-radius: 16px;
        border: 1px solid rgba(111, 255, 178, 0.14);
        background: rgba(111, 255, 178, 0.05);
        margin-bottom: 0.8rem;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #2f8f5a 0%, #56c781 100%);
        color: #041109;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        padding: 0.55rem 1rem;
        box-shadow: 0 10px 20px rgba(87, 255, 173, 0.18);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(87, 255, 173, 0.24);
    }
    .stAlert, .stInfo, .stSuccess, .stError {
        border-radius: 16px;
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
        "Confidence": [confidence],
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
        <div class="panel-card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <span style="font-size:1.16rem;">{icon}</span>
                <h4 style="margin:0; color:#f7fff8;">{title}</h4>
            </div>
            <div style="color:#8fdeae; line-height:1.65;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_home_page():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="section-label">◉ Cyber Vision Platform</div>
            <h1 class="hero-title">SmartVision AI</h1>
            <p class="hero-subtitle">A premium AI computer vision experience for real-time object recognition, intelligent analysis, and mission-ready visual insights.</p>
            <div style="margin-top:0.8rem;">
                <span class="pill">YOLO Detection</span>
                <span class="pill">Live Camera</span>
                <span class="pill">High Confidence</span>
                <span class="pill">Secure Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_col, side_col = st.columns([1.35, 0.65], gap="large")
    with hero_col:
        render_glass_card(
            "Vision Intelligence",
            "SmartVision AI fuses computer vision, deep learning, and polished interface design into a streamlined workflow for rapid scene understanding.",
            icon="🤖",
        )
        render_glass_card(
            "Operational Readiness",
            "The experience feels calibrated for modern AI systems with refined visuals, responsive controls, and a professional command-center aesthetic.",
            icon="⚙️",
        )
        
    with side_col:
        render_glass_card(
            "System Highlights",
            "<ul style='margin:0; padding-left:1rem; color:#8fdeae;'><li>Instant image analysis</li><li>Live webcam detection</li><li>Performance insights and reports</li></ul>",
            icon="🛰️",
        )


def render_image_detection_page(model):
    st.markdown(
        """
        <div class="hero-panel" style="margin-bottom:1rem;">
            <div class="section-label">◉ Image Analysis</div>
            <h2 style="margin:0; color:#f7fff8;">Upload an image and inspect detected objects with precision.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        left_col, right_col = st.columns([1, 1], gap="large")
        with left_col:
            render_glass_card("Source Image", "", icon="🖼️")
            st.image(image, use_container_width=True)

        with right_col:
            render_glass_card("Detection Output", "", icon="🎯")
            with st.spinner("Analyzing your image..."):
                results = model(image, conf=0.25)

            result_image = results[0].plot(conf=True)
            st.image(result_image, use_container_width=True)

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
                save_detection_history(names, round(sum(confidence_scores) / len(confidence_scores), 1))
                st.session_state["max_confidence"] = max(confidence_scores)
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Total Objects", len(boxes))
                summary_col2.metric("Categories", len(counts))
                summary_col3.metric("Confidence", f"{round(sum(confidence_scores) / len(confidence_scores), 1)}%")

                st.markdown("<div class='section-label' style='margin-top:1rem;'>◉ Detection Summary</div>", unsafe_allow_html=True)
                for name, count in counts.most_common(4):
                    conf_value = round(
                        sum(conf for detected_name, conf in zip(names, confidence_scores) if detected_name == name) / count,
                        1,
                    )
                    st.markdown(
                        f"""
                        <div class="panel-card" style="margin-top:0.7rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:#f7fff8;">{name}</strong>
                                <span style="color:#98f7b6;">{conf_value:.1f}% confidence</span>
                            </div>
                            <div style="margin-top:0.6rem; height:8px; border-radius:999px; overflow:hidden; background:rgba(255,255,255,0.10);">
                                <div style="height:100%; width:{conf_value:.1f}%; background:linear-gradient(90deg, #3bc775, #8ff5b8);"></div>
                            </div>
                            <div style="margin-top:0.45rem; color:#8fdeae;">Detected {count} object(s)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No objects detected in this image.")
    else:
        st.info("Upload an image to begin the detection experience.")


def render_webcam_page(model):
    st.markdown(
        """
        <div class="hero-panel" style="margin-bottom:1rem;">
            <div class="section-label">◉ Live Camera Feed</div>
            <h2 style="margin:0; color:#f7fff8;">Monitor the environment in real time with AI-guided analysis.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_glass_card(
        "Real-Time AI Camera",
        "Detect objects live using the YOLO model with the same intelligence applied to uploaded images.",
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
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame, channels="RGB")

        finally:
            cap.release()
    else:
        st.info("Enable Start Camera to begin live detection.")


def render_analytics_page():
    st.markdown(
        """
        <div class="hero-panel" style="margin-bottom:1rem;">
            <div class="section-label">◉ Intelligence Metrics</div>
            <h2 style="margin:0; color:#f7fff8;">Track detection performance through a refined operational dashboard.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    download_report()
    metrics = st.columns(6)
    metrics[0].metric("Objects Detected", st.session_state.get("total_objects", 0))
    metrics[1].metric("Unique Objects", st.session_state.get("unique_objects", 0))
    metrics[2].metric("Average Confidence", f"{st.session_state.get('avg_confidence', 0)}%")
    metrics[3].metric("Highest Confidence", f"{st.session_state.get('max_confidence', 0)}%")
    metrics[4].metric("Images Processed", st.session_state.get("images_processed", 0))
    metrics[5].metric("Most Detected", st.session_state.get("most_detected", "None"))

    trend_data = pd.DataFrame(
        {
            "Confidence": [st.session_state.get("avg_confidence", 0)],
            "Objects": [st.session_state.get("total_objects", 0)],
        },
        index=["Current Detection"],
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
            mime="text/csv",
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

        st.markdown(
            """
            <div class="nav-card">
                <h2 style='color:white;margin:0 0 0.2rem 0;'>🧠 SmartVision AI</h2>
                <p style='color:#8fdeae;font-size:15px;margin:0;'>AI Vision Intelligence Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["Home", "Image Detection", "Live Webcam Detection", "Analytics"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("<div class='nav-card'><strong style='color:#f7fff8;'>Built with</strong><br>" + "<span class='pill'>YOLO</span><span class='pill'>OpenCV</span><span class='pill'>Python</span><span class='pill'>Streamlit</span>" + "</div>", unsafe_allow_html=True)

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
        color:#7bc697;
        border-top:1px solid rgba(111,255,178,0.16);'>
        🚀 SmartVision AI | AI Object Detection Platform
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()