import streamlit as st
import cv2
import os
import time
import pandas as pd
from pathlib import Path
from deepface import DeepFace
import plotly.express as px
import base64

# ---------- CONFIG ----------
VIDEO_FOLDER = Path("videos")
COORDINATES_FOLDER = Path("COORDINATES")
EMOTIONS_FOLDER = Path("emotions")

for d in (VIDEO_FOLDER, COORDINATES_FOLDER, EMOTIONS_FOLDER):
    d.mkdir(exist_ok=True, parents=True)

# Emotions set (matching Affectiva colors)
EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "disgust", "fear"]
emotion_colors = {
    'happy': '#1f77b4',
    'sad': '#d62728',
    'angry': '#ff7f0e',
    'surprise': '#2ca02c',
    'neutral': '#9467bd',
    'disgust': '#8c564b',
    'fear': '#17becf'
}

SAMPLE_EVERY_SECONDS = 1

st.set_page_config(page_title="Facial Coding — Integrated App", layout="wide", page_icon="🎥")

# ---- Set black background + form labels ----
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    label, .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Logo + Title ----
logo_path = "assets/logo.png"  # Update with your actual logo path
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 20px;
            background-color: #FFEE8C;
            padding: 10px 15px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        ">
            <img src="data:image/png;base64,{encoded}" width="80">
            <h1 style="color:black; margin:0; font-size: 32px;">Facial Coding — Market Research Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("Facial Coding — Market Research Demo")

# ---------- FORM ----------
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

if not st.session_state["form_submitted"]:
    with st.form("user_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.text_input("ID")
            name = st.text_input("Name")
            age_group = st.selectbox("Age Group", ["20-24 Years", "25-30 Years", "31-35 Years"])
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
            city = st.text_input("City", value="Karachi")
            sec_cat = st.selectbox("SEC Category", ["SEC-A", "SEC-B", "SEC-C", "SEC-D"])
        
        # ✅ File uploader instead of dropdown
        uploaded_video = st.file_uploader(
            "Upload Video File", 
            type=["mp4", "mov", "avi", "mkv"], 
            help="Choose a video file from your system"
        )

        # ✅ Optional: Preview uploaded video before starting
        if uploaded_video:
            st.video(uploaded_video)

        start_btn = st.form_submit_button("Start Detection & Play Video")

    if start_btn:
        if not user_id or not name:
            st.warning("Please enter ID and Name before continuing.")
            st.stop()
        if not uploaded_video:
            st.error("Please upload a video file to continue.")
            st.stop()

        # Save uploaded video to local folder
        video_path = VIDEO_FOLDER / uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.session_state["form_submitted"] = True
        st.session_state["user"] = {
            "id": user_id,
            "name": name,
            "age_group": age_group,
            "gender": gender,
            "city": city,
            "sec_cat": sec_cat,
            "video": str(video_path),
        }
        st.rerun()

# ---------- LAYOUT ----------
if st.session_state["form_submitted"]:
    user = st.session_state["user"]

    # CSS for centered smaller video
    st.markdown(
        """
        <style>
        .centered-video { display: flex; justify-content: center; }
        .centered-video video { max-width: 70% !important; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, center_col, right_col = st.columns([1, 2, 1])

    # ---- Legend (left) ----
    with left_col:
        st.subheader("Legend")
        for emo, col in emotion_colors.items():
            st.markdown(
                f"""
                <div style="
                    border:2px solid white;
                    background-color:black;
                    color:{col};
                    font-weight:700;
                    font-size:16px;
                    padding:6px 10px;
                    margin:6px 0;
                    border-radius:8px;
                    text-align:center;">
                    {emo.title()}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ---- Video (center) ----
    with center_col:
        st.markdown('<div class="centered-video">', unsafe_allow_html=True)
        st.video(user["video"])
        st.markdown('</div>', unsafe_allow_html=True)
        chart_placeholder = center_col.empty()

    # ---- Emotion count table (right) ----
    with right_col:
        count_table_placeholder = right_col.empty()

    # ---------- DETECTION ----------
    cap_info = cv2.VideoCapture(str(user["video"]))
    fps = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    video_duration = frame_count / fps if fps else 0
    cap_info.release()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        st.error("Webcam could not be opened. Close other apps using webcam and retry.")
        st.stop()

    results_rows = []
    start_time = time.time()
    next_sample_time = start_time
    current_second = -1
    dominant_counts = {e: 0 for e in EMOTIONS}

    def analyze_frame_deepface(frame_rgb):
        try:
            resized = cv2.resize(frame_rgb, (224, 224))
            analysis = DeepFace.analyze(resized, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            raw_scores = analysis.get('emotion', {})
            normalized = {k.strip().lower(): float(v) for k, v in raw_scores.items()}
            dominant = max(normalized, key=normalized.get) if normalized else None
            return dominant, normalized
        except:
            return None, {e: 0.0 for e in EMOTIONS}

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= video_duration:   # stop when video ends
                break

            ret, frame_bgr = webcam.read()
            if not ret:
                time.sleep(0.02)
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if now >= next_sample_time:
                next_sample_time += SAMPLE_EVERY_SECONDS
                timestamp_sec = int(elapsed) + 1
                if timestamp_sec == current_second:
                    continue
                current_second = timestamp_sec

                dominant, scores = analyze_frame_deepface(rgb)
                row = {
                    "time": timestamp_sec,
                    "dominant_emotion": dominant if dominant else "neutral"
                }
                row.update(scores)
                results_rows.append(row)

                if dominant in dominant_counts:
                    dominant_counts[dominant] += 1

                # ---- Update timeline chart ----
                df = pd.DataFrame(results_rows)
                fig_line = px.line(
                    df, x="time", y=EMOTIONS,
                    color_discrete_map=emotion_colors,
                    labels={"value": "Score", "time": "Time (s)"},
                    title="Live Emotion Scores (Timeline)"
                )
                fig_line.update_layout(height=250, legend_title_text="Emotions")
                chart_placeholder.plotly_chart(fig_line, use_container_width=True)

                # ---- Update emotion count table ----
                stats_df = pd.DataFrame({"Emotion": list(dominant_counts.keys()), "Count": list(dominant_counts.values())})
                styled_table = stats_df.to_html(index=False, justify="center", border=0)\
                    .replace("<table", '<table style="color:white; background-color:black; border:1px solid white; width:100%; text-align:center; font-weight:bold;"')\
                    .replace("<th>", '<th style="color:white; background-color:#222; padding:8px;">')\
                    .replace("<td>", '<td style="color:white; background-color:black; padding:6px;">')
                count_table_placeholder.markdown(styled_table, unsafe_allow_html=True)

            time.sleep(0.02)

    finally:
        webcam.release()

    # ---------- SAVE ----------
    if results_rows:
        df = pd.DataFrame(results_rows)

        # Add user metadata
        df.insert(0, "ID", user["id"])
        df.insert(1, "Name", user["name"])
        df.insert(2, "Age_Group", user["age_group"])
        df.insert(3, "Gender", user["gender"])
        df.insert(4, "City", user["city"])
        df.insert(5, "SEC_Category", user["sec_cat"])
        df.insert(6, "Video", Path(user['video']).stem)

        # Reorder columns
        cols = ["ID","Name","Age_Group","Gender","City","SEC_Category","Video","time","dominant_emotion"] + EMOTIONS
        df = df[cols]

        # Save per-video Excel
        video_name = Path(user['video']).stem
        out_path = EMOTIONS_FOLDER / f"{video_name}.xlsx"

        if out_path.exists():
            existing_df = pd.read_excel(out_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df

        combined_df.to_excel(out_path, index=False)
        st.success(f"Results saved to {out_path}")
