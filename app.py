
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import hashlib
import sqlite3
import os
import io
import json
from datetime import datetime
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Image Authenticity Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stApp {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #58a6ff;
}
.subtitle {
    color: #aab4c3;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: #1c2430;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #303b4b;
}
div.stButton > button {
    border-radius: 8px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
# =====================================================

DB_NAME = "image_history.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            date TEXT,
            sha256 TEXT,
            width INTEGER,
            height INTEGER,
            ai_label TEXT,
            ai_confidence REAL,
            metadata TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_history(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history
        (filename, date, sha256, width, height,
         ai_label, ai_confidence, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["filename"],
        data["date"],
        data["sha256"],
        data["width"],
        data["height"],
        data["ai_label"],
        data["ai_confidence"],
        json.dumps(data["metadata"])
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT * FROM history ORDER BY id DESC",
        conn
    )
    conn.close()
    return df


init_database()

# =====================================================
# MODEL CONFIGURATION
# =====================================================

DEFAULT_MODEL = "umm-maybe/AI-image-detector"

def get_model_id():
    model_id = os.getenv("HF_MODEL_ID")

    if not model_id:
        try:
            model_id = st.secrets.get("HF_MODEL_ID")
        except Exception:
            pass

    return model_id or DEFAULT_MODEL


@st.cache_resource(show_spinner="Loading image detection model...")
def load_detector(model_id):
    return pipeline(
        "image-classification",
        model=model_id,
        device=-1
    )


def run_ai_detection(image):
    model_id = get_model_id()
    detector = load_detector(model_id)

    predictions = detector(image.convert("RGB"), top_k=None)

    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]

    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    top = predictions[0]
    label = str(top["label"])
    confidence = float(top["score"])

    normalized = label.lower()

    ai_words = [
        "ai", "artificial", "fake", "generated", "synthetic"
    ]
    real_words = [
        "real", "human", "natural", "authentic", "photo"
    ]

    if any(word in normalized for word in ai_words):
        category = "Likely AI-generated"
    elif any(word in normalized for word in real_words):
        category = "Likely real photograph"
    else:
        category = "Unclassified model result"

    return {
        "model": model_id,
        "label": label,
        "category": category,
        "confidence": confidence,
        "predictions": predictions
    }


# =====================================================
# IMAGE INFORMATION
# =====================================================

def get_sha256(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def get_image_metadata(image):
    metadata = {}

    try:
        exif = image.getexif()

        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))

            if tag_name in [
                "Make",
                "Model",
                "Software",
                "DateTime",
                "DateTimeOriginal",
                "Orientation"
            ]:
                metadata[tag_name] = str(value)

    except Exception:
        pass

    if not metadata:
        metadata["EXIF Information"] = "No selected EXIF information found"

    return metadata


def analyze_image_properties(image, image_bytes):
    width, height = image.size

    rgb = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    entropy = float(cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten().dot(
        -np.log2(
            (cv2.calcHist(
                [gray], [0], None, [256], [0, 256]
            ).flatten() / (width * height) + 1e-10)
        )
    ))

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3),
        "file_size_kb": round(len(image_bytes) / 1024, 2),
        "format": image.format or "Unknown",
        "color_mode": image.mode,
        "entropy": round(entropy, 3),
        "sha256": get_sha256(image_bytes)
    }


# =====================================================
# ERROR LEVEL ANALYSIS
# =====================================================

def perform_ela(image):
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    difference = ImageChops.difference(image, compressed)

    extrema = difference.getextrema()
    max_difference = max(
        channel[1] for channel in extrema
    )

    scale = 255 / max(max_difference, 1)

    ela_image = ImageEnhance.Brightness(
        difference
    ).enhance(scale)

    diff_array = np.array(difference).astype(np.float32)

    mean_difference = float(np.mean(diff_array))

    return ela_image, mean_difference


# =====================================================
# PDF REPORT
# =====================================================

def create_pdf_report(data):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "AI Image Authenticity Analysis Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Generated: " + data["date"],
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))

    rows = [
        ["Property", "Result"],
        ["Filename", data["filename"]],
        ["Image Dimensions",
         f'{data["width"]} x {data["height"]}'],
        ["SHA-256", data["sha256"]],
        ["AI Detection Result", data["ai_label"]],
        ["AI Model Confidence",
         f'{data["ai_confidence"] * 100:.2f}%'],
        ["Detection Model", data["model"]],
        ["ELA Mean Difference",
         str(data["ela_mean"])]
    ]

    table = Table(rows, colWidths=[160, 330])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263b55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Disclaimer: This report provides automated estimates "
            "and forensic indicators. It cannot prove that an image "
            "is genuine, AI-generated, or manipulated. ELA and "
            "metadata are not definitive authenticity tests.",
            styles["BodyText"]
        )
    )

    document.build(elements)

    return buffer.getvalue()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🔍 Image Forensics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Image Analysis",
        "Analysis History",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload an image to inspect its properties, metadata, "
    "compression patterns, and AI detector estimates."
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("🔍 AI Image Authenticity Checker")

    st.markdown(
        '<p class="subtitle">'
        'Analyze images using machine learning and digital '
        'image forensics.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Image Forensics")
        st.write(
            "Inspect image properties, metadata, file size, "
            "and Error Level Analysis."
        )

    with col2:
        st.subheader("🤖 AI Image Detection")
        st.write(
            "Use a pretrained image classifier to estimate "
            "whether an image may be AI-generated."
        )

    st.markdown("---")

    history = get_history()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Analyses", len(history))

    if len(history) > 0:
        c2.metric(
            "Latest File",
            str(history.iloc[0]["filename"])[:20]
        )
    else:
        c2.metric("Latest File", "None")

    c3.metric("Detection Engine", "Hybrid")

    st.info(
        "Go to Image Analysis from the sidebar to start."
    )

# =====================================================
# IMAGE ANALYSIS PAGE
# =====================================================

elif page == "Image Analysis":

    st.title("📤 Upload & Analyze Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:

        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > 15 * 1024 * 1024:
            st.error("Maximum upload size is 15 MB.")
            st.stop()

        try:
            original_image = Image.open(
                io.BytesIO(image_bytes)
            )
            original_image.verify()

            original_image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except Exception:
            st.error("Invalid or unsupported image.")
            st.stop()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)

        with col2:
            st.subheader("Image Information")

            properties = analyze_image_properties(
                original_image,
                image_bytes
            )

            st.write("**Filename:**", uploaded_file.name)
            st.write("**Format:**", properties["format"])
            st.write(
                "**Dimensions:**",
                f'{properties["width"]} x {properties["height"]}'
            )
            st.write(
                "**File Size:**",
                f'{properties["file_size_kb"]} KB'
            )
            st.write(
                "**Aspect Ratio:**",
                properties["aspect_ratio"]
            )

        st.markdown("---")

        st.subheader("⚙️ Run Analysis")

        if st.button("🚀 Analyze Image", type="primary"):

            with st.spinner("Analyzing image..."):

                metadata = get_image_metadata(original_image)

                ela_image, ela_mean = perform_ela(
                    original_image
                )

                ai_result = None
                ai_error = None

                try:
                    ai_result = run_ai_detection(
                        original_image
                    )
                except Exception as error:
                    ai_error = str(error)

                current_date = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if ai_result:
                    ai_label = ai_result["category"]
                    ai_confidence = ai_result["confidence"]
                    model_name = ai_result["model"]
                else:
                    ai_label = "Model unavailable"
                    ai_confidence = 0.0
                    model_name = get_model_id()

                report_data = {
                    "filename": uploaded_file.name,
                    "date": current_date,
                    "sha256": properties["sha256"],
                    "width": properties["width"],
                    "height": properties["height"],
                    "ai_label": ai_label,
                    "ai_confidence": ai_confidence,
                    "model": model_name,
                    "metadata": metadata,
                    "ela_mean": round(ela_mean, 3)
                }

                save_history(report_data)

                st.session_state["last_report"] = report_data
                st.session_state["last_ai_result"] = ai_result
                st.session_state["last_ela"] = ela_image
                st.session_state["last_metadata"] = metadata
                st.session_state["last_properties"] = properties
                st.session_state["last_error"] = ai_error

        # DISPLAY RESULTS AFTER ANALYSIS

        if "last_report" in st.session_state:

            report = st.session_state["last_report"]
            ai_result = st.session_state["last_ai_result"]

            st.markdown("---")
            st.header("📊 Analysis Results")

            if ai_result:

                r1, r2 = st.columns(2)

                with r1:
                    st.metric(
                        "AI Detector Result",
                        ai_result["category"]
                    )

                with r2:
                    st.metric(
                        "Model Confidence",
                        f'{ai_result["confidence"] * 100:.2f}%'
                    )

                st.caption(
                    "Model confidence is not a calibrated probability "
                    "that the image is actually AI-generated."
                )

                st.subheader("Model Predictions")

                prediction_df = pd.DataFrame(
                    ai_result["predictions"]
                )

                prediction_df["score"] = (
                    prediction_df["score"] * 100
                ).round(2)

                prediction_df = prediction_df.rename(
                    columns={
                        "label": "Model Label",
                        "score": "Score (%)"
                    }
                )

                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning(
                    "AI detection model could not be loaded. "
                    "Forensic analysis is still available."
                )

                st.code(
                    st.session_state["last_error"] or "Unknown error"
                )

                st.caption(
                    "Check your internet connection, model access, "
                    "and installed dependencies."
                )

            st.markdown("---")

            st.subheader("🧬 Error Level Analysis")

            st.image(
                st.session_state["last_ela"],
                caption="ELA Visualization",
                use_container_width=True
            )

            st.write(
                "Mean ELA pixel difference:",
                report["ela_mean"]
            )

            st.warning(
                "ELA highlights differences after JPEG recompression. "
                "These differences can occur naturally and do not "
                "prove manipulation."
            )

            st.markdown("---")

            st.subheader("📋 EXIF Metadata")

            metadata = st.session_state["last_metadata"]

            if metadata:
                metadata_df = pd.DataFrame(
                    list(metadata.items()),
                    columns=["Metadata Field", "Value"]
                )

                st.dataframe(
                    metadata_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")

            st.subheader("🔐 Image Integrity")

            st.code(report["sha256"])

            st.caption(
                "SHA-256 identifies the uploaded file's exact byte "
                "content. It does not establish who created the image."
            )

            pdf_bytes = create_pdf_report(report)

            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="image_analysis_report.pdf",
                mime="application/pdf"
            )

# =====================================================
# HISTORY PAGE
# =====================================================

elif page == "Analysis History":

    st.title("📚 Analysis History")

    history = get_history()

    if history.empty:
        st.info("No analysis history available yet.")

    else:
        st.dataframe(
            history.drop(columns=["metadata"]),
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History CSV",
            data=csv_data,
            file_name="analysis_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            conn = sqlite3.connect(DB_NAME)
            conn.execute("DELETE FROM history")
            conn.commit()
            conn.close()

            st.rerun()

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("ℹ️ About This Project")

    st.write("""
    AI Image Authenticity Checker is a digital image analysis
    application built with Python and Streamlit.

    It combines:

    - Pretrained image classification
    - EXIF metadata inspection
    - Error Level Analysis
    - Image property analysis
    - SHA-256 file identification
    - SQLite analysis history
    - PDF report generation

    **Limitations:**

    No single detector or forensic technique can guarantee
    whether an image is authentic, AI-generated, or manipulated.

    Always interpret results carefully and use independent
    evidence for important decisions.
    """)

    st.markdown("---")

    st.caption(
        "Developed using Python | Streamlit | OpenCV | PyTorch"
    )
