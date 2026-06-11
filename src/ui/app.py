from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from ui.client import API_URL, run_prediction


# -----------------------------
# Font helper (anti-aliased)
# -----------------------------
def _safe_font(size: int = 18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


# -----------------------------
# Resize helpers
# -----------------------------
def resize_image_and_detections(image: Image.Image, detections: pd.DataFrame, target_w: int):
    if image.width == target_w or target_w <= 0:
        return image, detections

    scale = target_w / image.width
    target_h = int(round(image.height * scale))

    resized = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    det = detections.copy()
    for c in ["x_min", "x_max"]:
        if c in det.columns:
            det[c] = (det[c].astype(float) * scale).round().astype(int)
    for c in ["y_min", "y_max"]:
        if c in det.columns:
            det[c] = (det[c].astype(float) * scale).round().astype(int)

    return resized, det


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_annotated(
    image: Image.Image,
    detections: pd.DataFrame,
    mode: Literal["all", "mitosis", "not_mitosis"] = "all",
    font_size: int = 18,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _safe_font(font_size)

    if detections.empty or "mitosis" not in detections.columns:
        return annotated

    df = detections.copy()
    if mode == "mitosis":
        df = df[df["mitosis"] == True]  # noqa: E712
    elif mode == "not_mitosis":
        df = df[df["mitosis"] == False]  # noqa: E712

    for _, row in df.iterrows():
        x_min, y_min = int(row["x_min"]), int(row["y_min"])
        x_max, y_max = int(row["x_max"]), int(row["y_max"])

        cls_name = row.get("class_name", "cell")
        mito = row.get("mitosis", None)
        mito_txt = "mitosis" if mito is True else ("not" if mito is False else "")
        label = f"{cls_name} | {mito_txt}".strip(" |")

        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_y = max(0, y_min - text_h - 2)

        draw.rectangle([x_min, text_y, x_min + text_w + 4, text_y + text_h + 4], fill="red")
        draw.text(
            (x_min + 2, text_y + 2),
            label,
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )

    return annotated


# -----------------------------
# Summary (post-filter counts)
# -----------------------------
def compute_summary(detections: pd.DataFrame) -> dict:
    total = int(len(detections))
    if total == 0 or "mitosis" not in detections.columns:
        return {"total": 0, "mitotic": 0, "non_mitotic": 0, "mitotic_index": 0.0}

    mitotic = int((detections["mitosis"] == True).sum())  # noqa: E712
    non_mitotic = int((detections["mitosis"] == False).sum())  # noqa: E712
    mitotic_index = (mitotic / total) * 100.0

    return {
        "total": total,
        "mitotic": mitotic,
        "non_mitotic": non_mitotic,
        "mitotic_index": float(mitotic_index),
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Allium cepa – Mitosis Detector", layout="wide")
st.title("Allium cepa – Cell Detection & Mitosis Index")

with st.sidebar:
    st.header("Options")

    anno_choice = st.radio(
        "Annotations",
        ["Off", "All cells", "Mitosis only", "Not in mitosis only"],
        index=1,
    )

    use_conf_filter = st.checkbox("Apply confidence threshold", value=True)
    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

    display_w = st.slider("Display width (px)", 400, 1600, 1100, 50)

    show_table = st.checkbox("Show detections table", value=False)
    download_csv = st.checkbox("Enable CSV download", value=True)

st.write("Upload an image to run detection + mitosis classification.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is None:
    st.stop()

image = Image.open(uploaded).convert("RGB")

with st.spinner("Running inference…"):
    try:
        counts, detections = run_prediction(uploaded)
    except Exception as e:
        st.error(f"API error — is the service running at {API_URL}?\n\n{e}")
        st.stop()

if use_conf_filter and "confidence" in detections.columns:
    detections = detections[detections["confidence"] >= conf_thr].reset_index(drop=True)

mode_map = {
    "Off": None,
    "All cells": "all",
    "Mitosis only": "mitosis",
    "Not in mitosis only": "not_mitosis",
}
anno_mode = mode_map[anno_choice]

disp_img, disp_det = resize_image_and_detections(image, detections, target_w=display_w)
font_size = max(14, display_w // 60)

col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.subheader("Image")
    if anno_mode is None:
        st.image(disp_img, width=display_w)
    else:
        annotated = draw_annotated(disp_img, disp_det, mode=anno_mode, font_size=font_size)
        st.image(annotated, width=display_w)

with col_right:
    st.subheader("Results")

    # Calibrated headline — server-side, computed over ALL detections
    st.metric("Mitotic index", f"{counts['mi']:.4f}")
    st.caption(f"95.45% CI: [{counts['ci_lower']:.4f}, {counts['ci_upper']:.4f}]")

    a, b, c = st.columns(3)
    a.metric("Total cells", counts["total_cells"])
    b.metric("Mitotic", counts["mitotic_cells"])
    c.metric("Non-mitotic", counts["non_mitotic_cells"])

    if use_conf_filter:
        st.caption(
            f"The MI ± CI above uses all {counts['total_cells']} detections. "
            f"The confidence filter (≥ {conf_thr:.2f}) only affects the image and table below."
        )

    if download_csv:
        csv_bytes = detections.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download detections CSV",
            data=csv_bytes,
            file_name="allium_cepa_detections.csv",
            mime="text/csv",
            use_container_width=True,
        )

if show_table:
    summary = compute_summary(detections)
    st.subheader(
        f"Detections table — {summary['total']} cells ({summary['mitotic']} mitotic) after filter"
    )
    st.dataframe(detections, use_container_width=True)
