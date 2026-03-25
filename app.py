"""
Streamlit front-end for the Computer-Aided Diagnosis Assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
import torch
from openai import OpenAI

from dataset import load_single_image_from_bytes
from model import build_unet, predict, predict_binary
from heatmap import probability_map_to_heatmap
from metadata import extract_metadata, format_metadata_for_prompt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CAD Assistant — Ultrasound Segmentation",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Build (or load) the U-Net once and cache it across reruns."""
    model = build_unet()
    weights_path = "unet_resnet34.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        st.toast("Loaded trained weights from disk.")
    else:
        st.toast("No trained weights found — using ImageNet-initialised encoder.")
    return model


def generate_report(meta_text: str, api_key: str) -> str:
    """Call OpenAI gpt-4o-mini to produce a 3-sentence radiology summary."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a board-certified radiologist assistant. "
                    "Given quantitative metadata extracted from an ultrasound "
                    "segmentation mask, write exactly 3 concise, professional "
                    "sentences summarising the findings. Do NOT hallucinate "
                    "information beyond what the metadata provides."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Below is the metadata extracted from the segmentation mask "
                    "of a breast ultrasound image.\n\n"
                    f"{meta_text}\n\n"
                    "Please provide a 3-sentence professional radiology summary."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Computer-Aided Diagnosis Assistant")
st.markdown(
    "Upload a breast-ultrasound image to run U-Net segmentation, "
    "view the diagnostic heatmap, and generate an AI radiology report."
)

with st.sidebar:
    st.header("Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")
    threshold = st.slider("Binarisation threshold", 0.0, 1.0, 0.5, 0.05)

uploaded = st.file_uploader(
    "Upload an ultrasound image (.png / .jpg)",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded is not None:
    raw_bytes = uploaded.read()
    image_tensor, image_rgb = load_single_image_from_bytes(raw_bytes)

    model = get_model()

    prob_map = predict(model, image_tensor)
    binary_mask = (prob_map >= threshold).float()

    overlay = probability_map_to_heatmap(prob_map, image_tensor, alpha=0.4)

    meta = extract_metadata(binary_mask)
    meta_text = format_metadata_for_prompt(meta)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Heatmap Overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("Extracted Metadata")
    st.json(meta)

    st.subheader("AI Radiology Report")
    if not openai_key:
        st.info("Enter your OpenAI API key in the sidebar to generate a report.")
        st.code(meta_text, language="text")
    else:
        with st.spinner("Generating report via GPT-4o-mini..."):
            report = generate_report(meta_text, openai_key)
        st.success(report)
