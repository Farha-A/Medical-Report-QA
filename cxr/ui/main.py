"""Streamlit app entrypoint: page config, sidebar, uploader, and tab delegation."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from cxr.ui.qa_tab import render_qa_tab
from cxr.ui.report_tab import render_report_tab
from cxr.ui.sidebar import render_sidebar


def main() -> None:
    st.set_page_config(page_title="CXR Report", layout="wide")
    st.title("Chest X-Ray Report Generator")
    st.caption("MedGemma (generative) or ColPali (retrieval). Research/educational use only.")

    settings = render_sidebar()

    uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        if st.session_state.get("xray_name") != uploaded.name:
            st.session_state["xray_image"] = image
            st.session_state["xray_name"] = uploaded.name
            st.session_state["qa_history"] = []
            st.session_state.pop("xray_description", None)
            st.session_state.pop("xray_description_for", None)
        st.image(image, caption=uploaded.name, width=320)
    else:
        st.info("Upload a chest X-ray image to begin.")

    image = st.session_state.get("xray_image")
    tab_report, tab_qa = st.tabs(["Report", "QA"])
    with tab_report:
        render_report_tab(image, settings)
    with tab_qa:
        render_qa_tab(image, settings)
