"""Sidebar widget rendering; returns a populated Settings dataclass."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from cxr.config import COLPALI_ADAPTER, COLPALI_MODEL, Settings


def render_sidebar() -> Settings:
    with st.sidebar:
        st.header("Settings")
        backend = st.radio("Backend", ["MedGemma (generative)", "ColPali (retrieval)"])
        st.markdown("---")

        with st.expander("MedGemma", expanded=backend.startswith("MedGemma")):
            max_new = st.slider("Max new tokens", 64, 1024, 256, step=32)
            temperature = st.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, step=0.05)

        with st.expander("ColPali", expanded=not backend.startswith("MedGemma")):
            colpali_id = st.text_input("ColPali model id", value=COLPALI_MODEL)
            use_colpali_adapter = st.checkbox(
                "Use fine-tuned ColPali adapter",
                value=Path(COLPALI_ADAPTER).exists(),
            )
            top_k = st.slider("Top-K findings", 1, 10, 5)

        with st.expander("Gemini LLM"):
            gemini_available = bool(os.environ.get("GOOGLE_API_KEY"))
            if not gemini_available:
                st.warning("GOOGLE_API_KEY not set — add it to .env.")
            use_gemini = st.checkbox(
                "Synthesize report with Gemini",
                value=gemini_available,
                disabled=not gemini_available,
                help=(
                    "Pass retrieved ColPali findings to Gemini for a freeform report. "
                    "Also used for ColPali QA answers. Requires GOOGLE_API_KEY in .env."
                ),
            )
            gemini_model = st.text_input(
                "Gemini model",
                value=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                disabled=not gemini_available,
            )
            show_retrieval = st.checkbox(
                "Show retrieved findings",
                value=True,
                help="Display the ranked candidate findings ColPali retrieved.",
            )

        st.markdown("---")
        st.caption("Outputs are not a substitute for clinical judgement.")

    return Settings(
        backend=backend,
        max_new=max_new,
        temperature=temperature,
        colpali_id=colpali_id,
        colpali_adapter_path=COLPALI_ADAPTER if use_colpali_adapter else None,
        top_k=top_k,
        gemini_available=gemini_available,
        use_gemini=use_gemini,
        gemini_model=gemini_model,
        show_retrieval=show_retrieval,
    )
