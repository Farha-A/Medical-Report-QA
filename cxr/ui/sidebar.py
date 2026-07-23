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

        with st.expander("Groq LLM"):
            groq_available = bool(os.environ.get("GROQ_API_KEY"))
            if not groq_available:
                st.warning("GROQ_API_KEY not set — add it to .env.")
            use_groq = st.checkbox(
                "Synthesize report with Groq",
                value=groq_available,
                disabled=not groq_available,
                help=(
                    "Pass retrieved ColPali findings to a Groq LLM for a freeform report. "
                    "Also used for ColPali QA answers. Requires GROQ_API_KEY in .env."
                ),
            )
            groq_model = st.text_input(
                "Groq model",
                value=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
                disabled=not groq_available,
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
        groq_available=groq_available,
        use_groq=use_groq,
        groq_model=groq_model,
        show_retrieval=show_retrieval,
    )
