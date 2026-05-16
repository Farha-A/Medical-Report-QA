"""Report tab: generate a radiology report via MedGemma or ColPali + Groq."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from cxr.colpali import colpali_rank, format_retrieval, load_colpali
from cxr.config import Settings
from cxr.groq_client import synthesize_with_groq
from cxr.medgemma import load_medgemma, medgemma_generate_stream


def render_report_tab(image: Image.Image | None, settings: Settings) -> None:
    if image is None:
        st.info("Upload an image above to generate a report.")
        return

    if not st.button("Generate report", type="primary"):
        return

    report = ""
    retrieval_to_show = None

    if settings.backend.startswith("MedGemma"):
        processor, model = load_medgemma()
        st.subheader("Generated report")
        with st.spinner("Generating report..."):
            report = st.write_stream(
                medgemma_generate_stream(processor, model, image, settings.max_new, settings.temperature)
            )
    else:
        processor, model = load_colpali(settings.colpali_id, settings.colpali_adapter_path)
        with st.spinner("Scoring findings..."):
            ranked = colpali_rank(processor, model, settings.colpali_id, image, settings.top_k)
        retrieved = format_retrieval(ranked)
        if settings.use_groq:
            with st.spinner(f"Synthesizing with Groq ({settings.groq_model})..."):
                try:
                    report = synthesize_with_groq(ranked, settings.groq_model)
                    if settings.show_retrieval:
                        retrieval_to_show = retrieved
                except Exception as e:
                    st.warning(f"Groq synthesis failed: {e}. Falling back to retrieved findings.")
                    report = retrieved
        else:
            report = retrieved
        st.subheader("Generated report")
        st.write(report)
        if retrieval_to_show is not None:
            with st.expander("Retrieved findings (ColPali)"):
                st.write(retrieval_to_show)

    st.download_button(
        "Download as .txt",
        data=report,
        file_name=f"{Path(st.session_state['xray_name']).stem}_report.txt",
    )
