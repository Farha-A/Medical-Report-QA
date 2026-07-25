"""QA tab: RAG-backed question answering using ColPali or MedGemma for image context."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from cxr.colpali import (
    colpali_rank,
    colpali_similarity_map,
    format_retrieval,
    heatmap_overlay,
    load_colpali,
)
from cxr.config import Settings
from cxr.gemini_client import rag_qa_answer
from cxr.medgemma import load_medgemma, medgemma_describe
from cxr.rag import retrieve_qa_pairs


def render_qa_tab(image: Image.Image | None, settings: Settings) -> None:
    if not settings.gemini_available:
        st.warning("GOOGLE_API_KEY not set — QA requires Gemini. Add it to .env.")
        return

    if image is None:
        st.info("Upload an image above to begin.")
        return

    if st.button("Clear chat", key="qa_clear"):
        st.session_state["qa_history"] = []

    qa_history: list[dict] = st.session_state.setdefault("qa_history", [])

    for turn in qa_history:
        with st.chat_message(turn["role"]):
            if turn.get("image") is not None:
                st.image(turn["image"], width="stretch")
            st.write(turn["text"])
            if turn.get("retrieved"):
                with st.expander("Retrieved reference Q/A pairs"):
                    _render_retrieved(turn["retrieved"])

    prompt = st.chat_input("Ask a question about this X-ray")
    if not prompt:
        return

    qa_history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if settings.backend.startswith("ColPali"):
        with st.spinner("Retrieving saliency patches and ranking findings..."):
            cp_processor, cp_model = load_colpali(
                settings.colpali_id, settings.colpali_adapter_path
            )
            ranked = colpali_rank(cp_processor, cp_model, settings.colpali_id, image, settings.top_k)
            saliency = colpali_similarity_map(cp_processor, cp_model, image, prompt)
            overlay = heatmap_overlay(image, saliency)
        image_for_llm = overlay
        image_context = format_retrieval(ranked)
    else:
        if st.session_state.get("xray_description_for") != st.session_state.get("xray_name"):
            with st.spinner("Generating image description (runs once per image)..."):
                mg_processor, mg_model = load_medgemma()
                desc = medgemma_describe(mg_processor, mg_model, image)
            st.session_state["xray_description"] = desc
            st.session_state["xray_description_for"] = st.session_state.get("xray_name")
        image_for_llm = image
        image_context = st.session_state["xray_description"]
        overlay = None

    with st.spinner("Retrieving reference Q/A pairs..."):
        retrieved = retrieve_qa_pairs(prompt)
    with st.spinner(f"Answering with Gemini ({settings.gemini_model})..."):
        answer = rag_qa_answer(image_for_llm, prompt, image_context, retrieved, settings.gemini_model)

    with st.chat_message("assistant"):
        if overlay is not None:
            st.image(overlay, width="stretch")
        st.write(answer)
        with st.expander("Retrieved reference Q/A pairs"):
            _render_retrieved(retrieved)

    qa_history.append({
        "role": "assistant",
        "text": answer,
        "image": overlay,
        "retrieved": retrieved,
    })


def _render_retrieved(pairs: list[dict]) -> None:
    for r in pairs:
        st.markdown(
            f"**Q:** {r['question']}  \n"
            f"**A:** {r['answer']}  \n"
            f"_Label: {r['label']} — score {r['score']:.2f}_"
        )
