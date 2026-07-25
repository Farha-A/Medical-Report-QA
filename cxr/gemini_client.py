"""Google AI Studio (Gemini) client calls for report synthesis and RAG-backed QA."""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from cxr.config import GROQ_SYSTEM_PROMPT, RAG_QA_SYSTEM_PROMPT


def _get_client():
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")
    genai.configure(api_key=api_key)
    return genai


def _pil_to_part(image: Image.Image):
    """Convert a PIL image to a Gemini-compatible inline image part."""
    import google.generativeai as genai

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(buf.getvalue()).decode(),
        }
    }


def rag_qa_answer(
    image: Image.Image,
    question: str,
    image_context: str,
    retrieved: list[dict],
    gemini_model_id: str,
) -> str:
    genai = _get_client()

    refs = "\n".join(
        f"- Q: {r['question']}\n  A: {r['answer']}  [label: {r['label']}, score {r['score']:.2f}]"
        for r in retrieved
    )
    user_text = (
        f"Image-side context:\n{image_context}\n\n"
        f"Reference Q/A pairs (top {len(retrieved)} by TF-IDF over a labelled corpus):\n{refs}\n\n"
        f"Question: {question}"
    )

    model = genai.GenerativeModel(
        model_name=gemini_model_id,
        system_instruction=RAG_QA_SYSTEM_PROMPT,
    )
    response = model.generate_content(
        [_pil_to_part(image), user_text],
        generation_config={"temperature": 0.2},
    )
    return response.text.strip()


def synthesize_with_gemini(ranked: list[tuple[str, float]], model_id: str) -> str:
    genai = _get_client()

    bullets = "\n".join(
        f"- {text}  (relevance score {score:.2f})" for text, score in ranked
    )
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=GROQ_SYSTEM_PROMPT,
    )
    response = model.generate_content(
        f"Retrieved candidate findings:\n{bullets}\n\nWrite the report.",
        generation_config={"temperature": 0.2},
    )
    return response.text.strip()
