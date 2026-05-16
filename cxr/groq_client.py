"""Groq LLM client calls for report synthesis and RAG-backed QA."""
from __future__ import annotations

import base64
import io

from PIL import Image

from cxr.config import GROQ_SYSTEM_PROMPT, RAG_QA_SYSTEM_PROMPT


def rag_qa_answer(
    image: Image.Image,
    question: str,
    image_context: str,
    retrieved: list[dict],
    groq_model_id: str,
) -> str:
    from groq import Groq

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    refs = "\n".join(
        f"- Q: {r['question']}\n  A: {r['answer']}  [label: {r['label']}, score {r['score']:.2f}]"
        for r in retrieved
    )
    user_text = (
        f"Image-side context:\n{image_context}\n\n"
        f"Reference Q/A pairs (top {len(retrieved)} by TF-IDF over a labelled corpus):\n{refs}\n\n"
        f"Question: {question}"
    )
    client = Groq()
    resp = client.chat.completions.create(
        model=groq_model_id,
        messages=[
            {"role": "system", "content": RAG_QA_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def synthesize_with_groq(ranked: list[tuple[str, float]], model_id: str) -> str:
    from groq import Groq

    client = Groq()
    bullets = "\n".join(
        f"- {text}  (relevance score {score:.2f})" for text, score in ranked
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user", "content": f"Retrieved candidate findings:\n{bullets}\n\nWrite the report."},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
