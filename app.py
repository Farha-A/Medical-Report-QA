"""Streamlit app: chest X-ray -> report.

Two backends:
- MedGemma (generative): writes a freeform FINDINGS/IMPRESSION report.
- ColPali  (retrieval):  scores the image against canonical finding snippets and
                         returns the top matches as a retrieval-style report.
"""
from __future__ import annotations

import logging
import warnings

# transformers 5.x walks its own image_processing_* registry on import and
# trips its own deprecated __path__ alias for every model. Filter that noise
# here; all other transformers log output is left intact.
warnings.filterwarnings("ignore", message=r".*Accessing `__path__` from.*")


class _DropPathDeprecation(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Accessing `__path__` from" not in record.getMessage()


logging.getLogger("transformers").addFilter(_DropPathDeprecation())

import base64
import io
import os
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

torch.set_num_threads(os.cpu_count() or 1)
CPU_ONLY = not torch.cuda.is_available()

MEDGEMMA_MODEL = "google/medgemma-4b-it"      # smallest multimodal MedGemma
COLPALI_MODEL = "vidore/colpali-v1.3"          # canonical ColPali (PaliGemma-based)
COLPALI_ADAPTER = "checkpoints/colpali-cxr-lora"

QA_CORPUS_PATH = "generated_questions_answers_0-24.csv"
RAG_TOP_K = 5

GEN_PROMPT = (
    "You are a radiologist. Examine the chest X-ray and write a concise "
    "report covering FINDINGS and IMPRESSION."
)

QA_PROMPT = (
    "You are a radiologist answering questions about the chest X-ray shown. "
    "Answer concisely and directly. If the image does not show enough detail "
    "to answer confidently, say so."
)

CANDIDATE_FINDINGS = [
    "Normal cardiomediastinal silhouette and clear lung fields. No acute cardiopulmonary findings.",
    "Cardiomegaly with enlarged cardiac silhouette.",
    "Bilateral pleural effusions.",
    "Right pleural effusion with associated atelectasis.",
    "Left pleural effusion with associated atelectasis.",
    "Right pneumothorax.",
    "Left pneumothorax.",
    "Pulmonary edema with bilateral interstitial and perihilar opacities.",
    "Right lower lobe consolidation, concerning for pneumonia.",
    "Left lower lobe consolidation, concerning for pneumonia.",
    "Right upper lobe opacity, suspicious for mass or infiltrate.",
    "Left upper lobe opacity, suspicious for mass or infiltrate.",
    "Hyperinflated lungs and flattened diaphragms, consistent with COPD/emphysema.",
    "Linear atelectasis at the lung bases.",
    "Endotracheal tube in appropriate position above the carina.",
    "Right internal jugular central venous catheter terminating in the superior vena cava.",
    "Nasogastric tube with tip in the stomach.",
    "Calcified granulomas, suggestive of prior granulomatous infection.",
    "Mild degenerative changes of the thoracic spine.",
    "Lungs are clear without focal consolidation, pleural effusion, or pneumothorax.",
]

st.set_page_config(page_title="CXR Report", layout="wide")


@st.cache_resource(show_spinner="Loading MedGemma...")
def load_medgemma():
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading ColPali...")
def load_colpali(model_id: str, adapter_path: str | None):
    from colpali_engine.models import ColPali, ColPaliProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ColPali.from_pretrained(model_id, dtype=dtype).to(device)
    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = ColPaliProcessor.from_pretrained(model_id)
    return processor, model


@st.cache_resource(show_spinner="Embedding candidate findings...")
def embed_candidates(_processor, _model, model_id: str, findings: tuple):
    inputs = _processor.process_queries(list(findings)).to(_model.device)
    with torch.inference_mode():
        return _model(**inputs)


@st.cache_resource(show_spinner="Indexing reference Q/A corpus...")
def load_qa_corpus(path: str = QA_CORPUS_PATH):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = pd.read_csv(path)
    df = df[df["answer"].astype(str).str.lower() != "error"].reset_index(drop=True)
    docs = df["question"].astype(str).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z]+")
    tfidf = vectorizer.fit_transform(docs)
    return df, vectorizer, tfidf


def retrieve_qa_pairs(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity
    df, vectorizer, tfidf = load_qa_corpus()
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf)[0]
    order = np.argsort(sims)[::-1][:top_k]
    out = []
    for rank_i in order:
        row = df.iloc[int(rank_i)]
        out.append({
            "question": row["question"],
            "answer": row["answer"],
            "label": row["question_label"],
            "score": float(sims[rank_i]),
        })
    return out


def _medgemma_stream(processor, model, messages: list, max_new_tokens: int, temperature: float):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs: dict = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        streamer=streamer,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9, top_k=50)
    else:
        gen_kwargs["do_sample"] = False

    def _run():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = Thread(target=_run, daemon=True)
    thread.start()
    for chunk in streamer:
        yield chunk
    thread.join()


def medgemma_generate_stream(processor, model, image: Image.Image, max_new_tokens: int, temperature: float):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": GEN_PROMPT},
        ],
    }]
    yield from _medgemma_stream(processor, model, messages, max_new_tokens, temperature)


DESCRIBE_PROMPT = (
    "Briefly describe what is visible in this chest X-ray in 2-3 sentences. "
    "Mention any obvious findings or note that the image appears normal."
)


def medgemma_describe(processor, model, image: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ],
    }]
    return "".join(_medgemma_stream(processor, model, messages, max_new_tokens=96, temperature=0.0))


def medgemma_chat(
    processor,
    model,
    image: Image.Image,
    history: list[tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
):
    messages = []
    first_user = True
    for turn in history:
        role, text = turn["role"], turn["text"]
        if role == "user" and first_user:
            content: list = [{"type": "image", "image": image}, {"type": "text", "text": f"{QA_PROMPT}\n\n{text}"}]
            first_user = False
        else:
            content = [{"type": "text", "text": text}]
        messages.append({"role": role, "content": content})
    yield from _medgemma_stream(processor, model, messages, max_new_tokens, temperature)


def colpali_rank(processor, model, model_id: str, image: Image.Image, top_k: int) -> list[tuple[str, float]]:
    img_inputs = processor.process_images([image]).to(model.device)
    with torch.inference_mode():
        img_emb = model(**img_inputs)
    q_emb = embed_candidates(processor, model, model_id, tuple(CANDIDATE_FINDINGS))
    scores = processor.score_multi_vector(q_emb, img_emb).squeeze(-1).float().cpu().tolist()
    return sorted(zip(CANDIDATE_FINDINGS, scores), key=lambda x: x[1], reverse=True)[:top_k]


def format_retrieval(ranked: list[tuple[str, float]]) -> str:
    body = "\n".join(f"- {text}  (score {score:.2f})" for text, score in ranked)
    return f"FINDINGS (top {len(ranked)} retrieved):\n{body}"


def colpali_similarity_map(
    processor, model, image: Image.Image, query: str
) -> torch.Tensor:
    """Return per-patch saliency (n_px, n_py) for the query against the image."""
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.inference_mode():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)
    n_px, n_py = processor.get_n_patches(image.size, model.patch_size)
    image_mask = processor.get_image_mask(batch_images)
    patch_emb = image_embeddings[0][image_mask[0]]            # (n_px*n_py, dim)
    grid = patch_emb.reshape(n_py, n_px, -1).permute(1, 0, 2)  # (n_px, n_py, dim)
    sim = torch.einsum("nk,ijk->nij", query_embeddings[0].float(), grid.float())
    return sim.max(dim=0).values.cpu()                        # (n_px, n_py)


def heatmap_overlay(image: Image.Image, saliency: torch.Tensor) -> Image.Image:
    """Overlay a colour heatmap (blue→red) representing patch saliency on the image."""
    sal = saliency.numpy().astype(np.float32)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_hw = sal.T                                            # (n_py, n_px) → H×W
    sal_img = Image.fromarray((sal_hw * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BICUBIC
    )
    sal_np = np.array(sal_img).astype(np.float32) / 255.0    # H×W in [0,1]

    # Anchor colours: blue → cyan → green → yellow → red
    anchors = np.array([
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    n = len(anchors) - 1
    idx = np.clip((sal_np * n).astype(int), 0, n - 1)        # H×W, which segment
    frac = (sal_np * n - idx)[..., None]                      # H×W×1, position in segment
    lo = anchors[idx]
    hi = anchors[np.minimum(idx + 1, n)]
    color = lo + frac * (hi - lo)                             # H×W×3 in [0,1]
    color_img = Image.fromarray((color * 255).astype(np.uint8), mode="RGB")

    base = image.convert("RGBA")
    overlay = color_img.convert("RGBA")
    r, g, b, _ = overlay.split()
    alpha = Image.fromarray((sal_np * 180).astype(np.uint8))
    overlay = Image.merge("RGBA", (r, g, b, alpha))
    composite = Image.alpha_composite(base, overlay)
    return composite.convert("RGB")


def answer_with_groq_vision(image: Image.Image, question: str, model_id: str) -> str:
    """Send the heatmap-composite image + question to a Groq vision model."""
    from groq import Groq
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"
    client = Groq()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": GROQ_QA_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def colpali_qa(
    processor, model, image: Image.Image, question: str, groq_model: str
) -> tuple[Image.Image, str]:
    """Retrieve important patches for the question; answer via Groq vision model."""
    saliency = colpali_similarity_map(processor, model, image, question)
    overlay = heatmap_overlay(image, saliency)
    answer = answer_with_groq_vision(overlay, question, groq_model)
    return overlay, answer


GROQ_QA_SYSTEM_PROMPT = (
    "You are a radiologist answering a specific question about a chest X-ray. "
    "The image you are shown is the X-ray with a colour heatmap overlay: brighter "
    "regions are the image patches a retrieval model flagged as most relevant to the "
    "question. Use the question and the highlighted regions to give a concise, direct "
    "answer. Do not fabricate findings that are not supported by what is visible."
)

RAG_QA_SYSTEM_PROMPT = (
    "You are a radiologist answering a question about a chest X-ray. "
    "You will be shown the image (or a saliency-highlighted version of it), "
    "a brief image-side description from a separate model, and a few "
    "reference question/answer pairs from a labelled corpus that resemble "
    "the user's question. Use the image as the primary evidence. Treat the "
    "reference pairs as style and phrasing examples — not as ground truth "
    "for the current image. Answer concisely and directly. If the image "
    "does not show enough detail to answer confidently, say so."
)


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


GROQ_SYSTEM_PROMPT = (
    "You are a radiologist. Given a ranked list of candidate findings retrieved "
    "from a chest X-ray by an image-text retrieval model (each with a relevance "
    "score), synthesize a concise, clinically plausible report with FINDINGS "
    "and IMPRESSION sections. Treat low-score items with appropriate skepticism. "
    "Do not fabricate findings that are not supported by the list."
)


def synthesize_with_groq(ranked: list[tuple[str, float]], model_id: str) -> str:
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from environment
    bullets = "\n".join(f"- {text}  (relevance score {score:.2f})" for text, score in ranked)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user", "content": f"Retrieved candidate findings:\n{bullets}\n\nWrite the report."},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    st.title("Chest X-Ray Report Generator")
    st.caption("MedGemma (generative) or ColPali (retrieval). Research/educational use only.")

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
                help="Pass retrieved ColPali findings to a Groq LLM for a freeform report. "
                     "Also used for ColPali QA answers. Requires GROQ_API_KEY in .env.",
            )
            groq_model = st.text_input(
                "Groq model",
                value=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
                disabled=not groq_available,
            )
            show_retrieval = st.checkbox(
                "Show retrieved findings",
                value=True,
                help="Display the ranked candidate findings ColPali retrieved.",
            )

        st.markdown("---")
        st.caption("Outputs are not a substitute for clinical judgement.")

    colpali_adapter_path = COLPALI_ADAPTER if use_colpali_adapter else None

    # Uploader above the tabs — shared by both Report and QA.
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

    tab_report, tab_qa = st.tabs(["Report", "QA"])

    with tab_report:
        image = st.session_state.get("xray_image")
        if image is None:
            st.info("Upload an image above to generate a report.")
        elif st.button("Generate report", type="primary"):
            report = ""
            retrieval_to_show = None
            if backend.startswith("MedGemma"):
                processor, model = load_medgemma()
                st.subheader("Generated report")
                with st.spinner("Generating report..."):
                    report = st.write_stream(
                        medgemma_generate_stream(processor, model, image, max_new, temperature)
                    )
            else:
                processor, model = load_colpali(colpali_id, colpali_adapter_path)
                with st.spinner("Scoring findings..."):
                    ranked = colpali_rank(processor, model, colpali_id, image, top_k)
                retrieved = format_retrieval(ranked)
                if use_groq:
                    with st.spinner(f"Synthesizing with Groq ({groq_model})..."):
                        try:
                            report = synthesize_with_groq(ranked, groq_model)
                            if show_retrieval:
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

    with tab_qa:
        if not groq_available:
            st.warning("GROQ_API_KEY not set — QA requires Groq. Add it to .env.")
        else:
            qa_image = st.session_state.get("xray_image")
            if qa_image is None:
                st.info("Upload an image above to begin.")
            else:
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
                                for r in turn["retrieved"]:
                                    st.markdown(
                                        f"**Q:** {r['question']}  \n"
                                        f"**A:** {r['answer']}  \n"
                                        f"_Label: {r['label']} — score {r['score']:.2f}_"
                                    )

                prompt = st.chat_input("Ask a question about this X-ray")
                if prompt:
                    qa_history.append({"role": "user", "text": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)

                    if backend.startswith("ColPali"):
                        with st.spinner("Retrieving saliency patches and ranking findings..."):
                            cp_processor, cp_model = load_colpali(colpali_id, colpali_adapter_path)
                            ranked = colpali_rank(cp_processor, cp_model, colpali_id, qa_image, top_k)
                            saliency = colpali_similarity_map(cp_processor, cp_model, qa_image, prompt)
                            overlay = heatmap_overlay(qa_image, saliency)
                        image_for_llm = overlay
                        image_context = format_retrieval(ranked)
                    else:
                        if st.session_state.get("xray_description_for") != st.session_state.get("xray_name"):
                            with st.spinner("Generating image description (runs once per image)..."):
                                mg_processor, mg_model = load_medgemma()
                                desc = medgemma_describe(mg_processor, mg_model, qa_image)
                            st.session_state["xray_description"] = desc
                            st.session_state["xray_description_for"] = st.session_state.get("xray_name")
                        image_for_llm = qa_image
                        image_context = st.session_state["xray_description"]

                    with st.spinner("Retrieving reference Q/A pairs..."):
                        retrieved = retrieve_qa_pairs(prompt)
                    with st.spinner(f"Answering with Groq ({groq_model})..."):
                        answer = rag_qa_answer(image_for_llm, prompt, image_context, retrieved, groq_model)

                    with st.chat_message("assistant"):
                        if backend.startswith("ColPali"):
                            st.image(overlay, width="stretch")
                        st.write(answer)
                        with st.expander("Retrieved reference Q/A pairs"):
                            for r in retrieved:
                                st.markdown(
                                    f"**Q:** {r['question']}  \n"
                                    f"**A:** {r['answer']}  \n"
                                    f"_Label: {r['label']} — score {r['score']:.2f}_"
                                )

                    qa_history.append({
                        "role": "assistant",
                        "text": answer,
                        "image": overlay if backend.startswith("ColPali") else None,
                        "retrieved": retrieved,
                    })


if __name__ == "__main__":
    main()
