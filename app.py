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

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MEDGEMMA_MODEL = "google/medgemma-4b-it"      # smallest multimodal MedGemma
COLPALI_MODEL = "vidore/colpali-v1.3"          # canonical ColPali (PaliGemma-based)
DEFAULT_ADAPTER = "checkpoints/medgemma-cxr-lora"

GEN_PROMPT = (
    "You are a radiologist. Examine the chest X-ray and write a concise "
    "report covering FINDINGS and IMPRESSION."
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
def load_medgemma(adapter_path: str | None):
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL,
        dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading ColPali...")
def load_colpali(model_id: str):
    from colpali_engine.models import ColPali, ColPaliProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColPali.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
    processor = ColPaliProcessor.from_pretrained(model_id)
    return processor, model


@st.cache_resource(show_spinner="Embedding candidate findings...")
def embed_candidates(_processor, _model, model_id: str, findings: tuple):
    inputs = _processor.process_queries(list(findings)).to(_model.device)
    with torch.inference_mode():
        return _model(**inputs)


def medgemma_generate(processor, model, image: Image.Image, max_new_tokens: int, temperature: float) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": GEN_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9, top_k=50)
    else:
        gen_kwargs["do_sample"] = False
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


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

        adapter = use_adapter = max_new = temperature = None
        colpali_id = top_k = use_groq = groq_model = show_retrieval = None
        if backend.startswith("MedGemma"):
            adapter = st.text_input("LoRA adapter path", value=DEFAULT_ADAPTER)
            use_adapter = st.checkbox("Use fine-tuned adapter", value=Path(adapter).exists())
            max_new = st.slider("Max new tokens", 64, 1024, 192, step=32)
            temperature = st.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, step=0.05)
        else:
            colpali_id = st.text_input("ColPali model id", value=COLPALI_MODEL)
            top_k = st.slider("Top-K findings", 1, 10, 5)
            groq_available = bool(os.environ.get("GROQ_API_KEY"))
            use_groq = st.checkbox(
                "Synthesize report with Groq LLM",
                value=groq_available,
                disabled=not groq_available,
                help="Pass the retrieved findings to a Groq-hosted LLM to write a freeform "
                     "FINDINGS/IMPRESSION report. Requires GROQ_API_KEY in .env.",
            )
            groq_model = st.text_input(
                "Groq model",
                value=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
                disabled=not use_groq,
            )
            show_retrieval = st.checkbox(
                "Show retrieved findings",
                value=True,
                help="Display the ranked candidate findings ColPali retrieved. "
                     "When Groq synthesis is on, these appear in an expander below "
                     "the report. When Groq is off, retrieval is the report itself "
                     "and this toggle has no effect.",
            )
            st.info(
                "ColPali ranks the X-ray against canonical findings. With Groq enabled, "
                "those ranked findings are turned into a synthesized report."
            )
        st.markdown("---")
        st.caption("Outputs are not a substitute for clinical judgement.")

    uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
    col_img, col_out = st.columns(2)

    if uploaded is None:
        col_img.info("Upload a chest X-ray image to begin.")
        return

    image = Image.open(uploaded).convert("RGB")
    col_img.image(image, caption=uploaded.name, width='stretch')

    if not col_out.button("Generate report", type="primary"):
        return

    if backend.startswith("MedGemma"):
        processor, model = load_medgemma(adapter if use_adapter else None)
        with st.spinner("Generating report..."):
            report = medgemma_generate(processor, model, image, max_new, temperature)
    else:
        processor, model = load_colpali(colpali_id)
        with st.spinner("Scoring findings..."):
            ranked = colpali_rank(processor, model, colpali_id, image, top_k)
        retrieved = format_retrieval(ranked)
        retrieval_to_show: str | None = None
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

    col_out.subheader("Generated report")
    col_out.write(report)
    if retrieval_to_show is not None:
        with col_out.expander("Retrieved findings (ColPali)"):
            st.write(retrieval_to_show)
    col_out.download_button(
        "Download as .txt",
        data=report,
        file_name=f"{Path(uploaded.name).stem}_report.txt",
    )


if __name__ == "__main__":
    main()
