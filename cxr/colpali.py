"""ColPali model loader, candidate ranking, saliency mapping, and heatmap overlay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from cxr.config import CANDIDATE_FINDINGS


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


def colpali_rank(
    processor, model, model_id: str, image: Image.Image, top_k: int
) -> list[tuple[str, float]]:
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
    patch_emb = image_embeddings[0][image_mask[0]]             # (n_px*n_py, dim)
    grid = patch_emb.reshape(n_py, n_px, -1).permute(1, 0, 2) # (n_px, n_py, dim)
    sim = torch.einsum("nk,ijk->nij", query_embeddings[0].float(), grid.float())
    return sim.max(dim=0).values.cpu()                         # (n_px, n_py)


def heatmap_overlay(image: Image.Image, saliency: torch.Tensor) -> Image.Image:
    """Overlay a colour heatmap (blue→red) representing patch saliency on the image."""
    sal = saliency.numpy().astype(np.float32)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_hw = sal.T                                             # (n_py, n_px) → H×W
    sal_img = Image.fromarray((sal_hw * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BICUBIC
    )
    sal_np = np.array(sal_img).astype(np.float32) / 255.0     # H×W in [0,1]

    # Anchor colours: blue → cyan → green → yellow → red
    anchors = np.array([
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    n = len(anchors) - 1
    idx = np.clip((sal_np * n).astype(int), 0, n - 1)         # H×W, which segment
    frac = (sal_np * n - idx)[..., None]                       # H×W×1, position in segment
    lo = anchors[idx]
    hi = anchors[np.minimum(idx + 1, n)]
    color = lo + frac * (hi - lo)                              # H×W×3 in [0,1]
    color_img = Image.fromarray((color * 255).astype(np.uint8), mode="RGB")

    base = image.convert("RGBA")
    overlay = color_img.convert("RGBA")
    r, g, b, _ = overlay.split()
    alpha = Image.fromarray((sal_np * 180).astype(np.uint8))
    overlay = Image.merge("RGBA", (r, g, b, alpha))
    composite = Image.alpha_composite(base, overlay)
    return composite.convert("RGB")
