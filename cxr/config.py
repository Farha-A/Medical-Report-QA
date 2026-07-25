"""Constants, prompts, finding candidates, and the Settings dataclass."""
from __future__ import annotations

from dataclasses import dataclass

MEDGEMMA_MODEL = "google/medgemma-4b-it"
COLPALI_MODEL = "vidore/colpali-v1.3"
COLPALI_ADAPTER = "checkpoints/colpali-cxr-lora"

QA_CORPUS_PATH = "data/generated_questions_answers_0-24.csv"
RAG_TOP_K = 5

GEN_PROMPT = (
    "You are a radiologist. Examine the chest X-ray and write a concise "
    "report covering FINDINGS and IMPRESSION."
)

DESCRIBE_PROMPT = (
    "Briefly describe what is visible in this chest X-ray in 2-3 sentences. "
    "Mention any obvious findings or note that the image appears normal."
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

GROQ_SYSTEM_PROMPT = (
    "You are a radiologist. Given a ranked list of candidate findings retrieved "
    "from a chest X-ray by an image-text retrieval model (each with a relevance "
    "score), synthesize a concise, clinically plausible report with FINDINGS "
    "and IMPRESSION sections. Treat low-score items with appropriate skepticism. "
    "Do not fabricate findings that are not supported by the list."
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


@dataclass
class Settings:
    backend: str
    max_new: int
    temperature: float
    colpali_id: str
    colpali_adapter_path: str | None
    top_k: int
    gemini_available: bool
    use_gemini: bool
    gemini_model: str
    show_retrieval: bool
