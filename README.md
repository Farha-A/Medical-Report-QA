# MedGemma CXR Report Generator

A Streamlit app that analyses chest X-rays using
[MedGemma 4B IT](https://huggingface.co/google/medgemma-4b-it) and
[ColPali v1.3](https://huggingface.co/vidore/colpali-v1.3).
Two tabs — **Report** generation and **QA** (question answering).

---

## Features

| Capability | How it works |
|---|---|
| **Generative reports** | MedGemma 4B IT produces freeform FINDINGS / IMPRESSION text from the uploaded X-ray. |
| **Retrieval-based reports** | ColPali scores the X-ray against 20 canonical candidate findings and returns the top-K matches. |
| **Groq LLM synthesis** | Optionally passes ColPali's ranked findings to a Groq-hosted LLM (default: Llama 4 Scout 17B) to write a polished report. |
| **RAG-backed QA** | A TF-IDF index over a labelled Q/A corpus retrieves reference pairs; Groq answers the question using the image, a model-generated description, and the retrieved context. |
| **Saliency heatmaps** | ColPali patch-level similarity maps are overlaid on the X-ray as a blue→red heatmap during QA, highlighting which regions drove the answer. |

---

## Project Structure

```
.
├── app.py                          # Entrypoint — delegates to cxr.ui.main
├── requirements.txt
├── .env                            # HF_TOKEN, GROQ_API_KEY, GROQ_MODEL (not committed)
│
├── cxr/                            # Core package
│   ├── __init__.py                 # Bootstrap: warning filters, dotenv, torch threads
│   ├── config.py                   # Constants, prompts, candidate findings, Settings dataclass
│   ├── medgemma.py                 # MedGemma loader, streaming generation, image description
│   ├── colpali.py                  # ColPali loader, candidate ranking, saliency maps, heatmap overlay
│   ├── groq_client.py              # Groq LLM calls — report synthesis & RAG QA
│   ├── rag.py                      # TF-IDF index + cosine-similarity retrieval over Q/A corpus
│   └── ui/                         # Streamlit interface
│       ├── __init__.py
│       ├── main.py                 # Page config, image uploader, tab routing
│       ├── sidebar.py              # Backend / model / Groq settings → Settings dataclass
│       ├── report_tab.py           # Report generation (MedGemma or ColPali + Groq)
│       └── qa_tab.py               # Chat-style QA with saliency overlays & retrieved references
│
├── data/
│   └── generated_questions_answers_0-24.csv  # Labelled Q/A corpus for RAG
│
└── checkpoints/
    └── colpali-cxr-lora/           # Fine-tuned ColPali LoRA adapter (if available)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_...           # Hugging Face — MedGemma is gated; accept the licence first
GROQ_API_KEY=gsk_...      # Required for report synthesis & QA tab
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct   # optional, this is the default
```

---

## Usage

```bash
streamlit run app.py
```

Upload a chest X-ray, configure settings in the sidebar, and use either tab:

### Report tab

Pick a backend in the sidebar:

- **MedGemma (generative)** — writes a freeform FINDINGS / IMPRESSION report. Configurable
  temperature and max-token length.
- **ColPali (retrieval)** — scores the X-ray against candidate findings and returns the
  top-K. Optionally loads a fine-tuned LoRA adapter from `checkpoints/colpali-cxr-lora/`.
  When Groq is enabled, the ranked findings are passed to a Groq-hosted LLM to produce a
  polished narrative report.

### QA tab

Type a question about the uploaded X-ray. The pipeline:

1. **Image context** — ColPali backend produces saliency heatmaps and ranked findings;
   MedGemma backend generates a short image description.
2. **Corpus retrieval** — TF-IDF cosine similarity retrieves the top-K reference Q/A pairs
   from `data/generated_questions_answers_0-24.csv`.
3. **Answer** — Groq LLM synthesises an answer from the image, the context, and the
   retrieved references.

Saliency heatmaps (ColPali mode) are displayed inline alongside the answer.

---

## Notes

- Outputs are for **research and educational use only** — not a substitute for clinical judgement.
- ColPali defaults to `vidore/colpali-v1.3`. To use other ColPali variants (SmolVLM/Qwen2),
  swap the import class in `cxr/colpali.py` (`ColIdefics3` / `ColQwen2`).
