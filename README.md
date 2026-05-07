# MedGemma CXR Report Generator

Fine-tunes [MedGemma 4B IT](https://huggingface.co/google/medgemma-4b-it) on the
[MIMIC-CXR Kaggle mirror](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset)
to generate a radiology report from a chest X-ray, and serves it via Streamlit.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Authentication:

- **Hugging Face**: MedGemma is gated. Accept the licence on the model page, then `huggingface-cli login`.
- **Kaggle**: place `kaggle.json` at `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`).

## 1. Download the dataset

```bash
python download_data.py --out data/manifest.json
```

This downloads the Kaggle dataset and emits a JSON manifest of `(image, report)` pairs.

## 2. Fine-tune (LoRA, 4-bit)

```bash
python train.py \
  --manifest data/manifest.json \
  --output_dir checkpoints/medgemma-cxr-lora \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8
```

Requires a CUDA GPU with ~12GB+ VRAM (4-bit quantisation + LoRA). Use `--max_samples 2000`
for a quick smoke test.

Add `--cpu` to skip 4-bit quantisation and run in float32 on CPU (smoke-test only; each
step takes minutes and the 4B model needs ~16 GB RAM). Keep `--max_samples` to 4–16.

## 3. Run the app

```bash
streamlit run app.py
```

Upload a chest X-ray, pick a backend in the sidebar, and the app returns a report.

Two backends are available via the sidebar:

- **MedGemma (generative)** — `google/medgemma-4b-it`, the lightest multimodal MedGemma.
  Writes a freeform FINDINGS/IMPRESSION report. Optionally loads the LoRA adapter from
  step 2.
- **ColPali (retrieval)** — defaults to `vidore/colpali-v1.3`. ColPali is a multi-vector
  retrieval model, *not* a generator: it scores the X-ray against a fixed list of
  canonical findings (defined in `app.py` as `CANDIDATE_FINDINGS`) and returns the
  top-K matches. Override the model id in the sidebar to use other ColPali variants
  (note: SmolVLM/Qwen2 variants need a different class — `ColIdefics3` / `ColQwen2` —
  swap the import in `load_colpali` if you change architectures).

## Notes

- Outputs are for research and educational use only — not a substitute for clinical judgement.
- If the manifest builder doesn't pair files in your Kaggle download, inspect the unpacked
  folder under `~/.cache/kagglehub/...` and adapt `find_pairs` in `download_data.py`.
