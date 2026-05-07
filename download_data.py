"""Download the MIMIC-CXR Kaggle mirror and build an image/report manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import kagglehub
import pandas as pd

KAGGLE_SLUG = "simhadrisadaram/mimic-cxr-dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
REPORT_EXTS = {".txt"}


def find_pairs(root: Path) -> list[dict]:
    images: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS:
            images[p.stem] = p
        elif suffix in REPORT_EXTS:
            reports[p.stem] = p

    pairs: list[dict] = []
    paired_stems = set(images) & set(reports)
    for stem in sorted(paired_stems):
        text = reports[stem].read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        pairs.append({"id": stem, "image": str(images[stem]), "report": text})

    if not pairs:
        csvs = list(root.rglob("*.csv"))
        for csv in csvs:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            text_cols = [c for c in df.columns if c.lower() in {"report", "findings", "impression", "text"}]
            path_cols = [c for c in df.columns if c.lower() in {"image", "image_path", "path", "filename"}]
            if not text_cols or not path_cols:
                continue
            for _, row in df.iterrows():
                rel = str(row[path_cols[0]])
                img_path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
                if not img_path.exists():
                    matches = list(root.rglob(Path(rel).name))
                    if not matches:
                        continue
                    img_path = matches[0]
                pairs.append({
                    "id": img_path.stem,
                    "image": str(img_path),
                    "report": str(row[text_cols[0]]).strip(),
                })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/manifest.json")
    args = parser.parse_args()

    print(f"Pulling {KAGGLE_SLUG} via kagglehub...")
    root = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    print(f"Dataset root: {root}")

    pairs = find_pairs(root)
    print(f"Found {len(pairs)} image/report pairs.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Manifest written to {out_path}")


if __name__ == "__main__":
    main()
