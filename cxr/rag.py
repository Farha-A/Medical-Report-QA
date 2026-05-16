"""TF-IDF index over the labelled Q/A corpus and cosine-similarity retrieval."""
from __future__ import annotations

import numpy as np
import streamlit as st

from cxr.config import QA_CORPUS_PATH, RAG_TOP_K


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
