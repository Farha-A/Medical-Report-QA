"""MedGemma model loader and text-generation helpers."""
from __future__ import annotations

from threading import Thread

import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

from cxr.config import DESCRIBE_PROMPT, GEN_PROMPT, MEDGEMMA_MODEL


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


def medgemma_generate_stream(
    processor, model, image: Image.Image, max_new_tokens: int, temperature: float
):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": GEN_PROMPT},
        ],
    }]
    yield from _medgemma_stream(processor, model, messages, max_new_tokens, temperature)


def medgemma_describe(processor, model, image: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ],
    }]
    return "".join(_medgemma_stream(processor, model, messages, max_new_tokens=96, temperature=0.0))
