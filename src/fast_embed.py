"""Embedding with ONNX Runtime instead of PyTorch: same model, less overhead.

The model's publisher ships ONNX exports of all-MiniLM-L6-v2 (plain, graph-
optimised and 8-bit quantised). This module runs one of them directly with
onnxruntime and reproduces what sentence-transformers does around the model:
tokenise (truncating at 256 tokens), mean-pool the token vectors over the real
tokens, and L2-normalise. It needs only `onnxruntime`, not `optimum`, which
would pin an older `transformers` and change Part 1's tokeniser (DECISIONS.md D7).

    encoder = OnnxEncoder("onnx/model_quint8_avx2.onnx")
    vectors = encoder.encode(texts, batch_size=64)
"""

from __future__ import annotations

import os

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # the model's own limit, as set in its sentence-transformers config
ONNX_FILES = {
    "onnx-fp32": "onnx/model.onnx",
    "onnx-optimised": "onnx/model_O3.onnx",
    "onnx-int8": "onnx/model_quint8_avx2.onnx",  # 8-bit weights; runs on any x86-64 CPU with AVX2
}


class OnnxEncoder:
    def __init__(self, onnx_file: str = ONNX_FILES["onnx-int8"], threads: int | None = None,
                 repo: str = MODEL_REPO):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count() or 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(hf_hub_download(repo, onnx_file), options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """One normalised 384-dimension vector per text, in the input order."""
        out = np.zeros((len(texts), 384), dtype=np.float32)
        # Longest first, so each batch pads to a similar length (as Part 1 does).
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for begin in range(0, len(order), batch_size):
            rows = order[begin:begin + batch_size]
            batch = self.tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                                   max_length=MAX_SEQ_LENGTH, return_tensors="np")
            feeds = {name: batch[name].astype(np.int64) for name in
                     ("input_ids", "attention_mask", "token_type_ids")
                     if name in self.input_names and name in batch}
            if "token_type_ids" in self.input_names and "token_type_ids" not in feeds:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
            tokens = self.session.run(None, feeds)[0]           # (batch, length, 384)
            mask = batch["attention_mask"][..., None].astype(np.float32)
            pooled = (tokens * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out[rows] = pooled
        return out
