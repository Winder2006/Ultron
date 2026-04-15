from __future__ import annotations

import functools
from typing import List, Tuple
import numpy as np
import onnxruntime as ort


_SESSION: ort.InferenceSession | None = None
_TOKENIZER = None


def _load_tokenizer():
    """Try to load a proper tokenizer, fall back to simple if unavailable."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from tokenizers import Tokenizer
        import os
        tok_path = "assistant/nlp/tokenizer.json"
        if os.path.exists(tok_path):
            _TOKENIZER = Tokenizer.from_file(tok_path)
            return _TOKENIZER
    except Exception:
        pass
    return None


def _simple_tokenize(texts: List[str], max_length: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple whitespace tokenizer fallback with fixed vocab.

    Returns input_ids, attention_mask, and token_type_ids shaped (N, L).
    """
    # Basic vocab - common words get consistent IDs
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 101, "[SEP]": 102}
    next_id = 1000
    tokenized: list[list[int]] = []
    max_len = 0
    for t in texts:
        # Add [CLS] at start, [SEP] at end like BERT
        tokens = t.lower().split()[:max_length - 2]
        ids = [101]  # [CLS]
        for tok in tokens:
            # Simple character-based hashing for consistent IDs
            if tok not in vocab:
                vocab[tok] = (hash(tok) % 28000) + 1000  # Map to vocab range
            ids.append(vocab.get(tok, 1))
        ids.append(102)  # [SEP]
        tokenized.append(ids)
        max_len = max(max_len, len(ids))
    
    max_len = min(max_len, max_length)
    input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    attn = np.zeros((len(texts), max_len), dtype=np.int64)
    token_type_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    
    for i, ids in enumerate(tokenized):
        length = min(len(ids), max_len)
        input_ids[i, :length] = np.array(ids[:length], dtype=np.int64)
        attn[i, :length] = 1
    
    return input_ids, attn, token_type_ids


@functools.lru_cache(maxsize=1)
def load_encoder(model_path: str = "assistant/nlp/minilm.onnx") -> ort.InferenceSession | None:
    """Load/cached ONNX encoder session.

    The default path can be replaced by any MiniLM/BGE-small ONNX model.
    """
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return sess
    except Exception:
        return None


def _tokenize_with_hf(texts: List[str], max_length: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use HuggingFace tokenizers if available for proper subword tokenization."""
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        return _simple_tokenize(texts, max_length)
    
    tokenizer.enable_padding(length=max_length, pad_id=0)
    tokenizer.enable_truncation(max_length=max_length)
    
    encoded = tokenizer.encode_batch(texts)
    
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attn = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
    
    return input_ids, attn, token_type_ids


def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """Embed texts to float32 vectors (N, D). Uses HF tokenizer or fallback.

    This expects an ONNX model with inputs: input_ids, attention_mask, token_type_ids;
    and output named "last_hidden_state" or "sentence_embedding". We perform
    simple mean pooling when needed.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    sess = load_encoder()
    if sess is not None:
        out_name = None
        for o in sess.get_outputs():
            out_name = o.name
            break
        assert out_name is not None
        # Check which inputs the model needs
        input_names = [i.name for i in sess.get_inputs()]

    embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if sess is not None:
            input_ids, attn, token_type_ids = _tokenize_with_hf(batch)
            feed = {"input_ids": input_ids, "attention_mask": attn}
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = token_type_ids
            outputs = sess.run([out_name], feed)
            arr = outputs[0]
            if arr.ndim == 3:
                mask = attn.astype(np.float32)
                mask = np.expand_dims(mask, -1)
                summed = (arr * mask).sum(axis=1)
                lens = np.maximum(mask.sum(axis=1), 1.0)
                arr = summed / lens
            embs.append(arr.astype(np.float32))
        else:
            # Hash-based fallback to 384-dim embeddings
            dim = 384
            fb = np.zeros((len(batch), dim), dtype=np.float32)
            for j, text in enumerate(batch):
                h = 0
                for tok in text.split():
                    h = (h * 1315423911 + hash(tok)) & 0xFFFFFFFF
                    idx = h % dim
                    fb[j, idx] += 1.0
            # L2 normalize
            norms = np.linalg.norm(fb, axis=1, keepdims=True) + 1e-8
            fb = fb / norms
            embs.append(fb)
    return np.vstack(embs)


