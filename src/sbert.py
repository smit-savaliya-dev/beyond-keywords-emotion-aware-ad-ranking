"""M2 Sentence-BERT: resumable corpus encoding and history-profile cosine scores. Encoder pretrained, never fitted (D50)."""
import json

import numpy as np
import pandas as pd

from .config import PROCESSED, SBERT_CHUNK, SBERT_MAX_LENGTH, SBERT_MODEL, SBERT_REVISION
from .congruence import cosine_rows
from .emotion import texts_fingerprint


def mean_pool(hidden, mask):
    """Attention-mask mean of token vectors, then L2 normalised: the stack in the model's modules.json."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)


def load_encoder(device, model_name=SBERT_MODEL, revision=SBERT_REVISION, max_length=SBERT_MAX_LENGTH):
    """-> encode(list of texts) -> (n, d) float32 array."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).eval()
    target = f"cuda:{device}" if device >= 0 else "cpu"
    model.to(target)

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(target)
        with torch.no_grad():
            out = model(**batch)
        return mean_pool(out.last_hidden_state, batch["attention_mask"]).cpu().numpy().astype(np.float32)

    return encode


def encode_batches(texts, encode, batch_size):
    """Sorted by length so batches pad little; returned in input order."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    parts = [encode([texts[i] for i in order[s:s + batch_size]]) for s in range(0, len(texts), batch_size)]
    out = np.empty((len(texts), parts[0].shape[1]), dtype=np.float32)
    out[order] = np.concatenate(parts)
    return out


def encode_corpus(news, encode, chunk_dir, settings, text_col, chunk_size=2000, batch_size=64, log=print):
    """(n_items, d) embeddings of news[text_col]. Resumable; refuses to mix chunks made with different settings."""
    if (news[text_col].str.strip() == "").any():
        raise ValueError(f"empty {text_col}: an empty string is never encoded")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    settings = {**settings, "text_col": text_col, "chunk_size": chunk_size, "n_items": len(news),
                "texts_sha": texts_fingerprint(news, text_col)}
    meta = chunk_dir / "settings.json"
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"settings differ from the saved chunks in {chunk_dir}; delete that folder to encode from scratch")
    else:
        meta.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    n_chunks = -(-len(news) // chunk_size)
    frames = []
    for i in range(n_chunks):
        path = chunk_dir / f"chunk_{i:04d}.npy"
        if path.exists():
            frames.append(np.load(path))
            log(f"[skip] chunk {i + 1}/{n_chunks}")
            continue
        texts = news[text_col].iloc[i * chunk_size:(i + 1) * chunk_size].tolist()
        mat = encode_batches(texts, encode, batch_size)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, mat)
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written chunk
        frames.append(mat)
        log(f"[ok]   chunk {i + 1}/{n_chunks}")
    return np.concatenate(frames)


def compare_encoders(a, b):
    """Row-wise cosine of two embedding matrices of the same items. -> min and mean, plus the largest norm gap."""
    cos = cosine_rows(a, b)
    gap = np.abs(np.linalg.norm(a, axis=1) - np.linalg.norm(b, axis=1))
    return {"min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()), "max_norm_gap": float(gap.max())}


def embedding_paths(text, processed=PROCESSED):
    return processed / f"sbert_{text}.npy", processed / f"sbert_{text}_ids.parquet"


def load_embeddings(text, news_ids, processed=PROCESSED):
    """Embeddings in news_ids order; refuses a cache built on other items."""
    mat_path, ids_path = embedding_paths(text, processed)
    ids = pd.read_parquet(ids_path)["news_id"].tolist()
    if ids != list(news_ids):
        raise ValueError(f"{ids_path.name} does not match the news table")
    return np.load(mat_path)


def profile_cosine(profiles, items, groups, item_idx, chunk_size=SBERT_CHUNK):
    """cosine(profiles[groups[i]], items[item_idx[i]]) per candidate; 0 for a zero profile (cold start)."""
    out = np.zeros(len(groups), dtype=np.float64)
    for s in range(0, len(groups), chunk_size):
        out[s:s + chunk_size] = cosine_rows(profiles[groups[s:s + chunk_size]], items[item_idx[s:s + chunk_size]])
    return out
