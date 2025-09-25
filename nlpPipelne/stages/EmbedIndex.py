import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Optional: pretty progress
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback no-op

# Embeddings
import torch
from sentence_transformers import SentenceTransformer

# FAISS
try:
    import faiss
except ImportError as e:
    raise SystemExit(
        "faiss-cpu is not installed. Install with:\n\n  pip install faiss-cpu\n"
    ) from e


# -----------------------------
# Config
# -----------------------------
INDEX_DIR = "vectorStore"
INDEX_NAME = "faiss_index"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.jsonl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 128
NORMALIZE = True  # cosine sim behavior with Inner Product index


# -----------------------------
# Helpers
# -----------------------------
def _device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _combine_chunk_text(sentences: List[str], summary: str) -> str:
    base = (summary or "").strip()
    body = " ".join((sentences or []))
    if base and body:
        return f"{base}\n\n{body}"
    return base or body


def _iter_stage4_chunks(doc) -> Tuple[List[str], List[Dict]]:
    texts, metas = [], []
    doc_id = doc.get("doc_id")
    file_type = doc.get("file_type")
    file_path = doc.get("file_path")
    doc_summary = doc.get("doc_summary", "")

    for ch in doc.get("chunks", []):
        chunk_id = ch.get("chunk_id")
        sentences = ch.get("sentences", [])
        summary = ch.get("summary", "") or doc_summary

        text = _combine_chunk_text(sentences, summary).strip()
        if not text:
            continue

        text_hash = _hash_text(text)

        meta = {
            "doc_id": doc_id,
            "file_type": file_type,
            "file_path": file_path,
            "chunk_id": chunk_id,
            "text_hash": text_hash,
            "summary": summary,
            "sentences": sentences,
        }
        texts.append(text)
        metas.append(meta)
    return texts, metas


def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i: i + batch_size]
        with torch.inference_mode():
            vecs = model.encode(
                batch,
                batch_size=min(batch_size, 256),
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE,
                show_progress_bar=False,
            )
        all_vecs.append(vecs)
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 384), dtype=np.float32)


def _build_faiss_index(embeddings: np.ndarray):
    if embeddings.size == 0:
        raise ValueError("No embeddings to index.")
    dim = embeddings.shape[1]
    if NORMALIZE:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def _save_index(index, embeddings: np.ndarray, metadatas: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / f"{INDEX_NAME}.faiss"))
    np.save(out_dir / EMBEDDINGS_FILE, embeddings.astype(np.float32))
    with open(out_dir / METADATA_FILE, "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def _load_index(out_dir: Path):
    index_path = out_dir / f"{INDEX_NAME}.faiss"
    metadata_path = out_dir / METADATA_FILE

    # Load FAISS index (create empty if not exists)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        # create empty index (adjust dimension as needed)
        index = faiss.IndexFlatL2(384)  # replace 384 with your embedding dim

    metas = []

    # Load metadata safely
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    metas.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: skipping invalid JSON line: {line}")

    return index, metas

# -----------------------------
# Public: build + save with append + dedup
# -----------------------------
def indexing(input_json: dict, index_dir: str = INDEX_DIR, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
    out_dir = Path(index_dir)

    print("Collecting chunks…")
    new_texts, new_metas = _iter_stage4_chunks(input_json)
    if not new_texts:
        raise ValueError("No chunks found to embed.")

    device = _device_str()
    print(f"Loading embedding model on {device}: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    print(f"Embedding {len(new_texts)} new chunks (batch_size={batch_size}, normalize={NORMALIZE})…")
    new_embeddings = _embed_texts(model, new_texts, batch_size=batch_size)

    faiss_path = out_dir / f"{INDEX_NAME}.faiss"
    metadata_path = out_dir / METADATA_FILE
    embeddings_path = out_dir / EMBEDDINGS_FILE

    if faiss_path.exists() and metadata_path.exists() and embeddings_path.exists():
        print("Loading existing FAISS index + metadata + embeddings…")
        index, existing_metas = _load_index(out_dir)
        existing_embeddings = np.load(embeddings_path)

        existing_hashes = {m["text_hash"] for m in existing_metas}

        filtered_texts, filtered_metas, filtered_embeddings = [], [], []
        for t, m, e in zip(new_texts, new_metas, new_embeddings):
            if m["text_hash"] not in existing_hashes:
                filtered_texts.append(t)
                filtered_metas.append(m)
                filtered_embeddings.append(e)

        if filtered_embeddings:
            filtered_embeddings = np.vstack(filtered_embeddings)
            index.add(filtered_embeddings.astype(np.float32))
            metas = existing_metas + filtered_metas
            all_embeddings = np.vstack([existing_embeddings, filtered_embeddings])
            print(f"Added {len(filtered_embeddings)} new vectors. Total vectors: {index.ntotal}")
        else:
            metas = existing_metas
            all_embeddings = existing_embeddings
            print("No new unique vectors to add.")
    else:
        print("No existing index found. Creating new FAISS index…")
        index = _build_faiss_index(new_embeddings)
        metas = new_metas
        all_embeddings = new_embeddings
        print(f"Created index with {index.ntotal} vectors.")

    print(f"Saving index, embeddings & metadata to: {out_dir.resolve()}")
    _save_index(index, all_embeddings, metas, out_dir)
    print("✅ Stage 5 complete.")
    print(f"- Index: {out_dir / (INDEX_NAME + '.faiss')}")
    print(f"- Embeddings: {out_dir / EMBEDDINGS_FILE}")
    print(f"- Metadata: {out_dir / METADATA_FILE}")


# -----------------------------
# Tiny demo search
# -----------------------------
def search(query: str, top_k: int = 3, index_dir: str = INDEX_DIR, model_name: str = MODEL_NAME):
    out_dir = Path(index_dir)
    print(f"{out_dir}/{INDEX_NAME}.faiss")
    if not (out_dir / f"{INDEX_NAME}.faiss").exists():
        raise FileNotFoundError("Index not built yet.")

    device = _device_str()
    model = SentenceTransformer(model_name, device=device)

    print("Loading FAISS index + metadata…")
    index, metas = _load_index(out_dir)

    print(f"Encoding query on {device}: {query}")
    with torch.inference_mode():
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=NORMALIZE, show_progress_bar=False).astype(np.float32)

    distances, ids = index.search(q, top_k)
    results = []
    for i, (idx, score) in enumerate(zip(ids[0], distances[0]), start=1):
        m = metas[idx]
        results.append({
            "rank": i,
            "score": float(score),
            "doc_id": m.get("doc_id"),
            "chunk_id": m.get("chunk_id"),
            "file_path": m.get("file_path"),
            "summary": m.get("summary"),
        })
    print(json.dumps({"query": query, "results": results}, indent=2, ensure_ascii=False))
    return results
