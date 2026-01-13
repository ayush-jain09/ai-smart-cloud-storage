import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path(__file__).resolve().parent.parent / ".index"
INDEX_DIR.mkdir(exist_ok=True)

def _index_path(user_id: int) -> Path:
    return INDEX_DIR / f"user_{user_id}_index.json"

def _safe_read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        # Try UTF-8 first; fall back to ignore errors
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        try:
            return path.read_text(errors="ignore")[:max_chars]
        except Exception:
            return ""

def build_index_for_user(user_id: int, FileObjectModel, user_storage_path: Path) -> None:
    # Build a lightweight local index: filename + tags + category + (text snippet when safe)
    files = FileObjectModel.query.filter_by(user_id=user_id).all()

    docs = []
    meta = []
    for f in files:
        p = user_storage_path / f.stored_name
        text = ""
        # Only extract text from small text/code files
        if (f.mime or "").startswith("text/") or f.category in ("Text", "Code"):
            if p.exists() and p.stat().st_size <= 800_000:  # <= ~0.8MB
                text = _safe_read_text(p)
        combined = f"{f.original_name} {f.tags or ''} {f.category} {text}"
        docs.append(combined)
        meta.append({"id": f.id, "name": f.original_name, "category": f.category, "tags": f.tags or ""})

    payload = {"meta": meta, "docs": docs}
    _index_path(user_id).write_text(json.dumps(payload), encoding="utf-8")

def search_index(user_id: int, query: str, top_k: int = 25) -> List[int]:
    p = _index_path(user_id)
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    meta = payload.get("meta", [])
    docs = payload.get("docs", [])
    if not docs:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs + [query])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    ranked = np.argsort(-sims)[:top_k]

    result_ids = []
    for idx in ranked:
        if idx < len(meta) and sims[idx] > 0:
            result_ids.append(int(meta[idx]["id"]))
    return result_ids
