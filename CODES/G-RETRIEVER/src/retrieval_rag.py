from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ══════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════
def chunk_articles(
    articles_df: pd.DataFrame,
    tokenizer,
    chunk_size: int = 512,
    overlap: int = 64,
    text_col: str = "content",
    id_col: str = "article_id",
) -> pd.DataFrame:
    """Split chaque article en chunks de tokens avec overlap."""
    rows = []
    for _, art in articles_df.iterrows():
        text = str(art.get(text_col, ""))
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        step = chunk_size - overlap
        for start in range(0, len(tokens), step):
            sub = tokens[start:start + chunk_size]
            if not sub:
                break
            chunk_text = tokenizer.decode(sub, skip_special_tokens=True)
            rows.append({
                "article_id": art[id_col],
                "chunk_id": f"{art[id_col]}__{start}",
                "text": chunk_text,
            })
            if start + chunk_size >= len(tokens):
                break
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Index FAISS : build / load
# ══════════════════════════════════════════════════════════════════════
def _index_paths(index_dir: str):
    p = Path(index_dir)
    return p / "index.faiss", p / "chunks.parquet"


def rag_index_exists(index_dir: str) -> bool:
    idx, chk = _index_paths(index_dir)
    return idx.exists() and chk.exists()


def build_rag_index(cfg, articles_csv: str):
    """Construit l'index FAISS et le sauvegarde."""
    print("🔨 Construction de l'index FAISS RAG...")
    Path(cfg.rag_index_dir).mkdir(parents=True, exist_ok=True)

    articles_df = pd.read_csv(articles_csv)
    print(f"  • {len(articles_df)} articles chargés")

    model = SentenceTransformer(cfg.rag_embedder_id, device=cfg.device,trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(cfg.rag_embedder_id)

    chunks_df = chunk_articles(
        articles_df, tok,
        chunk_size=cfg.chunk_size_tokens,
        overlap=cfg.chunk_overlap_tokens,
    )
    print(f"  • {len(chunks_df)} chunks générés")

    texts = chunks_df["text"].tolist()
    embs = model.encode(
        texts, 
        batch_size=64, 
        normalize_embeddings=True,
        convert_to_numpy=True, 
        show_progress_bar=True,
        task="retrieval.passage"  # <-- Indique à Jina qu'il s'agit de documents à indexer
    ).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    idx_path, chk_path = _index_paths(cfg.rag_index_dir)
    faiss.write_index(index, str(idx_path))
    chunks_df.to_parquet(chk_path)
    print(f"✅ Index sauvegardé dans {cfg.rag_index_dir}")


def load_rag_index(cfg):
    idx_path, chk_path = _index_paths(cfg.rag_index_dir)
    index = faiss.read_index(str(idx_path))
    chunks_df = pd.read_parquet(chk_path)
    print(f"✅ Index FAISS chargé ({index.ntotal} vecteurs, {len(chunks_df)} chunks)")
    return index, chunks_df


def ensure_rag_index(cfg):
    if cfg.force_rebuild_rag_index or not rag_index_exists(cfg.rag_index_dir):
        build_rag_index(cfg, cfg.articles_csv)
    return load_rag_index(cfg)


# ══════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════
def encode_questions_e5(questions, model, batch_size: int = 64) -> np.ndarray:
    texts = ["query: " + q for q in questions]
    return model.encode(
        questions, 
        batch_size=batch_size,
        normalize_embeddings=True, 
        convert_to_numpy=True,
        show_progress_bar=False,
        task="retrieval.query"   # <-- Indique à Jina qu'il s'agit d'une question
    ).astype("float32")


def faiss_retrieve(q_embs: np.ndarray, faiss_index, chunks_df: pd.DataFrame, k: int = 3,
                   return_scores: bool = False):
    scores, idxs = faiss_index.search(q_embs, k)
    chunks = [chunks_df.iloc[row]["text"].tolist() for row in idxs]
    if return_scores:
        chunk_ids = [chunks_df.iloc[row]["chunk_id"].tolist() for row in idxs]
        return chunks, scores, chunk_ids
    return chunks

