"""
Standalone script to evaluate G-RETRIEVER on 8 categories (musica, literatura,
folclore, gastronomia, danza, pintura, artesania, cine).

- Builds a dedicated `Config` per category (`make_config`) from the shared `FIXED_CONFIG` parameters, deriving all the paths (questions, articles subset, graph, provenance, RAG index).
- Loads the embedder and the LLM once, then reuses them across every category to save GPU memory.
- For each category: loads/filters/samples the questions, then safely loads the graph artifacts, the RAG index and the provenance (missing resources simply disable the incompatible modes).
- Runs different modes (`zero_shot`, `rag`, `kaping`, graph modes...) and collects predictions, contexts and, for graph modes, the share of context triplets actually coming from the target article.
- Can't run G-Retriever with trained modules (GNN or Linear).
- Writes a detailed CSV and an accuracy summary per category, plus global concatenated files (wide + long format) in `<CLEAN>/DATA/RESULTS_CATEGORIES/<llm_slang>`.

Usage:
    cd CODES/G-RETRIEVER
    python run_8_categories.py
"""

import gc
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Add the parent folder to PYTHONPATH so that src/ can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # CLEAN/
DATA = PROJECT_ROOT / "DATA"

from src.config import Config
from src.data_loading import load_article_ids, load_questions, filter_questions_by_articles, sample_df
from src.evaluation import SUPPORTED_MODES, run_eval_for_mode
from src.graph_utils import build_provenance_embeddings, ensure_graph_artifacts, load_provenance
from src.llm import load_llm
from src.retrieval_rag import ensure_rag_index

warnings.filterwarnings("ignore")

# ── Fixed parameters ─────────────────────────────────────────────────────────
CATEGORIES: List[str] = [
    "artesania", 
    # "literatura", 
    # "folclore",
    # "gastronomia",
    # "danza",
    # "pintura",
    # "musica", 
    # "cine",
    #"global"
]

FIXED_CONFIG = {
    "llm_id": "Qwen/Qwen2.5-3B-Instruct",              # generative model used to answer the MCQs
    "embedder_id": "jinaai/jina-embeddings-v3",         # shared encoder for the graph and RAG embeddings
    "modes_to_run": [                                   # retrieval strategies evaluated for every category
        "zero_shot", 
        "rag", 
        "kaping", 
    ],
    "retrieval_query_mode": "question_options",          # retrieval query = question + all the options
    "n_samples": None,                                   # number of sampled questions (None = all of them)
    "random_seed": 1,                                    # seed making the sampling reproducible
    "batch_size": 8,                                     # number of prompts generated in parallel
    "max_new_tokens": 5,                                 # short generation: only the answer letter is needed
    "k_rag": 5,                                          # number of chunks retrieved in RAG mode
    "chunk_size_tokens": 512,                            # chunk size when building the RAG index
    "chunk_overlap_tokens": 64,                          # overlap between two consecutive chunks
    "topk_nodes": 15,                                    # nodes kept by the graph retrieval
    "topk_edges": 20,                                    # edges kept by the graph retrieval
    "pcst_cost_e": 0.1,                                  # per-edge cost in the PCST objective
    "pcst_workers": 4,                                   # processes used to solve the PCSTs
    "topk_triplets": 15,        # default Config value    # triplets kept in KAPING mode
    "top_k_provenance": 20,                              # provenance triplets kept per question
    "force_rebuild_graph_artifacts": False,              # True = recompute the graph artifacts (embeddings, indexes)
    "force_rebuild_rag_index": False,                    # True = rebuild the RAG index from scratch
    "save_all": False,                                   # verbose logging (full prompts/contexts) in the outputs
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",  # device used for the embedder and the LLM
}

llm_slang=FIXED_CONFIG["llm_id"].split("/")[-1].replace("-", "_")
OUTPUT_DIR = DATA / "RESULTS_CATEGORIES" / llm_slang

def make_config(category: str) -> Config:
    """Builds a Config dedicated to one category using the fixed parameters."""
    cfg_dict = {
        **FIXED_CONFIG,
        "CATEGORY": category,
        # Questions/articles
        "questions_csv": str(DATA / "QUESTIONS" / "mcq_es.csv"),
        "articles_csv": str(DATA / "SUBSETS_ES" / "ARTICLES_SUBSETS_ES" / f"{category}_articles_es_disjoint.csv"),

        # Graph
        "graph_dir": str(DATA / "SUBSETS_ES" / "GRAPHS" / category),
        "graph_json_name": f"{category}_clustered_graph.json",
        "artifacts_subdir": "clustered_artifacts",
        "provenance_subdir": "provenance",
        "provenance_name": f"{category}_clustered_provenance.pkl",
        "ln_name": f"{category}_ln_triplets.csv",
        # RAG index
        "rag_index_dir": str(DATA / "RAG_INDEX" / f"rag_index_{category}"),
        "device": "cuda:1"
    }
    # Config is a dataclass: drop the fields that do not exist
    cfg_fields = {f.name for f in Config.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    return Config(**cfg_dict)

def safe_load_graph_artifacts(cfg: Config, graph_embedder) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[object], Optional[object]]:
    """Loads the graph artifacts, returns None on error."""
    try:
        nodes_df, edges_df, node2id, graph = ensure_graph_artifacts(cfg, graph_embedder)
        print(f"✅ Graph loaded: {len(nodes_df)} nodes, {len(edges_df)} edges")
        return nodes_df, edges_df, node2id, graph
    except Exception as e:
        print(f"❌ Unable to load the graph: {e}")
        return None, None, None, None

def safe_load_rag_index(cfg: Config):
    """Loads the RAG index, returns None on error."""
    try:
        return ensure_rag_index(cfg)
    except Exception as e:
        print(f"❌ Unable to load/build the RAG index: {e}")
        return None, None

def safe_load_provenance(cfg: Config, graph_embedder):
    """Loads the provenance and prepares the triplet embeddings."""
    try:
        provenance_index = load_provenance(cfg.provenance_path)
        # Encode every unique triplet once, for the oracle/KAPING similarity search
        provenance_keys, provenance_embeddings, provenance2idx = build_provenance_embeddings(
            provenance_index, graph_embedder, cfg
        )
        print(f"✅ Provenance loaded: {len(provenance_keys)} unique triplets")
        return provenance_index, provenance_keys, provenance_embeddings, provenance2idx
    except Exception as e:
        print(f"❌ Unable to load the provenance: {e}")
        return None, None, None, None

def compute_pct_good_triplets(context_str: Optional[str], article_id: str, provenance_index: Optional[Dict]) -> Optional[float]:
    """
    Computes the percentage of context triplets coming from the target article_id.
    Only for the graph/oracle modes whose context follows the
    'src,relation,dst' format, one triplet per line.
    """
    if context_str is None or provenance_index is None:
        return None
    lines = [l.strip() for l in context_str.splitlines() if l.strip()]
    if not lines or lines[0].lower().startswith("src,"):
        # drop the header
        lines = lines[1:]
    if not lines:
        return 0.0

    n_good = 0
    n_total = 0
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        # The destination may itself contain commas -> rejoin the remaining parts
        triplet = (parts[0], parts[1], ",".join(parts[2:]))
        sources = provenance_index.get(triplet, [])
        if article_id in sources:
            n_good += 1
        n_total += 1
    return 100.0 * n_good / n_total if n_total > 0 else 0.0

def format_context_split(context_str: Optional[str], mode: str, provenance_index: Optional[Dict] = None) -> str:
    """
    Normalizes the context into a single string: for RAG, separated by '--- EXTRACT k ---';
    for graph/oracle, one triplet per line.
    """
    if context_str is None:
        return ""

    if mode == "rag":
        # The RAG context is already formatted as [Extract k]\ntext
        return context_str.strip()
    elif mode in {"graph_rag_pcst", "graph_rag_topk", "kaping", "oracle_triplets"}:
        # Graph context: CSV triplets
        return context_str.strip()
    return context_str.strip()

def run_category(category: str, graph_embedder, rag_embedder, llm, tokenizer) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Runs every mode for one category and returns (details_df, summary_df)."""
    print(f"\n{'='*70}\n  CATEGORY: {category.upper()}\n{'='*70}")

    cfg = make_config(category)

    # ── Loading the questions ────────────────────────────────────────────
    df_questions = load_questions(cfg.questions_csv)
    if cfg.filter_by_articles:
        article_ids_set = load_article_ids(cfg.articles_csv)
        df_questions = filter_questions_by_articles(df_questions, article_ids_set)
    df_eval = sample_df(df_questions, cfg.n_samples, cfg.random_seed).reset_index(drop=True)
    print(f"📊 {len(df_eval)} questions for '{category}'")

    if len(df_eval) == 0:
        print(f"⚠️ No question for '{category}', category skipped.")
        return None, None

    # ── Loading graph / RAG / provenance ─────────────────────────────────
    nodes_df, edges_df, node2id, graph = safe_load_graph_artifacts(cfg, graph_embedder)
    rag_index, rag_chunks_df = safe_load_rag_index(cfg)

    # Provenance is only required for oracle_triplets, but we load it whenever possible
    provenance_index, provenance_keys, provenance_embeddings, provenance2idx = (None, None, None, None)
    if nodes_df is not None:
        provenance_index, provenance_keys, provenance_embeddings, provenance2idx = safe_load_provenance(cfg, graph_embedder)

    # Preliminary check: if the graph is missing, some modes become impossible
    modes_possible = list(cfg.modes_to_run)
    if nodes_df is None or graph is None:
        modes_possible = [m for m in modes_possible if m in {"zero_shot", "rag"}]
        print(f"⚠️ Graph unavailable, modes restricted to: {modes_possible}")
    if rag_index is None:
        modes_possible = [m for m in modes_possible if m != "rag"]
        print(f"⚠️ RAG unavailable, modes restricted to: {modes_possible}")
    if provenance_index is None:
        modes_possible = [m for m in modes_possible if m != "oracle_triplets"]

    if not modes_possible:
        print(f"❌ No possible mode for '{category}', category skipped.")
        gc.collect()
        torch.cuda.empty_cache()
        return None, None

    # ── Evaluation per mode ──────────────────────────────────────────────
    results: Dict[str, pd.DataFrame] = {}
    for mode in modes_possible:
        print(f"\n  ▶ MODE: {mode}")
        try:
            df_mode = run_eval_for_mode(
                df_eval, mode, cfg, tokenizer, llm,
                rag_index=rag_index,
                rag_chunks_df=rag_chunks_df,
                rag_embedder=rag_embedder,
                graph=graph,
                nodes_df=nodes_df,
                edges_df=edges_df,
                graph_embedder=graph_embedder,
                provenance_index=provenance_index,
                provenance_triplet_keys=provenance_keys,
                provenance_triplet_emb=provenance_embeddings,
                provenance_triplet2idx=provenance2idx,
            )
            results[mode] = df_mode
            acc = df_mode["correct"].mean()
            print(f"    ✅ Accuracy [{mode}]: {acc:.2%}")
        except Exception as e:
            print(f"    ❌ Error in mode '{mode}': {e}")
            import traceback
            traceback.print_exc()

        # Free the GPU between two modes
        torch.cuda.empty_cache()
        gc.collect()

    if not results:
        return None, None

    # ── Building the detailed DataFrame ──────────────────────────────────
    detail_cols = {
        "category": category,
        "article_id": df_eval["article_id"].values,
        "question": df_eval["question"].values,
        "option_A": df_eval["option A"].values,
        "option_B": df_eval["option B"].values,
        "option_C": df_eval["option C"].values,
        "option_D": df_eval["option D"].values,
        "correct_letter": df_eval["correct_letter"].values,
    }
    detail = pd.DataFrame(detail_cols)

    for mode, df in results.items():
        # One column block per mode: prediction, correctness, context
        detail[f"{mode}_predicted_letter"] = df["predicted_letter"].values
        detail[f"{mode}_correct"] = df["correct"].values
        contexts = df["context"].values
        detail[f"{mode}_context"] = [
            format_context_split(str(ctx) if ctx is not None else "", mode, provenance_index)
            for ctx in contexts
        ]
        if mode in {"graph_rag_pcst", "graph_rag_topk", "kaping", "oracle_triplets"}:
            # Retrieval precision w.r.t. the gold article
            detail[f"{mode}_pct_correct_article"] = [
                compute_pct_good_triplets(
                    str(ctx) if ctx is not None else None,
                    str(aid),
                    provenance_index,
                )
                for ctx, aid in zip(contexts, df_eval["article_id"].values)
            ]

    # ── Building the summary ─────────────────────────────────────────────
    summary = pd.DataFrame([
        {
            "category": category,
            "mode": mode,
            "accuracy": df["correct"].mean(),
            "n_questions": len(df),
            "n_correct": int(df["correct"].sum()),
        }
        for mode, df in results.items()
    ])

    return detail, summary

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── Single loading of the shared models ──────────────────────────────
    cfg_first = make_config(CATEGORIES[0])
    print(f"\n🔌 Loading the embedder: {cfg_first.embedder_id}")
    embedder = SentenceTransformer(cfg_first.embedder_id, device=cfg_first.device, trust_remote_code=True)

    print(f"🔌 Loading the LLM: {cfg_first.llm_id}")
    llm, tokenizer = load_llm(cfg_first.llm_id, device=cfg_first.device)

    all_details: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []

    for category in CATEGORIES:
        # The same embedder is reused for the graph and for RAG
        detail, summary = run_category(category, embedder, embedder, llm, tokenizer)
        if detail is None or summary is None:
            continue

        # Per-category saves
        detail_path = OUTPUT_DIR / f"{category}_details.csv"
        summary_path = OUTPUT_DIR / f"{category}_accuracy_summary.csv"
        detail.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"💾 {category}_details.csv")
        print(f"💾 {category}_accuracy_summary.csv")

        all_details.append(detail)
        all_summaries.append(summary)

    # ── Global saves ─────────────────────────────────────────────────────
    if all_details:
        total_detail = pd.concat(all_details, ignore_index=True)
        total_detail_path = OUTPUT_DIR / "total_details.csv"
        total_detail.to_csv(total_detail_path, index=False)
        print(f"\n💾 total_details.csv: {len(total_detail)} rows")

    if all_summaries:
        total_summary = pd.concat(all_summaries, ignore_index=True)
        # Pivot to get categories × modes
        pivot_summary = total_summary.pivot(index="category", columns="mode", values="accuracy").reset_index()
        pivot_summary_path = OUTPUT_DIR / "total_accuracy_summary.csv"
        pivot_summary.to_csv(pivot_summary_path, index=False)
        total_summary_path = OUTPUT_DIR / "total_accuracy_summary_long.csv"
        total_summary.to_csv(total_summary_path, index=False)
        print(f"💾 total_accuracy_summary.csv (wide format)")
        print(f"💾 total_accuracy_summary_long.csv (long format)")
        print("\n📊 Global recap:")
        print(pivot_summary.to_string(index=False))

if __name__ == "__main__":
    main()