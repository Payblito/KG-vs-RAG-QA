"""
Script autonome pour évaluer G-RETRIEVER sur 8 catégories (musica, literatura,
folclore, gastronomia, danza, pintura, artesania, cine).

Pour chaque catégorie, tous les modes supportés sont lancés avec les paramètres
fixes définis dans FIXED_CONFIG. Les résultats sont enregistrés dans :
    <CLEAN>/DATA/RESULTS_8_CATEGORIES

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

# Ajout du dossier parent au PYTHONPATH pour importer src/
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

# ── Paramètres fixes ─────────────────────────────────────────────────────────
CATEGORIES: List[str] = [
    # "artesania", 
    "literatura", 
    # "folclore",
    # "gastronomia",
    # "danza",
    # "pintura",
    # "musica", 
    # "cine",
    #"global"
]

FIXED_CONFIG = {
    "llm_id": "Qwen/Qwen2.5-3B-Instruct",
    "embedder_id": "jinaai/jina-embeddings-v3",
    "modes_to_run": [
        "zero_shot", 
        "rag", 
        "kaping", 
    ],
    "retrieval_query_mode": "question_options",
    "n_samples": None,
    "random_seed": 1,
    "batch_size": 8,
    "max_new_tokens": 5,
    "k_rag": 5,
    "chunk_size_tokens": 512,
    "chunk_overlap_tokens": 64,
    "topk_nodes": 15,
    "topk_edges": 20,
    "pcst_cost_e": 0.1,
    "pcst_workers": 4,
    "topk_triplets": 15,        # valeur Config par défaut
    "top_k_provenance": 20,
    "force_rebuild_graph_artifacts": False,
    "force_rebuild_rag_index": False,
    "save_all": False,
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
}

llm_slang=FIXED_CONFIG["llm_id"].split("/")[-1].replace("-", "_")
OUTPUT_DIR = DATA / "RESULTS_CATEGORIES" / llm_slang


def make_config(category: str) -> Config:
    """Construit une Config dédiée à une catégorie avec les paramètres fixes."""
    cfg_dict = {
        **FIXED_CONFIG,
        "CATEGORY": category,
        # Questions/articles
        "questions_csv": str(DATA / "QUESTIONS" / "mcq_es.csv"),
        "articles_csv": str(DATA / "SUBSETS_ES" / "ARTICLES_SUBSETS_ES" / f"{category}_articles_es_disjoint.csv"),
        
        # Graphe
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
    # Config est un dataclass : on enlève les champs qui n'existent pas
    cfg_fields = {f.name for f in Config.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    return Config(**cfg_dict)


def safe_load_graph_artifacts(cfg: Config, graph_embedder) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[object], Optional[object]]:
    """Charge les artéfacts graphe, retourne None en cas d'erreur."""
    try:
        nodes_df, edges_df, node2id, graph = ensure_graph_artifacts(cfg, graph_embedder)
        print(f"✅ Graphe chargé : {len(nodes_df)} nœuds, {len(edges_df)} arêtes")
        return nodes_df, edges_df, node2id, graph
    except Exception as e:
        print(f"❌ Impossible de charger le graphe : {e}")
        return None, None, None, None


def safe_load_rag_index(cfg: Config):
    """Charge l'index RAG, retourne None en cas d'erreur."""
    try:
        return ensure_rag_index(cfg)
    except Exception as e:
        print(f"❌ Impossible de charger/construire l'index RAG : {e}")
        return None, None


def safe_load_provenance(cfg: Config, graph_embedder):
    """Charge la provenance et prépare les embeddings de triplets."""
    try:
        provenance_index = load_provenance(cfg.provenance_path)
        provenance_keys, provenance_embeddings, provenance2idx = build_provenance_embeddings(
            provenance_index, graph_embedder, cfg
        )
        print(f"✅ Provenance chargée : {len(provenance_keys)} triplets uniques")
        return provenance_index, provenance_keys, provenance_embeddings, provenance2idx
    except Exception as e:
        print(f"❌ Impossible de charger la provenance : {e}")
        return None, None, None, None


def compute_pct_good_triplets(context_str: Optional[str], article_id: str, provenance_index: Optional[Dict]) -> Optional[float]:
    """
    Calcule le pourcentage de triplets du contexte provenant de l'article_id cible.
    Seulement pour les modes graphe/oracle dont le contexte est au format
    'src,relation,dst' avec un triplet par ligne.
    """
    if context_str is None or provenance_index is None:
        return None
    lines = [l.strip() for l in context_str.splitlines() if l.strip()]
    if not lines or lines[0].lower().startswith("src,"):
        # on retire l'en-tête
        lines = lines[1:]
    if not lines:
        return 0.0

    n_good = 0
    n_total = 0
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        triplet = (parts[0], parts[1], ",".join(parts[2:]))
        sources = provenance_index.get(triplet, [])
        if article_id in sources:
            n_good += 1
        n_total += 1
    return 100.0 * n_good / n_total if n_total > 0 else 0.0


def format_context_split(context_str: Optional[str], mode: str, provenance_index: Optional[Dict] = None) -> str:
    """
    Normalise le contexte en une chaîne : pour RAG, séparé par '--- EXTRACT k ---' ;
    pour graphe/oracle, un triplet par ligne.
    """
    if context_str is None:
        return ""

    if mode == "rag":
        # Le contexte RAG est déjà formaté avec [Extract k]\ntext
        return context_str.strip()
    elif mode in {"graph_rag_pcst", "graph_rag_topk", "kaping", "oracle_triplets"}:
        # Contexte graphe : triplets CSV
        return context_str.strip()
    return context_str.strip()


def run_category(category: str, graph_embedder, rag_embedder, llm, tokenizer) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Lance tous les modes pour une catégorie et retourne (details_df, summary_df)."""
    print(f"\n{'='*70}\n  CATÉGORIE : {category.upper()}\n{'='*70}")

    cfg = make_config(category)

    # ── Chargement des questions ─────────────────────────────────────────
    df_questions = load_questions(cfg.questions_csv)
    if cfg.filter_by_articles:
        article_ids_set = load_article_ids(cfg.articles_csv)
        df_questions = filter_questions_by_articles(df_questions, article_ids_set)
    df_eval = sample_df(df_questions, cfg.n_samples, cfg.random_seed).reset_index(drop=True)
    print(f"📊 {len(df_eval)} questions pour '{category}'")

    if len(df_eval) == 0:
        print(f"⚠️ Aucune question pour '{category}', catégorie ignorée.")
        return None, None

    # ── Chargement graphe / RAG / provenance ─────────────────────────────
    nodes_df, edges_df, node2id, graph = safe_load_graph_artifacts(cfg, graph_embedder)
    rag_index, rag_chunks_df = safe_load_rag_index(cfg)

    # Provenance nécessaire seulement pour oracle_triplets, mais on la charge si possible
    provenance_index, provenance_keys, provenance_embeddings, provenance2idx = (None, None, None, None)
    if nodes_df is not None:
        provenance_index, provenance_keys, provenance_embeddings, provenance2idx = safe_load_provenance(cfg, graph_embedder)

    # Vérification préalable : si graphe absent, certains modes sont impossibles
    modes_possible = list(cfg.modes_to_run)
    if nodes_df is None or graph is None:
        modes_possible = [m for m in modes_possible if m in {"zero_shot", "rag"}]
        print(f"⚠️ Graphe indisponible, modes restreints : {modes_possible}")
    if rag_index is None:
        modes_possible = [m for m in modes_possible if m != "rag"]
        print(f"⚠️ RAG indisponible, modes restreints : {modes_possible}")
    if provenance_index is None:
        modes_possible = [m for m in modes_possible if m != "oracle_triplets"]

    if not modes_possible:
        print(f"❌ Aucun mode possible pour '{category}', catégorie ignorée.")
        gc.collect()
        torch.cuda.empty_cache()
        return None, None

    # ── Évaluation par mode ──────────────────────────────────────────────
    results: Dict[str, pd.DataFrame] = {}
    for mode in modes_possible:
        print(f"\n  ▶ MODE : {mode}")
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
            print(f"    ✅ Accuracy [{mode}] : {acc:.2%}")
        except Exception as e:
            print(f"    ❌ Erreur mode '{mode}' : {e}")
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()
        gc.collect()

    if not results:
        return None, None

    # ── Construction du DataFrame détaillé ───────────────────────────────
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
        detail[f"{mode}_predicted_letter"] = df["predicted_letter"].values
        detail[f"{mode}_correct"] = df["correct"].values
        contexts = df["context"].values
        detail[f"{mode}_context"] = [
            format_context_split(str(ctx) if ctx is not None else "", mode, provenance_index)
            for ctx in contexts
        ]
        if mode in {"graph_rag_pcst", "graph_rag_topk", "kaping", "oracle_triplets"}:
            detail[f"{mode}_pct_correct_article"] = [
                compute_pct_good_triplets(
                    str(ctx) if ctx is not None else None,
                    str(aid),
                    provenance_index,
                )
                for ctx, aid in zip(contexts, df_eval["article_id"].values)
            ]

    # ── Construction du résumé ───────────────────────────────────────────
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
    print(f"📁 Répertoire de sortie : {OUTPUT_DIR}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── Chargement unique des modèles communs ────────────────────────────
    cfg_first = make_config(CATEGORIES[0])
    print(f"\n🔌 Chargement de l'embedder : {cfg_first.embedder_id}")
    embedder = SentenceTransformer(cfg_first.embedder_id, device=cfg_first.device, trust_remote_code=True)

    print(f"🔌 Chargement du LLM : {cfg_first.llm_id}")
    llm, tokenizer = load_llm(cfg_first.llm_id, device=cfg_first.device)

    all_details: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []

    for category in CATEGORIES:
        detail, summary = run_category(category, embedder, embedder, llm, tokenizer)
        if detail is None or summary is None:
            continue

        # Sauvegardes par catégorie
        detail_path = OUTPUT_DIR / f"{category}_details.csv"
        summary_path = OUTPUT_DIR / f"{category}_accuracy_summary.csv"
        detail.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"💾 {category}_details.csv")
        print(f"💾 {category}_accuracy_summary.csv")

        all_details.append(detail)
        all_summaries.append(summary)

    # ── Sauvegardes globales ─────────────────────────────────────────────
    if all_details:
        total_detail = pd.concat(all_details, ignore_index=True)
        total_detail_path = OUTPUT_DIR / "total_details.csv"
        total_detail.to_csv(total_detail_path, index=False)
        print(f"\n💾 total_details.csv : {len(total_detail)} lignes")

    if all_summaries:
        total_summary = pd.concat(all_summaries, ignore_index=True)
        # Pivot pour avoir catégories × modes
        pivot_summary = total_summary.pivot(index="category", columns="mode", values="accuracy").reset_index()
        pivot_summary_path = OUTPUT_DIR / "total_accuracy_summary.csv"
        pivot_summary.to_csv(pivot_summary_path, index=False)
        total_summary_path = OUTPUT_DIR / "total_accuracy_summary_long.csv"
        total_summary.to_csv(total_summary_path, index=False)
        print(f"💾 total_accuracy_summary.csv (format large)")
        print(f"💾 total_accuracy_summary_long.csv (format long)")
        print("\n📊 Récapitulatif global :")
        print(pivot_summary.to_string(index=False))


if __name__ == "__main__":
    main()
