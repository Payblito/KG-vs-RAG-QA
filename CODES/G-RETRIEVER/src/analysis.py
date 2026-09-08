"""Analyse qualitative comparée RAG vs Graph-RAG PCST."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .retrieval_rag import encode_questions_e5, faiss_retrieve
from .retrieval_graph import retrieval_via_pcst


# ══════════════════════════════════════════════════════════════════════
# 1. Build df_analysis
# ══════════════════════════════════════════════════════════════════════
def _options_from_row(row):
    return {"A": row["option A"], "B": row["option B"],
            "C": row["option C"], "D": row["option D"]}


def _query_text(question, options):
    return (f"{question} A) {options['A']} B) {options['B']} "
            f"C) {options['C']} D) {options['D']}")


def _categorize(correct_rag, correct_graph):
    if correct_rag and correct_graph:       return "A"  # both correct
    if correct_rag and not correct_graph:   return "B"  # RAG only
    if not correct_rag and correct_graph:   return "C"  # Graph only
    return "D"                                          # both wrong


def run_retrieval_with_diagnostics(
    df_eval, cfg,
    rag_results_csv: str,
    graph_results_csv: str,
    rag_index, rag_chunks_df, rag_embedder,
    graph, nodes_df, edges_df, graph_embedder,
    save_dir: str = None,
):
    """
    Recharge les prédictions LLM depuis CSV, rejoue le retrieval (RAG + Graph PCST)
    en mode 'question_options' et collecte tous les diagnostics.
    """
    print("📂 Chargement des résultats LLM précédents...")
    rag_res = pd.read_csv(rag_results_csv)
    graph_res = pd.read_csv(graph_results_csv)
    assert len(rag_res) == len(df_eval) == len(graph_res), \
        f"Tailles incohérentes : eval={len(df_eval)}, rag={len(rag_res)}, graph={len(graph_res)}"

    df_eval = df_eval.reset_index(drop=True)
    questions = df_eval["question"].tolist()
    options_list = [_options_from_row(r) for _, r in df_eval.iterrows()]
    query_texts = [_query_text(q, o) for q, o in zip(questions, options_list)]

    # ── RAG retrieval avec scores ──
    print(f"🔎 RAG retrieval (k={cfg.k_rag})...")
    q_embs_rag = encode_questions_e5(query_texts, rag_embedder)
    rag_chunks, rag_scores, rag_chunk_ids = faiss_retrieve(
        q_embs_rag, rag_index, rag_chunks_df, k=cfg.k_rag, return_scores=True
    )

    # ── Graph retrieval avec diagnostics (séquentiel, plus simple) ──
    print(f"🔎 Graph-RAG PCST retrieval (topk_n={cfg.topk_nodes}, topk_e={cfg.topk_edges})...")
    q_embs_graph = graph_embedder.encode(
        query_texts, batch_size=64, convert_to_tensor=True,
        show_progress_bar=True, device=cfg.device,
    ).cpu()

    graph_contexts, graph_diags = [], []
    for i in tqdm(range(len(query_texts)), desc="PCST + diagnostics"):
        desc, _, diag = retrieval_via_pcst(
            graph, q_embs_graph[i], nodes_df, edges_df,
            topk=cfg.topk_nodes, topk_e=cfg.topk_edges,
            cost_e=cfg.pcst_cost_e, mode_pcst=True,
            return_diagnostics=True,
        )
        graph_contexts.append(desc)
        graph_diags.append(diag)

    # ── Assemblage du df_analysis ──
    print("🧩 Assemblage de df_analysis...")
    rows = []
    for i in range(len(df_eval)):
        correct_letter = df_eval["correct_letter"].iloc[i]
        pred_rag = rag_res["predicted_letter"].iloc[i]
        pred_graph = graph_res["predicted_letter"].iloc[i]
        c_rag = (pred_rag == correct_letter)
        c_graph = (pred_graph == correct_letter)
        d = graph_diags[i]

        rows.append({
            "question": questions[i],
            "options": options_list[i],
            "correct_letter": correct_letter,
            "article_id": df_eval["article_id"].iloc[i],
            "pred_rag": pred_rag,
            "pred_graph": pred_graph,
            "correct_rag": c_rag,
            "correct_graph": c_graph,
            "category": _categorize(c_rag, c_graph),
            # RAG
            "rag_context": "\n\n".join(
                f"[Extract {j+1}]\n{c}" for j, c in enumerate(rag_chunks[i])
            ),
            "rag_scores": rag_scores[i].tolist(),
            "rag_chunk_ids": rag_chunk_ids[i],
            # Graph
            "graph_context": graph_contexts[i],
            "graph_topk_node_sims": d["topk_node_sims"],
            "graph_topk_edge_sims": d["topk_edge_sims"],
            "graph_selected_node_sims": d["selected_node_sims"],
            "graph_selected_edge_sims": d["selected_edge_sims"],
            "graph_n_selected_nodes": d["n_selected_nodes"],
            "graph_n_selected_edges": d["n_selected_edges"],
            "pcst_fallback": d["pcst_fallback"],
        })

    df_analysis = pd.DataFrame(rows)
    print(f"✅ df_analysis : {len(df_analysis)} lignes")
    print(df_analysis["category"].value_counts().sort_index())

    # ── Sauvegarde ──
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = save_dir / "df_analysis.pkl"
        csv_path = save_dir / "df_analysis_light.csv"
        with open(pkl_path, "wb") as f:
            pickle.dump(df_analysis, f)
        # CSV allégé : sans les listes
        light_cols = ["question", "correct_letter", "article_id",
                      "pred_rag", "pred_graph", "correct_rag", "correct_graph",
                      "category", "graph_n_selected_nodes", "graph_n_selected_edges",
                      "pcst_fallback"]
        df_analysis[light_cols].to_csv(csv_path, index=False)
        print(f"💾 Sauvegardé : {pkl_path}")
        print(f"💾 Sauvegardé : {csv_path}")

    return df_analysis


# ══════════════════════════════════════════════════════════════════════
# 2. Stats agrégées
# ══════════════════════════════════════════════════════════════════════
def _stats(lst):
    if lst is None or len(lst) == 0:
        return dict(mean=np.nan, median=np.nan, max=np.nan, min=np.nan, std=np.nan, count=0)
    a = np.asarray(lst, dtype=float)
    return dict(mean=a.mean(), median=np.median(a), max=a.max(),
                min=a.min(), std=a.std(), count=len(a))


def compute_similarity_stats(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des colonnes de stats pour chaque liste de scores."""
    df = df_analysis.copy()

    score_cols = {
        "rag":           "rag_scores",
        "graph_topk_n":  "graph_topk_node_sims",
        "graph_topk_e":  "graph_topk_edge_sims",
        "graph_sel_n":   "graph_selected_node_sims",
        "graph_sel_e":   "graph_selected_edge_sims",
    }
    for prefix, col in score_cols.items():
        stats = df[col].apply(_stats)
        for stat_name in ["mean", "median", "max", "min", "std"]:
            df[f"{prefix}_{stat_name}"] = stats.apply(lambda d: d[stat_name])

    return df


def summarize_by_category(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """Tableau récap : moyennes des métriques principales par catégorie A/B/C/D."""
    metric_cols = [c for c in df_analysis.columns
                   if any(c.startswith(p) for p in
                          ["rag_", "graph_topk_n_", "graph_topk_e_",
                           "graph_sel_n_", "graph_sel_e_"])
                   and any(c.endswith(s) for s in ["_mean", "_max", "_min", "_std", "_median"])]
    extra = ["graph_n_selected_nodes", "graph_n_selected_edges", "pcst_fallback"]
    cols = metric_cols + [c for c in extra if c in df_analysis.columns]

    summary = df_analysis.groupby("category")[cols].mean(numeric_only=True)
    summary.insert(0, "n_questions", df_analysis.groupby("category").size())
    print("\n📊 Synthèse par catégorie (A=both ok, B=RAG only, C=Graph only, D=both wrong)")
    print(summary.round(3).T)
    return summary


# ══════════════════════════════════════════════════════════════════════
# 3. Visualisations
# ══════════════════════════════════════════════════════════════════════
def plot_distributions_by_category(df_analysis: pd.DataFrame):
    """Boxplots des métriques principales par catégorie + histogrammes RAG vs Graph."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    metrics = [
        ("rag_max",          "RAG — max similarité (top-k chunks)"),
        ("graph_sel_n_max",  "Graph — max sim (nodes retenus par PCST)"),

        ("rag_mean",         "RAG — mean similarité"),
        ("graph_sel_n_mean", "Graph — mean sim (nodes retenus par PCST)"),
        
        ("graph_topk_n_max", "Graph — max sim (top-k nodes, avant PCST)"),
        ("graph_sel_n_max", "Graph — max sim (nodes retenus par PCST)"),

        ("graph_topk_e_max", "Graph — max sim (top-k edges, avant PCST)"),
        ("graph_sel_e_max", "Graph — max sim (edges retenues par PCST)"),

    ]

    fig, axes = plt.subplots(4, 2, figsize=(10, 22))

    order = ["A", "B", "C", "D"]
    palette = {"A": "#4CAF50", "B": "#FF9800", "C": "#2196F3", "D": "#F44336"}

    for ax, (col, title) in zip(axes.flat, metrics):
        if col not in df_analysis.columns:
            ax.set_visible(False); continue
        sns.boxplot(data=df_analysis, x="category", y=col, order=order,
                    palette=palette, ax=ax, showfliers=False)
        sns.stripplot(data=df_analysis, x="category", y=col, order=order,
                      color="black", size=2, alpha=0.4, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0.75, top=0.92)
        ax.set_xlabel("")
    plt.suptitle("Distributions des scores de similarité par catégorie",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    # Histogramme focus sur catégorie B : RAG vs Graph (max sim)
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_B = df_analysis[df_analysis["category"] == "B"]
    if len(cat_B) > 0:
        ax.hist(cat_B["rag_max"].dropna(), bins=20, alpha=0.6,
                label=f"RAG max (n={len(cat_B)})", color="#4CAF50")
        ax.hist(cat_B["graph_sel_n_max"].dropna(), bins=20, alpha=0.6,
                label="Graph max (nodes retenus)", color="#F44336")
        ax.set_title("Catégorie B (RAG ✅ / Graph ❌) — max similarité")
        ax.set_xlabel("Max similarité"); ax.set_ylabel("# questions")
        ax.legend()
        plt.tight_layout(); plt.show()

    # Taux de fallback par catégorie
    if "pcst_fallback" in df_analysis.columns:
        fb = df_analysis.groupby("category")["pcst_fallback"].mean()
        print("\n⚠️ Taux de fallback PCST par catégorie :")
        print(fb.round(3))


# ══════════════════════════════════════════════════════════════════════
# 4. Inspection cas par cas
# ══════════════════════════════════════════════════════════════════════
def inspect_question(idx: int, df_analysis: pd.DataFrame, max_ctx_chars: int = 1500):
    """Affiche en détail une question + contextes + scores RAG et Graph."""
    if idx not in df_analysis.index:
        print(f"❌ Index {idx} absent."); return
    row = df_analysis.loc[idx]

    cat_label = {
        "A": "A — both correct ✅✅",
        "B": "B — RAG ✅ / Graph ❌",
        "C": "C — RAG ❌ / Graph ✅",
        "D": "D — both wrong ❌❌",
    }[row["category"]]

    print("═" * 90)
    print(f"Q[{idx}] — Catégorie {cat_label}")
    print("═" * 90)
    print(f"Question : {row['question']}")
    for L in "ABCD":
        marker = " ← bonne réponse" if L == row["correct_letter"] else ""
        print(f"  {L}) {row['options'][L]}{marker}")
    print(f"\n→ correct={row['correct_letter']}  | "
          f"RAG={row['pred_rag']} {'✅' if row['correct_rag'] else '❌'}  | "
          f"Graph={row['pred_graph']} {'✅' if row['correct_graph'] else '❌'}")

    # ── RAG ──
    print("\n" + "─" * 90)
    print("── RAG ──")
    scores = np.asarray(row["rag_scores"], dtype=float)
    print(f"Scores top-{len(scores)} : {[round(s, 3) for s in scores.tolist()]}")
    print(f"  mean={scores.mean():.3f}  max={scores.max():.3f}  min={scores.min():.3f}")
    print("Contexte :")
    ctx = row["rag_context"]
    print(ctx[:max_ctx_chars] + ("..." if len(ctx) > max_ctx_chars else ""))

    # ── Graph ──
    print("\n" + "─" * 90)
    print("── Graph-RAG PCST ──")
    print(f"PCST fallback : {row['pcst_fallback']}")
    tn = np.asarray(row["graph_topk_node_sims"], dtype=float)
    te = np.asarray(row["graph_topk_edge_sims"], dtype=float)
    sn = np.asarray(row["graph_selected_node_sims"], dtype=float)
    se = np.asarray(row["graph_selected_edge_sims"], dtype=float)

    print(f"Avant PCST : top-{len(tn)} nodes  → mean={tn.mean():.3f}  max={tn.max():.3f}")
    print(f"             top-{len(te)} edges  → mean={te.mean():.3f}  max={te.max():.3f}")
    sn_mean = sn.mean() if len(sn) else float('nan')
    sn_max  = sn.max()  if len(sn) else float('nan')
    se_mean = se.mean() if len(se) else float('nan')
    se_max  = se.max()  if len(se) else float('nan')
    print(f"Après PCST : {len(sn)} nodes retenus → mean={sn_mean:.3f}  max={sn_max:.3f}")
    print(f"             {len(se)} edges retenues → mean={se_mean:.3f}  max={se_max:.3f}")

    print("Contexte (sous-graphe) :")
    ctx = row["graph_context"] or ""
    print(ctx[:max_ctx_chars] + ("..." if len(ctx) > max_ctx_chars else ""))
    print()


def list_category(df_analysis: pd.DataFrame, category: str, n: int = None):
    """Liste les indices d'une catégorie donnée."""
    idx = df_analysis[df_analysis["category"] == category].index.tolist()
    print(f"Catégorie {category} : {len(idx)} questions")
    return idx[:n] if n else idx
