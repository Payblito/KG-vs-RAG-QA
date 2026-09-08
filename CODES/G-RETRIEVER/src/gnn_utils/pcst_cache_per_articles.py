"""
Build and cache PCST subgraphs per article: reconstruct local article graph from provenance,
run PCST for each question, and store selected global node/edge indices + context desc.
"""
import os
import pickle

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..graph_utils import load_provenance
from ..retrieval_graph import retrieval_via_pcst
from .pcst_cache import _build_query_embedding_mode


def build_pcst_cache_per_articles(
    df_questions, cfg, graph, nodes_df, edges_df,
    graph_embedder, cache_path, query_mode=None
):
    """
    Build PCST cache per article: for each article, reconstruct subgraph from provenance,
    run PCST on the article-specific graph for each question, and record selected global
    node/edge indices plus the context description.

    Args:
        df_questions: DataFrame with columns 'article_id', 'question', 'option A'..'D', etc.
        cfg: config object (provides topk_nodes, topk_edges, pcst_cost_e, device, provenance_path).
        graph: global PyG Data graph.
        nodes_df: DataFrame of global nodes (with 'node_id' and 'node_attr').
        edges_df: DataFrame of global edges (with 'src','edge_attr','dst').
        graph_embedder: embedder for queries.
        cache_path: path to serialize the resulting cache (pickle list of dicts keyed by article/question).
        query_mode: optional override for cfg.retrieval_query_mode.

    Returns:
        List[dict] with keys {article_id, question, selected_nodes, selected_edges, desc}.
    """
    # load global provenance: article_id -> list of (s,p,o)
    provenance_index = load_provenance(cfg.provenance_path)
    # normalize provenance_index keys to str for consistent lookup
    provenance_index = {str(aid): triplets for aid, triplets in provenance_index.items()}
    print(f"[pcst_cache_per_articles] provenance loaded and normalized: {len(provenance_index)} articles indexed (str keys)")

    # map global node_id -> label for matching provenance triplets and build triplet->edge index map
    id2label = dict(zip(nodes_df["node_id"], nodes_df["node_attr"]))
    triplet2edge_idx = {}
    for eidx, row in edges_df.iterrows():
        key = (id2label[row["src"]], row["edge_attr"], id2label[row["dst"]])
        triplet2edge_idx.setdefault(key, []).append(eidx)
    print(f"[pcst_cache_per_articles] built triplet->edge map for {len(triplet2edge_idx)} unique triplets")

    # determine query_mode (fallback to config)
    if query_mode is None:
        query_mode = getattr(cfg, "retrieval_query_mode", "question_options")

    entries = []
    df = df_questions.reset_index(drop=True)
    # group by article to rebuild per-article subgraph once
    num_articles = df["article_id"].nunique()
    for article_id, grp in tqdm(df.groupby("article_id"), desc="Articles", total=num_articles):
        aid = str(article_id)
        prov_triplets = provenance_index.get(aid, [])
        # collect global edge indices for this article from provenance
        edge_idx_article = sorted({e for t in prov_triplets for e in triplet2edge_idx.get(t, [])})
        # collect global node indices if edges exist
        if edge_idx_article:
            srcs = edges_df.loc[edge_idx_article, "src"].tolist()
            dsts = edges_df.loc[edge_idx_article, "dst"].tolist()
            node_idx_article = sorted(set(srcs + dsts))
            # build PyG subgraph for this article
            sub_x = graph.x[node_idx_article]
            sub_ea = graph.edge_attr[edge_idx_article]
            sub_ei_raw = graph.edge_index[:, edge_idx_article]
            remap = {old: new for new, old in enumerate(node_idx_article)}
            sub_ei = torch.tensor([
                [remap[i] for i in sub_ei_raw[0].tolist()],
                [remap[i] for i in sub_ei_raw[1].tolist()]
            ], dtype=torch.long)
            sub_graph = Data(x=sub_x, edge_index=sub_ei, edge_attr=sub_ea)
            sub_nodes_df = nodes_df.iloc[node_idx_article].reset_index(drop=True)
            sub_edges_df = edges_df.iloc[edge_idx_article].reset_index(drop=True)
        else:
            node_idx_article = []
            edge_idx_article = []
        #print(f"[pcst_cache_per_articles] Article {aid}: provenance edges={len(edge_idx_article)}, nodes={len(node_idx_article)}")
        # run PCST for each question in this article
        num_q = len(grp)
        for qidx, (_, row) in enumerate(grp.iterrows(), start=1):
            options = {L: row[f"option {L}"] for L in ["A", "B", "C", "D"]}
            correct = row.get("correct_letter", None)
            q_emb = _build_query_embedding_mode(
                row["question"], options, graph_embedder, query_mode,
                correct_letter=correct, device=cfg.device
            )
            if edge_idx_article:
                desc, _, diag = retrieval_via_pcst(
                    sub_graph, q_emb, sub_nodes_df, sub_edges_df,
                    topk=cfg.topk_nodes, topk_e=cfg.topk_edges,
                    cost_e=cfg.pcst_cost_e, mode_pcst=True,
                    return_diagnostics=True
                )
                sel_nodes_glob = [node_idx_article[i] for i in diag["selected_nodes"]]
                sel_edges_glob = [edge_idx_article[i] for i in diag["selected_edges"]]
                #print(f"  [article={aid} q={qidx}/{num_q}] selected_nodes={len(sel_nodes_glob)}, selected_edges={len(sel_edges_glob)}")
            else:
                desc = None
                sel_nodes_glob = []
                sel_edges_glob = []
            entries.append({
                "article_id": aid,
                "question": row["question"],
                "selected_nodes": sel_nodes_glob,
                "selected_edges": sel_edges_glob,
                "desc": desc,
            })

    # serialize cache to distinct path
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(entries, f)
    print(f"[pcst_cache_per_articles] {len(entries)} entries -> {cache_path}")
    return entries


def load_pcst_cache_per_articles(cache_path):
    """Load per-article PCST cache (list of dicts keyed by article/question)."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def reorder_pcst_cache(entries, df_questions):
    """
    Reorder a keyed PCST cache to align positionally with df_questions.

    entries: list of dicts with keys 'article_id','question',...
    df_questions: DataFrame (reset_index(drop=True)) with matching 'article_id','question'.

    Returns a list of entries in the same order as df_questions; missing keys yield None.
    """
    df = df_questions.reset_index(drop=True)
    # build lookup {(article_id, question): entry}
    lookup = {(e['article_id'], e['question']): e for e in entries}
    ordered = []
    missing = []
    for aid, q in zip(df['article_id'].astype(str), df['question']):
        key = (aid, q)
        if key in lookup:
            ordered.append(lookup[key])
        else:
            missing.append(key)
            ordered.append(None)
    if missing:
        print(f"[reorder_pcst_cache] missing {len(missing)} entries, example key: {missing[0]}")
    return ordered
