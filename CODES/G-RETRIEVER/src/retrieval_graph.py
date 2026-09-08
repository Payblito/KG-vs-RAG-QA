from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import csv
import numpy as np
import pcst_fast
import torch
from torch_geometric.data import Data


# ══════════════════════════════════════════════════════════════════════
# Helpers LN (transfo_ln)
# ══════════════════════════════════════════════════════════════════════
def _normalize_triplet(s, p, o):
    """Normalise un triplet pour la comparaison (strip + strip quotes)."""
    return (
        str(s).strip().strip('"').strip(),
        str(p).strip().strip('"').strip(),
        str(o).strip().strip('"').strip(),
    )


def load_ln_lookup(ln_path):
    """Charge le CSV LN et retourne un dict {(s,p,o): sentence}.

    Le CSV doit avoir les colonnes : subject, predicate, object, sentence.
    Retourne None si le fichier n'existe pas ou est vide.
    """
    ln_path = Path(ln_path)
    if not ln_path.exists():
        print(f"⚠️ Fichier LN introuvable : {ln_path}")
        return None

    lookup = {}
    with open(ln_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _normalize_triplet(row["subject"], row["predicate"], row["object"])
            lookup[key] = row["sentence"].strip()
    print(f"✅ LN chargé : {len(lookup)} triplets depuis {ln_path.name}")
    return lookup


def _triplet_to_ln(ln_lookup, s, p, o, fallback_csv=True):
    """Retourne la phrase LN si disponible, sinon fallback au format CSV.

    Args:
        ln_lookup: dict {(s,p,o): sentence} ou None.
        s, p, o: labels du triplet.
        fallback_csv: si True et pas de LN, retourne "s,p,o". Sinon retourne None.
    """
    if ln_lookup is not None:
        key = _normalize_triplet(s, p, o)
        if key in ln_lookup:
            return ln_lookup[key]
    if fallback_csv:
        return f"{s},{p},{o}"
    return None


# ══════════════════════════════════════════════════════════════════════
# Retrieval principal
# ══════════════════════════════════════════════════════════════════════
def retrieval_via_pcst(
    graph, q_emb, nodes_df, edges_df,
    topk: int = 10, topk_e: int = 10, cost_e: float = 0.5,
    mode_pcst: bool = True,
    return_diagnostics: bool = False,
    return_selected_indices: bool = False,
    ln_lookup=None,
):
    """
    mode_pcst=False : top-k direct (rapide, ignore la topologie)
    mode_pcst=True  : PCST (sous-graphe connecté), à la G-Retriever
    return_diagnostics=True : retourne aussi un dict avec scores avant/après PCST
    return_selected_indices=True : retourne (selected_nodes, selected_edges) (indices globaux)
                                    au lieu de (desc, sub_data). Utile pour fusionner
                                    plusieurs sous-graphes (mode question+1option).
    """
    n_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), graph.x, dim=-1)
    e_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), graph.edge_attr, dim=-1)

    topk = min(topk, n_sim.size(0))
    topk_e = min(topk_e, e_sim.size(0))

    # Top-k initiaux (pour diagnostics)
    topk_n_vals, topk_n_idx = torch.topk(n_sim, topk, largest=True)
    topk_e_vals, topk_e_idx = torch.topk(e_sim, topk_e, largest=True)

    def _make_diag(selected_nodes, selected_edges, fallback):
        return {
            "topk_node_idx": topk_n_idx.tolist(),
            "topk_node_sims": topk_n_vals.tolist(),
            "topk_edge_idx": topk_e_idx.tolist(),
            "topk_edge_sims": topk_e_vals.tolist(),
            "selected_nodes": list(selected_nodes),
            "selected_edges": list(selected_edges),
            "selected_node_sims": n_sim[list(selected_nodes)].tolist() if len(selected_nodes) else [],
            "selected_edge_sims": e_sim[list(selected_edges)].tolist() if len(selected_edges) else [],
            "n_selected_nodes": len(selected_nodes),
            "n_selected_edges": len(selected_edges),
            "pcst_fallback": fallback,
        }

    # ── Mode TOP-K direct ──────────────────────────────────────────────
    if not mode_pcst:
        desc, sub_data, sel_nodes, sel_edges = _retrieve_topk(
            graph, n_sim, e_sim, topk, topk_e, nodes_df, edges_df, return_selected=True, ln_lookup=ln_lookup
        )
        if return_selected_indices:
            return sel_nodes, sel_edges
        if return_diagnostics:
            return desc, sub_data, _make_diag(sel_nodes, sel_edges, fallback=False)
        return desc, sub_data

    # ── Mode PCST ──────────────────────────────────────────────────────
    n_prizes = torch.zeros(graph.x.size(0))
    n_prizes[topk_n_idx] = torch.linspace(1.0, 0.1, topk)

    e_prizes = torch.zeros(graph.edge_attr.size(0))
    e_prizes[topk_e_idx] = torch.linspace(1.0, 0.1, topk_e)

    n_prizes_np = n_prizes.numpy()
    e_prizes_np = e_prizes.numpy()
    edge_index_np = graph.edge_index.numpy().T

    costs_np = np.full(len(e_prizes_np), cost_e, dtype=np.float64)
    virtual_costs = costs_np - e_prizes_np

    neg_mask = virtual_costs < 0
    if neg_mask.any():
        src = graph.edge_index[0].numpy()
        dst = graph.edge_index[1].numpy()
        surplus = -virtual_costs[neg_mask]
        np.add.at(n_prizes_np, src[neg_mask], surplus / 2.0)
        np.add.at(n_prizes_np, dst[neg_mask], surplus / 2.0)
        virtual_costs[neg_mask] = 0.0

    root = -1
    pruning = 'gw'
    n_clusters = 2
    try:
        selected_nodes, selected_edges = pcst_fast.pcst_fast(
            edge_index_np, n_prizes_np, virtual_costs, root, n_clusters, pruning, 0,
        )
    except Exception as ex:
        print(f"⚠️ PCST a échoué ({ex}) → fallback top-k")
        desc, sub_data, sel_nodes, sel_edges = _retrieve_topk(
            graph, n_sim, e_sim, topk, topk_e, nodes_df, edges_df, return_selected=True, ln_lookup=ln_lookup
        )
        if return_selected_indices:
            return sel_nodes, sel_edges
        if return_diagnostics:
            return desc, sub_data, _make_diag(sel_nodes, sel_edges, fallback=True)
        return desc, sub_data

    if len(selected_nodes) < 3:
        desc, sub_data, sel_nodes, sel_edges = _retrieve_topk(
            graph, n_sim, e_sim, topk, topk_e, nodes_df, edges_df, return_selected=True, ln_lookup=ln_lookup
        )
        if return_selected_indices:
            return sel_nodes, sel_edges
        if return_diagnostics:
            return desc, sub_data, _make_diag(sel_nodes, sel_edges, fallback=True)
        return desc, sub_data

    selected_nodes = sorted(selected_nodes.tolist())
    selected_edges = sorted(selected_edges.tolist())

    if return_selected_indices:
        return selected_nodes, selected_edges

    desc, sub_data = _build_subgraph(
        graph, selected_nodes, selected_edges, nodes_df, edges_df, ln_lookup=ln_lookup
    )
    if return_diagnostics:
        return desc, sub_data, _make_diag(selected_nodes, selected_edges, fallback=False)
    return desc, sub_data


# ══════════════════════════════════════════════════════════════════════
# Retrieval KAPING : top-k triplets par similarité directe query/triplet
# ══════════════════════════════════════════════════════════════════════
def _triplets_desc(triplet_idx, nodes_df, edges_df, ln_lookup=None):
    """Construit le contexte textuel pour une liste d'indices de triplets (arêtes).

    Si ln_lookup est fourni, utilise les phrases en langage naturel (LN).
    Sinon, utilise le format "src,relation,dst".
    """
    id2label = dict(zip(nodes_df["node_id"], nodes_df["node_attr"]))
    MAX_LINE_CHARS = 500
    lines = []
    for ei in triplet_idx:
        row = edges_df.iloc[ei]
        src = id2label.get(row["src"], str(row["src"]))
        dst = id2label.get(row["dst"], str(row["dst"]))
        text = _triplet_to_ln(ln_lookup, src, row["edge_attr"], dst)
        if text is not None:
            if len(text) > MAX_LINE_CHARS:
                text = text[:MAX_LINE_CHARS]
            lines.append(text)
    if not lines:
        return None
    if ln_lookup is not None:
        return "\n".join(lines)
    return "src,relation,dst\n" + "\n".join(lines)


def triplets_to_context(triplets, ln_lookup=None):
    """Construit le contexte textuel depuis une liste de triplets (s, p, o).

    Si ln_lookup est fourni, utilise les phrases en langage naturel (LN).
    Sinon, utilise le format "src,relation,dst".
    Retourne None si la liste est vide.
    """
    if not triplets:
        return None
    MAX_LINE_CHARS = 500
    lines = []
    for s, p, o in triplets:
        text = _triplet_to_ln(ln_lookup, s, p, o)
        if text is not None:
            if len(text) > MAX_LINE_CHARS:
                text = text[:MAX_LINE_CHARS]
            lines.append(text)
    if not lines:
        return None
    if ln_lookup is not None:
        return "\n".join(lines)
    return "src,relation,dst\n" + "\n".join(lines)


def retrieval_kaping(graph, q_emb, nodes_df, edges_df, topk_triplets: int = 10,
                     return_selected_indices: bool = False, ln_lookup=None):
    """
    Méthode KAPING : cosine similarity entre la query et chaque triplet
    (src relation dst), puis sélection des top-k triplets.

    Les embeddings de triplets sont déjà calculés dans graph.edge_attr
    (cf. build_pyg_graph : edge_texts = "src rel dst").

    return_selected_indices=True : retourne la liste des indices de triplets
        (triée par similarité décroissante) au lieu du contexte textuel.
        Utile pour fusionner (union) les top-k de plusieurs requêtes
        (mode question+1option).

    Sinon retourne le contexte textuel au format "src,relation,dst" (une ligne
    par triplet), trié par similarité décroissante.
    """
    e_sim = torch.nn.functional.cosine_similarity(
        q_emb.unsqueeze(0), graph.edge_attr, dim=-1
    )
    k = min(topk_triplets, e_sim.size(0))
    top_vals, top_idx = torch.topk(e_sim, k, largest=True)
    top_idx = top_idx.tolist()

    if return_selected_indices:
        return top_idx

    return _triplets_desc(top_idx, nodes_df, edges_df, ln_lookup=ln_lookup)




# ══════════════════════════════════════════════════════════════════════
# Helpers internes
# ══════════════════════════════════════════════════════════════════════
def _retrieve_topk(graph, n_sim, e_sim, topk, topk_e, nodes_df, edges_df, return_selected=False, ln_lookup=None):
    _, top_n_idx = torch.topk(n_sim, topk, largest=True)

    selected_nodes = top_n_idx.tolist()
    sel_set = set(selected_nodes)

    edge_src = graph.edge_index[0].tolist()
    edge_dst = graph.edge_index[1].tolist()

    connected_edges = [
        i for i, (s, d) in enumerate(zip(edge_src, edge_dst))
        if s in sel_set and d in sel_set
    ]
    _, top_e_idx = torch.topk(e_sim, topk_e, largest=True)
    selected_edges = list(set(connected_edges) | set(top_e_idx.tolist()))

    for ei in selected_edges:
        s, d = edge_src[ei], edge_dst[ei]
        if s not in sel_set:
            selected_nodes.append(s); sel_set.add(s)
        if d not in sel_set:
            selected_nodes.append(d); sel_set.add(d)

    selected_nodes = sorted(selected_nodes)
    selected_edges = sorted(selected_edges)

    desc, sub_data = _build_subgraph(graph, selected_nodes, selected_edges, nodes_df, edges_df, ln_lookup=ln_lookup)
    if return_selected:
        return desc, sub_data, selected_nodes, selected_edges
    return desc, sub_data



def _build_subgraph(graph, selected_nodes, selected_edges, nodes_df, edges_df, ln_lookup=None):
    node_mask = torch.zeros(graph.x.size(0), dtype=torch.bool)
    node_mask[selected_nodes] = True
    edge_mask = torch.zeros(graph.edge_index.size(1), dtype=torch.bool)
    edge_mask[selected_edges] = True

    sub_x = graph.x[node_mask]
    sub_ea = graph.edge_attr[edge_mask]
    sub_ei_raw = graph.edge_index[:, edge_mask]

    remap = {old: new for new, old in enumerate(selected_nodes)}
    sub_ei = torch.tensor(
        [[remap[i] for i in sub_ei_raw[0].tolist()],
         [remap[i] for i in sub_ei_raw[1].tolist()]],
        dtype=torch.long,
    )

    node_df_sub = nodes_df.iloc[selected_nodes].reset_index(drop=True)
    edge_df_sub = edges_df.iloc[selected_edges].reset_index(drop=True)

    id2label = dict(zip(nodes_df["node_id"], nodes_df["node_attr"]))
    MAX_LINE_CHARS = 500
    lines = []
    for _, row in edge_df_sub.iterrows():
        src = id2label.get(row["src"], str(row["src"]))
        dst = id2label.get(row["dst"], str(row["dst"]))
        text = _triplet_to_ln(ln_lookup, src, row["edge_attr"], dst)
        if text is not None:
            if len(text) > MAX_LINE_CHARS:
                text = text[:MAX_LINE_CHARS]
            lines.append(text)

    if not lines:
        return None, Data(x=sub_x, edge_index=sub_ei, edge_attr=sub_ea)

    if ln_lookup is not None:
        desc = "\n".join(lines)
    else:
        desc = "src,relation,dst\n" + "\n".join(lines)

    sub_data = Data(x=sub_x, edge_index=sub_ei, edge_attr=sub_ea)
    return desc, sub_data


# ══════════════════════════════════════════════════════════════════════
# Multiprocessing
# ══════════════════════════════════════════════════════════════════════
_W_GRAPH = _W_NODES = _W_EDGES = _W_CFG = _W_LN_LOOKUP = None


def _pcst_init(graph, nodes_df, edges_df, cfg_dict, ln_lookup=None):
    global _W_GRAPH, _W_NODES, _W_EDGES, _W_CFG, _W_LN_LOOKUP
    import torch as _t
    _t.set_num_threads(1)
    _W_GRAPH, _W_NODES, _W_EDGES, _W_CFG, _W_LN_LOOKUP = graph, nodes_df, edges_df, cfg_dict, ln_lookup


def _pcst_task(q_emb, mode_pcst=True):
    desc, _ = retrieval_via_pcst(
        _W_GRAPH, q_emb, _W_NODES, _W_EDGES,
        topk=_W_CFG["topk_nodes"],
        topk_e=_W_CFG["topk_edges"],
        cost_e=_W_CFG["pcst_cost_e"],
        mode_pcst=mode_pcst,
        ln_lookup=_W_LN_LOOKUP,
    )
    return desc


def _pcst_task_keyed(q_emb, mode_pcst=True):
    """Variante de _pcst_task retournant (selected_nodes, selected_edges, desc).

    Utile pour construire/mettre à jour un cache PCST avec indices (compatible GNN).
    """
    desc, _sub, diag = retrieval_via_pcst(
        _W_GRAPH, q_emb, _W_NODES, _W_EDGES,
        topk=_W_CFG["topk_nodes"],
        topk_e=_W_CFG["topk_edges"],
        cost_e=_W_CFG["pcst_cost_e"],
        mode_pcst=mode_pcst,
        return_diagnostics=True,
        ln_lookup=_W_LN_LOOKUP,
    )
    return list(diag["selected_nodes"]), list(diag["selected_edges"]), desc


def _pcst_task_fusion(q_embs_per_option, mode_pcst=True):
    """
    q_embs_per_option : liste de q_emb (un par option : question+optionA, ...).
    On fait un retrieval PCST pour chaque combinaison question+1option, puis on
    fusionne les sous-graphes (union des nœuds et arêtes sélectionnés) avant de
    construire le contexte textuel final.
    """
    all_nodes = set()
    all_edges = set()
    for q_emb in q_embs_per_option:
        sel_nodes, sel_edges = retrieval_via_pcst(
            _W_GRAPH, q_emb, _W_NODES, _W_EDGES,
            topk=_W_CFG["topk_nodes"],
            topk_e=_W_CFG["topk_edges"],
            cost_e=_W_CFG["pcst_cost_e"],
            mode_pcst=mode_pcst,
            return_selected_indices=True,
        )
        all_nodes.update(sel_nodes)
        all_edges.update(sel_edges)

    merged_nodes = sorted(all_nodes)
    merged_edges = sorted(all_edges)

    desc, _ = _build_subgraph(
        _W_GRAPH, merged_nodes, merged_edges, _W_NODES, _W_EDGES, ln_lookup=_W_LN_LOOKUP
    )
    return desc



def _kaping_task(q_emb):
    desc = retrieval_kaping(
        _W_GRAPH, q_emb, _W_NODES, _W_EDGES,
        topk_triplets=_W_CFG["topk_triplets"],
        ln_lookup=_W_LN_LOOKUP,
    )
    return desc


def _kaping_task_fusion(q_embs_per_option):
    """
    q_embs_per_option : liste de q_emb (un par option : question+optionA, ...).
    On prend les top-k triplets pour chaque combinaison question+1option, puis on
    fait l'UNION des triplets. L'ordre final est trié par la similarité maximale
    obtenue sur l'une des options.
    """
    topk = _W_CFG["topk_triplets"]
    best_sim = {}  # triplet_idx -> meilleure similarité (sur toutes les options)
    for q_emb in q_embs_per_option:
        e_sim = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), _W_GRAPH.edge_attr, dim=-1
        )
        k = min(topk, e_sim.size(0))
        top_vals, top_idx = torch.topk(e_sim, k, largest=True)
        for ei, val in zip(top_idx.tolist(), top_vals.tolist()):
            if ei not in best_sim or val > best_sim[ei]:
                best_sim[ei] = val

    merged_idx = sorted(best_sim, key=lambda ei: best_sim[ei], reverse=True)
    return _triplets_desc(merged_idx, _W_NODES, _W_EDGES, ln_lookup=_W_LN_LOOKUP)


def parallel_graph_retrieval(q_embs, graph, nodes_df, edges_df, cfg, mode_pcst: bool, ln_lookup=None):
    """Lance le retrieval en parallèle pour une liste de q_embs."""
    from tqdm import tqdm

    cfg_dict = {
        "topk_nodes": cfg.topk_nodes,
        "topk_edges": cfg.topk_edges,
        "pcst_cost_e": cfg.pcst_cost_e,
    }
    task = partial(_pcst_task, mode_pcst=mode_pcst)

    with ProcessPoolExecutor(
        max_workers=cfg.pcst_workers,
        initializer=_pcst_init,
        initargs=(graph, nodes_df, edges_df, cfg_dict, ln_lookup),
    ) as ex:
        contexts = list(tqdm(
            ex.map(task, q_embs, chunksize=8),
            total=len(q_embs),
            desc=f"PCST (mode_pcst={mode_pcst})",
        ))
    return contexts


def parallel_graph_retrieval_keyed(q_embs, graph, nodes_df, edges_df, cfg, mode_pcst: bool = True, ln_lookup=None):
    """Retrieval PCST parallèle renvoyant (selected_nodes, selected_edges, desc) par question.

    Symétrique de parallel_graph_retrieval mais conserve les indices globaux des
    nœuds/arêtes sélectionnés (pour construire/mettre à jour un cache PCST
    compatible GNN).
    """
    from tqdm import tqdm

    cfg_dict = {
        "topk_nodes": cfg.topk_nodes,
        "topk_edges": cfg.topk_edges,
        "pcst_cost_e": cfg.pcst_cost_e,
    }
    task = partial(_pcst_task_keyed, mode_pcst=mode_pcst)

    with ProcessPoolExecutor(
        max_workers=cfg.pcst_workers,
        initializer=_pcst_init,
        initargs=(graph, nodes_df, edges_df, cfg_dict, ln_lookup),
    ) as ex:
        results = list(tqdm(
            ex.map(task, q_embs, chunksize=8),
            total=len(q_embs),
            desc=f"PCST-keyed (mode_pcst={mode_pcst})",
        ))
    return results


def parallel_graph_retrieval_fusion(q_embs_per_question, graph, nodes_df, edges_df, cfg, mode_pcst: bool, ln_lookup=None):
    """Retrieval avec fusion de sous-graphes (mode question+1option).

    Args:
        q_embs_per_question : liste (une entrée par question) de listes de q_emb,
                              un q_emb par combinaison question+1option.
    Pour chaque question, on fait un retrieval PCST par option puis on fusionne
    (union) les sous-graphes sélectionnés.
    """
    from tqdm import tqdm

    cfg_dict = {
        "topk_nodes": cfg.topk_nodes,
        "topk_edges": cfg.topk_edges,
        "pcst_cost_e": cfg.pcst_cost_e,
    }
    task = partial(_pcst_task_fusion, mode_pcst=mode_pcst)

    with ProcessPoolExecutor(
        max_workers=cfg.pcst_workers,
        initializer=_pcst_init,
        initargs=(graph, nodes_df, edges_df, cfg_dict, ln_lookup),
    ) as ex:
        contexts = list(tqdm(
            ex.map(task, q_embs_per_question, chunksize=4),
            total=len(q_embs_per_question),
            desc=f"PCST-FUSION (mode_pcst={mode_pcst})",
        ))
    return contexts


def parallel_graph_retrieval_kaping(q_embs, graph, nodes_df, edges_df, cfg, ln_lookup=None):
    """Retrieval KAPING en parallèle : top-k triplets par similarité query/triplet."""
    from tqdm import tqdm

    cfg_dict = {
        "topk_triplets": cfg.topk_triplets,
    }

    with ProcessPoolExecutor(
        max_workers=cfg.pcst_workers,
        initializer=_pcst_init,
        initargs=(graph, nodes_df, edges_df, cfg_dict, ln_lookup),
    ) as ex:
        contexts = list(tqdm(
            ex.map(_kaping_task, q_embs, chunksize=8),
            total=len(q_embs),
            desc="KAPING (top-k triplets)",
        ))
    return contexts


def parallel_graph_retrieval_kaping_fusion(q_embs_per_question, graph, nodes_df, edges_df, cfg, ln_lookup=None):
    """Retrieval KAPING avec fusion (mode question+1option) : pour chaque question,
    union des top-k triplets obtenus sur chaque combinaison question+1option.

    Args:
        q_embs_per_question : liste (une entrée par question) de listes de q_emb,
                              un q_emb par combinaison question+1option.
    """
    from tqdm import tqdm

    cfg_dict = {
        "topk_triplets": cfg.topk_triplets,
    }

    with ProcessPoolExecutor(
        max_workers=cfg.pcst_workers,
        initializer=_pcst_init,
        initargs=(graph, nodes_df, edges_df, cfg_dict, ln_lookup),
    ) as ex:
        contexts = list(tqdm(
            ex.map(_kaping_task_fusion, q_embs_per_question, chunksize=4),
            total=len(q_embs_per_question),
            desc="KAPING-FUSION (union top-k triplets)",
        ))
    return contexts
