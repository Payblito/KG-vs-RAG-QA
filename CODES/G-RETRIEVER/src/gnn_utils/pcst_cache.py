"""Pré-calcul et cache des sous-graphes PCST (1 fois pour toutes).

On stocke, par question, les indices GLOBAUX des noeuds et aretes selectionnes
par PCST + le `desc` (CSV src,relation,dst) deja produit par ta fonction.
Le GNN reconstruit ensuite le sous-graphe via graph.x[sel_nodes], etc.

Deux formats de cache coexistent :
  - legacy positionnel : list[dict] aligné sur df_questions.reset_index(drop=True),
    chaque dict = {selected_nodes, selected_edges, desc}.
  - keyed (recommandé) : list[dict] avec clés explicites
    {article_id, question, selected_nodes, selected_edges, desc}, alignement par
    clé (article_id, question) et non plus par position.

`load_pcst_cache_as_dict` gère les deux formats (déduction des clés pour le
format legacy via re-chargement du df source).
"""
import os
import pickle
import torch
from tqdm import tqdm

from ..retrieval_graph import retrieval_via_pcst   # ta fonction existante


# ── Mémoïsation des dicts de cache (path -> dict{(article_id, question): entry}) ──
_PCST_CACHE_DICTS: dict = {}


def _build_query_embedding(question, options, graph_embedder):
    """Mode 'question_options' : on concatene question + les 4 options,
    puis on encode avec ton embedder (Jina v3). Retourne un tensor [d]."""
    opts = " ".join(f"{L}) {options[L]}" for L in ["A", "B", "C", "D"])
    text = f"{question} {opts}"
    q_emb = graph_embedder.encode([text])          # adapte si ton API differe
    if isinstance(q_emb, torch.Tensor):
        return q_emb.squeeze(0).float()
    return torch.tensor(q_emb, dtype=torch.float).squeeze(0)


def _build_query_embedding_mode(question, options, graph_embedder, query_mode,
                                correct_letter=None, device=None):
    """Généralise _build_query_embedding à tous les retrieval_query_mode.

    Modes supportés :
      - "question"          : question seule
      - "question_options"  : question + 4 options (équivalent _build_query_embedding)
      - "oracle"            : question + bonne réponse (correct_letter requis)

    Retourne un tensor [d].
    """
    if query_mode == "question":
        text = question
    elif query_mode == "question_options":
        opts = " ".join(f"{L}) {options[L]}" for L in ["A", "B", "C", "D"])
        text = f"{question} {opts}"
    elif query_mode == "oracle":
        assert correct_letter is not None, "correct_letter requis pour oracle"
        text = f"{question} {options[correct_letter]}"
    else:
        raise ValueError(f"query_mode non supporté pour le cache PCST : {query_mode}")

    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    q_emb = graph_embedder.encode([text], **kwargs)
    if isinstance(q_emb, torch.Tensor):
        return q_emb.squeeze(0).float()
    return torch.tensor(q_emb, dtype=torch.float).squeeze(0)


def build_pcst_cache(df_questions, cfg, graph, nodes_df, edges_df,
                     graph_embedder, cache_path):
    """Construit le cache PCST pour toutes les questions et le serialise.

    Retourne: list[dict] aligne sur df_questions.reset_index(drop=True),
    chaque dict = {selected_nodes, selected_edges, desc}.
    """
    # if os.path.exists(cache_path):
    #     print(f"[pcst_cache] cache existant -> {cache_path}")
    #     return load_pcst_cache(cache_path)

    df = df_questions.reset_index(drop=True)
    cache = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PCST cache"):
        options = {L: row[f"option {L}"] for L in ["A", "B", "C", "D"]}
        q_emb = _build_query_embedding(row["question"], options, graph_embedder)

        out = retrieval_via_pcst(
            graph, q_emb, nodes_df, edges_df,
            topk=cfg.topk_nodes, topk_e=cfg.topk_edges,
            cost_e=cfg.pcst_cost_e, mode_pcst=True,
            return_diagnostics=True,
        )
        desc, _sub_data, diag = out          # 3 elements car return_diagnostics=True

        cache.append({
            "selected_nodes": list(diag["selected_nodes"]),
            "selected_edges": list(diag["selected_edges"]),
            "desc": desc,
        })  

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[pcst_cache] {len(cache)} sous-graphes caches -> {cache_path}")
    return cache


def load_pcst_cache(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════
# Format keyed : alignement par (article_id, question)
# ══════════════════════════════════════════════════════════════════════
def _cache_entry_key(entry):
    """Retourne la clé (article_id, question) d'une entrée keyed, ou None."""
    if "article_id" in entry and "question" in entry:
        return (str(entry["article_id"]), entry["question"])
    return None


def _deduce_keys_positional(cache, cfg):
    """Re-déduit les clés (article_id, question) d'un cache legacy positionnel.

    On re-charge le df_questions source (questions_csv) et on applique le même
    filtrage par articles que dans le notebook, puis zip positionnel avec le
    cache. La clé (article_id, question) est unique dans le CSV source.
    """
    from ..data_loading import load_questions, load_article_ids, filter_questions_by_articles

    df = load_questions(cfg.questions_csv)
    if getattr(cfg, "filter_by_articles", False):
        article_ids = load_article_ids(cfg.articles_csv)
        df = filter_questions_by_articles(df, article_ids)
    df = df.reset_index(drop=True)

    if len(df) != len(cache):
        raise ValueError(
            f"[pcst_cache] Cache legacy positionnel : len(df_source)={len(df)} "
            f"!= len(cache)={len(cache)}. Impossible de déduire les clés. "
            f"Reconstruisez le cache avec build_pcst_cache_keyed."
        )
    keys = [(str(aid), q) for aid, q in zip(df["article_id"], df["question"])]
    return {k: e for k, e in zip(keys, cache)}


def load_pcst_cache_as_dict(cache_path, cfg=None):
    """Charge un cache PCST et retourne un dict {(article_id, question): entry}.

    Gère les deux formats :
      - keyed    : entrées avec clés article_id/question (forward-compatible).
      - legacy   : entrées positionnelles sans clés ; déduction via cfg + df source.

    Mémoïsé au niveau module (_PCST_CACHE_DICTS) pour éviter les relectures pkl.

    Args:
        cache_path: chemin du pkl.
        cfg: Config (requis pour le format legacy afin de déduire les clés).
    """
    if cache_path in _PCST_CACHE_DICTS:
        return _PCST_CACHE_DICTS[cache_path]

    if not os.path.exists(cache_path):
        _PCST_CACHE_DICTS[cache_path] = {}
        return {}

    cache = load_pcst_cache(cache_path)
    if not cache:
        _PCST_CACHE_DICTS[cache_path] = {}
        return {}

    # Format keyed ?
    first_key = _cache_entry_key(cache[0])
    if first_key is not None:
        d = {}
        for entry in cache:
            k = _cache_entry_key(entry)
            if k is not None:
                d[k] = entry
        _PCST_CACHE_DICTS[cache_path] = d
        return d

    # Format legacy positionnel -> déduction des clés
    if cfg is None:
        raise ValueError(
            "[pcst_cache] Cache legacy positionnel détecté : cfg requis pour "
            "déduire les clés (article_id, question)."
        )
    print("[pcst_cache] Cache legacy positionnel -> déduction des clés via cfg")
    d = _deduce_keys_positional(cache, cfg)
    _PCST_CACHE_DICTS[cache_path] = d
    return d


def invalidate_pcst_cache_dict(cache_path=None):
    """Invalide le dict de cache mémoïsé (pour forcer une relecture)."""
    if cache_path is None:
        _PCST_CACHE_DICTS.clear()
    else:
        _PCST_CACHE_DICTS.pop(cache_path, None)


def build_pcst_cache_keyed(df_missing, cfg, graph, nodes_df, edges_df,
                           graph_embedder, query_mode=None, ln_lookup=None):
    """Construit des entrées PCST keyed pour df_missing (parallèle).

    Args:
        df_missing: DataFrame des questions à calculer (reset_index(drop=True)).
        cfg: Config.
        graph, nodes_df, edges_df: artifacts du graphe.
        graph_embedder: SentenceTransformer (Jina).
        query_mode: mode de construction de la query (défaut: cfg.retrieval_query_mode).
        ln_lookup: optional LN lookup.

    Returns:
        list[dict] avec clés {article_id, question, selected_nodes, selected_edges, desc}.
    """
    from ..retrieval_graph import parallel_graph_retrieval_keyed

    if query_mode is None:
        query_mode = getattr(cfg, "retrieval_query_mode", "question_options")

    df = df_missing.reset_index(drop=True)

    # Construction des query_texts selon query_mode
    query_texts = []
    for _, row in df.iterrows():
        options = {L: row[f"option {L}"] for L in ["A", "B", "C", "D"]}
        if query_mode == "question":
            query_texts.append(row["question"])
        elif query_mode == "question_options":
            opts = " ".join(f"{L}) {options[L]}" for L in ["A", "B", "C", "D"])
            query_texts.append(f"{row['question']} {opts}")
        elif query_mode == "oracle":
            query_texts.append(f"{row['question']} {options[row['correct_letter']]}")
        else:
            raise ValueError(f"query_mode non supporté : {query_mode}")

    # Encodage batch
    q_embs_t = graph_embedder.encode(
        query_texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=cfg.device,
        task="retrieval.query",
    ).cpu()
    q_list = [q_embs_t[i] for i in range(q_embs_t.size(0))]

    # Retrieval parallèle keyed
    results = parallel_graph_retrieval_keyed(
        q_list, graph, nodes_df, edges_df, cfg, mode_pcst=True, ln_lookup=ln_lookup
    )

    entries = []
    for (_, row), (sel_nodes, sel_edges, desc) in zip(df.iterrows(), results):
        entries.append({
            "article_id": str(row["article_id"]),
            "question": row["question"],
            "selected_nodes": list(sel_nodes),
            "selected_edges": list(sel_edges),
            "desc": desc,
        })
    return entries


def save_pcst_cache(entries, cache_path, existing_dict=None):
    """Sauvegarde un cache PCST keyed (format liste de dicts avec clés).

    Si existing_dict est fourni (dict {(article_id, question): entry}), on
    fusionne : les nouvelles entrées écrasent les clés identiques, les autres
    sont conservées. Le résultat est sérialisé en liste de dicts keyed
    (forward-compatible avec load_pcst_cache_as_dict et le GNN).

    Invalide le dict mémoïsé pour forcer une relecture au prochain appel.
    """
    merged = {}
    if existing_dict:
        merged.update(existing_dict)
    for e in entries:
        k = _cache_entry_key(e)
        if k is not None:
            merged[k] = e

    out_list = list(merged.values())
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out_list, f)
    print(f"[pcst_cache] {len(out_list)} sous-graphes caches (keyed) -> {cache_path}")

    invalidate_pcst_cache_dict(cache_path)
    return out_list
