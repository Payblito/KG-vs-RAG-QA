import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


# ──────────────────────────────────────────────────────────────────────
# Construction depuis le JSON
# ──────────────────────────────────────────────────────────────────────
def load_graph_from_json(json_path: str):
    """
    Charge un graphe sauvegardé au format kg_gen.Graph (sérialisé en JSON).
    
    Format attendu :
        {
            "entities": [...],
            "edges": [...],            # labels de relations (non utilisés ici)
            "relations": [[h, r, t], ...],
            "entity_clusters": {...},  # ignoré
            "edge_clusters": {...},    # ignoré
            "entity_metadata": {...}   # ignoré
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Collecte des nœuds : entities + tout ce qui apparaît dans les relations ──
    nodes_set = set(data.get("entities", []))
    for rel in data["relations"]:
        h, r, t = rel  # tuple/list de 3 éléments
        nodes_set.add(h)
        nodes_set.add(t)

    nodes_list = sorted(nodes_set)
    node2id = {n: i for i, n in enumerate(nodes_list)}

    nodes_df = pd.DataFrame({
        "node_id": list(range(len(nodes_list))),
        "node_attr": nodes_list,
    })

    edges = [
        {"src": node2id[h], "edge_attr": r, "dst": node2id[t]}
        for h, r, t in data["relations"]
    ]
    edges_df = pd.DataFrame(edges)

    print(f"  • {len(nodes_df)} nœuds, {len(edges_df)} relations")

    import networkx as nx


# ⚠️ Reconstruire le mapping node_id → index et reconstruire `graph` (PyG Data)

    return nodes_df, edges_df, node2id


def build_pyg_graph(nodes_df, edges_df, embedder) -> Data:
    # Nœuds : on utilise la tâche d'indexation (retrieval.passage)
    node_texts = [n for n in nodes_df["node_attr"].tolist()]
    
    x = embedder.encode(
        node_texts, 
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        task="retrieval.passage"  # <-- AJOUT POUR JINA
    )

    

    node_attr = nodes_df["node_attr"].tolist()
    edge_texts = [
        f"{node_attr[src]} {rel} {node_attr[dst]}"
        for src, rel, dst in zip(edges_df["src"], edges_df["edge_attr"], edges_df["dst"])
    ]
    
    # Arêtes/Relations : même tâche d'indexation
    edge_attr = embedder.encode(
        edge_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,  # Fortement recommandé aussi pour les arêtes
        show_progress_bar=True,
        batch_size=128,
        task="retrieval.passage"  # <-- AJOUT POUR JINA
    )

    edge_index = torch.tensor(
        [edges_df["src"].tolist(), edges_df["dst"].tolist()],
        dtype=torch.long,
    )

    return Data(
        x=x.cpu(),
        edge_index=edge_index,
        edge_attr=edge_attr.cpu(),
        num_nodes=len(nodes_df),
    )



# ──────────────────────────────────────────────────────────────────────
# Save / Load artifacts
# ──────────────────────────────────────────────────────────────────────
def save_graph_artifacts(nodes_df, edges_df, node2id, graph, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes_df.to_parquet(out / "nodes_df.parquet")
    edges_df.to_parquet(out / "edges_df.parquet")
    with open(out / "node2id.pkl", "wb") as f:
        pickle.dump(node2id, f)
    torch.save(graph, out / "graph.pt")

    print(f"✅ Artifacts sauvegardés dans {out.resolve()}")


def load_graph_artifacts(out_dir: str):
    out = Path(out_dir)
    nodes_df = pd.read_parquet(out / "nodes_df.parquet")
    edges_df = pd.read_parquet(out / "edges_df.parquet")
    with open(out / "node2id.pkl", "rb") as f:
        node2id = pickle.load(f)
    graph = torch.load(out / "graph.pt", weights_only=False)

    print(f"✅ Artifacts chargés depuis {out.resolve()}")
    return nodes_df, edges_df, node2id, graph


def artifacts_exist(out_dir: str) -> bool:
    out = Path(out_dir)
    return all((out / f).exists() for f in [
        "nodes_df.parquet", "edges_df.parquet", "node2id.pkl", "graph.pt"
    ])


def ensure_graph_artifacts(cfg, embedder):
    """Construit les artifacts si absents (ou si force_rebuild), sinon les charge."""
    if cfg.force_rebuild_graph_artifacts or not artifacts_exist(cfg.artifacts_dir):
        print("🔨 Construction des artifacts du graphe...")
        nodes_df, edges_df, node2id = load_graph_from_json(cfg.graph_json_path)
        graph = build_pyg_graph(nodes_df, edges_df, embedder)
        save_graph_artifacts(nodes_df, edges_df, node2id, graph, cfg.artifacts_dir)
    return load_graph_artifacts(cfg.artifacts_dir)


# ──────────────────────────────────────────────────────────────────────
# Provenance : index article_id -> liste de (subject, predicate, object)
# ──────────────────────────────────────────────────────────────────────
def load_provenance(pkl_path: str):
    """Charge le fichier de provenance (.pkl) et construit un index
    article_id -> liste de triplets (subject, predicate, object).

    Formats supportés :
      - dict {(subject, predicate, object): [article_id, ...], ...}
      - liste de dicts [{"subject":..., "predicate":..., "object":..., "articles":[...]}]
    """
    with open(pkl_path, "rb") as f:
        provenance = pickle.load(f)

    index = {}
    if isinstance(provenance, dict):
        for (s, p, o), art_ids in provenance.items():
            for art_id in art_ids:
                index.setdefault(art_id, []).append((s, p, o))
        nb_relations = len(provenance)
    else:
        for entry in provenance:
            s, p, o = entry["subject"], entry["predicate"], entry["object"]
            for art_id in entry["articles"]:
                index.setdefault(art_id, []).append((s, p, o))
        nb_relations = len(provenance)

    print(f"✅ Provenance chargée : {len(index)} articles indexés, "
          f"{nb_relations} relations au total")
    return index


# ──────────────────────────────────────────────────────────────────────
# Embeddings des triplets de provenance (calculés une fois pour toute)
# ──────────────────────────────────────────────────────────────────────
def build_provenance_embeddings(provenance_index: dict, embedder, cfg):
    """Encode tous les triplets uniques de la provenance avec l'embedder.

    Retourne :
        triplet_keys : list[(s, p, o)]   — liste ordonnée des triplets uniques
        triplet_emb  : Tensor[N, D]      — embeddings normalisés (CPU)
        triplet2idx  : dict[(s,p,o) -> int] — lookup rapide
    """
    # Collecte des triplets uniques
    unique = set()
    for triplets in provenance_index.values():
        for t in triplets:
            unique.add(t)
    triplet_keys = list(unique)
    triplet2idx = {t: i for i, t in enumerate(triplet_keys)}

    # Textes à encoder (format cohérent avec triplets_to_context)
    texts = [f"{s},{p},{o}" for (s, p, o) in triplet_keys]

    print(f"🔨 Encodage de {len(texts)} triplets de provenance...")
    triplet_emb = embedder.encode(
        texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=cfg.device,
        task="retrieval.passage",
    ).cpu()

    print(f"✅ {triplet_emb.size(0)} triplets encodés (dim={triplet_emb.size(1)})")
    return triplet_keys, triplet_emb, triplet2idx
