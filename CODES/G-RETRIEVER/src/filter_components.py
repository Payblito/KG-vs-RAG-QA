import json
import networkx as nx

def filter_graph_json_to_giant_component(input_json_path: str, output_json_path: str):
    """
    Charge un graphe au format kg_gen.Graph (JSON), conserve uniquement
    la plus grande composante connexe, et réenregistre au même format.

    Format JSON attendu/produit :
        {
            "entities": [...],
            "edges": [...],
            "relations": [[h, r, t], ...],
            "entity_clusters": {...},
            "edge_clusters": {...},
            "entity_metadata": {...}
        }
    """
    # ── Chargement ──
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = list(data.get("entities", []))
    relations = data.get("relations", [])

    # Tous les nœuds (entities + ceux apparaissant dans relations)
    all_nodes = set(entities)
    for h, r, t in relations:
        all_nodes.add(h)
        all_nodes.add(t)

    # ── Construction du graphe NetworkX ──
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from((h, t) for h, r, t in relations)

    # ── Composante géante ──
    components = list(nx.connected_components(G))
    giant = max(components, key=len)
    giant_set = set(giant)

    print(f"  • Nœuds avant : {len(all_nodes)} | après : {len(giant_set)}")
    print(f"  • Relations avant : {len(relations)}", end="")

    # ── Filtrage ──
    new_entities = [e for e in entities if e in giant_set]
    new_relations = [
        [h, r, t] for h, r, t in relations
        if h in giant_set and t in giant_set
    ]

    print(f" | après : {len(new_relations)}")
    print(f"  • Composantes supprimées : {len(components) - 1}")

    # Filtrage des champs optionnels (clusters / metadata) s'ils existent
    entity_clusters = data.get("entity_clusters", {})
    new_entity_clusters = {
        k: [v for v in vs if v in giant_set]
        for k, vs in entity_clusters.items()
        if k in giant_set
    }
    # Nettoyage des clés vides
    new_entity_clusters = {k: v for k, v in new_entity_clusters.items() if v}

    # edge_clusters : on garde tel quel (ce sont des labels de relations, pas des entités)
    edge_clusters = data.get("edge_clusters", {})

    entity_metadata = data.get("entity_metadata", {})
    new_entity_metadata = {
        k: v for k, v in entity_metadata.items() if k in giant_set
    }

    # edges (labels de relations) : on garde ceux encore présents dans new_relations
    used_edge_labels = {r for h, r, t in new_relations}
    new_edges = [e for e in data.get("edges", []) if e in used_edge_labels]

    # ── Écriture ──
    new_data = {
        "entities": new_entities,
        "edges": new_edges,
        "relations": new_relations,
        "entity_clusters": new_entity_clusters,
        "edge_clusters": edge_clusters,
        "entity_metadata": new_entity_metadata,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Graphe filtré sauvegardé : {output_json_path}")

