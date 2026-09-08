from typing import List
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from kg_gen.models import Graph
import logging
import threading
import time
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans


class LLMDeduplicate:
    graph: Graph
    nodes: list[str]
    edges: list[str]
    node_clusters: list[list[str]]
    edge_clusters: list[list[str]]
    retrieval_model: SentenceTransformer
    lm: dspy.LM

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        retrieval_model: SentenceTransformer,
        lm: dspy.LM,
        graph: Graph,
        cache_dir: str | Path = "./kg_dedup_cache",  # 🆕 dossier des checkpoints
    ):
        print("\n" + "=" * 70)
        print("🏗️  INITIALISATION de LLMDeduplicate")
        print("=" * 70)
        self.graph = graph
        self.nodes = sorted(graph.entities)
        self.edges = sorted(graph.edges)
        self.node_clusters = graph.entity_clusters or []
        self.edge_clusters = graph.edge_clusters or []
        self.retrieval_model = retrieval_model
        self.lm = lm

        # 🆕 Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  💾 Cache checkpoints : {self.cache_dir.resolve()}")

        print(f"  📊 Nombre de nodes (entités) : {len(self.nodes)}")
        print(f"  📊 Nombre d'edges (prédicats) : {len(self.edges)}")
        print(f"  📊 Nombre de relations       : {len(graph.relations)}")

        # Embeddings nodes
        print(f"\n  🔢 Encodage embeddings des {len(self.nodes)} nodes...")
        t0 = time.time()
        self.node_embeddings = retrieval_model.encode(self.nodes, show_progress_bar=True)
        print(f"     ✅ Fait en {time.time() - t0:.1f}s | shape={self.node_embeddings.shape}")

        print(f"  🔤 Construction BM25 nodes...")
        self.node_bm25_tokenized = [text.lower().split() for text in self.nodes]
        self.node_bm25 = BM25Okapi(self.node_bm25_tokenized)
        print(f"     ✅ BM25 nodes prêt")

        # Embeddings edges
        print(f"\n  🔢 Encodage embeddings des {len(self.edges)} edges...")
        t0 = time.time()
        self.edge_embeddings = retrieval_model.encode(self.edges, show_progress_bar=True)
        print(f"     ✅ Fait en {time.time() - t0:.1f}s | shape={self.edge_embeddings.shape}")

        print(f"  🔤 Construction BM25 edges...")
        self.edge_bm25_tokenized = [text.lower().split() for text in self.edges]
        self.edge_bm25 = BM25Okapi(self.edge_bm25_tokenized)
        print(f"     ✅ BM25 edges prêt")

        dspy.configure(lm=lm)
        print(f"\n  ⚙️  DSPy configuré avec LM : {lm}")
        print("=" * 70 + "\n")

        self._progress_lock = threading.Lock()
        self._llm_calls_done = 0
        self._llm_calls_total = 0

    def get_relevant_items(self, query: str, top_k: int = 50, type: str = "node") -> list[str]:
        query_tokens = query.lower().split()
        bm25_scores = (
            self.node_bm25.get_scores(query_tokens)
            if type == "node"
            else self.edge_bm25.get_scores(query_tokens)
        )
        query_embedding = self.retrieval_model.encode([query], show_progress_bar=False)
        embeddings = self.node_embeddings if type == "node" else self.edge_embeddings
        embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()
        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        items = self.nodes if type == "node" else self.edges
        return [items[i] for i in top_indices]

    def cluster(self):
        # ... (inchangé, je ne le recopie pas pour la lisibilité)
        print("\n" + "=" * 70)
        print("🚀 ÉTAPE 1 : CLUSTERING KMEANS")
        print("=" * 70)

        cluster_size = 128
        embedding_sets = {"node": self.node_embeddings, "edge": self.edge_embeddings}

        for embedding_type, embeddings in embedding_sets.items():
            n_samples = len(embeddings)
            num_clusters = max(1, n_samples // cluster_size)
            print(f"\n  ▶ Type='{embedding_type}' : {n_samples} items → {num_clusters} clusters (taille max ~{cluster_size})")
            t0 = time.time()
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=1, max_iter=20, tol=0.0, algorithm="lloyd", verbose=False)
            kmeans.fit(embeddings.astype(np.float32))
            centroids = kmeans.cluster_centers_
            print(f"     ⏱️  KMeans fit fait en {time.time() - t0:.1f}s")
            distances = cdist(embeddings, centroids)
            assignments = np.argsort(distances, axis=1)
            clusters: List[List[int]] = [[] for _ in range(num_clusters)]
            assigned = np.zeros(n_samples, dtype=bool)
            for rank in range(num_clusters):
                for i in range(n_samples):
                    if assigned[i]:
                        continue
                    cluster_id = assignments[i, rank]
                    if len(clusters[cluster_id]) < cluster_size:
                        clusters[cluster_id].append(i)
                        assigned[i] = True
            unassigned = np.where(~assigned)[0]
            if len(unassigned) > 0:
                print(f"     ⚠️  {len(unassigned)} items non-assignés → ajoutés dans un cluster séparé")
                clusters.append(unassigned.tolist())
            sizes = [len(c) for c in clusters]
            print(f"     📦 Clusters obtenus : {len(clusters)}")
            print(f"     📏 Tailles : min={min(sizes)}, max={max(sizes)}, moy={sum(sizes)/len(sizes):.1f}")
            if embedding_type == "node":
                clusters_data = [[self.nodes[idx] for idx in c] for c in clusters]
                self.node_clusters = clusters_data
                print(f"     ✅ node_clusters enregistrés ({len(clusters_data)} clusters)")
                print(f"     🔎 Exemple cluster[0] (3 premiers) : {clusters_data[0][:3]}")
            else:
                clusters_data = [[self.edges[idx] for idx in c] for c in clusters]
                self.edge_clusters = clusters_data
                print(f"     ✅ edge_clusters enregistrés ({len(clusters_data)} clusters)")
                if clusters_data and clusters_data[0]:
                    print(f"     🔎 Exemple cluster[0] (3 premiers) : {clusters_data[0][:3]}")

        print("\n" + "=" * 70)
        print("✅ FIN ÉTAPE 1 : CLUSTERING KMEANS")
        print("=" * 70 + "\n")

    # 🆕 ════════════════════════════════════════════════════════════════
    def _checkpoint_path(self, cluster_idx: int, type: str) -> Path:
        """Chemin du fichier checkpoint pour un cluster donné."""
        return self.cache_dir / f"{type}_cluster_{cluster_idx:04d}.pkl"

    def _load_checkpoint(self, cluster_idx: int, type: str):
        """Charge un checkpoint s'il existe, sinon retourne None."""
        path = self._checkpoint_path(cluster_idx, type)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"     ⚠️  Checkpoint corrompu {path.name} : {e} → recompute")
                path.unlink(missing_ok=True)
        return None

    def _save_checkpoint(self, cluster_idx: int, type: str, result):
        """Sauve le résultat d'un cluster sur disque (atomique)."""
        path = self._checkpoint_path(cluster_idx, type)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(result, f)
        tmp.replace(path)  # rename atomique → pas de fichier corrompu en cas de crash
    # ═════════════════════════════════════════════════════════════════════

    def deduplicate_cluster_with_cache(
        self, cluster_idx: int, cluster: list[str], type: str
    ) -> tuple[set, dict[str, list[str]]]:
        """🆕 Wrapper avec checkpoint persistant."""
        # 1. Tentative de chargement
        cached = self._load_checkpoint(cluster_idx, type)
        if cached is not None:
            # On incrémente quand même le compteur de progression
            with self._progress_lock:
                self._llm_calls_done += len(cluster)
            return cached

        # 2. Calcul réel
        result = self.deduplicate_cluster(cluster, type)

        # 3. Sauvegarde
        self._save_checkpoint(cluster_idx, type, result)
        return result

    def deduplicate_cluster(
        self, cluster: list[str], type: str = "node"
    ) -> tuple[set, dict[str, list[str]]]:
        cluster = cluster.copy()
        items = set()
        item_clusters = {}
        plural_type = "entities" if type == "node" else "edges"
        singular_type = "entity" if type == "node" else "edge"

        while len(cluster) > 0:
            item = cluster.pop()
            relevant_items = self.get_relevant_items(item, 16, type)

            class Deduplicate(dspy.Signature):
                __doc__ = f"""Find duplicate {plural_type} for the item and an alias that best represents the duplicates. Duplicates are those that are the same in meaning, such as with variation in tense, plural form, stem form, case, abbreviation, shorthand. Return an empty list if there are none."""
                item: str = dspy.InputField()
                set: list[str] = dspy.InputField()
                duplicates: list[str] = dspy.OutputField(description="Exact matches to items in {plural_type} set")
                alias: str = dspy.OutputField(description=f"Best {singular_type} name to represent the duplicates, ideally from the {plural_type} set")

            deduplicate = dspy.Predict(Deduplicate)
            result = deduplicate(item=item, set=relevant_items)

            with self._progress_lock:
                self._llm_calls_done += 1
                done = self._llm_calls_done
                total = self._llm_calls_total
            if done % 10 == 0 or done == total:
                pct = (done / total * 100) if total else 0
                print(f"  ⏳ Progression LLM : {done}/{total} ({pct:.1f}%) - dernier '{type}' traité : '{item[:40]}'")

            items.add(result.alias)
            duplicates = [dup for dup in result.duplicates if dup in cluster]

            if len(duplicates) > 0:
                item_clusters[result.alias] = {item}
                for duplicate in duplicates:
                    cluster.remove(duplicate)
                    item_clusters[result.alias].add(duplicate)
            else:
                item_clusters[item] = {item}

        return items, item_clusters

    def deduplicate(self) -> Graph:
        print("\n" + "=" * 70)
        print("🚀 ÉTAPE 2 : DÉDUPLICATION VIA LLM")
        print("=" * 70)

        entities = set()
        edges = set()
        entity_clusters = {}
        edge_clusters = {}

        cnt_nodes = sum(len(c) for c in self.node_clusters)
        cnt_edges = sum(len(c) for c in self.edge_clusters)
        self._llm_calls_done = 0
        self._llm_calls_total = cnt_nodes + cnt_edges

        # 🆕 Détection des checkpoints existants
        existing_node_ckpts = sum(
            1 for i in range(len(self.node_clusters))
            if self._checkpoint_path(i, "node").exists()
        )
        existing_edge_ckpts = sum(
            1 for i in range(len(self.edge_clusters))
            if self._checkpoint_path(i, "edge").exists()
        )

        print(f"  📦 {len(self.node_clusters)} clusters de nodes ({cnt_nodes} items au total)")
        print(f"  📦 {len(self.edge_clusters)} clusters d'edges ({cnt_edges} items au total)")
        print(f"  🔮 Nombre total d'appels LLM attendus : {self._llm_calls_total}")
        print(f"  💾 Checkpoints détectés : {existing_node_ckpts}/{len(self.node_clusters)} nodes, "
              f"{existing_edge_ckpts}/{len(self.edge_clusters)} edges")
        if existing_node_ckpts + existing_edge_ckpts > 0:
            print(f"     → ces clusters seront chargés depuis le disque (skip LLM)")
        print(f"  🧵 Pool de 5 threads en parallèle\n")

        t_start = time.time()
        pool = ThreadPoolExecutor(max_workers=5)

        # 🆕 On passe l'index du cluster
        node_futures = [
            pool.submit(self.deduplicate_cluster_with_cache, i, cluster, "node")
            for i, cluster in enumerate(self.node_clusters)
        ]
        edge_futures = [
            pool.submit(self.deduplicate_cluster_with_cache, i, cluster, "edge")
            for i, cluster in enumerate(self.edge_clusters)
        ]

        print(f"  ✅ {len(node_futures)} jobs nodes + {len(edge_futures)} jobs edges soumis au pool\n")

        print("  ⏳ En attente des résultats des clusters de NODES...")
        for i, future in enumerate(node_futures):
            try:
                cluster_entities, cluster_entity_map = future.result()
                entities.update(cluster_entities)
                entity_clusters.update(cluster_entity_map)
                print(f"     ✓ Cluster node #{i+1}/{len(node_futures)} terminé "
                      f"({len(cluster_entities)} entités uniques)")
            except Exception as e:
                print(f"     ✗ ERREUR cluster node #{i} : {e}")
                self.logger.error("Error processing node cluster %s: %s", i, e)

        print("\n  ⏳ En attente des résultats des clusters d'EDGES...")
        for i, future in enumerate(edge_futures):
            try:
                cluster_edges, cluster_edge_map = future.result()
                edges.update(cluster_edges)
                edge_clusters.update(cluster_edge_map)
                print(f"     ✓ Cluster edge #{i+1}/{len(edge_futures)} terminé "
                      f"({len(cluster_edges)} edges uniques)")
            except Exception as e:
                print(f"     ✗ ERREUR cluster edge #{i} : {e}")
                self.logger.error("Error processing edge cluster %s: %s", i, e)

        elapsed = time.time() - t_start
        print(f"\n  🎉 Tous les clusters traités en {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"     → {len(entities)} entités uniques (vs {len(self.nodes)} initiales, "
              f"réduction de {(1-len(entities)/max(1,len(self.nodes)))*100:.1f}%)")
        print(f"     → {len(edges)} edges uniques (vs {len(self.edges)} initiaux, "
              f"réduction de {(1-len(edges)/max(1,len(self.edges)))*100:.1f}%)")

        # ── Mise à jour des relations ────────────────────────────────────
        print(f"\n  🔄 Mise à jour des {len(self.graph.relations)} relations avec les alias...")
        relations: set[tuple[str, str, str]] = set()
        relation_clusters: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
        remapped = 0
        for s, p, o in self.graph.relations:
            orig = (s, p, o)
            if s not in entities:
                for rep, cluster in entity_clusters.items():
                    if s in cluster:
                        s = rep
                        break
            if p not in edges:
                for rep, cluster in edge_clusters.items():
                    if p in cluster:
                        p = rep
                        break
            if o not in entities:
                for rep, cluster in entity_clusters.items():
                    if o in cluster:
                        o = rep
                        break
            dedup_triplet = (s, p, o)
            if dedup_triplet != orig:
                remapped += 1
            relations.add(dedup_triplet)
            relation_clusters.setdefault(dedup_triplet, set()).add(orig)

        serialized_relation_clusters = {
            "\t".join(k): [list(t) for t in v]
            for k, v in relation_clusters.items()
        }

        print(f"     ✅ {len(relations)} relations finales (après déduplication des triplets)")
        print(f"     🔁 {remapped} relations re-mappées vers un alias")
        print(f"     🔗 {len(serialized_relation_clusters)} clusters de triplets traçés")
        new_entity_metadata: dict[str, set[str]] | None = None
        if self.graph.entity_metadata:
            print(f"\n  📝 Mise à jour des entity_metadata ({len(self.graph.entity_metadata)} entrées)...")
            new_entity_metadata = {}
            for original_entity, metadata_set in self.graph.entity_metadata.items():
                deduped_entity = original_entity
                for rep, cluster in entity_clusters.items():
                    if original_entity in cluster:
                        deduped_entity = rep
                        break
                if deduped_entity in new_entity_metadata:
                    new_entity_metadata[deduped_entity].update(metadata_set)
                else:
                    new_entity_metadata[deduped_entity] = metadata_set.copy()
            print(f"     ✅ {len(new_entity_metadata)} entrées de metadata après fusion")

        deduped_graph = Graph(
            entities=entities,
            edges=edges,
            relations=relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
            entity_metadata=new_entity_metadata,
            relation_clusters=serialized_relation_clusters,
        )

        print("\n" + "=" * 70)
        print("✅ FIN ÉTAPE 2 : DÉDUPLICATION TERMINÉE")
        print("=" * 70)
        print(f"  📊 Graph final :")
        print(f"     • entités  : {len(deduped_graph.entities)}")
        print(f"     • edges    : {len(deduped_graph.edges)}")
        print(f"     • relations: {len(deduped_graph.relations)}")
        print("=" * 70 + "\n")

        return deduped_graph
