from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[3]  # CLEAN/
DATA = ROOT / "DATA"


@dataclass
class Config:
    CATEGORY: str = "global"  # catégorie testée (ex: global, gastronomia...)

    # ── Paths : données ────────────────────────────────────────────────
    questions_csv: str = str(DATA / "QUESTIONS" / "mcq_eval_results_es__ministral-small.csv")
    articles_csv: Optional[str] = None
    filter_by_articles: bool = True   # restreint aux questions liées aux articles ci-dessus

    # ── Paths : graphe ─────────────────────────────────────────────────
    graph_dir: Optional[str] = None
    graph_json_name: Optional[str] = None
    artifacts_subdir: str = "clustered_artifacts"
    # ── Paths : provenance ──────────────────────────────────────────────
    provenance_subdir: str = "provenance"
    provenance_name: Optional[str] = None

    force_rebuild_graph_artifacts: bool = False

    # Paths : LN
    ln_subdir: str = "LN_triplets"
    ln_name: Optional[str] = None
    transfo_ln: bool = False
    # ── Paths : index RAG ──────────────────────────────────────────────
    rag_index_dir: Optional[str] = None
    force_rebuild_rag_index: bool = False

    # ── Paths : sortie ─────────────────────────────────────────────────
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.articles_csv is None:
            self.articles_csv = str(
                DATA / "SUBSETS_ES" / "ARTICLES_SUBSETS_ES"
                / f"{self.CATEGORY}_articles_es_disjoint.csv"
            )
        if self.graph_dir is None:
            self.graph_dir = str(DATA / "SUBSETS_ES" / "GRAPHS" / self.CATEGORY)
        if self.graph_json_name is None:
            self.graph_json_name = f"{self.CATEGORY}_clustered_graph.json"
        if self.provenance_name is None:
            self.provenance_name = f"{self.CATEGORY}_clustered_provenance.pkl"
        if self.ln_name is None:
            self.ln_name = f"{self.CATEGORY}_ln_triplets.csv"
        if self.rag_index_dir is None:
            self.rag_index_dir = str(DATA / "RAG_INDEX" / f"rag_index_{self.CATEGORY}")
        if self.output_dir is None:
            self.output_dir = str(DATA / "RESULTS")
    # ── Modèles ────────────────────────────────────────────────────────
    llm_id: str = "Qwen/Qwen2.5-3B-Instruct"
    # Un seul embedder pour RAG et Graph
    embedder_id: str = "jinaai/jina-embeddings-v3"

    # Garde les anciens pour compatibilité si besoin
    @property
    def graph_embedder_id(self): return self.embedder_id
    @property
    def rag_embedder_id(self): return self.embedder_id


    # ── Échantillonnage ────────────────────────────────────────────────
    n_samples: Optional[int] = None            # None = tout
    random_seed: int = 1

    # ── Génération LLM ─────────────────────────────────────────────────
    batch_size: int = 15
    max_new_tokens: int = 5

    # ── RAG ────────────────────────────────────────────────────────────
    k_rag: int = 5
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ── Graph-RAG ──────────────────────────────────────────────────────
    topk_nodes: int = 15
    topk_edges: int = 20
    pcst_cost_e: float = 0.5
    pcst_workers: int = 4

    # ── Cache PCST (graph_rag_pcst) ────────────────────────────────────
    # Si True, graph_rag_pcst utilise un cache PCST précalculé (pkl) au lieu
    # de recalculer le PCST pour chaque question. Le cache est aligné par clé
    # (article_id, question). En cas de cache partiel/absent et
    # save_pcst_cache_on_recompute=True, on recalcule les manquants (parallèle)
    # et on sauvegarde le cache mis à jour.

    use_pcst_cache: bool = False
    save_pcst_cache_on_recompute: bool = True

    # ── KAPING ─────────────────────────────────────────────────────────
    # Nombre de triplets (src relation dst) retenus par similarité directe
    # query/triplet, donnés tels quels en contexte.
    topk_triplets: int = 15

    # ── Oracle triplets (provenance) ────────────────────────────────────
    # Nombre de triplets pertinents retenus par similarité query/triplet
    # parmi les triplets de l'article (filtrage via provenance).
    top_k_provenance: int = 10

    # ── Modes à évaluer ────────────────────────────────────────────────
    # Valeurs possibles :
    #   "zero_shot"
    #   "graph_rag_pcst"      (PCST connecté)
    #   "graph_rag_topk"      (top-k direct)
    #   "kaping"              (top-k triplets par similarité directe query/triplet)
    modes_to_run: List[str] = field(default_factory=lambda: [
        "zero_shot", "rag", "graph_rag_pcst", "graph_rag_topk"
    ]) 

    retrieval_query_mode: str = "question_options"  # valeurs possibles : "question", "question_options", "oracle", "question+1option"
    # valeurs possibles : "question", "question_options", "oracle", "question+1option"
    #   "question+1option" : pour les modes graphe uniquement. On fait un retrieval
    #   PCST (ou top-k) pour chaque combinaison question+1option (A, B, C, D) puis on
    #   fusionne (union) les sous-graphes obtenus pour construire le contexte.


    # ── Sauvegarde ─────────────────────────────────────────────────────
    # Si True, on enregistre TOUT pour chaque question et chaque mode :
    # prompt complet, contexte (liste de tous les triplets donnés au LLM),
    # sortie brute, prédiction, etc. (un CSV par mode).
    save_all: bool = False

    # ── Divers ─────────────────────────────────────────────────────────
    device: str = "cuda:0"
    # ── Helpers ────────────────────────────────────────────────────────
    @property
    def graph_json_path(self) -> str:
        return str(Path(self.graph_dir) / self.graph_json_name)

    @property
    def artifacts_dir(self) -> str:
        return str(Path(self.graph_dir) / self.artifacts_subdir)

    @property
    def provenance_path(self) -> str:
        return str(Path(self.graph_dir) / self.provenance_subdir / self.provenance_name)

    @property
    def ln_path(self) -> str:
        return str(Path(self.graph_dir) / self.ln_subdir / self.ln_name)

    @property
    def pcst_cache_path(self) -> str:
        return str(Path(self.graph_dir) / "pcst_cache_for_gnn" / "pcst_cache.pkl")
