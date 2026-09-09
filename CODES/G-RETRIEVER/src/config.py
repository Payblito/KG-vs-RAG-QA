"""
Central configuration for the QA evaluation pipeline.

- Resolves the project root (CLEAN/) from this file's location and derives the DATA folder.
- Exposes a single `Config` dataclass holding every path, model id and hyper-parameter of the pipeline.
- Paths left as `None` are auto-filled in `__post_init__` from the `CATEGORY` field (articles CSV, graph dir, provenance, LN triplets, RAG index, output dir).
- Selects which retrieval strategies are evaluated through `modes_to_run` and how the query is built through `retrieval_query_mode`.
- Read-only `@property` helpers rebuild the full file paths (graph JSON, artifacts, provenance, LN, PCST cache).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Project root = 3 levels above this file (CLEAN/), then the shared DATA folder
ROOT = Path(__file__).resolve().parents[3]  # CLEAN/
DATA = ROOT / "DATA"


@dataclass
class Config:
    CATEGORY: str = "global"  # tested category (e.g. global, gastronomia...) ; drives all default paths

    # ── Paths: data ────────────────────────────────────────────────────
    questions_csv: str = str(DATA / "QUESTIONS" / "mcq_es.csv")  # MCQ evaluation file (questions + answer + distractors)
    articles_csv: Optional[str] = None  # articles subset of the category (None -> auto-filled from CATEGORY)
    filter_by_articles: bool = True   # restricts the evaluation to the questions linked to the articles above

    # ── Paths: graph ───────────────────────────────────────────────────
    graph_dir: Optional[str] = None  # root folder of the category graph (None -> auto)
    graph_json_name: Optional[str] = None  # file name of the clustered graph JSON (None -> auto)
    artifacts_subdir: str = "clustered_artifacts"  # subfolder for the precomputed graph artifacts (embeddings, indexes...)
    # ── Paths: provenance ──────────────────────────────────────────────
    provenance_subdir: str = "provenance"  # subfolder holding the provenance pickle
    provenance_name: Optional[str] = None  # provenance file name = triplet -> source article mapping (None -> auto)

    force_rebuild_graph_artifacts: bool = False  # True = recompute the graph artifacts even if they already exist

    # Paths: LN
    ln_subdir: str = "LN_triplets"  # subfolder of the LN (natural language) triplets
    ln_name: Optional[str] = None  # CSV file name of the LN triplets (None -> auto)
    transfo_ln: bool = False  # True = feed the LLM with the LN-verbalized triplets instead of the raw ones
    # ── Paths: RAG index ───────────────────────────────────────────────
    rag_index_dir: Optional[str] = None  # folder of the vector index used by the classic RAG (None -> auto)
    force_rebuild_rag_index: bool = False  # True = rebuild the RAG index from scratch

    # ── Paths: output ──────────────────────────────────────────────────
    output_dir: Optional[str] = None  # folder where the result CSVs are written (None -> DATA/RESULTS)

    def __post_init__(self):
        # Fill in every path left to None using the current CATEGORY
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

    # ── Models ─────────────────────────────────────────────────────────
    llm_id: str = "Qwen/Qwen2.5-3B-Instruct"  # HF id of the LLM answering the MCQs
    # A single embedder shared by RAG and Graph
    embedder_id: str = "jinaai/jina-embeddings-v3"  # HF id of the embedding model (queries, chunks, nodes, edges)

    # Kept for backward compatibility if needed
    @property
    def graph_embedder_id(self): return self.embedder_id  # alias: embedder used on the graph side

    @property
    def rag_embedder_id(self): return self.embedder_id  # alias: embedder used on the RAG side

    # ── Sampling ───────────────────────────────────────────────────────
    n_samples: Optional[int] = None # None = all questions ; otherwise number of questions sampled
    random_seed: int = 1  # seed making the sampling reproducible

    # ── LLM generation ─────────────────────────────────────────────────
    batch_size: int = 15  # number of prompts sent per generation batch
    max_new_tokens: int = 5  # very short: only the answer letter (A/B/C/D) is expected

    # ── RAG ────────────────────────────────────────────────────────────
    k_rag: int = 5  # number of chunks retrieved per question
    chunk_size_tokens: int = 512  # chunk size (in tokens) when indexing the articles
    chunk_overlap_tokens: int = 64  # token overlap between two consecutive chunks

    # ── Graph-RAG ──────────────────────────────────────────────────────
    topk_nodes: int = 15  # number of nodes kept by similarity with the query before the PCST is computed (if applicable)
    topk_edges: int = 20  # number of edges kept by similarity with the query before the PCST is computed (if applicable)
    pcst_cost_e: float = 0.5  # edge cost in the PCST (higher = smaller/sparser subgraph)
    pcst_workers: int = 4  # number of parallel processes for the PCST computation

    # ── PCST cache (graph_rag_pcst) ────────────────────────────────────
    # If True, graph_rag_pcst uses a precomputed PCST cache (pkl) instead of recomputing the PCST for every question.

    use_pcst_cache: bool = False  # True = read the PCST subgraphs from the cache
    save_pcst_cache_on_recompute: bool = True  # True = persist the newly computed subgraphs to the cache

    # ── KAPING ─────────────────────────────────────────────────────────
    # Number of triplets (src relation dst) kept by direct query/triplet
    # similarity, given as-is in the context.
    topk_triplets: int = 15  # size of the triplet context in KAPING mode

    # ── Oracle triplets (provenance) ───────────────────────────────────
    # Number of relevant triplets kept by query/triplet similarity among the
    # triplets of the article (filtered through provenance).
    top_k_provenance: int = 10  # size of the oracle context (gold article only)

    # ── Modes to evaluate ──────────────────────────────────────────────
    # Possible values:
    #   "zero_shot"            (no context given to the LLM)
    #   "graph_rag_pcst"      (connected PCST)
    #   "kaping"              (top-k triplets by direct query/triplet similarity)
    #   "rag"                 (retrieval augmented generation)

    modes_to_run: List[str] = field(default_factory=lambda: [
        "zero_shot", "rag", "graph_rag_pcst", "kaping"
    ])  # list of retrieval strategies actually run

    retrieval_query_mode: str = "question_options"  # how the retrieval query is built
    # possible values: "question", "question_options", "oracle", "question+1option"
    #   "question+1option": for the graph modes only. 
    #   This mode performs a retrieval for each question and one of its options (A, B, C, D), then merges the resulting subgraphs.

    # ── Misc ───────────────────────────────────────────────────────────
    device: str = "cuda:0"  # device used for the LLM and the embedder

    # ── Helpers ────────────────────────────────────────────────────────
    @property
    def graph_json_path(self) -> str:
        # Full path of the clustered graph JSON
        return str(Path(self.graph_dir) / self.graph_json_name)

    @property
    def artifacts_dir(self) -> str:
        # Full path of the precomputed graph artifacts folder
        return str(Path(self.graph_dir) / self.artifacts_subdir)

    @property
    def provenance_path(self) -> str:
        # Full path of the provenance pickle
        return str(Path(self.graph_dir) / self.provenance_subdir / self.provenance_name)

    @property
    def ln_path(self) -> str:
        # Full path of the LN triplets CSV
        return str(Path(self.graph_dir) / self.ln_subdir / self.ln_name)

    @property
    def pcst_cache_path(self) -> str:
        # Full path of the PCST cache used by the GNN
        return str(Path(self.graph_dir) / "pcst_cache_for_gnn" / "pcst_cache.pkl")