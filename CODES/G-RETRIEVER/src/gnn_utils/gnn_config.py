"""Configuration of the GNN / G-Retriever module.

- Resolves the project root (CLEAN/) and the DATA folder from this file's location, plus the local `artifacts/` folder of G-RETRIEVER.
- Exposes a single `GNNConfig` dataclass gathering every hyper-parameter of the G-Retriever pipeline.
- Groups the settings by concern: GraphTransformer architecture, projector (graph token -> LLM space), soft prompt, training, text lengths, cache/checkpoint paths.
- Handles result saving (root folder, category, ablation name) and the train/val/test split (ratios + seed).
- Enables the ablation studies through `use_graph_token` / `use_text_graph`, and selects the target format with `train_gen_mode` (letter only vs. letter + option text).
"""
from dataclasses import dataclass
from pathlib import Path

# Project root = 4 levels above this file (CLEAN/), then the shared DATA folder
ROOT = Path(__file__).resolve().parents[4]  # CLEAN/
DATA = ROOT / "DATA"
_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"  # G-RETRIEVER/artifacts/

@dataclass
class GNNConfig:
    # --- GraphTransformer ---
    in_dim: int = 1024          # node embedding dim (Jina v3)
    hidden_dim: int = 1024      # hidden dim of the GNN layers
    gnn_out_dim: int = 1024     # output dim of the graph readout
    edge_dim: int = 1024        # edge embedding dim (same encoder as the nodes)
    num_layers: int = 4         # number of GraphTransformer layers
    num_heads: int = 8          # number of attention heads per layer
    dropout: float = 0.1        # dropout applied inside the GNN

    # --- Projector (graph token -> LLM space) ---
    projector_hidden_dim: int = 2048  # hidden dim of the MLP mapping the graph readout to the LLM space
    # llm_hidden_dim is read dynamically from the LLM (Qwen2.5-3B = 2048)

    # --- Soft prompt ---
    num_graph_tokens: int = 1   # number of virtual tokens produced by the projector

    # --- Training ---
    lr: float = 5e-5              # learning rate (GNN + projector)
    weight_decay: float = 1e-5    # L2 regularization of the optimizer
    batch_size: int = 4           # number of graphs/questions per batch
    grad_accum_steps: int = 2     # gradient accumulation steps (effective batch = batch_size * this)
    num_epochs: int = 10          # maximum number of training epochs
    warmup_ratio: float = 0.05    # fraction of steps used for the LR warmup
    max_grad_norm: float = 1.0    # gradient clipping threshold
    patience: int = 3       # early stopping (on val accuracy)

    # --- Text ---
    max_txt_len: int = 8192       # aligned with G-Retriever (llm.py max_length=8192)
    max_target_len: int = 256   # max tokens for the "X) option" target

    # --- Cache / paths ---
    pcst_cache_path: str = str(_ARTIFACTS / "pcst_cache.pkl")   # precomputed PCST subgraphs, keyed by question
    ckpt_path: str = str(_ARTIFACTS / "g_retriever_best.pt")    # checkpoint of the best model (best val accuracy)

    # --- Saving of the training results ---
    # Output root (distinct from RESULTS_HYPER_OPTI / RESULTS_TEST_AUGMENT_TOKEN).
    results_root: str = str(DATA / "RESULTS_GNN")
    category: str = "global"        # tested category (e.g. global, gastronomia...)
    ablation_name: str = "both"     # ablation name (text_only, graph_only, both)
    # llm_slang is derived dynamically from the LLM (e.g. Qwen2.5_3B_Instruct).

    # --- Split ---
    val_ratio: float = 0.15   # share of the data used for validation
    test_ratio: float = 0.15  # share of the data used for the final test
    seed: int = 101           # seed making the split reproducible

    # --- Ablations ---
    use_graph_token: bool = True   # inject the GNN soft-token(s)
    use_text_graph: bool = True    # inject the textualized graph {desc} into the prompt

    # --- Generation mode (consistent training + eval) ---
    # "gen_letter" : target "A)"           (direct MCQ)
    # "gen_full"   : target "A) option..." (letter + text)
    train_gen_mode: str = "gen_letter"

    device: str = "cuda:0"  # device used for the GNN, the projector and the LLM