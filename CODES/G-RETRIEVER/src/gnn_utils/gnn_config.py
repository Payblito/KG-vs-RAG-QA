"""Configuration du module GNN / G-Retriever."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # CLEAN/
DATA = ROOT / "DATA"
_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"  # G-RETRIEVER/artifacts/


@dataclass
class GNNConfig:
    # --- GraphTransformer ---
    in_dim: int = 1024          # dim embeddings de noeuds (Jina v3)
    hidden_dim: int = 1024
    gnn_out_dim: int = 1024
    edge_dim: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # --- Projector (graph token -> espace LLM) ---
    projector_hidden_dim: int = 2048
    # llm_hidden_dim est lu dynamiquement depuis le LLM (Qwen2.5-3B = 2048)

    # --- Soft prompt ---
    num_graph_tokens: int = 1   # nb de tokens virtuels produits par le projector

    # --- Entrainement ---
    lr: float = 5e-5
    weight_decay: float = 1e-5
    batch_size: int = 4
    grad_accum_steps: int = 2
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 3       # early stopping (sur val accuracy)

    # --- Texte ---
    max_txt_len: int = 8192       # aligné sur G-Retriever (llm.py max_length=8192)
    max_target_len: int = 256   # tokens max pour la cible "X) option"

    # --- Cache / chemins ---
    pcst_cache_path: str = str(_ARTIFACTS / "pcst_cache.pkl")
    ckpt_path: str = str(_ARTIFACTS / "g_retriever_best.pt")

    # --- Sauvegarde des resultats d'entrainement ---
    # Racine des sorties (distincte de RESULTS_HYPER_OPTI / RESULTS_TEST_AUGMENT_TOKEN).
    results_root: str = str(DATA / "RESULTS_GNN")
    category: str = "global"        # categorie testee (ex: global, gastronomia...)
    ablation_name: str = "both"     # nom de l'ablation (text_only, graph_only, both)
    # llm_slang est derive dynamiquement du LLM (ex: Qwen2.5_3B_Instruct).

    # --- Split ---
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 101

    # --- Ablations ---
    use_graph_token: bool = True   # injecter le(s) soft-token(s) du GNN
    use_text_graph: bool = True    # injecter le graphe textualisé {desc} dans le prompt

    # --- Mode de génération (entraînement + éval cohérents) ---
    # "gen_letter" : cible "A)"          (MCQ direct)
    # "gen_full"   : cible "A) option..." (lettre + texte)
    train_gen_mode: str = "gen_letter"


    device: str = "cuda:0"
