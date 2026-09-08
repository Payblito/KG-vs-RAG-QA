"""Boucle d'entrainement + evaluation de G-Retriever (GNN + projector).

Sauvegarde automatique des resultats dans :
    {cfg.results_root}/{llm_slang}/{cfg.category}/{cfg.ablation_name}_{gen_mode}/
        summary.csv         - 1 ligne (metrics globales)
        epoch_history.csv   - 1 ligne/epoch (train_loss, val_acc, lr, temps)
        train_loss.csv      - 1 ligne/step (loss, lr)
        details.csv         - 1 ligne/question x {val,test} sur le best model
        best.pt             - checkpoint GNN + projector
        run_meta.json       - config + timestamps
"""
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import GRetrieverMCQDataset, mcq_collate
from .gnn_config import GNNConfig


# =====================================================================
# Splits
# =====================================================================
def make_splits(n, cfg: GNNConfig):
    """Split par question (train/val/test). Retourne 3 listes d'indices."""
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)
    n_test = int(n * cfg.test_ratio)
    n_val = int(n * cfg.val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


# =====================================================================
# Utilitaires chemins / sauvegarde
# =====================================================================
def _llm_slang(model) -> str:
    """Nom court du LLM (ex: Qwen2.5-3B-Instruct -> Qwen2.5_3B_Instruct)."""
    name = getattr(model.llm, "name_or_path", "") or getattr(model.llm, "name", "")
    if not name:
        name = "unknown_llm"
    return name.split("/")[-1].replace("-", "_")


def _base_run_dir(cfg: GNNConfig, model) -> Path:
    """Chemin de base (sans timestamp) du dossier de sortie."""
    slang = _llm_slang(model)
    gen_tag = cfg.train_gen_mode  # gen_letter ou gen_full
    return (Path(cfg.results_root) / slang / gen_tag / cfg.category
            / f"{cfg.ablation_name}_{gen_tag}")


def _make_run_dir(cfg: GNNConfig, model, resume=False) -> Path:
    """Construit le dossier de sortie, ajoute un timestamp si existant.

    Si resume=True et que le dossier de base existe, le réutilise sans timestamp
    (pour reprendre un entraînement interrompu).
    """
    base = _base_run_dir(cfg, model)
    if resume and base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    if base.exists() and any(base.iterdir()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = base.with_name(f"{base.name}_{ts}")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _to_device(batch, device):
    for k in ("x", "edge_index", "edge_attr", "batch"):
        batch[k] = batch[k].to(device)


def _cfg_to_dict(cfg: GNNConfig) -> dict:
    """Serialise la config (dataclass) en dict JSON-serializable."""
    out = {}
    for k in cfg.__dataclass_fields__:
        v = getattr(cfg, k)
        if isinstance(v, Path):
            v = str(v)
        out[k] = v
    return out


# =====================================================================
# Evaluation
# =====================================================================
@torch.no_grad()
def evaluate_g_retriever(model, loader, device, eval_method="gen_letter",
                         split_name="val", collect=False, debug=False):
    """Evaluation generative (gen_letter / gen_full) ou par scoring.

    Retourne un dict :
        {"accuracy": float, "records": [...]}   si collect=True
        {"accuracy": float, "records": []}      sinon

    records (collect=True) : un dict par question contenant
        split, idx, article_id, question, option_A..D, desc,
        correct_letter, pred_raw, pred_parsed, correct, n_nodes, n_edges
    """
    model.eval()
    correct, total, n_unparsed = 0, 0, 0
    records = []
    idx_counter = 0

    for batch in tqdm(loader, desc=f"eval[{eval_method}/{split_name}]",
                      leave=False):
        _to_device(batch, device)

        if eval_method == "score":
            preds = model.score_options(batch, debug=debug)
            raws = [""] * len(preds)  # pas de raw en mode score
        elif eval_method == "gen_letter":
            preds, raws = model.generate_answers(
                batch, gen_mode="letter", debug=debug)
        elif eval_method == "gen_full":
            preds, raws = model.generate_answers(
                batch, gen_mode="full", debug=debug)
        else:
            raise ValueError(f"eval_method inconnu : {eval_method}")

        for i, (p, gt) in enumerate(zip(preds, batch["correct_letter"])):
            if p is None:
                n_unparsed += 1
            is_correct = (p == gt)
            correct += int(is_correct)
            total += 1

            if collect:
                records.append({
                    "split": split_name,
                    "idx": idx_counter,
                    "article_id": batch["article_id"][i],
                    "question": batch["question"][i],
                    "option_A": batch["options"][i]["A"],
                    "option_B": batch["options"][i]["B"],
                    "option_C": batch["options"][i]["C"],
                    "option_D": batch["options"][i]["D"],
                    "desc": batch["desc"][i],
                    "correct_letter": gt,
                    "pred_raw": raws[i],
                    "pred_parsed": p if p is not None else "",
                    "correct": bool(is_correct),
                    "n_nodes": int(batch["num_nodes"][i]),
                    "n_edges": int(batch["num_edges"][i]),
                })
            idx_counter += 1

    if n_unparsed:
        print(f"   ⚠️ {n_unparsed}/{total} generations non parsees "
              f"(comptees comme fausses) [{eval_method}/{split_name}]")
    return {"accuracy": correct / max(total, 1), "records": records}


# =====================================================================
# Sauvegarde fichiers
# =====================================================================
def _save_csv(rows, path: Path, fieldnames=None):
    """Ecrit une liste de dicts en CSV (pandas pour simplicite)."""
    if not rows:
        print(f"  (vide) {path.name} non ecrit")
        return
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path.name} ({len(rows)} lignes)")


def _save_summary(path: Path, cfg: GNNConfig, model, slang: str,
                  best_val, test_acc, best_epoch, n_epochs_run,
                  n_train, n_val, n_test, total_sec):
    import pandas as pd
    n_trainable = sum(p.numel() for p in model.trainable_parameters()
                      if p.requires_grad)
    row = {
        "ablation_name": cfg.ablation_name,
        "category": cfg.category,
        "llm_slang": slang,
        "gen_mode": cfg.train_gen_mode,
        "use_graph_token": cfg.use_graph_token,
        "use_text_graph": cfg.use_text_graph,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "n_epochs_run": n_epochs_run,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "num_graph_tokens": cfg.num_graph_tokens,
        "n_trainable_params": n_trainable,
        "total_time_sec": round(total_sec, 1),
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"  saved {path.name}")


def _save_run_meta(path: Path, cfg: GNNConfig, t_start, t_end):
    meta = {
        "config": _cfg_to_dict(cfg),
        "timestamp_start": t_start.isoformat(),
        "timestamp_end": t_end.isoformat(),
        "duration_sec": (t_end - t_start).total_seconds(),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  saved {path.name}")


# =====================================================================
# Boucle principale
# =====================================================================
def train_g_retriever(model, df, pcst_cache, graph, cfg: GNNConfig,
                      splits=None):
    """Entrainement + evaluation + sauvegarde complete.

    Retourne un dict avec les metriques cles et le chemin de sortie.
    """
    import pandas as pd

    t_start = datetime.now()
    device = cfg.device
    model.gnn.to(device)
    model.projector.to(device)

    # --- Détection checkpoint pour reprise ---
    base_dir = _base_run_dir(cfg, model)
    ckpt_dir_candidate = base_dir / "epoch"
    resume_ckpt = None
    start_epoch = 0
    if ckpt_dir_candidate.exists():
        epoch_ckpts = sorted(
            ckpt_dir_candidate.glob("epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if epoch_ckpts:
            resume_ckpt = epoch_ckpts[-1]
            start_epoch = int(resume_ckpt.stem.split("_")[1])
            print(f"♻️  Checkpoint trouvé : {resume_ckpt.name} (epoch {start_epoch})")

    # --- Dossier de sortie ---
    run_dir = _make_run_dir(cfg, model, resume=(resume_ckpt is not None))
    slang = _llm_slang(model)
    print(f"\n📁 Dossier de sortie : {run_dir}")

    # --- Dataset / loaders ---
    dataset = GRetrieverMCQDataset(df, pcst_cache, graph)
    if splits is None:
        tr_idx, va_idx, te_idx = make_splits(len(dataset), cfg)
    else:
        tr_idx, va_idx, te_idx = splits

    train_loader = DataLoader(
        Subset(dataset, tr_idx), batch_size=cfg.batch_size,
        shuffle=True, collate_fn=mcq_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, va_idx), batch_size=cfg.batch_size,
        shuffle=False, collate_fn=mcq_collate,
    )
    test_loader = DataLoader(
        Subset(dataset, te_idx), batch_size=cfg.batch_size,
        shuffle=False, collate_fn=mcq_collate,
    )

    # --- Garde-fou : aucun parametre trainable (mode baseline text_only) ---
    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable if p.requires_grad)

    gm = cfg.train_gen_mode

    if not cfg.use_graph_token or n_trainable == 0:
        print("⚠️  Aucun paramètre entraînable (mode baseline) -> "
              "évaluation directe sans entraînement.")
        val_res = evaluate_g_retriever(model, val_loader, device, gm,
                                       split_name="val", collect=True)
        test_res = evaluate_g_retriever(model, test_loader, device, gm,
                                        split_name="test", collect=True)
        val_acc, test_acc = val_res["accuracy"], test_res["accuracy"]
        print(f"🎯 [baseline] val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  [{gm}]")

        # --- Sauvegardes ---
        _save_csv(val_res["records"] + test_res["records"],
                  run_dir / "details.csv")
        _save_summary(run_dir / "summary.csv", cfg, model, slang,
                      val_acc, test_acc, best_epoch=0, n_epochs_run=0,
                      n_train=len(tr_idx), n_val=len(va_idx),
                      n_test=len(te_idx),
                      total_sec=(datetime.now() - t_start).total_seconds())
        _save_run_meta(run_dir / "run_meta.json", cfg, t_start, datetime.now())
        return {
            "best_val_acc": val_acc,
            "test_acc": test_acc,
            "run_dir": str(run_dir),
            "splits": {"train": tr_idx, "val": va_idx, "test": te_idx},
        }

    # --- Optimiseur + scheduler ---
    optim = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_val, best_state, no_improve = -1.0, None, 0
    best_epoch = 0
    epoch_history = []
    train_loss_rows = []
    global_step = 0
    epoch_start_time = time.time()

    # --- Restauration si reprise ---
    if resume_ckpt is not None:
        print(f"♻️  Reprise depuis epoch {start_epoch}...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.gnn.load_state_dict(ckpt["gnn"])
        model.projector.load_state_dict(ckpt["projector"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        best_val = ckpt["best_val"]
        best_state = ckpt["best_state"]
        best_epoch = ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        global_step = ckpt["global_step"]
        epoch_history = ckpt["epoch_history"]
        train_loss_rows = ckpt["train_loss_rows"]
        print(f"   best_val={best_val:.4f} best_epoch={best_epoch} "
              f"no_improve={no_improve} global_step={global_step}")

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        model.llm.eval()
        running = 0.0
        optim.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.num_epochs}")
        total_iters = len(train_loader)
        for it, batch in enumerate(pbar):
            _to_device(batch, device)
            torch.cuda.reset_peak_memory_stats(device)

            loss = model(batch) / cfg.grad_accum_steps
            loss.backward()
            

            alloc = torch.cuda.memory_allocated(device) / 2**20
            reserved = torch.cuda.memory_reserved(device) / 2**20
            peak = torch.cuda.max_memory_allocated(device) / 2**20

            if reserved > 16000:  # 16 GiB → fragmentation critique
                            gc.collect()
                            torch.cuda.empty_cache()

            running += loss.item() * cfg.grad_accum_steps

            if (it + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), cfg.max_grad_norm)
                optim.step(); sched.step(); optim.zero_grad()
                global_step += 1
                train_loss_rows.append({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": loss.item() * cfg.grad_accum_steps,
                    "lr": sched.get_last_lr()[0],
                })

            # --- Debug mémoire ---
            if (it + 1) % 50 == 0:
                seq_len = getattr(model, '_last_seq_len', 0)
                logit_mb = seq_len * 151936 * 2 / 2**20
                # print(f"[iter {it+1}/{total_iters}] "
                #       f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB "
                #       f"seq_len={seq_len} logits_est={logit_mb:.0f}MB")
                gc.collect()
                torch.cuda.empty_cache()

            pbar.set_postfix(loss=f"{running / (it + 1):.4f}",
                             alloc=f"{alloc:.0f}M", reserved=f"{reserved:.0f}M",
                             peak=f"{peak:.0f}M")

        epoch_loss = running / max(1, len(train_loader))
        val_acc = evaluate_g_retriever(model, val_loader, device, gm,
                                       split_name="val", collect=False)["accuracy"]
        epoch_time = time.time() - epoch_start_time
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_acc": val_acc,
            "lr": sched.get_last_lr()[0],
            "elapsed_sec": round(epoch_time, 1),
        })
        print(f"[epoch {epoch+1}] train_loss={epoch_loss:.4f} "
              f"val_acc={val_acc:.4f}  [{gm}]  ({epoch_time:.1f}s)")

        if val_acc > best_val:
            best_val, no_improve = val_acc, 0
            best_epoch = epoch + 1
            best_state = {
                "gnn": model.gnn.state_dict(),
                "projector": model.projector.state_dict(),
            }
            print(f"  ✅ nouveau meilleur (val_acc={val_acc:.4f})")
        else:
            no_improve += 1

        # --- Checkpoint toutes les 2 epochs ---
        if (epoch + 1) % 2 == 0:
            ckpt_dir = run_dir / "epoch"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for old in ckpt_dir.glob("epoch_*.pt"):
                old.unlink()
            torch.save({
                "gnn": model.gnn.state_dict(),
                "projector": model.projector.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch + 1,
                "best_val": best_val,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "global_step": global_step,
                "epoch_history": epoch_history,
                "train_loss_rows": train_loss_rows,
            }, ckpt_dir / f"epoch_{epoch+1}.pt")
            print(f"  💾 checkpoint epoch {epoch+1} -> epoch/")

        if no_improve >= cfg.patience:
            print(f"  ⏹ early stopping (patience={cfg.patience})")
            break
        epoch_start_time = time.time()

    # --- Restauration du best model ---
    if best_state is not None:
        model.gnn.load_state_dict(best_state["gnn"])
        model.projector.load_state_dict(best_state["projector"])

    # --- Evaluation finale (best model) avec collecte ---
    val_res = evaluate_g_retriever(model, val_loader, device, gm,
                                   split_name="val", collect=True)
    test_res = evaluate_g_retriever(model, test_loader, device, gm,
                                    split_name="test", collect=True)
    test_acc = test_res["accuracy"]
    print(f"\n🎯 Test (best model, epoch {best_epoch}) | {gm}={test_acc:.4f}")

    # --- Sauvegardes ---
    _save_csv(train_loss_rows, run_dir / "train_loss.csv")
    _save_csv(epoch_history, run_dir / "epoch_history.csv")
    _save_csv(val_res["records"] + test_res["records"],
              run_dir / "details.csv")

    # Checkpoint
    ckpt_path = run_dir / "best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  saved {ckpt_path.name}")

    total_sec = (datetime.now() - t_start).total_seconds()
    _save_summary(run_dir / "summary.csv", cfg, model, slang,
                  best_val, test_acc, best_epoch,
                  len(epoch_history),
                  len(tr_idx), len(va_idx), len(te_idx), total_sec)
    _save_run_meta(run_dir / "run_meta.json", cfg, t_start, datetime.now())

    print(f"\n⏱  Temps total : {total_sec:.1f}s")

    return {
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "run_dir": str(run_dir),
        "splits": {"train": tr_idx, "val": va_idx, "test": te_idx},
    }
