import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def save_results(results: dict, cfg, df_eval: pd.DataFrame):
    """Sauve un CSV détaillé (1 ligne / question, colonnes par mode) + un CSV résumé."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_slug = re.sub(r"[^\w\-]", "_", cfg.llm_id.split("/")[-1])
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Détaillé ───────────────────────────────────────────────────────
    detailed = df_eval[["question", "correct_letter", "article_id"]].copy().reset_index(drop=True)
    for mode, df in results.items():
        detailed[f"pred_{mode}"] = df["predicted_letter"].values
        detailed[f"correct_{mode}"] = df["correct"].values
        detailed[f"context_{mode}"] = df["context"].values

    detailed_path = out_dir / f"eval_detailed_{model_slug}_{ts}.csv"
    detailed.to_csv(detailed_path, index=False)
    print(f"💾 Détaillé : {detailed_path}")

    # ── Résumé ─────────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {
            "model": cfg.llm_id,
            "mode": mode,
            "accuracy": df["correct"].mean(),
            "n_questions": len(df),
            "n_correct": int(df["correct"].sum()),
            "timestamp": ts,
        }
        for mode, df in results.items()
    ])
    summary_path = out_dir / f"eval_summary_{model_slug}_{ts}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"📊 Résumé   : {summary_path}\n")
    print(summary.to_string(index=False))
    return detailed_path, summary_path


def save_all_results(results: dict, cfg):
    """Enregistre TOUT pour chaque question et chaque mode.

    Pour chaque mode, dump le DataFrame complet renvoyé par run_eval_for_mode
    (question, prédiction, correct, sortie brute, contexte = liste de tous les
    triplets donnés au LLM, prompt complet, ...). On écrit :
      - un CSV par mode    : eval_full_<mode>_<model>_<ts>.csv
      - un CSV concaténé   : eval_full_ALL_<model>_<ts>.csv

    Renvoie le dossier de sortie.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_slug = re.sub(r"[^\w\-]", "_", cfg.llm_id.split("/")[-1])
    out_dir = Path(cfg.output_dir) / f"eval_full_{model_slug}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for mode, df in results.items():
        mode_slug = re.sub(r"[^\w\-]", "_", mode)
        path = out_dir / f"eval_full_{mode_slug}.csv"
        df.to_csv(path, index=False)
        print(f"💾 [save_all] {mode} : {path}")
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = out_dir / "eval_full_ALL.csv"
        combined.to_csv(combined_path, index=False)
        print(f"💾 [save_all] tous modes : {combined_path}")

    print(f"📂 [save_all] Dossier : {out_dir.resolve()}\n")
    return out_dir
