"""
Analyse qualitative automatique via Mistral Large comme juge.

Pour chaque question de catégorie B (RAG ✅ / Graph ❌), demande au LLM-juge
de classer le cas en :
  - ABSENT  : info nécessaire absente du sous-graphe (échec retrieval)
  - PARTIEL : info partielle / noyée dans le bruit
  - PRESENT : info présente, mais LLM générateur n'a pas su l'exploiter
"""

import os
import json
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import litellm


# ─────────────────────────────────────────────────────────────────────
# Prompt du juge
# ─────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """Tu es un évaluateur expert en RAG (Retrieval-Augmented Generation) sur graphe de connaissances.

Ton rôle : pour une question à choix multiples où le modèle "Graph-RAG" s'est trompé alors que le "RAG textuel" a réussi, tu dois diagnostiquer la cause de l'échec en analysant le sous-graphe (triplets src/relation/dst) qui a été fourni au LLM.

Tu dois classer le cas dans EXACTEMENT UNE de ces 3 catégories :

1. ABSENT : L'information factuelle nécessaire pour répondre correctement n'est PAS présente dans le sous-graphe (aucun triplet ne contient le fait clé). C'est un échec de RETRIEVAL.

2. PARTIEL : Un ou plusieurs triplets touchent au sujet mais l'information exacte est ambiguë, incomplète, ou noyée dans du bruit massif. Un humain pourrait éventuellement deviner, mais le signal est faible.

3. PRESENT : Le triplet exact (ou quasi-exact) répondant à la question est présent dans le sous-graphe. C'est un échec de GÉNÉRATION (le LLM n'a pas su exploiter l'info disponible).

Réponds STRICTEMENT en JSON valide avec ce format :
{
  "category": "ABSENT" | "PARTIEL" | "PRESENT",
  "key_fact_needed": "<le fait précis nécessaire pour répondre correctement, en 1 phrase>",
  "evidence_in_subgraph": "<triplet(s) pertinent(s) trouvé(s), ou 'aucun'>",
  "reasoning": "<justification en 1-2 phrases>"
}
"""

JUDGE_USER_TEMPLATE = """QUESTION : {question}

OPTIONS :
{options}

BONNE RÉPONSE : {correct_letter}) {correct_option}
RÉPONSE DU GRAPH-RAG : {graph_letter}) {graph_option}

SOUS-GRAPHE FOURNI AU LLM (format CSV src,relation,dst) :
{subgraph}

Analyse ce cas et réponds en JSON."""


# ─────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────
def build_options_str(row):
    opts = row["options"]
    # Au cas où c'est une string (lue depuis CSV) : parser
    if isinstance(opts, str):
        import ast
        opts = ast.literal_eval(opts)
    return "\n".join(
        f"  {letter}) {opts[letter]}"
        for letter in ["A", "B", "C", "D"]
        if letter in opts and pd.notna(opts[letter])
    )



def get_option_text(row, letter):
    col = f"option_{letter.lower()}"
    return row[col] if col in row and pd.notna(row[col]) else ""


def truncate_subgraph(context, max_chars=8000):
    """Tronque le sous-graphe si trop long pour le prompt."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n... [tronqué]"


def call_judge(model, question, options_str, correct_letter, correct_opt,
               graph_letter, graph_opt, subgraph, max_retries=3):
    """Appelle Mistral et parse le JSON. Retries en cas d'erreur."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        options=options_str,
        correct_letter=correct_letter,
        correct_option=correct_opt,
        graph_letter=graph_letter,
        graph_option=graph_opt,
        subgraph=truncate_subgraph(subgraph),
    )
    print(user_msg)  # Affiche la requête envoyée au juge (pour debug)

    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=f"mistral/{model}",   # ex: "mistral/mistral-large-latest"
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "category": "ERROR",
                    "key_fact_needed": "",
                    "evidence_in_subgraph": "",
                    "reasoning": f"Erreur API: {e}",
                }


# ─────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────
def run_judge_analysis(
    df_analysis: pd.DataFrame,
    category: str = "B",
    api_key: str = None,
    model: str = "mistral-large-latest",
    output_csv: str = None,
    sleep_between: float = 0.5,
):
    """
    Lance le LLM-juge sur toutes les questions d'une catégorie donnée.

    df_analysis : doit contenir les colonnes
        question, option_a/b/c/d, correct_letter,
        rag_pred, graph_pred, category,
        graph_context (le sous-graphe CSV)
    """
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Fournir api_key ou définir MISTRAL_API_KEY")

    df_cat = df_analysis[df_analysis["category"] == category].copy().reset_index(drop=True)
    print(f"▶ Analyse LLM-juge sur {len(df_cat)} questions (catégorie {category})")

    results = []
    for i, row in tqdm(df_cat.iterrows(), total=len(df_cat), desc="Judging"):
        options_str = build_options_str(row)
        correct_letter = row["correct_letter"]
        graph_letter = row["pred_graph"]

        verdict = call_judge( model,
            question=row["question"],
            options_str=options_str,
            correct_letter=correct_letter,
            correct_opt=get_option_text(row, correct_letter),
            graph_letter=graph_letter,
            graph_opt=get_option_text(row, graph_letter),
            subgraph=row["graph_context"],
        )

        results.append({
            "idx": i,
            "question": row["question"][:120],
            "options": options_str,
            "correct_letter": correct_letter,
            "graph_pred": graph_letter,
            "judge_category": verdict.get("category", "ERROR"),
            "key_fact_needed": verdict.get("key_fact_needed", ""),
            "evidence_in_subgraph": verdict.get("evidence_in_subgraph", ""),
            "reasoning": verdict.get("reasoning", ""),
        })
        time.sleep(sleep_between)

    df_results = pd.DataFrame(results)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        print(f"✅ Sauvegardé : {output_csv}")

    return df_results


# ─────────────────────────────────────────────────────────────────────
# Résumé / visualisation
# ─────────────────────────────────────────────────────────────────────
def summarize_judge_results(df_results: pd.DataFrame):
    """Affiche les stats et un graphique."""
    import matplotlib.pyplot as plt

    counts = df_results["judge_category"].value_counts()
    total = len(df_results)

    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DU LLM-JUGE")
    print("=" * 60)
    for cat in ["ABSENT", "PARTIEL", "PRESENT", "ERROR"]:
        n = counts.get(cat, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cat:8s} : {n:3d} ({pct:5.1f}%)")
    print("=" * 60)

    # Diagnostic
    n_retrieval = counts.get("ABSENT", 0) + counts.get("PARTIEL", 0)
    n_generation = counts.get("PRESENT", 0)
    print(f"\n🔴 Échec de RETRIEVAL  : {n_retrieval} ({100*n_retrieval/total:.1f}%)")
    print(f"🟢 Échec de GÉNÉRATION : {n_generation} ({100*n_generation/total:.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"ABSENT": "#d62728", "PARTIEL": "#ff7f0e",
              "PRESENT": "#2ca02c", "ERROR": "#7f7f7f"}
    order = ["ABSENT", "PARTIEL", "PRESENT", "ERROR"]
    vals = [counts.get(c, 0) for c in order]
    bars = ax.bar(order, vals, color=[colors[c] for c in order])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, str(v),
                ha="center", fontweight="bold")
    ax.set_ylabel("Nombre de questions")
    ax.set_title("Diagnostic LLM-juge : pourquoi Graph-RAG a échoué ?")
    plt.tight_layout()
    plt.show()

    return counts
