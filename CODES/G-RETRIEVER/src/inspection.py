"""Helpers pour inspecter prompts / contextes / sorties sur quelques exemples."""

from .prompts import build_prompt_no_context, build_prompt_with_context
from .retrieval_graph import retrieval_via_pcst
from .retrieval_rag import encode_questions_e5, faiss_retrieve


def inspect_examples(
    df_questions, cfg, tokenizer,
    n: int = 3,
    rag_index=None, rag_chunks_df=None, rag_embedder=None,
    graph=None, nodes_df=None, edges_df=None, graph_embedder=None,
):
    """Affiche pour les n premières questions le contexte de chaque mode."""
    sub = df_questions.head(n).reset_index(drop=True)

    for i, row in sub.iterrows():
        q = row["question"]
        opts = {k: row[f"option {k}"] for k in "ABCD"}
        print("\n" + "═" * 100)
        print(f"Q{i+1}: {q}")
        for L in "ABCD":
            print(f"  {L}. {opts[L]}")
        print(f"✅ Bonne réponse : {row.get('correct_letter', '?')}")

        for mode in cfg.modes_to_run:
            print(f"\n── MODE : {mode} ──")
            if mode == "zero_shot":
                ctx = None
                prompt = build_prompt_no_context(tokenizer, q, opts)

            elif mode == "rag":
                q_emb = encode_questions_e5([q], rag_embedder)
                chunks = faiss_retrieve(q_emb, rag_index, rag_chunks_df, k=cfg.k_rag)[0]
                ctx = "\n\n".join(f"[Extract {j+1}]\n{c}" for j, c in enumerate(chunks))
                prompt = build_prompt_with_context(tokenizer, q, opts, ctx)

            elif mode in {"graph_rag_pcst", "graph_rag_topk"}:
                q_emb = graph_embedder.encode(
                    [q], convert_to_tensor=True, show_progress_bar=False,
                    device=cfg.device,
                ).cpu()[0]
                ctx, _ = retrieval_via_pcst(
                    graph, q_emb, nodes_df, edges_df,
                    topk=cfg.topk_nodes, topk_e=cfg.topk_edges,
                    cost_e=cfg.pcst_cost_e,
                    mode_pcst=(mode == "graph_rag_pcst"),
                )
                prompt = build_prompt_with_context(tokenizer, q, opts, ctx)
            else:
                continue

            print(f"Contexte : {ctx}")
            print(f"Tokens prompt : {len(tokenizer.encode(prompt))}")
