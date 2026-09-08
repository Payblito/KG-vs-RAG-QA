import pandas as pd
import torch
import torch.nn.functional as F

from .llm import generate_letters_batch
from .prompts import build_prompt_no_context, build_prompt_with_context
from .retrieval_graph import (
    parallel_graph_retrieval,
    parallel_graph_retrieval_fusion,
    parallel_graph_retrieval_kaping,
    parallel_graph_retrieval_kaping_fusion,
    triplets_to_context,
)
from .retrieval_rag import encode_questions_e5, faiss_retrieve


def _pcst_cache_contexts(questions, article_ids, cfg,
                         options_list=None, correct_letters=None,
                         graph=None, nodes_df=None, edges_df=None, graph_embedder=None,
                         ln_lookup=None):
    """Retourne les contextes PCST depuis le cache (alignement par clé).

    Branche utilisée quand cfg.use_pcst_cache=True et mode == "graph_rag_pcst".
    En cas de cache partiel/absent :
      - si cfg.save_pcst_cache_on_recompute et (graph, graph_embedder, nodes_df,
        edges_df) disponibles : on recalcule les manquants (parallèle) et on
        sauvegarde le cache mis à jour (format keyed, compatible GNN).
      - sinon : warning + retour None pour signaler un fallback au caller.

    Retourne (contexts, ok) où ok=True si tous les contextes ont été résolus
    via le cache, False si un fallback calcul est nécessaire (caller gère).
    """
    from .gnn_utils.pcst_cache import (
        load_pcst_cache_as_dict,
        build_pcst_cache_keyed,
        save_pcst_cache,
    )

    cache_path = cfg.pcst_cache_path
    cache_dict = load_pcst_cache_as_dict(cache_path, cfg)

    keys = [(str(aid), q) for aid, q in zip(article_ids, questions)]
    contexts = [None] * len(keys)
    miss_idx = []
    for i, k in enumerate(keys):
        if k in cache_dict:
            contexts[i] = cache_dict[k].get("desc")
        else:
            miss_idx.append(i)

    if not miss_idx:
        print(f"[pcst_cache] {len(keys)}/{len(keys)} contextes résolus via cache "
              f"({cache_path})")
        return contexts, True

    print(f"[pcst_cache] {len(keys) - len(miss_idx)}/{len(keys)} résolus via cache, "
          f"{len(miss_idx)} manquants.")

    if not getattr(cfg, "save_pcst_cache_on_recompute", False):
        print(f"[pcst_cache] save_pcst_cache_on_recompute=False -> pas de recompute, "
              f"fallback calcul standard pour les {len(miss_idx)} manquants.")
        return contexts, False

    if graph is None or graph_embedder is None or nodes_df is None or edges_df is None:
        print("[pcst_cache] ⚠️ Cache partiel/absent et artifacts graphe non fournis "
              "(graph/nodes_df/edges_df/graph_embedder=None). Recharge les artifacts "
              "ou désactive cfg.use_pcst_cache. Fallback calcul standard.")
        return contexts, False

    # Reconstruction d'un df minimal pour build_pcst_cache_keyed
    import pandas as pd
    rows = []
    for i in miss_idx:
        opts = options_list[i] if options_list is not None else {
            L: "" for L in ["A", "B", "C", "D"]
        }
        rows.append({
            "article_id": article_ids[i],
            "question": questions[i],
            "option A": opts.get("A", ""),
            "option B": opts.get("B", ""),
            "option C": opts.get("C", ""),
            "option D": opts.get("D", ""),
            "correct_letter": correct_letters[i] if correct_letters else "",
        })
    df_miss = pd.DataFrame(rows)

    print(f"[pcst_cache] Recompute PCST (parallèle) pour {len(df_miss)} questions "
          f"(query_mode={cfg.retrieval_query_mode})...")
    new_entries = build_pcst_cache_keyed(
        df_miss, cfg, graph, nodes_df, edges_df, graph_embedder,
        query_mode=cfg.retrieval_query_mode, ln_lookup=ln_lookup,
    )
    save_pcst_cache(new_entries, cache_path, existing_dict=cache_dict)

    for j, idx in enumerate(miss_idx):
        contexts[idx] = new_entries[j].get("desc")

    return contexts, True


# Modes supportés
SUPPORTED_MODES = {"zero_shot", "rag", "graph_rag_pcst", "graph_rag_topk", "kaping", "oracle_triplets"}

def _query_text(question: str, options: dict) -> str:
    """Concatène la question et les 4 options pour le retrieval."""
    return (
        f"{question} "
        f"A) {options['A']} "
        f"B) {options['B']} "
        f"C) {options['C']} "
        f"D) {options['D']}"
    )
def _query_text_oracle(question: str, options: dict, correct_letter: str) -> str:
    """Requête oracle : question + bonne réponse uniquement (pour upper bound retrieval)."""
    return f"{question} {options[correct_letter]}"


def _query_texts_question_plus_one_option(question: str, options: dict) -> list:
    """Une requête par option : [question+A, question+B, question+C, question+D].

    Utilisé pour le mode 'question+1option' : on fait un retrieval par combinaison
    puis on fusionne les sous-graphes obtenus.
    """
    return [f"{question} {options[letter]}" for letter in ("A", "B", "C", "D")]



def _options_from_row(row):
    return {
        "A": row["option A"], "B": row["option B"],
        "C": row["option C"], "D": row["option D"],
    }


def _build_contexts(
    mode, questions, cfg,
    options_list=None,
    correct_letters=None,                    # ← NOUVEAU
    article_ids=None,                        # ← NOUVEAU (oracle_triplets)
    rag_index=None, rag_chunks_df=None, rag_embedder=None,
    graph=None, nodes_df=None, edges_df=None, graph_embedder=None,
    provenance_index=None,                   # ← NOUVEAU (oracle_triplets)
    provenance_triplet_keys=None,            # ← NOUVEAU (embeddings triplets)
    provenance_triplet_emb=None,             # ← NOUVEAU (matrice [N, D])
    provenance_triplet2idx=None,             # ← NOUVEAU (lookup (s,p,o)->int)
    ln_lookup=None,                          # ← NOUVEAU (transfo_ln)
):
    if mode == "zero_shot":
        return [None] * len(questions)

    # ── Mode oracle_triplets : sélection par article_id via la provenance ──
    # Filtrage par similarité query/triplet (uniquement sur les triplets de
    # l'article), top_k_provenance triplets retenus.
    if mode == "oracle_triplets":
        assert provenance_index is not None, "provenance_index requis pour oracle_triplets"
        assert article_ids is not None, "article_ids requis pour oracle_triplets"
        assert provenance_triplet_emb is not None, "provenance_triplet_emb requis pour oracle_triplets"
        assert provenance_triplet2idx is not None, "provenance_triplet2idx requis pour oracle_triplets"
        assert graph_embedder is not None, "graph_embedder requis pour oracle_triplets"

        top_k = getattr(cfg, "top_k_provenance", 5)
        query_mode = getattr(cfg, "retrieval_query_mode", "question_options")

        # ── Cas question+1option : 4 requêtes par question, union des top-k ──
        if query_mode == "question+1option":
            assert options_list is not None
            flat_query_texts = []
            for q, opts in zip(questions, options_list):
                flat_query_texts.extend(_query_texts_question_plus_one_option(q, opts))

            q_embs_t = graph_embedder.encode(
                flat_query_texts,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=cfg.device,
                task="retrieval.query",
            ).cpu()

            n_opts = 4
            q_embs_per_question = [
                [q_embs_t[i * n_opts + j] for j in range(n_opts)]
                for i in range(len(questions))
            ]

            contexts = []
            for q_idx, (aid, q_embs_list) in enumerate(zip(article_ids, q_embs_per_question)):
                triplets = provenance_index.get(aid, [])
                if not triplets:
                    contexts.append(None)
                    continue
                # Indices des embeddings des triplets de cet article
                tri_idx = [provenance_triplet2idx[t] for t in triplets if t in provenance_triplet2idx]
                if not tri_idx:
                    contexts.append(None)
                    continue
                art_emb = provenance_triplet_emb[tri_idx]  # [M, D]

                # Union des top-k par option : score = max sur les 4 options
                best_scores = torch.full((len(tri_idx),), -1.0)
                for q_emb in q_embs_list:
                    sims = F.cosine_similarity(art_emb, q_emb.unsqueeze(0).expand_as(art_emb), dim=1)
                    best_scores = torch.maximum(best_scores, sims)

                k = min(top_k, len(tri_idx))
                topk_scores, topk_local = torch.topk(best_scores, k)
                selected = [triplets[i] for i in topk_local.tolist()]
                contexts.append(triplets_to_context(selected, ln_lookup=ln_lookup))
            return contexts

        # ── Cas question / question_options / oracle : une seule requête ──
        if query_mode == "question":
            query_texts = questions
        elif query_mode == "question_options":
            assert options_list is not None
            query_texts = [_query_text(q, opts) for q, opts in zip(questions, options_list)]
        elif query_mode == "oracle":
            assert options_list is not None and correct_letters is not None
            query_texts = [
                _query_text_oracle(q, opts, c)
                for q, opts, c in zip(questions, options_list, correct_letters)
            ]
        else:
            raise ValueError(f"retrieval_query_mode inconnu : {query_mode}")

        q_embs_t = graph_embedder.encode(
            query_texts,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=cfg.device,
            task="retrieval.query",
        ).cpu()

        contexts = []
        for q_idx, (aid, q_emb) in enumerate(zip(article_ids, q_embs_t)):
            triplets = provenance_index.get(aid, [])
            if not triplets:
                contexts.append(None)
                continue
            tri_idx = [provenance_triplet2idx[t] for t in triplets if t in provenance_triplet2idx]
            if not tri_idx:
                contexts.append(None)
                continue
            art_emb = provenance_triplet_emb[tri_idx]  # [M, D]
            sims = F.cosine_similarity(art_emb, q_emb.unsqueeze(0).expand_as(art_emb), dim=1)
            k = min(top_k, len(tri_idx))
            topk_scores, topk_local = torch.topk(sims, k)
            selected = [triplets[i] for i in topk_local.tolist()]
            contexts.append(triplets_to_context(selected, ln_lookup=ln_lookup))
        return contexts

    # ✅ Construction de la requête selon cfg.retrieval_query_mode
    query_mode = getattr(cfg, "retrieval_query_mode", "question_options")

    # ── Cas spécial : question+1option (retrieval par option puis fusion) ──
    # Ce mode n'a de sens que pour les modes graphe (fusion de sous-graphes).
    if query_mode == "question+1option":
        if mode not in {"graph_rag_pcst", "graph_rag_topk", "kaping"}:
            raise ValueError(
                "retrieval_query_mode='question+1option' n'est supporté que pour "
                f"les modes graphe (graph_rag_pcst, graph_rag_topk, kaping), pas pour '{mode}'."
            )
        assert options_list is not None

        # Liste plate : 4 requêtes par question (A, B, C, D) → encodage en un batch
        flat_query_texts = []
        for q, opts in zip(questions, options_list):
            flat_query_texts.extend(_query_texts_question_plus_one_option(q, opts))

        q_embs_t = graph_embedder.encode(
            flat_query_texts,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=cfg.device,
            task="retrieval.query",
        ).cpu()

        # Regroupe par question : [[A,B,C,D], [A,B,C,D], ...]
        n_opts = 4
        q_embs_per_question = [
            [q_embs_t[i * n_opts + j] for j in range(n_opts)]
            for i in range(len(questions))
        ]

        if mode == "kaping":
            # KAPING : union des top-k triplets de chaque option
            return parallel_graph_retrieval_kaping_fusion(
                q_embs_per_question, graph, nodes_df, edges_df, cfg, ln_lookup=ln_lookup
            )

        # PCST / top-k : fusion (union) des sous-graphes de chaque option
        mode_pcst = (mode == "graph_rag_pcst")
        return parallel_graph_retrieval_fusion(
            q_embs_per_question, graph, nodes_df, edges_df, cfg, mode_pcst=mode_pcst, ln_lookup=ln_lookup
        )

    if query_mode == "question":
        query_texts = questions
    elif query_mode == "question_options":
        assert options_list is not None
        query_texts = [_query_text(q, opts) for q, opts in zip(questions, options_list)]
    elif query_mode == "oracle":
        assert options_list is not None and correct_letters is not None
        query_texts = [
            _query_text_oracle(q, opts, c)
            for q, opts, c in zip(questions, options_list, correct_letters)
        ]
    else:
        raise ValueError(f"retrieval_query_mode inconnu : {query_mode}")

    if mode == "rag":

        q_embs = encode_questions_e5(query_texts, rag_embedder)   # ← query_texts
        retrieved = faiss_retrieve(q_embs, rag_index, rag_chunks_df, k=cfg.k_rag)
        return [
            "\n\n".join(f"[Extract {j+1}]\n{c}" for j, c in enumerate(chks))
            for chks in retrieved
        ]

    if mode in {"graph_rag_pcst", "graph_rag_topk", "kaping"}:
        # ── Cache PCST (graph_rag_pcst uniquement, non-fusion) ──
        if mode == "graph_rag_pcst" and getattr(cfg, "use_pcst_cache", False):
            ctx_cache, ok = _pcst_cache_contexts(
                questions, article_ids, cfg,
                options_list=options_list,
                correct_letters=correct_letters,
                graph=graph, nodes_df=nodes_df, edges_df=edges_df,
                graph_embedder=graph_embedder, ln_lookup=ln_lookup,
            )
            if ok:
                return ctx_cache
            # ok=False : cache partiel et recompute impossible/désactivé.
            # On tombe sur le calcul standard, en réutilisant les contextes
            # déjà résolus via le cache (où ctx_cache[i] is not None).
            resolved = ctx_cache
            miss_idx = [i for i, c in enumerate(resolved) if c is None]
            if not miss_idx:
                return resolved
            if graph is None or graph_embedder is None or nodes_df is None or edges_df is None:
                raise RuntimeError(
                    "[pcst_cache] Cache PCST partiel/absent et artifacts graphe non "
                    "fournis (graph/nodes_df/edges_df/graph_embedder=None). "
                    "Recharge les artifacts avant run_all_modes, ou désactive "
                    "cfg.use_pcst_cache."
                )
            print(f"[pcst_cache] Fallback calcul standard pour {len(miss_idx)} questions.")
            # Encodage uniquement des manquants
            miss_questions = [query_texts[i] for i in miss_idx]
            q_embs_t = graph_embedder.encode(
                miss_questions,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=cfg.device,
                task="retrieval.query",
            ).cpu()
            q_list = [q_embs_t[i] for i in range(q_embs_t.size(0))]
            fallback_ctx = parallel_graph_retrieval(
                q_list, graph, nodes_df, edges_df, cfg, mode_pcst=True, ln_lookup=ln_lookup
            )
            for j, idx in enumerate(miss_idx):
                resolved[idx] = fallback_ctx[j]
            return resolved

        q_embs_t = graph_embedder.encode(
            query_texts,
            batch_size=64, 
            convert_to_tensor=True,
            normalize_embeddings=True,                     
            show_progress_bar=True, 
            device=cfg.device,
            task="retrieval.query"   # <-- Spécifique pour Jina
        ).cpu()

        q_list = [q_embs_t[i] for i in range(q_embs_t.size(0))]

        if mode == "kaping":
            return parallel_graph_retrieval_kaping(
                q_list, graph, nodes_df, edges_df, cfg, ln_lookup=ln_lookup
            )

        mode_pcst = (mode == "graph_rag_pcst")
        return parallel_graph_retrieval(
            q_list, graph, nodes_df, edges_df, cfg, mode_pcst=mode_pcst, ln_lookup=ln_lookup
        )

    raise ValueError(f"Mode inconnu : {mode}")



def run_eval_for_mode(
    df_questions, mode, cfg, tokenizer, llm,
    rag_index=None, rag_chunks_df=None, rag_embedder=None,
    graph=None, nodes_df=None, edges_df=None, graph_embedder=None,
    provenance_index=None,
    provenance_triplet_keys=None,
    provenance_triplet_emb=None,
    provenance_triplet2idx=None,
    ln_lookup=None,
):
    assert mode in SUPPORTED_MODES, f"Mode invalide : {mode}"

    questions = df_questions["question"].tolist()
    options_list = [_options_from_row(r) for _, r in df_questions.iterrows()]
    correct_letters = df_questions["correct_letter"].tolist()
    article_ids = df_questions["article_id"].tolist()

    # 1. Contextes
    contexts = _build_contexts(
        mode, questions, cfg,
        options_list=options_list,
        correct_letters=correct_letters,                    # ← NOUVEAU
        article_ids=article_ids,                            # ← NOUVEAU
        rag_index=rag_index, rag_chunks_df=rag_chunks_df, rag_embedder=rag_embedder,
        graph=graph, nodes_df=nodes_df, edges_df=edges_df, graph_embedder=graph_embedder,
        provenance_index=provenance_index,                  # ← NOUVEAU
        provenance_triplet_keys=provenance_triplet_keys,
        provenance_triplet_emb=provenance_triplet_emb,
        provenance_triplet2idx=provenance_triplet2idx,
        ln_lookup=ln_lookup,
    )
    # 2. Prompts
    prompts = []
    for q, opts, ctx in zip(questions, options_list, contexts):
        if mode == "zero_shot":
            prompts.append(build_prompt_no_context(tokenizer, q, opts))
        else:
            prompts.append(build_prompt_with_context(tokenizer, q, opts, ctx))

    lengths = [len(tokenizer.encode(p)) for p in prompts]
    print(f"[{mode}] Token lengths : min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")
    # print("3 Premiers prompts :")
    # for i, p in enumerate(prompts[:3]):
    #     print(f"  {i+1}. {p}")

    # 3. Génération
    raws, preds = generate_letters_batch(
        prompts, llm, tokenizer,
        batch_size=cfg.batch_size, max_new_tokens=cfg.max_new_tokens,
    )

    # 4. Résultats
    out = pd.DataFrame({
        "question": questions,
        "article_id": df_questions["article_id"].values,
        "correct_letter": df_questions["correct_letter"].values,
        "predicted_letter": preds,
        "correct": [p == c for p, c in zip(preds, df_questions["correct_letter"])],
        "raw_output": raws,
        "context": contexts,
        "prompt": prompts,
        "mode": mode,
    })
    return out



def run_all_modes(df_questions, cfg, save_all=None, **kwargs):
    """Lance tous les modes listés dans cfg.modes_to_run et renvoie un dict.

    Args:
        save_all: si True, enregistre TOUT pour chaque question et chaque mode
                  (prompt, contexte/triplets, sortie brute, prédiction...).
                  Si None, on utilise cfg.save_all.
    """
    if save_all is None:
        save_all = getattr(cfg, "save_all", False)

    results = {}
    for mode in cfg.modes_to_run:
        print(f"\n{'='*60}\n  MODE : {mode}\n{'='*60}")
        results[mode] = run_eval_for_mode(df_questions, mode, cfg, **kwargs)
        torch.cuda.empty_cache()
        acc = results[mode]["correct"].mean()
        print(f"✅ Accuracy [{mode}] : {acc:.2%}")

    if save_all:
        from .io_utils import save_all_results
        save_all_results(results, cfg)

    return results

def run_modes(df_questions, cfg, modes, save_all=None, **kwargs):
    """Lance les modes spécifiés (liste passée en argument) et renvoie un dict.
    
    Args:
        df_questions: DataFrame des questions à évaluer
        cfg: Config
        modes: liste de modes à exécuter (ex: ["zero_shot", "rag"])
        save_all: si True, enregistre TOUT pour chaque question et chaque mode.
                  Si None, on utilise cfg.save_all.
        **kwargs: tokenizer, llm, rag_index, rag_chunks_df, rag_embedder,
                  graph, nodes_df, edges_df, graph_embedder
    """
    invalid = set(modes) - SUPPORTED_MODES
    if invalid:
        raise ValueError(f"Modes invalides : {invalid}. Supportés : {SUPPORTED_MODES}")

    if save_all is None:
        save_all = getattr(cfg, "save_all", False)

    results = {}
    for mode in modes:
        print(f"\n{'='*60}\n  MODE : {mode}\n{'='*60}")
        results[mode] = run_eval_for_mode(df_questions, mode, cfg, **kwargs)
        torch.cuda.empty_cache()
        acc = results[mode]["correct"].mean()
        print(f"✅ Accuracy [{mode}] : {acc:.2%}")

    if save_all:
        from .io_utils import save_all_results
        save_all_results(results, cfg)

    return results
