"""G-Retriever : GNN + projector + LLM gele.

- forward()        : loss CE (teacher forcing) sur la cible 'X) option' correcte.
                     Gradient -> graph token -> projector -> GNN.
- score_options()  : log-vraisemblance moyenne de chacune des 4 options,
                     argmax = prediction (eval methode b).

Le soft prompt place le(s) graph token(s) AVANT le texte (desc+question+options).

Les prompts sont alignes sur ceux de G-Retriever (src/prompts.py) :
meme SYSTEM_PROMPT, meme chat template, meme wording. La seule difference est
l'injection optionnelle des graph tokens avant les embeddings du prompt.
"""
import torch
import torch.nn as nn

from .gnn import GraphTransformer
from .gnn_config import GNNConfig
from ..prompts import (
    wrap_chat_prompt,
    build_user_msg_with_context,
    build_user_msg_no_context,
    build_user_msg_with_context_full,
    build_user_msg_no_context_full,
)
import re
import difflib

IGNORE_INDEX = -100
_NUM2LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}

def parse_letter(text: str):
    if not text:
        return None
    text = text.strip().upper()
    # lettre au début
    m = re.match(r"\s*([ABCD])\s*[\)\.\:]?", text)
    if m:
        return m.group(1)
    # chiffre au début (1-4) -> lettre
    m = re.match(r"\s*([1-4])\s*[\)\.\:]?", text)
    if m:
        return _NUM2LETTER[m.group(1)]
    # fallback lettre isolée
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    # fallback chiffre isolé
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return _NUM2LETTER[m.group(1)]
    return None


def parse_full(text: str, options: dict):
    """Pour le mode 'reponse complete'.
    1) tente d'extraire une lettre (regex),
    2) sinon fallback : similarite textuelle contre les 4 options.
    """
    letter = parse_letter(text)
    if letter is not None:
        return letter

    # fallback : similarite difflib avec chaque option
    if not text:
        return None
    text_low = text.strip().lower()
    best_letter, best_ratio = None, -1.0
    for L in ("A", "B", "C", "D"):
        opt = str(options[L]).strip().lower()
        ratio = difflib.SequenceMatcher(None, text_low, opt).ratio()
        if ratio > best_ratio:
            best_ratio, best_letter = ratio, L
    return best_letter


class Projector(nn.Module):
    """MLP : graph token (gnn_out_dim) -> num_graph_tokens * llm_hidden_dim."""

    def __init__(self, in_dim, hidden_dim, llm_dim, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim * num_tokens),
        )

    def forward(self, graph_emb):              # [B, in_dim]
        out = self.net(graph_emb)              # [B, num_tokens * llm_dim]
        return out.view(graph_emb.size(0), self.num_tokens, self.llm_dim)


class GRetriever(nn.Module):
    def __init__(self, llm, tokenizer, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.llm = llm                          # cause-LM HF, GELE
        self.tokenizer = tokenizer
        self.llm_dim = llm.config.hidden_size

        # LLM gelé + static FP16 pour réduire la mémoire GPU
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        self.llm.half()

        self.gnn = GraphTransformer(
            in_dim=cfg.in_dim, hidden_dim=cfg.hidden_dim,
            out_dim=cfg.gnn_out_dim, edge_dim=cfg.edge_dim,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.projector = Projector(
            cfg.gnn_out_dim, cfg.projector_hidden_dim,
            self.llm_dim, cfg.num_graph_tokens,
        )

        self.word_embedding = self.llm.get_input_embeddings()

    # ---------- utilitaires ----------
    def _embed_text(self, text_ids):
        return self.word_embedding(text_ids)

    def _graph_tokens(self, batch_data):
        graph_emb = self.gnn(
            batch_data["x"], batch_data["edge_index"],
            batch_data["edge_attr"], batch_data["batch"],
        )                                       # [B, gnn_out_dim]
        return self.projector(graph_emb)        # [B, num_graph_tokens, llm_dim]

    def _build_prompt(self, desc, question, opts, mode="score"):
        """Construit le prompt complet (chat template + SYSTEM_PROMPT).

        Aligné sur src/prompts.py (G-Retriever) : même wording, même chat
        template. La seule différence GNN est l'injection optionnelle des
        graph tokens avant les embeddings (gérée par forward/generate_answers).

        Modes :
            "score" / "gen_letter" -> instruction "Answer (single letter):"
            "gen_full"             -> instruction "Copy the letter AND the full text..."
        """
        if mode == "gen_full":
            if self.cfg.use_text_graph:
                user_msg = build_user_msg_with_context_full(question, opts, desc)
            else:
                user_msg = build_user_msg_no_context_full(question, opts)
        else:  # "score" ou "gen_letter"
            if self.cfg.use_text_graph:
                user_msg = build_user_msg_with_context(question, opts, desc)
            else:
                user_msg = build_user_msg_no_context(question, opts)
        return wrap_chat_prompt(self.tokenizer, user_msg)



    # ---------- entrainement ----------
    def forward(self, batch_data, gen_mode=None):
        """Loss CE (teacher forcing) sur la cible correcte.

        gen_mode = "gen_letter" -> cible "A)"          (prompt gen_letter)
        gen_mode = "gen_full"   -> cible "A) option..." (prompt gen_full)
        Defaults to cfg.train_gen_mode.
        """
        if gen_mode is None:
            gen_mode = self.cfg.train_gen_mode
        device = next(self.parameters()).device
        llm_dtype = next(self.llm.parameters()).dtype 
        B = len(batch_data["desc"])
        if self.cfg.use_graph_token:
            graph_tokens = self._graph_tokens(batch_data).to(llm_dtype)  # [B, T, d]
            T = graph_tokens.size(1)
        else:
            graph_tokens = None
            T = 0

        eos = self.tokenizer.eos_token_id
        bos_emb = None  # Qwen n'a pas de BOS obligatoire

        inputs_embeds_list, labels_list, attn_list = [], [], []
        for i in range(B):
            prompt = self._build_prompt(
                batch_data["desc"][i], batch_data["question"][i],
                batch_data["options"][i], mode=gen_mode,
            )
            correct = batch_data["correct_letter"][i]
            if gen_mode == "gen_letter":
                target = f"{correct})"
            else:
                target = batch_data["targets"][i][correct]
            p_ids = self.tokenizer(
                prompt, add_special_tokens=False,
                truncation=True, max_length=self.cfg.max_txt_len,
            ).input_ids
            t_ids = self.tokenizer(
                target, add_special_tokens=False,
                truncation=True, max_length=self.cfg.max_target_len,
            ).input_ids + [eos]

            p_ids = torch.tensor(p_ids, device=device)
            t_ids = torch.tensor(t_ids, device=device)

            p_emb = self._embed_text(p_ids)
            t_emb = self._embed_text(t_ids)

            if self.cfg.use_graph_token:
                g_emb = graph_tokens[i]                       # [T, d]
                full_emb = torch.cat([g_emb, p_emb, t_emb], dim=0)
            else:
                full_emb = torch.cat([p_emb, t_emb], dim=0)

            labels = torch.cat([
                torch.full((T + p_ids.size(0),), IGNORE_INDEX, device=device),
                t_ids,
            ])
            attn = torch.ones(full_emb.size(0), device=device, dtype=torch.long)


            inputs_embeds_list.append(full_emb)
            labels_list.append(labels)
            attn_list.append(attn)

        # padding a gauche-droite : on pad a DROITE (labels -100, attn 0)
        maxL = max(e.size(0) for e in inputs_embeds_list)
        pad_emb = self._embed_text(torch.tensor([eos], device=device))[0]
        emb_b, lab_b, att_b = [], [], []
        for emb, lab, att in zip(inputs_embeds_list, labels_list, attn_list):
            pad = maxL - emb.size(0)
            if pad > 0:
                emb = torch.cat([emb, pad_emb.unsqueeze(0).repeat(pad, 1)], 0)
                lab = torch.cat([lab, torch.full((pad,), IGNORE_INDEX, device=device)])
                att = torch.cat([att, torch.zeros(pad, device=device, dtype=torch.long)])
            emb_b.append(emb); lab_b.append(lab); att_b.append(att)

        inputs_embeds = torch.stack(emb_b)
        labels = torch.stack(lab_b)
        attention_mask = torch.stack(att_b)

        self._last_seq_len = inputs_embeds.size(1)
        self._last_batch_size = B

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = out.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != IGNORE_INDEX
        sel_logits = shift_logits[mask]
        sel_labels = shift_labels[mask]

        loss = torch.nn.functional.cross_entropy(sel_logits.float(), sel_labels)

        return loss
    
    
    @torch.no_grad()
    def generate_answers(self, batch_data, gen_mode="letter",
                         max_new_tokens=None, debug=False):
        """Evaluation generative (item par item, pas de batch).

        gen_mode = "letter" -> le LLM doit generer 'A)' ; parse_letter.
        gen_mode = "full"   -> le LLM genere la reponse complete ; parse_full.

        Retourne : (preds, raws)
            preds : liste de lettres predites (ou None si parsing echoue).
            raws  : liste des generations brutes (str) decodees.
        """
        device = next(self.llm.parameters()).device
        llm_dtype = next(self.llm.parameters()).dtype
        B = len(batch_data["desc"])

        if gen_mode == "letter":
            prompt_mode = "gen_letter"
            mnt = max_new_tokens if max_new_tokens is not None else 5
            parse_fn = lambda txt, opts: parse_letter(txt)
        elif gen_mode == "full":
            prompt_mode = "gen_full"
            mnt = max_new_tokens if max_new_tokens is not None else 64
            parse_fn = lambda txt, opts: parse_full(txt, opts)
        else:
            raise ValueError(f"gen_mode inconnu : {gen_mode}")

        if self.cfg.use_graph_token:
            graph_tokens = self._graph_tokens(batch_data).to(llm_dtype)
        else:
            graph_tokens = None

        preds, raws = [], []
        for i in range(B):
            prompt = self._build_prompt(
                batch_data["desc"][i], batch_data["question"][i],
                batch_data["options"][i], mode=prompt_mode,
            )
            p_ids = self.tokenizer(
                prompt, add_special_tokens=False,
                truncation=True, max_length=self.cfg.max_txt_len,
            ).input_ids
            p_ids = torch.tensor(p_ids, device=device)
            p_emb = self._embed_text(p_ids)

            if self.cfg.use_graph_token and graph_tokens is not None:
                g_emb = graph_tokens[i]
                full_emb = torch.cat([g_emb, p_emb], dim=0).unsqueeze(0)
            else:
                full_emb = p_emb.unsqueeze(0)
            full_emb = full_emb.to(llm_dtype)

            attn = torch.ones(full_emb.size(1), device=device,
                              dtype=torch.long).unsqueeze(0)

            gen = self.llm.generate(
                inputs_embeds=full_emb,
                attention_mask=attn,
                max_new_tokens=mnt,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            raw = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            pred = parse_fn(raw, batch_data["options"][i])
            preds.append(pred)
            raws.append(raw)

            if debug:
                print("=" * 80)
                print(f"[gen_mode={gen_mode}] PROMPT :\n{prompt}")
                print("-" * 80)
                print("RAW GENERATION :", repr(raw))
                print("PRED LETTER    :", pred)
                print("GOLD LETTER    :", batch_data["correct_letter"][i])
                print("CORRECT ?      :", pred == batch_data["correct_letter"][i])
                print("=" * 80)

        return preds, raws


    # ---------- evaluation (methode b) ----------
    @torch.no_grad()
    def score_options(self, batch_data, debug=True):
        """Pour chaque item, calcule la log-vraisemblance moyenne par token de
        chaque option et renvoie la lettre argmax. Retourne liste de lettres."""
        device = next(self.parameters()).device
        llm_dtype = next(self.llm.parameters()).dtype
        B = len(batch_data["desc"])
        if self.cfg.use_graph_token:
            graph_tokens = self._graph_tokens(batch_data).to(llm_dtype)
            T = graph_tokens.size(1)
        else:
            graph_tokens = None
            T = 0

        eos = self.tokenizer.eos_token_id
        letters = ["A", "B", "C", "D"]
        preds = []

        for i in range(B):
            prompt = self._build_prompt(
                batch_data["desc"][i], batch_data["question"][i],
                batch_data["options"][i],
            )
            p_ids = self.tokenizer(
                prompt, add_special_tokens=False,
                truncation=True, max_length=self.cfg.max_txt_len,
            ).input_ids
            p_ids = torch.tensor(p_ids, device=device)
            p_emb = self._embed_text(p_ids)
            g_emb = graph_tokens[i] if self.cfg.use_graph_token else None

            scores = {}
            best_letter, best_score = None, -1e30
            for L in letters:
                target = batch_data["targets"][i][L]
                t_ids = self.tokenizer(
                    target, add_special_tokens=False,
                    truncation=True, max_length=self.cfg.max_target_len,
                ).input_ids + [eos]
                t_ids = torch.tensor(t_ids, device=device)
                t_emb = self._embed_text(t_ids)

                if self.cfg.use_graph_token:
                    full_emb = torch.cat([g_emb, p_emb, t_emb], dim=0).unsqueeze(0)
                else:
                    full_emb = torch.cat([p_emb, t_emb], dim=0).unsqueeze(0)
                full_emb = full_emb.to(llm_dtype)
                attn = torch.ones(full_emb.size(1), device=device,
                                dtype=torch.long).unsqueeze(0)
                out = self.llm(inputs_embeds=full_emb, attention_mask=attn)
                logits = out.logits[0]                      # [Lfull, V]

                start = T + p_ids.size(0)
                pred_logits = logits[start - 1: start - 1 + t_ids.size(0)]
                logprobs = torch.log_softmax(pred_logits, dim=-1)
                tok_lp = logprobs.gather(1, t_ids.unsqueeze(1)).squeeze(1)
                score = tok_lp.mean().item()                # moyenne par token

                scores[L] = score
                if score > best_score:
                    best_score, best_letter = score, L

            preds.append(best_letter)

            if debug:
                print("=" * 80)
                print("PROMPT ENVOYÉ AU LLM :")
                print(prompt)
                print("-" * 80)
                print("Log-vraisemblances moyennes par option (tri décroissant) :")
                for L, s in sorted(scores.items(), key=lambda x: -x[1]):
                    tag = "  <-- choisie" if L == best_letter else ""
                    print(f"   {L}) {batch_data['options'][i][L]!r:40s}  "
                        f"LL={s:.4f}{tag}")
                print("-" * 80)
                raw = self._debug_generate(batch_data, i, graph_tokens)
                print("RÉPONSE RAW (génération libre) :", repr(raw))
                print("PRÉDICTION EXTRAITE (argmax LL):", best_letter)
                print("BONNE RÉPONSE                  :", batch_data["correct_letter"][i])
                print("CORRECT ?                      :",
                    best_letter == batch_data["correct_letter"][i])
                print("=" * 80)

        return preds
    
    @torch.no_grad()

    def _debug_generate(self, batch_data, i, graph_tokens, max_new_tokens=20):
        device = next(self.gnn.parameters()).device  # cf. PB device

        prompt = self._build_prompt(
            batch_data["desc"][i],
            batch_data["question"][i],
            batch_data["options"][i],
        )
        p_ids = self.tokenizer(
            prompt, add_special_tokens=False,
            truncation=True, max_length=self.cfg.max_txt_len,
        ).input_ids
        p_ids = torch.tensor(p_ids, device=device)
        p_emb = self._embed_text(p_ids)              # [Lp, d]

        if self.cfg.use_graph_token and graph_tokens is not None:
            g_emb = graph_tokens[i]                  # [T, d]
            full_emb = torch.cat([g_emb, p_emb], dim=0).unsqueeze(0)
        else:
            full_emb = p_emb.unsqueeze(0)

        # important : cast au dtype du LLM (comme dans score_options)
        full_emb = full_emb.to(next(self.llm.parameters()).dtype)

        attn = torch.ones(full_emb.size(1), device=device, dtype=torch.long).unsqueeze(0)

        gen = self.llm.generate(
            inputs_embeds=full_emb,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


  

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
