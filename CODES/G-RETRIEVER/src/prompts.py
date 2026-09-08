SYSTEM_PROMPT = """You are an expert assistant answering multiple-choice questions.

Instructions:
- Answer ONLY with the letter (A, B, C, or D) corresponding to the correct answer.
- If you are unsure, make your best guess.
- Do not add any explanation or additional text."""


# ══════════════════════════════════════════════════════════════════════
# User-message builders (sans SYSTEM_PROMPT, sans chat template)
# Réutilisés côté GNN (g_retriever_model) pour aligner les prompts.
# ══════════════════════════════════════════════════════════════════════
def build_user_msg_no_context(question, options):
    return f"""Answer the following multiple-choice question based on your general knowledge.

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Answer (single letter):"""


def build_user_msg_with_context(question, options, context_block):
    return f"""Here is some context that *may* help answer the following multiple-choice question.
If the context is irrelevant or unclear, rely on your general knowledge.

Context:
{context_block}

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Answer (single letter):"""


def build_user_msg_no_context_full(question, options):
    return f"""Answer the following multiple-choice question based on your general knowledge.

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Copy the letter AND the full text of the correct option, for example: A) option text."""


def build_user_msg_with_context_full(question, options, context_block):
    return f"""Here is some context that *may* help answer the following multiple-choice question.
If the context is irrelevant or unclear, rely on your general knowledge.

Context:
{context_block}

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Copy the letter AND the full text of the correct option, for example: A) option text."""


# ══════════════════════════════════════════════════════════════════════
# Chat template wrapper
# ══════════════════════════════════════════════════════════════════════
def wrap_chat_prompt(tokenizer, user_msg: str) -> str:
    """Applique le chat template du tokenizer avec le SYSTEM_PROMPT.

    Le SYSTEM_PROMPT est concaténé au user_msg (comme G-Retriever le faisait
    historiquement), puis le tout est passé en role 'user'.
    """
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# Alias interne pour backward compat
_wrap_chat = wrap_chat_prompt


# ══════════════════════════════════════════════════════════════════════
# API publique G-Retriever (signatures inchangées)
# ══════════════════════════════════════════════════════════════════════
def build_prompt_no_context(tokenizer, question, options):
    return wrap_chat_prompt(tokenizer, build_user_msg_no_context(question, options))


def build_prompt_with_context(tokenizer, question, options, context_block):
    return wrap_chat_prompt(tokenizer, build_user_msg_with_context(question, options, context_block))
