from typing import List, Optional
from pathlib import Path
import time
import threading
import dspy
import litellm
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# Compteur global thread-safe pour monitoring
# ─────────────────────────────────────────────────────────────
_ENT_STATS = {"calls": 0, "ok": 0, "fail": 0, "t0": time.time()}
_ENT_LOCK = threading.Lock()

def reset_entity_stats():
    """Reset compteurs (à appeler avant un nouveau run)."""
    with _ENT_LOCK:
        _ENT_STATS["calls"] = 0
        _ENT_STATS["ok"] = 0
        _ENT_STATS["fail"] = 0
        _ENT_STATS["t0"] = time.time()


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    entities: List[str]


def _load_entities_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text()


def _format_entity_hints(entity_hints: Optional[List[str]]) -> str:
    if not entity_hints:
        return ""
    hints_str = "\n".join(f"- {hint}" for hint in entity_hints)
    return (
        "\n\n## Suggested Entities\n\n"
        "If any of the following entities are present or relevant in the text, "
        "please include them. You may also include other relevant entities found in the text.\n\n"
        f"{hints_str}\n"
    )


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    entity_hints: Optional[List[str]] = None,
) -> List[str]:
    with _ENT_LOCK:
        _ENT_STATS["calls"] += 1
        call_id = _ENT_STATS["calls"]

    n_chars = len(input_data)
    n_tokens_approx = n_chars // 4
    hints_note = f" | hints={len(entity_hints)}" if entity_hints else ""
    # print(f"   📤 #{call_id} [litellm] model={model} | input={n_chars} chars (~{n_tokens_approx} tok){hints_note}")

    prompt_template = _load_entities_prompt()
    hints_section = _format_entity_hints(entity_hints)
    user_prompt = f"""
Here is the text to extract entities from:

<article>
{input_data}
</article>
{hints_section}
    """

    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    t0 = time.time()
    try:
        response = litellm.responses(**kwargs)
        dt = time.time() - t0
        parsed = EntitiesResponse.model_validate_json(
            response.output[-1].content[0].text
        )
        with _ENT_LOCK:
            _ENT_STATS["ok"] += 1
            elapsed = time.time() - _ENT_STATS["t0"]
            rate = _ENT_STATS["calls"] / elapsed if elapsed > 0 else 0
        # print(f"   ✅ #{call_id} OK en {dt:.2f}s | {len(parsed.entities)} entités "
        #       f"| total: {_ENT_STATS['ok']} ok / {_ENT_STATS['fail']} fail "
        #       f"| {rate:.2f} req/s")
        if parsed.entities:
            sample = parsed.entities[:3]
            # print(f"      🔎 ex: {sample}")
        return parsed.entities
    except Exception as e:
        dt = time.time() - t0
        with _ENT_LOCK:
            _ENT_STATS["fail"] += 1
        err_type = type(e).__name__
        err_msg = str(e)[:120]
        print(f"   ❌ #{call_id} {err_type} en {dt:.2f}s : {err_msg}")
        raise


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    entity_hints: Optional[List[str]] = None,
) -> List[str]:
    if use_litellm_prompt and not is_conversation:
        return _get_entities_litellm(
            input_data, model=model, api_key=api_key,
            api_base=api_base, temperature=temperature,
            entity_hints=entity_hints,
        )

    with _ENT_LOCK:
        _ENT_STATS["calls"] += 1
        call_id = _ENT_STATS["calls"]

    mode = "conversation" if is_conversation else "text"
    n_chars = len(input_data)
    # print(f"   📤 #{call_id} [dspy/{mode}] input={n_chars} chars (~{n_chars//4} tok)")

    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )

    t0 = time.time()
    try:
        result = extract(source_text=input_data)
        dt = time.time() - t0
        entities = result.entities
        with _ENT_LOCK:
            _ENT_STATS["ok"] += 1
            elapsed = time.time() - _ENT_STATS["t0"]
            rate = _ENT_STATS["calls"] / elapsed if elapsed > 0 else 0
        # if call_id % 10 == 0:
        #     print(f"   ✅ #{call_id} OK en {dt:.2f}s | {len(entities)} entités "
        #         f"| total: {_ENT_STATS['ok']} ok / {_ENT_STATS['fail']} fail "
        #         f"| {rate:.2f} req/s")
        # if entities:
            # print(f"      🔎 ex: {entities[:3]}")
            
        return entities
    except Exception as e:
        dt = time.time() - t0
        with _ENT_LOCK:
            _ENT_STATS["fail"] += 1
        err_type = type(e).__name__
        err_msg = str(e)[:120]
        print(f"   ❌ #{call_id} {err_type} en {dt:.2f}s : {err_msg}")
        raise
