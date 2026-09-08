from typing import List, Tuple, Optional, Literal, Type
from pathlib import Path
import json
import time
import threading
import dspy
import litellm
from pydantic import BaseModel, create_model, ValidationError

# ─────────────────────────────────────────────────────────────
# Compteur global thread-safe
# ─────────────────────────────────────────────────────────────
_REL_STATS = {"calls": 0, "ok": 0, "fail": 0, "fallback": 0, "fix": 0, "t0": time.time()}
_REL_LOCK = threading.Lock()

def reset_relation_stats():
    """Reset compteurs (à appeler avant un nouveau run)."""
    with _REL_LOCK:
        _REL_STATS["calls"] = 0
        _REL_STATS["ok"] = 0
        _REL_STATS["fail"] = 0
        _REL_STATS["fallback"] = 0
        _REL_STATS["fix"] = 0
        _REL_STATS["t0"] = time.time()

def _log_progress(call_id: int, status: str, dt: float, extra: str = ""):
    with _REL_LOCK:
        elapsed = time.time() - _REL_STATS["t0"]
        rate = _REL_STATS["calls"] / elapsed if elapsed > 0 else 0
        ok, fail, fb, fix = _REL_STATS["ok"], _REL_STATS["fail"], _REL_STATS["fallback"], _REL_STATS["fix"]
    print(f"   {status} #{call_id} en {dt:.2f}s {extra} "
          f"| ok={ok} fail={fail} fb={fb} fix={fix} | {rate:.2f} req/s")


def parse_relations_response(
    raw_json: str,
    entities: List[str],
    response_model: Optional[Type[BaseModel]] = None,
) -> List[Tuple[str, str, str]]:
    entities_set = set(entities)

    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(raw_json)
            return [(r.subject, r.predicate, r.object) for r in parsed.relations]
        except ValidationError as e:
            print(f"      ⚠️  ValidationError strict → fallback JSON ({str(e)[:80]})")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"      ❌ JSONDecodeError : {e}")
        return []

    items = data.get("relations", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        print(f"      ❌ Format inattendu (pas une liste) : {type(items).__name__}")
        return []

    relations = []
    skipped_missing = 0
    skipped_invalid_entity = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        if not all([subject, predicate, obj]):
            skipped_missing += 1
            continue
        if subject not in entities_set or obj not in entities_set:
            skipped_invalid_entity += 1
            continue
        relations.append((subject, predicate, obj))

    if skipped_missing or skipped_invalid_entity:
        print(f"      ⚠️  skipped: {skipped_missing} missing fields, "
              f"{skipped_invalid_entity} invalid entities")
    return relations


def _load_relations_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "relations.txt"
    return prompt_path.read_text()


def _format_relation_hints(relation_hints: Optional[List[str]]) -> str:
    if not relation_hints:
        return ""
    hints_str = "\n".join(f"- {hint}" for hint in relation_hints)
    return (
        "\n\n## Suggested Predicates\n\n"
        "If any of the following predicates are a relevant fit for a relationship between entities in the text, "
        "prefer them. If none fit well, you may propose another precise predicate.\n\n"
        f"{hints_str}\n"
    )


def _create_relations_model(entities: List[str]):
    EntityLiteral = Literal[tuple(entities)]  # type: ignore
    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),
        predicate=(str, ...),
        object=(EntityLiteral, ...),
    )
    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )
    return RelationItem, RelationsResponse


def _get_relations_litellm(
    input_data: str,
    entities: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    relation_hints: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    with _REL_LOCK:
        _REL_STATS["calls"] += 1
        call_id = _REL_STATS["calls"]

    n_chars = len(input_data)
    hints_note = f" | hints={len(relation_hints)}" if relation_hints else ""
    print(f"   📤 #{call_id} [litellm/relations] model={model} | "
          f"text={n_chars} chars (~{n_chars//4} tok) | {len(entities)} entités{hints_note}")

    prompt_template = _load_relations_prompt()
    entities_str = "\n".join(f"- {e}" for e in entities)
    hints_section = _format_relation_hints(relation_hints)
    user_prompt = f"""
Here is the list of entities that were previously extracted from the source text:

<entities>
{entities_str}
</entities>
{hints_section}
Here is the source text to analyze:

<text>
{input_data}
</text>
    """

    _, RelationsResponse = _create_relations_model(entities)

    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False

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
                "name": "relations_response",
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
        raw_json = response.output[-1].content[0].text
        relations = parse_relations_response(raw_json, entities, RelationsResponse)
        dt = time.time() - t0
        with _REL_LOCK:
            _REL_STATS["ok"] += 1
        _log_progress(call_id, "✅", dt, f"| {len(relations)} relations")
        # if relations:
        #     # print(f"      🔎 ex: {relations[0]}")
        return relations
    except Exception as e:
        dt = time.time() - t0
        with _REL_LOCK:
            _REL_STATS["fail"] += 1
        err_type = type(e).__name__
        _log_progress(call_id, "❌", dt, f"| {err_type}: {str(e)[:80]}")
        raise


def extraction_sig(
    Relation: BaseModel,
    is_conversation: bool,
    context: str = "",
    relation_hints: Optional[List[str]] = None,
) -> dspy.Signature:
    hints_section = _format_relation_hints(relation_hints) if relation_hints else ""
    full_context = f"{context}{hints_section}"
    if not is_conversation:
        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the source text.
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {full_context}"""
            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples. Be thorough."
            )
        return ExtractTextRelations
    else:
        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {full_context}"""
            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough"
            )
        return ExtractConversationRelations


def fallback_extraction_sig(
    entities,
    is_conversation,
    context: str = "",
    relation_hints: Optional[List[str]] = None,
) -> dspy.Signature:
    entities_str = "\n- ".join(entities)

    class Relation(BaseModel):
        __doc__ = f"""Knowledge graph subject-predicate-object tuple. Subject and object entities must be one of: {entities_str}"""
        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    return Relation, extraction_sig(Relation, is_conversation, context, relation_hints)


def _filter_entities(entities: List[str]) -> List[str]:
    filtered = [e for e in entities if '"' not in e]
    dropped = len(entities) - len(filtered)
    if dropped:
        print(f"      ⚠️  {dropped} entités filtrées (contiennent des guillemets)")
    return filtered


def get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    relation_hints: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    entities = _filter_entities(entities)

    if use_litellm_prompt and not is_conversation:
        return _get_relations_litellm(
            input_data, entities, model=model,
            api_key=api_key, api_base=api_base, temperature=temperature,
            relation_hints=relation_hints,
        )

    with _REL_LOCK:
        _REL_STATS["calls"] += 1
        call_id = _REL_STATS["calls"]

    mode = "conversation" if is_conversation else "text"
    n_chars = len(input_data)
    hints_note = f" | hints={len(relation_hints)}" if relation_hints else ""
    # print(f"   📤 #{call_id} [dspy/{mode}] text={n_chars} chars "
    #       f"(~{n_chars//4} tok) | {len(entities)} entités{hints_note}")

    class Relation(BaseModel):
        """Knowledge graph subject-predicate-object tuple."""
        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    ExtractRelations = extraction_sig(Relation, is_conversation, context, relation_hints)

    t0 = time.time()
    try:
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        relations = [(r.subject, r.predicate, r.object) for r in result.relations]
        dt = time.time() - t0
        with _REL_LOCK:
            _REL_STATS["ok"] += 1
        _log_progress(call_id, "✅", dt, f"| {len(relations)} relations (path: primary)")
        if relations:
            print(f"      🔎 ex: {relations[0]}")
        return relations

    except Exception as e:
        dt_primary = time.time() - t0
        err_type = type(e).__name__
        print(f"   ⚠️  #{call_id} primary FAILED en {dt_primary:.2f}s "
              f"({err_type}: {str(e)[:80]}) → fallback extraction")

        with _REL_LOCK:
            _REL_STATS["fallback"] += 1

        try:
            t1 = time.time()
            Relation, ExtractRelations = fallback_extraction_sig(
                entities, is_conversation, context, relation_hints
            )
            extract = dspy.Predict(ExtractRelations)
            result = extract(source_text=input_data, entities=entities)
            print(f"      ↳ fallback OK en {time.time()-t1:.2f}s "
                  f"({len(result.relations)} relations brutes) → fix step...")

            class FixedRelations(dspy.Signature):
                """Fix the relations so that every subject and object of the relations are exact matches to an entity. Keep the predicate the same. The meaning of every relation should stay faithful to the reference text. If you cannot maintain the meaning of the original relation relative to the source text, then do not return it."""
                source_text: str = dspy.InputField()
                entities: list[str] = dspy.InputField()
                relations: list[Relation] = dspy.InputField()
                fixed_relations: list[Relation] = dspy.OutputField()

            with _REL_LOCK:
                _REL_STATS["fix"] += 1

            t2 = time.time()
            fix = dspy.ChainOfThought(FixedRelations)
            fix_res = fix(
                source_text=input_data, entities=entities, relations=result.relations
            )
            print(f"      ↳ fix OK en {time.time()-t2:.2f}s "
                  f"({len(fix_res.fixed_relations)} relations corrigées)")

            good_relations = []
            dropped = 0
            for rel in fix_res.fixed_relations:
                if rel.subject in entities and rel.object in entities:
                    good_relations.append(rel)
                else:
                    dropped += 1
            if dropped:
                print(f"      ⚠️  {dropped} relations corrigées rejetées "
                      f"(entités toujours invalides)")

            dt_total = time.time() - t0
            with _REL_LOCK:
                _REL_STATS["ok"] += 1
            _log_progress(call_id, "✅", dt_total,
                          f"| {len(good_relations)} relations (path: fallback+fix)")
            return [(r.subject, r.predicate, r.object) for r in good_relations]

        except Exception as e2:
            dt_total = time.time() - t0
            with _REL_LOCK:
                _REL_STATS["fail"] += 1
            err2_type = type(e2).__name__
            _log_progress(call_id, "❌", dt_total,
                          f"| fallback aussi KO ({err2_type}: {str(e2)[:80]})")
            raise
