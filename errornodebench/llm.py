"""Thin wrapper around litellm for structured-output calls.

We route through litellm so the benchmark can compare consolidators / judges
across providers (Anthropic, OpenAI, Gemini, etc.) without changing call
sites. The pattern is: build messages, hand a pydantic class to
`response_format`, parse the JSON returned in the first text block.

The `LLMConfig` adds per-call `api_base` / `api_key` overrides so a local
OpenAI-compatible proxy (e.g. ../mgpt serving `gpt-5.5` at
`http://localhost:8080/v1`) can be used by passing
`LLMConfig(model="openai/gpt-5.5", api_base="http://localhost:8080/v1")`.

The mgpt-served names (`gpt-5.5`, `gpt-5.4`, `gpt-5.3-codex`,
`gpt-5.3-codex-spark`) are auto-detected: if you pass one of them as a bare
string or with the `openai/` prefix, `MGPT_BASE_URL` is used as the base URL.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import litellm
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# Client-facing slot names mgpt accepts on /v1/chat/completions.
# Keep in sync with mgpt/src/mgpt/config.py:961 (map_model).
MGPT_MODELS = {
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
}


@dataclass
class LLMConfig:
    """Per-call routing config — model + optional base URL + key.

    `model` follows litellm's provider-prefixed format (e.g.
    `openai/gpt-5.5`, `anthropic/claude-opus-4-7`). If you pass a bare mgpt
    slot name like `gpt-5.5`, we auto-prefix `openai/` and use
    `MGPT_BASE_URL` from the environment.
    """

    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    @classmethod
    def resolve(cls, model: str) -> "LLMConfig":
        # bare mgpt slot? e.g. "gpt-5.5"
        if model in MGPT_MODELS:
            return cls(
                model=f"openai/{model}",
                api_base=os.environ.get("MGPT_BASE_URL"),
                # mgpt doesn't auth /v1/chat/completions, but litellm
                # insists on a non-empty key.
                api_key=os.environ.get("OPENAI_API_KEY") or "mgpt-local",
            )
        # already-prefixed mgpt slot? e.g. "openai/gpt-5.5"
        m = re.match(r"^openai/(.+)$", model)
        if m and m.group(1) in MGPT_MODELS:
            return cls(
                model=model,
                api_base=os.environ.get("MGPT_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY") or "mgpt-local",
            )
        # everything else — litellm reads provider keys from the env
        return cls(model=model)


def _strip_code_fence(raw: str) -> str:
    """Some providers wrap JSON in ```json ... ``` fences."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.removeprefix("```")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.lstrip()
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _build_schema_prompt(user: str, response_model: Type[BaseModel]) -> str:
    """Inline the JSON schema into the prompt so providers that don't honor
    `response_format=json_schema` (notably the upstream ChatGPT API that mgpt
    proxies for) still produce the exact field names we need.

    We strip the schema's `title` field because some smaller open-weights
    models (Llama-3.1-8B in particular) treat the title as a wrapper key
    and produce `{"_ReflexionResponse": {...}}` instead of the bare schema.
    The example uses concrete field names from the schema to make the
    intended shape unambiguous for both directions of over/under-wrapping.
    """
    schema = response_model.model_json_schema()
    schema.pop("title", None)
    for v in schema.get("$defs", {}).values():
        v.pop("title", None) if isinstance(v, dict) else None
    top_props = list(schema.get("properties", {}).keys())
    top_key_list = ", ".join(f'"{k}"' for k in top_props)
    return (
        f"{user}\n\n"
        "Reply with ONLY a JSON object — no prose, no markdown fences. "
        "The top-level keys of your JSON object must be exactly: "
        f"{top_key_list}. Do NOT wrap the JSON in a class-name key (no "
        '`{"_FooResponse": {...}}`). Do NOT flatten a nested object into '
        "the top level (if the schema has top-level key `entry`, your "
        'object must literally start with `{"entry": ...}`). Use the '
        "field names from the schema verbatim.\n\n"
        f"{json.dumps(schema, indent=2)}"
    )


def _try_recover_schema(raw_obj: dict, response_model: Type[BaseModel]):
    """If parsing fails, try common recovery patterns:
    - Model unwrapped a single top-level wrapper key (e.g. returned the
      `entry` payload directly without the `entry` wrapper).
    - Model wrapped in an extra class-name key (e.g. `_ReflexionResponse`).
    Return a re-wrapped/unwrapped dict, or None if no recovery is possible.
    """
    schema = response_model.model_json_schema()
    top_props = list(schema.get("properties", {}).keys())

    # Case A: the response has exactly one outer key that's NOT in the
    # schema's top properties — probably a class-name wrapper. Unwrap.
    if isinstance(raw_obj, dict) and len(raw_obj) == 1:
        only_key = next(iter(raw_obj))
        if only_key not in top_props and isinstance(raw_obj[only_key], dict):
            inner = raw_obj[only_key]
            if any(k in inner for k in top_props):
                return inner

    # Case B: the response looks like the inner payload of a single
    # top-level wrapper (e.g. schema is {"entry": MemoryEntry} and the
    # model returned the MemoryEntry fields directly).
    if isinstance(raw_obj, dict) and len(top_props) == 1:
        wrapper_key = top_props[0]
        if wrapper_key not in raw_obj and all(
            k in raw_obj
            for k in schema.get("$defs", {})
            .get(
                # Resolve the $ref target name for the wrapper field's type.
                _ref_name(schema["properties"][wrapper_key]),
                {},
            )
            .get("required", [])
        ):
            return {wrapper_key: raw_obj}

    return None


def _ref_name(field_schema: dict) -> str:
    """Extract the $defs target name from a {"$ref": "#/$defs/Foo"} pointer."""
    ref = field_schema.get("$ref") or ""
    return ref.rsplit("/", 1)[-1]


def structured_call(
    *,
    model: str,
    system: str,
    user: str,
    response_model: Type[T],
    temperature: float = 0.0,
    max_tokens: int = 4000,
    max_retries: int = 1,
    seed: int | None = None,
) -> T:
    """Call `model` and parse the response as `response_model`.

    We send `response_format={"type": "json_object"}` (the lowest common
    denominator that mgpt's upstream honors) and inline the schema in the
    prompt. On a validation failure we retry once with the pydantic error
    fed back in — this catches the common "model renamed `steps` to
    `trajectory`" failure mode without aborting the whole run.
    """
    from errornodebench.claude_cli import is_claude_cli_model, structured_call_via_cli
    from errornodebench.vllm_backend import is_vllm_model, resolve_vllm

    if is_claude_cli_model(model) or (
        model.startswith("anthropic/") and is_claude_cli_model(model[len("anthropic/"):])
    ):
        return structured_call_via_cli(
            model=model,
            system=system,
            user=user,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    if is_vllm_model(model):
        # Modal-hosted vLLM: pre-resolve the api_base + real model id and
        # build a synthetic LLMConfig so we re-use the litellm path below.
        litellm_model, api_base, api_key = resolve_vllm(model)
        cfg = LLMConfig(model=litellm_model, api_base=api_base, api_key=api_key)
        schema_user = _build_schema_prompt(user, response_model)
    else:
        cfg = LLMConfig.resolve(model)
        schema_user = _build_schema_prompt(user, response_model)

    last_error: str | None = None
    last_raw: str | None = None
    for attempt in range(max_retries + 1):
        user_msg = schema_user
        if last_error is not None:
            user_msg = (
                f"{schema_user}\n\n"
                "Your previous reply failed validation:\n"
                f"  raw: {last_raw[:600] if last_raw else '(empty)'}\n"
                f"  error: {last_error}\n"
                "Return a JSON object that matches the schema exactly."
            )

        kwargs: dict = dict(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if seed is not None:
            kwargs["seed"] = seed

        resp = litellm.completion(**kwargs)
        raw = resp.choices[0].message.content or ""
        last_raw = raw
        try:
            data = json.loads(_strip_code_fence(raw))
            try:
                return response_model.model_validate(data)
            except Exception as inner:
                # Try common over/under-wrapping recovery patterns before
                # giving up — open models often drift in both directions.
                recovered = _try_recover_schema(data, response_model)
                if recovered is not None:
                    return response_model.model_validate(recovered)
                raise inner
        except (json.JSONDecodeError, Exception) as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt == max_retries:
                raise RuntimeError(
                    f"structured_call failed after {attempt + 1} attempts. "
                    f"Last error: {last_error}\nLast raw: {raw[:1000]}"
                ) from e

    # unreachable — the loop either returns or raises
    raise RuntimeError("structured_call: unreachable")
