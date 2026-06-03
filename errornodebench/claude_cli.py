"""Backend: shell out to `claude -p` with --json-schema for structured output.

Used by `llm.structured_call` when the model name starts with one of the
`claude-` slot prefixes recognized here. The Claude Code CLI returns a JSON
envelope on stdout whose `structured_output` field holds the
schema-validated payload — we parse that directly and feed it into the
caller's Pydantic schema.

We deliberately do NOT pass `--bare`: that mode requires ANTHROPIC_API_KEY,
which we don't set in this environment. Without `--bare`, the CLI relies on
OAuth and inherits the user-level system prompt (~30k tokens). That overhead
caches across invocations within ~5min, so back-to-back calls are cheap;
single calls in isolation are not.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# Claude Code slot names we recognize. The user types e.g. `claude-haiku`,
# and we map that to the `--model` flag the CLI accepts. Keep these short
# so they fit in our CLI flags too.
CLAUDE_MODELS = {
    "claude-opus":   "opus",
    "claude-sonnet": "sonnet",
    "claude-haiku":  "haiku",
}


def is_claude_cli_model(model: str) -> bool:
    return model in CLAUDE_MODELS


def _strip_provider_prefix(model: str) -> str:
    """Allow `anthropic/claude-haiku` as an alias."""
    if model.startswith("anthropic/"):
        return model[len("anthropic/"):]
    return model


def structured_call_via_cli(
    *,
    model: str,
    system: str,
    user: str,
    response_model: Type[T],
    temperature: float = 0.0,    # ignored — claude -p doesn't expose temperature
    max_tokens: int = 4000,      # ignored
    seed: int | None = None,     # ignored
    timeout_s: int = 600,
) -> T:
    cli_model = CLAUDE_MODELS[_strip_provider_prefix(model)]
    schema = response_model.model_json_schema()

    # Compose the user prompt with the schema spelled out, mirroring what
    # `llm._build_schema_prompt` does for litellm. claude -p honors
    # --json-schema directly, but we also tell the model in-prompt so it
    # produces clean fields.
    full_user = (
        f"{user}\n\n"
        "Reply with ONLY a JSON object — no prose, no markdown fences — "
        "matching this JSON Schema exactly. Use the field names from "
        "the schema verbatim; do not invent synonyms.\n\n"
        f"{json.dumps(schema, indent=2)}"
    )

    cmd = [
        "claude",
        "-p",
        "--model", cli_model,
        "--output-format", "json",
        "--disable-slash-commands",
        "--system-prompt", system,
        "--json-schema", json.dumps(schema),
        full_user,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"claude -p timed out after {timeout_s}s for model {cli_model}"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p exited {proc.returncode}: "
            f"stderr={proc.stderr[:500]!r} stdout={proc.stdout[:500]!r}"
        )

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude -p returned non-JSON stdout: {proc.stdout[:500]!r}"
        ) from e

    if envelope.get("is_error"):
        raise RuntimeError(
            f"claude -p reported error: "
            f"{envelope.get('result') or envelope}"
        )

    # Prefer structured_output (schema-validated); fall back to parsing the
    # `result` string if for some reason structured_output is absent.
    data = envelope.get("structured_output")
    if data is None:
        result_text = envelope.get("result", "")
        try:
            data = json.loads(result_text)
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(
                f"claude -p envelope had no structured_output and "
                f"result wasn't JSON: {result_text[:500]!r}"
            ) from e

    return response_model.model_validate(data)
