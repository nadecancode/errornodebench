"""Backend: route `vllm-*` model names to Modal-hosted vLLM endpoints.

We host three open-weights models on Modal (see ../modal_app.py); each one
exposes an OpenAI-compatible /v1/chat/completions endpoint. From this side,
we just pre-resolve the model name to (api_base, real_model_id) and let
litellm do the rest --- the existing `structured_call` path handles
schema-prompted JSON output and the standard retry.

The endpoint URLs are filled in after `modal deploy modal_app.py` and are
also overridable via the VLLM_<MODEL>_URL environment variables, e.g.:

    export VLLM_LLAMA_70B_URL=https://my-ws--errornodebench-vllm-serve-llama70b.modal.run
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMTarget:
    """One Modal-hosted vLLM endpoint: HF model id, default URL, override env var."""

    # The litellm-prefixed model id the OpenAI-compatible proxy expects in
    # /v1/chat/completions `model` field. vLLM accepts whatever id you
    # started it with; we keep the full HF repo path for clarity.
    real_model_id: str
    # The default endpoint URL (no /v1 suffix; we append it ourselves).
    # Override per-model with the env var key below.
    default_url: str
    env_var: str


# Defaults assume the Modal workspace is `allenzhg` and the app name is
# `errornodebench-vllm` (matching modal_app.py). Override with env vars
# (MODAL_WORKSPACE and the per-model VLLM_*_URL) if your deployment differs.
_DEFAULT_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "allenzhg")
_BASE = f"https://{_DEFAULT_WORKSPACE}--errornodebench-vllm"

VLLM_ENDPOINTS: dict[str, VLLMTarget] = {
    "vllm-llama-3.1-8b": VLLMTarget(
        real_model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        default_url=f"{_BASE}-serve-llama8b.modal.run",
        env_var="VLLM_LLAMA_8B_URL",
    ),
    "vllm-qwen3.5-27b": VLLMTarget(
        real_model_id="Qwen/Qwen3.5-27B",
        default_url=f"{_BASE}-serve-qwen35-27b.modal.run",
        env_var="VLLM_QWEN35_27B_URL",
    ),
    "vllm-qwen2.5-32b": VLLMTarget(
        # The original substitution slot; kept available for reproducibility
        # of the prior runs/v7_qwen32b.json file.
        real_model_id="Qwen/Qwen2.5-32B-Instruct",
        default_url=f"{_BASE}-serve-qwen27b.modal.run",
        env_var="VLLM_QWEN_27B_URL",
    ),
    "vllm-deepseek-v4-flash": VLLMTarget(
        real_model_id="deepseek-ai/DeepSeek-V4-Flash",
        default_url=f"{_BASE}-serve-deepseek-v4-flash.modal.run",
        env_var="VLLM_DEEPSEEK_V4_FLASH_URL",
    ),
}


def is_vllm_model(model: str) -> bool:
    """True if ``model`` is a ``vllm-*`` slot served by a Modal vLLM endpoint."""
    return model in VLLM_ENDPOINTS


def resolve_vllm(model: str) -> tuple[str, str, str]:
    """Return (litellm_model, api_base, api_key) for a vllm-* slot."""
    t = VLLM_ENDPOINTS[model]
    url = os.environ.get(t.env_var, t.default_url).rstrip("/")
    api_base = f"{url}/v1"
    # vLLM doesn't require auth, but litellm insists on a non-empty key.
    api_key = os.environ.get("VLLM_API_KEY") or "modal-vllm"
    return f"openai/{t.real_model_id}", api_base, api_key
