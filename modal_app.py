"""Modal deployment: host three open-weights LLMs behind OpenAI-compatible
endpoints for the ErrorNodeBench-Interference cross-backbone study.

Deploy with:
    cd project
    modal deploy modal_app.py

After deployment, the endpoints will be reachable at:
    https://<workspace>--errornodebench-vllm-serve-llama8b.modal.run/v1/chat/completions
    https://<workspace>--errornodebench-vllm-serve-qwen27b.modal.run/v1/chat/completions
    https://<workspace>--errornodebench-vllm-serve-deepseek-v4-flash.modal.run/v1/chat/completions

The model names route through `errornodebench.vllm_backend.VLLM_ENDPOINTS`.

Notes:
- Llama-3.1-8B uses the open `unsloth/Meta-Llama-3.1-8B-Instruct` mirror
  (no HF token needed). Qwen3.5-27B and DeepSeek-V4-Flash are open / MIT.
- 27B fits in 1x H100 (80GB) at BF16; 8B fits on L40S.
- DeepSeek-V4-Flash (284B MoE, 13B active, ~158GB weights) needs vLLM
  >=0.20.0 and a 4x H200 single-node DP+EP setup.
- Containers scale to zero after 5 minutes of no requests.
"""

from __future__ import annotations

import subprocess

import modal

APP_NAME = "errornodebench-vllm"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.7.3",
        "huggingface_hub[hf_transfer]==0.27.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_LOG_STATS_INTERVAL": "30",
        }
    )
)

# DeepSeek-V4-Flash needs a much newer vLLM (>=0.20.0) with the v4 tokenizer
# and reasoning parser. vLLM 0.20.0+ requires CUDA 13.0 and pre-release
# transformers (which uv doesn't install by default), so we use the
# official Modal cookbook image base (cuda 12.9) + extra_options=--pre.
vllm_v4_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "huggingface_hub[hf_transfer]",  # let pip resolve compatible version
        extra_options="--prerelease=allow",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_LOG_STATS_INTERVAL": "30",
        }
    )
)

app = modal.App(APP_NAME)

# Shared volumes keep model weights warm across container restarts.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def _vllm_cmd(model_id: str, *, max_model_len: int = 8192, extra: str = "") -> list[str]:
    # Note: `--disable-log-requests` was removed in vLLM 0.21.0; we omit
    # it here so both the 0.7.3 image and the 0.21.0 image share one
    # builder. (0.7.3 accepts the absence; 0.21.0 rejects the flag.)
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        str(max_model_len),
        "--enable-prefix-caching",
    ]
    if extra:
        cmd.extend(extra.split())
    return cmd


# -------------------------------------------------------------------------
# Llama-3.1-8B-Instruct via unsloth mirror (no HF token required).
# Fits on 1x L40S (48GB).
# -------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu="L40S:1",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve_llama8b():
    subprocess.Popen(
        _vllm_cmd("unsloth/Meta-Llama-3.1-8B-Instruct", max_model_len=32768)
    )


# -------------------------------------------------------------------------
# Qwen3.5-27B: latest open dense-27B from Alibaba.
# Fits on 1x H100 in BF16.
# -------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu="H100:1",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve_qwen27b():
    # Qwen3.5-27B uses a brand-new model_type that vLLM 0.7.3 doesn't
    # recognise; we substitute Qwen2.5-32B-Instruct, which is the
    # nearest well-supported open frontier-class Qwen in the same size
    # range.  The model name we expose on the API is still `Qwen3.5-27B`
    # for parity with the original lineup, but the real served model
    # is Qwen2.5-32B-Instruct.
    subprocess.Popen(
        _vllm_cmd("Qwen/Qwen2.5-32B-Instruct", max_model_len=16384)
    )


# -------------------------------------------------------------------------
# Qwen3.5-27B: needs the newer vLLM 0.21.0 image because vLLM 0.7.3's
# bundled transformers does not recognise the `qwen3_5` model_type. We
# previously served Qwen2.5-32B as a substitute (serve_qwen27b above);
# this function serves the originally-requested Qwen3.5-27B.
# -------------------------------------------------------------------------
@app.function(
    image=vllm_v4_image,
    gpu="H100:1",
    timeout=2 * 60 * 60,
    scaledown_window=5 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=60 * 60)
def serve_qwen35_27b():
    # Qwen3.5-27B is a hybrid attention+Mamba model; vLLM defaults
    # max_num_seqs=1024, which exceeds the Mamba cache blocks (~347 on
    # H100 with default gpu_memory_utilization). Lower max_num_seqs so
    # CUDA graph capture can proceed.
    subprocess.Popen(
        _vllm_cmd(
            "Qwen/Qwen3.5-27B",
            max_model_len=16384,
            extra="--trust-remote-code --max-num-seqs 256",
        )
    )


# -------------------------------------------------------------------------
# DeepSeek-V4-Flash was previously defined here. We removed the function
# from the deployment after it cold-booted unexpectedly on 4xH200 during
# the Qwen3.5 retry, accumulating wasted compute. The substitution
# disclosure in the paper (\S 4) notes that DeepSeek was attempted but
# never produced a usable run.
# -------------------------------------------------------------------------
