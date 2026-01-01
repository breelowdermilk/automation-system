"""
Model Router - Routes requests to Claude Code CLI, OpenRouter, or Local models
"""

import subprocess
import json
import time
import os
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import httpx


class Backend(Enum):
    CLAUDE_CODE = "claude_code"
    OPENROUTER = "openrouter"
    LOCAL = "local"


@dataclass
class ModelConfig:
    backend: Backend
    display_name: str
    model_id: str
    supports_vision: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


# Model registry
MODELS: Dict[str, ModelConfig] = {
    # Claude Code CLI models (subscription - no marginal cost)
    "sonnet": ModelConfig(
        backend=Backend.CLAUDE_CODE,
        display_name="Claude Sonnet",
        model_id="sonnet",
        supports_vision=True,
    ),
    "opus": ModelConfig(
        backend=Backend.CLAUDE_CODE,
        display_name="Claude Opus",
        model_id="opus",
        supports_vision=True,
    ),
    "haiku": ModelConfig(
        backend=Backend.CLAUDE_CODE,
        display_name="Claude Haiku",
        model_id="haiku",
        supports_vision=True,
    ),
    # OpenRouter models (pay per use)
    "gemini-flash": ModelConfig(
        backend=Backend.OPENROUTER,
        display_name="Gemini 1.5 Flash",
        model_id="google/gemini-flash-1.5",
        supports_vision=True,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
    "gpt4o-mini": ModelConfig(
        backend=Backend.OPENROUTER,
        display_name="GPT-4o Mini",
        model_id="openai/gpt-4o-mini",
        supports_vision=True,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "claude-haiku-openrouter": ModelConfig(
        backend=Backend.OPENROUTER,
        display_name="Claude Haiku (API)",
        model_id="anthropic/claude-3-haiku-20240307",
        supports_vision=True,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
    ),
    # Local models (free)
    "llama": ModelConfig(
        backend=Backend.LOCAL,
        display_name="Llama 3.2",
        model_id="llama3.2",
        supports_vision=False,
    ),
    "qwen": ModelConfig(
        backend=Backend.LOCAL,
        display_name="Qwen 2.5 7B",
        model_id="qwen2.5:7b",
        supports_vision=False,
    ),
}


@dataclass
class ModelResponse:
    text: str
    model: str
    backend: str
    latency_ms: int
    success: bool = True
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = field(default_factory=dict)


class ModelRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openrouter_api_key = os.environ.get(
            config.get("models", {}).get("openrouter", {}).get("api_key_env", "OPENROUTER_API_KEY")
        )
        self.ollama_host = config.get("models", {}).get("local", {}).get("ollama_host", "http://localhost:11434")
        self.timeout = config.get("models", {}).get("claude_code", {}).get("timeout_seconds", 60)

    async def process(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        model_name: str = "sonnet",
        task_type: str = "default",
    ) -> ModelResponse:
        """
        Route request to appropriate backend based on model.
        """
        if model_name not in MODELS:
            return ModelResponse(
                text="",
                model=model_name,
                backend="unknown",
                latency_ms=0,
                success=False,
                error=f"Unknown model: {model_name}",
            )

        model_config = MODELS[model_name]

        # Check vision support
        if image_path and not model_config.supports_vision:
            return ModelResponse(
                text="",
                model=model_name,
                backend=model_config.backend.value,
                latency_ms=0,
                success=False,
                error=f"Model {model_name} does not support vision",
            )

        start_time = time.time()

        try:
            if model_config.backend == Backend.CLAUDE_CODE:
                response = await self._call_claude_code(prompt, image_path, model_config)
            elif model_config.backend == Backend.OPENROUTER:
                response = await self._call_openrouter(prompt, image_path, model_config)
            elif model_config.backend == Backend.LOCAL:
                response = await self._call_local(prompt, model_config)
            else:
                raise ValueError(f"Unknown backend: {model_config.backend}")

            response.latency_ms = int((time.time() - start_time) * 1000)
            return response

        except Exception as e:
            return ModelResponse(
                text="",
                model=model_name,
                backend=model_config.backend.value,
                latency_ms=int((time.time() - start_time) * 1000),
                success=False,
                error=str(e),
            )

    async def _call_claude_code(
        self, prompt: str, image_path: Optional[str], model_config: ModelConfig
    ) -> ModelResponse:
        """
        Shell out to claude CLI with --print flag.
        Uses subscription, no API cost.
        """
        # Build the prompt with image reference if needed
        full_prompt = prompt
        if image_path:
            full_prompt = f"{prompt}\n\nAnalyze this image file: {image_path}"

        cmd = [
            "claude",
            "-p",
            full_prompt,
            "--print",
            "--output-format", "json",
            "--model", model_config.model_id,
        ]

        # Run in executor to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            ),
        )

        if result.returncode != 0:
            return ModelResponse(
                text="",
                model=model_config.model_id,
                backend=Backend.CLAUDE_CODE.value,
                latency_ms=0,
                success=False,
                error=result.stderr or "Claude Code CLI failed",
            )

        # Parse JSON response
        try:
            response_data = json.loads(result.stdout)
            text = response_data.get("result", result.stdout)

            return ModelResponse(
                text=text,
                model=model_config.model_id,
                backend=Backend.CLAUDE_CODE.value,
                latency_ms=0,
                success=True,
                input_tokens=response_data.get("usage", {}).get("input_tokens"),
                output_tokens=response_data.get("usage", {}).get("output_tokens"),
                cost_usd=None,  # Subscription, no marginal cost
                raw_response=response_data,
            )
        except json.JSONDecodeError:
            # If not JSON, just use stdout directly
            return ModelResponse(
                text=result.stdout.strip(),
                model=model_config.model_id,
                backend=Backend.CLAUDE_CODE.value,
                latency_ms=0,
                success=True,
            )

    async def _call_openrouter(
        self, prompt: str, image_path: Optional[str], model_config: ModelConfig
    ) -> ModelResponse:
        """
        Call OpenRouter API. Tracks cost.
        """
        if not self.openrouter_api_key:
            return ModelResponse(
                text="",
                model=model_config.model_id,
                backend=Backend.OPENROUTER.value,
                latency_ms=0,
                success=False,
                error="OPENROUTER_API_KEY not set",
            )

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8420",
            "X-Title": "Automation Control Panel",
        }

        # Build messages
        content = [{"type": "text", "text": prompt}]

        if image_path:
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Detect mime type
            ext = image_path.lower().split(".")[-1]
            mime_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
            mime_type = mime_types.get(ext, "image/png")

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            })

        payload = {
            "model": model_config.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Calculate cost
        cost = (
            (input_tokens / 1000) * model_config.cost_per_1k_input +
            (output_tokens / 1000) * model_config.cost_per_1k_output
        )

        return ModelResponse(
            text=text,
            model=model_config.model_id,
            backend=Backend.OPENROUTER.value,
            latency_ms=0,
            success=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            raw_response=data,
        )

    async def _call_local(self, prompt: str, model_config: ModelConfig) -> ModelResponse:
        """
        Call local Ollama. No vision support.
        """
        payload = {
            "model": model_config.model_id,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        return ModelResponse(
            text=data.get("response", ""),
            model=model_config.model_id,
            backend=Backend.LOCAL.value,
            latency_ms=0,
            success=True,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            cost_usd=0.0,
            raw_response=data,
        )

    def get_model_for_task(self, task_type: str) -> str:
        """Get the configured model for a task type."""
        task_defaults = self.config.get("task_defaults", {})
        task_config = task_defaults.get(task_type, {})
        return task_config.get("model", "sonnet")

    def get_fallback_model(self, task_type: str) -> Optional[str]:
        """Get the fallback model for a task type."""
        task_defaults = self.config.get("task_defaults", {})
        task_config = task_defaults.get(task_type, {})
        return task_config.get("fallback")

    async def process_with_fallback(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        task_type: str = "default",
    ) -> ModelResponse:
        """Process with automatic fallback on failure."""
        primary_model = self.get_model_for_task(task_type)
        response = await self.process(prompt, image_path, primary_model, task_type)

        if not response.success:
            fallback_model = self.get_fallback_model(task_type)
            if fallback_model:
                response = await self.process(prompt, image_path, fallback_model, task_type)

        return response
