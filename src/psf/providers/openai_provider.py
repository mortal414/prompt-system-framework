from __future__ import annotations

from typing import Any

from ..exceptions import ProviderError
from ..models import ModelConfig
from .base import LLMProvider, LLMResult


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client or self._build_default_client()

    @staticmethod
    def _build_default_client() -> Any:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only if deps missing
            raise ProviderError(
                "openai package is not installed. Install 'openai' to use the OpenAI provider"
            ) from exc
        return OpenAI()

    def complete(self, prompt: str, config: ModelConfig) -> LLMResult:
        params: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        params.update(config.extra_params)
        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:  # pragma: no cover - network errors not exercised in tests
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        return _response_to_result(response)


def _response_to_result(response: Any) -> LLMResult:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None) if choice else None
    content = getattr(message, "content", "")
    raw_text = _coerce_content_to_text(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    metadata = {
        "response_id": getattr(response, "id", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
    }
    return LLMResult(
        raw_text=raw_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=getattr(response, "model", None),
        provider_metadata=metadata,
    )


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text_value = getattr(part, "text", None)
            if text_value:
                parts.append(text_value)
        return "".join(parts)
    return str(content or "")
