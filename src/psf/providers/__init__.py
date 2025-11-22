from __future__ import annotations

from .base import LLMProvider, LLMResult, ProviderRegistry

_default_registry = ProviderRegistry()


def get_provider(name: str) -> LLMProvider:
    _ensure_defaults(name)
    return _default_registry.get(name)


def registry() -> ProviderRegistry:
    return _default_registry


def _ensure_defaults(provider_name: str) -> None:
    if provider_name == "openai" and provider_name not in _default_registry.available():
        from .openai_provider import OpenAIProvider

        _default_registry.register("openai", OpenAIProvider())


__all__ = [
    "LLMProvider",
    "LLMResult",
    "ProviderRegistry",
    "OpenAIProvider",
    "get_provider",
    "registry",
]
