from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

from ..exceptions import ProviderError
from ..models import ModelConfig


@dataclass
class LLMResult:
    raw_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    provider_metadata: Dict[str, object] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str, config: ModelConfig) -> LLMResult:
        ...


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderError(f"Unknown provider '{name}'") from exc

    def available(self) -> Dict[str, LLMProvider]:
        return dict(self._providers)
