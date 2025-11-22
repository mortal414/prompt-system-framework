from __future__ import annotations


class PromptSystemError(Exception):
    """Base exception for the Prompt System Framework."""


class ConfigError(PromptSystemError):
    """Raised when a system definition fails validation."""


class ExecutionError(PromptSystemError):
    """Raised when execution fails at runtime."""


class RecoverableExecutionError(ExecutionError):
    """Execution error types that allow retries."""


class NodeExecutionError(ExecutionError):
    """Execution error that carries contextual node data (e.g., raw outputs)."""

    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ProviderError(PromptSystemError):
    """Raised when an LLM provider returns an error."""
