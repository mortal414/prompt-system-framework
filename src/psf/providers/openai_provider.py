from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


class OpenAIBatchProvider(LLMProvider):
    """
    Batch-enabled provider that submits chat completions via the OpenAI Batch API.

    Requests are grouped by (provider, model) and flushed when enough requests are queued
    to fill a batch, or when max_wait_seconds passes without any new requests (idle flush).
    """

    name = "openai"

    def __init__(
        self,
        *,
        batch_size: int,
        client: Any | None = None,
        logger: logging.Logger | None = None,
        max_wait_seconds: float = 0.5,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        if batch_size < 1:
            raise ProviderError("batch_size must be >= 1")
        self._client = client or OpenAIProvider._build_default_client()
        self._batch_size = batch_size
        self._logger = logger or logging.getLogger("psf.providers.openai.batch")
        self._max_wait = max_wait_seconds
        self._poll_interval = poll_interval_seconds
        self._locks: Dict[Tuple[str, str], threading.Condition] = {}
        self._queues: Dict[Tuple[str, str], List["_PendingRequest"]] = {}
        self._last_enqueue_at: Dict[Tuple[str, str], float] = {}

    def complete(self, prompt: str, config: ModelConfig) -> LLMResult:
        key = (config.provider or "openai", config.model)
        condition = self._locks.setdefault(key, threading.Condition())
        request = _PendingRequest(prompt=prompt, config=config)
        with condition:
            queue = self._queues.setdefault(key, [])
            queue.append(request)
            self._last_enqueue_at[key] = request.enqueued_at
            condition.notify_all()

            while not request.completed:
                queue = self._queues.setdefault(key, [])
                if not queue:
                    condition.wait(timeout=self._max_wait)
                    continue

                should_flush = False
                reason = "unknown"
                if len(queue) >= self._batch_size:
                    should_flush = True
                    reason = "size"
                elif queue[0] is request:
                    last_enqueue = self._last_enqueue_at.get(key, request.enqueued_at)
                    idle_for = time.monotonic() - last_enqueue
                    if idle_for >= self._max_wait:
                        should_flush = True
                        reason = "max_wait"

                if should_flush:
                    batch_len = min(self._batch_size, len(queue))
                    batch = queue[:batch_len]
                    del queue[:batch_len]
                    condition.release()
                    try:
                        self._flush_batch(key, batch, reason=reason)
                    finally:
                        condition.acquire()
                    condition.notify_all()
                    continue

                timeout = self._max_wait
                if queue[0] is request:
                    last_enqueue = self._last_enqueue_at.get(key, request.enqueued_at)
                    idle_for = time.monotonic() - last_enqueue
                    timeout = max(0.0, self._max_wait - idle_for)
                condition.wait(timeout=timeout)

        if request.error:
            raise request.error
        if not request.result:
            raise ProviderError("Batch request completed without a result")
        return request.result

    def _flush_batch(
        self,
        key: Tuple[str, str],
        batch: List["_PendingRequest"],
        *,
        reason: str,
    ) -> None:
        requests: list[dict[str, Any]] = []
        for req in batch:
            body: dict[str, Any] = {
                "model": req.config.model,
                "messages": [{"role": "user", "content": req.prompt}],
            }
            if req.config.temperature is not None:
                body["temperature"] = req.config.temperature
            if req.config.max_tokens is not None:
                body["max_tokens"] = req.config.max_tokens
            if req.config.extra_params:
                body.update(req.config.extra_params)
            requests.append(
                {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )

        started = time.monotonic()
        provider_name, model = key
        self._logger.info(
            "Submitting OpenAI batch (%s): model=%s requests=%s reason=%s",
            provider_name,
            model,
            len(batch),
            reason,
        )
        try:
            input_file_id = self._upload_batch_file(requests)
            batch_job = self._client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_id = getattr(batch_job, "id", None)
            if batch_id:
                self._logger.info(
                    "OpenAI batch created (%s): id=%s model=%s requests=%s",
                    provider_name,
                    batch_id,
                    model,
                    len(batch),
                )
            batch_job = self._wait_for_batch(batch_job.id)
            if getattr(batch_job, "status", None) != "completed":
                raise ProviderError(f"Batch job failed with status '{getattr(batch_job, 'status', None)}'")
            output_file_id = getattr(batch_job, "output_file_id", None)
            if not output_file_id:
                raise ProviderError("Batch completed without output_file_id")
            duration = time.monotonic() - started
            self._logger.info(
                "OpenAI batch completed (%s): id=%s status=%s model=%s requests=%s duration=%.1fs",
                provider_name,
                getattr(batch_job, "id", None),
                getattr(batch_job, "status", None),
                model,
                len(batch),
                duration,
            )
            output_lines = self._download_batch_output(output_file_id)
            responses = {entry["custom_id"]: entry for entry in output_lines}
            for req in batch:
                entry = responses.get(req.custom_id)
                if not entry:
                    req.error = ProviderError(f"No response for custom_id {req.custom_id}")
                else:
                    try:
                        result = _batch_entry_to_result(entry)
                        req.result = result
                    except Exception as exc:  # pragma: no cover - parsing failures rare
                        req.error = ProviderError(str(exc))
                req.completed = True
                req.event.set()
        except Exception as exc:  # pragma: no cover - network/batch errors
            for req in batch:
                req.error = ProviderError(str(exc))
                req.completed = True
                req.event.set()

    def _upload_batch_file(self, requests: list[dict[str, Any]]) -> str:
        fd, path_str = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        path = Path(path_str)
        try:
            with path.open("w", encoding="utf-8") as fh:
                for item in requests:
                    fh.write(json.dumps(item) + "\n")
            with path.open("rb") as fh:
                upload = self._client.files.create(file=("batch.jsonl", fh), purpose="batch")
            file_id = getattr(upload, "id", None)
            if not file_id:
                raise ProviderError("Failed to upload batch input file")
            return file_id
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _wait_for_batch(self, batch_id: str) -> Any:
        while True:
            batch_job = self._client.batches.retrieve(batch_id)
            status = getattr(batch_job, "status", None)
            if status in {"completed", "failed", "canceled"}:
                return batch_job
            time.sleep(self._poll_interval)

    def _download_batch_output(self, file_id: str) -> list[dict[str, Any]]:
        # Handle client differences: some return a response with .text, others with .read().
        file_content = self._client.files.content(file_id)
        if hasattr(file_content, "text"):
            raw = file_content.text
        elif hasattr(file_content, "read"):
            raw = file_content.read().decode("utf-8")
        else:
            raw = str(file_content)
        lines = raw.splitlines()
        entries: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries


class _PendingRequest:
    def __init__(self, *, prompt: str, config: ModelConfig) -> None:
        self.prompt = prompt
        self.config = config
        self.custom_id = secrets.token_hex(8)
        self.enqueued_at = time.monotonic()
        self.result: LLMResult | None = None
        self.error: ProviderError | None = None
        self.completed: bool = False
        self.event = threading.Event()


def _batch_entry_to_result(entry: dict[str, Any]) -> LLMResult:
    response = entry.get("response", {})
    body = response.get("body", {})
    choices = body.get("choices") or []
    message = choices[0].get("message") if choices else {}
    content = message.get("content", "")
    raw_text = _coerce_content_to_text(content)
    usage = body.get("usage") or {}
    return LLMResult(
        raw_text=raw_text,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        model=body.get("model"),
        provider_metadata={"batch_custom_id": entry.get("custom_id"), "finish_reason": choices[0].get("finish_reason") if choices else None},
    )
