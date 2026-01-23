from __future__ import annotations

import argparse
import json
import logging
import random
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import chess

from chess_data import GeneratedPosition, generate_illegal_san, generate_illegal_uci, play_random_plies
from psf.config_loader import load_system_definition
from psf.engine import (
    ContextStore,
    ExecutionEngine,
    ExecutionResult,
    NodeRunRecord,
    _build_metrics,
    _handle_structured_output,
    _normalize_system_input,
)
from psf.exceptions import ExecutionError, ProviderError, RecoverableExecutionError
from psf.logging import RunLogger
from psf.models import ModelConfig, NodeDef, SystemDef
from psf.providers import get_provider
from psf.providers.openai_provider import OpenAIBatchProvider, _batch_entry_to_result


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    status: str
    final_output: Optional[dict[str, Any]]
    error: Optional[str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class GeneratorSystem:
    system_path: Path
    system_def: SystemDef
    move_notation: str
    move_output_field: str


@dataclass(frozen=True)
class ValidatorSystem:
    system_path: Path
    system_def: SystemDef
    move_notation: str


@dataclass(frozen=True)
class PositionOutcome:
    index: int
    record: dict[str, Any]
    token_totals: dict[str, int]
    validator_baseline: dict[str, dict[str, int]]
    generator: dict[str, dict[str, int]]
    validator_on_generator: dict[str, dict[str, int]]
    pipeline: dict[str, int]


@dataclass(frozen=True)
class SystemTaskOutcome:
    index: int
    kind: str
    system_id: str
    entry: dict[str, Any]
    token_totals: dict[str, int]
    counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class _BatchTaskSpec:
    task_id: int
    kind: str
    index: int
    system_def: SystemDef
    system_path: Path
    runs_root: Path
    run_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _BatchRunState:
    spec: _BatchTaskSpec
    engine: ExecutionEngine
    context: ContextStore
    run_logger: RunLogger
    current_node_id: str
    current_node_attempt: int = 0
    records: list[NodeRunRecord] = field(default_factory=list)
    models_used: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _PendingNodeCall:
    state_id: int
    node: NodeDef
    prompt: str
    model_config: ModelConfig
    attempt: int


@dataclass
class _BatchJobInfo:
    batch_id: str
    provider: str
    model: str
    request_count: int
    submitted_at: float
    custom_ids: list[str]


def _setup_logging(*, level: str, log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("psf.chess.benchmark")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _generate_benchmark_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"chess_random_{timestamp}_{suffix}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _oracle_san_legality(*, fen: str, move_san: str) -> tuple[bool, Optional[str]]:
    board = chess.Board(fen)
    try:
        parsed = board.parse_san(move_san)
    except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError) as exc:
        return False, str(exc)
    return board.is_legal(parsed), None


def _oracle_uci_legality(*, fen: str, move_uci: str) -> tuple[bool, Optional[str]]:
    board = chess.Board(fen)
    try:
        parsed = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        return False, str(exc)
    return board.is_legal(parsed), None


def _execute_with_run_logger(
    *,
    system_def: SystemDef,
    system_path: Path,
    payload: dict[str, Any],
    runs_root: Path,
    run_id: str,
    provider_resolver: Callable[[str], Any] | None = None,
) -> RunArtifacts:
    try:
        local_engine = ExecutionEngine(system_def, provider_resolver=provider_resolver)
        result = local_engine.run(payload)
    except ExecutionError as exc:
        result = ExecutionResult(
            status="failure",
            final_output=None,
            context={"runtime": {}},
            node_results=[],
            metrics={},
            error=str(exc),
        )

    runs_root.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(runs_root, system_def, run_id=run_id, use_system_subdir=False)
    run_logger.snapshot_system()
    run_logger.snapshot_input(payload)
    for record in result.node_results:
        run_logger.log_node(record)
    run_logger.finalize(result, config_path=system_path)
    return RunArtifacts(
        run_dir=run_logger.run_dir,
        status=result.status,
        final_output=result.final_output,
        error=result.error,
        metrics=result.metrics,
    )


def _execute_task_specs_openai_batch(
    *,
    task_specs: list[_BatchTaskSpec],
    openai_batch_provider: OpenAIBatchProvider,
    batch_size: int,
    max_wait_seconds: float,
    poll_interval_seconds: float,
    logger: logging.Logger,
) -> Any:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    pending_results: list[tuple[_BatchTaskSpec, RunArtifacts]] = []

    states: dict[int, _BatchRunState] = {}
    ready_by_group: dict[tuple[str, str], list[_PendingNodeCall]] = {}
    last_enqueue_at: dict[tuple[str, str], float] = {}

    def _finalize_state(state_id: int, *, status: str, error_message: str | None, final_output: dict[str, Any] | None) -> None:
        state = states.pop(state_id)
        finished_at = datetime.now(timezone.utc)
        state.context.set_runtime("finished_at", finished_at.isoformat())
        state.context.set_runtime("status", status)
        metrics = _build_metrics(state.records, state.models_used)
        result = ExecutionResult(
            status=status,
            final_output=final_output,
            context=state.context.data,
            node_results=state.records,
            metrics=metrics,
            error=error_message,
        )
        state.run_logger.finalize(result, config_path=state.spec.system_path)
        pending_results.append(
            (
                state.spec,
                RunArtifacts(
                    run_dir=state.run_logger.run_dir,
                    status=result.status,
                    final_output=result.final_output,
                    error=result.error,
                    metrics=result.metrics,
                ),
            )
        )

    def _fail_state_with_record(state_id: int, *, node: NodeDef, prompt: str, model_config: ModelConfig, error_message: str, raw_output: str | None) -> None:
        state = states[state_id]
        record = NodeRunRecord(node_id=node.id)
        record.prompt = prompt
        record.provider = model_config.provider
        record.model = model_config.model
        record.raw_output = raw_output
        record.error = error_message
        state.records.append(record)
        state.run_logger.log_node(record)
        _finalize_state(state_id, status="failure", error_message=error_message, final_output=None)

    def _enqueue_current_node(state_id: int) -> None:
        state = states[state_id]
        node = state.spec.system_def.get_node(state.current_node_id)
        try:
            inputs = state.engine._resolve_inputs(node, state.context)
            prompt = state.engine._render_prompt(node, inputs)
            model_config = state.engine._build_model_config(node)
        except ExecutionError as exc:
            record = NodeRunRecord(node_id=node.id)
            record.error = str(exc)
            record.provider = node.provider
            record.model = node.model
            state.records.append(record)
            state.run_logger.log_node(record)
            _finalize_state(state_id, status="failure", error_message=str(exc), final_output=None)
            return

        attempt = state.current_node_attempt + 1
        state.current_node_attempt = attempt
        group_key = (model_config.provider or "openai", model_config.model)
        ready_by_group.setdefault(group_key, []).append(
            _PendingNodeCall(
                state_id=state_id,
                node=node,
                prompt=prompt,
                model_config=model_config,
                attempt=attempt,
            )
        )
        last_enqueue_at[group_key] = time.monotonic()

    # Initialize run states (and queue their first node call).
    for spec in task_specs:
        spec.runs_root.mkdir(parents=True, exist_ok=True)
        run_logger = RunLogger(spec.runs_root, spec.system_def, run_id=spec.run_id, use_system_subdir=False)
        run_logger.snapshot_system()
        run_logger.snapshot_input(spec.payload)
        try:
            normalized = _normalize_system_input(spec.system_def, spec.payload)
        except ExecutionError as exc:
            result = ExecutionResult(
                status="failure",
                final_output=None,
                context={"runtime": {}},
                node_results=[],
                metrics={},
                error=str(exc),
            )
            run_logger.finalize(result, config_path=spec.system_path)
            pending_results.append(
                (
                    spec,
                    RunArtifacts(
                        run_dir=run_logger.run_dir,
                        status=result.status,
                        final_output=result.final_output,
                        error=result.error,
                        metrics=result.metrics,
                    ),
                )
            )
            while pending_results:
                yield pending_results.pop()
            continue

        engine = ExecutionEngine(spec.system_def)
        context = ContextStore(normalized)
        context.set_runtime("started_at", datetime.now(timezone.utc).isoformat())
        states[spec.task_id] = _BatchRunState(
            spec=spec,
            engine=engine,
            context=context,
            run_logger=run_logger,
            current_node_id=spec.system_def.entry_node_id,
        )
        _enqueue_current_node(spec.task_id)
        while pending_results:
            yield pending_results.pop()

    inflight_calls: dict[str, _PendingNodeCall] = {}
    inflight_jobs: dict[str, _BatchJobInfo] = {}

    def _submit_ready(now: float) -> None:
        nonlocal inflight_calls, inflight_jobs
        for group_key, queue in list(ready_by_group.items()):
            provider_name, model = group_key
            if provider_name != "openai":
                for call in list(queue):
                    state_id = call.state_id
                    if state_id not in states:
                        continue
                    _fail_state_with_record(
                        state_id,
                        node=call.node,
                        prompt=call.prompt,
                        model_config=call.model_config,
                        error_message=f"OpenAI batch mode does not support provider '{provider_name}'",
                        raw_output=None,
                    )
                ready_by_group[group_key] = []
                continue

            while len(queue) >= batch_size:
                chunk = queue[:batch_size]
                del queue[:batch_size]
                _submit_chunk(group_key, chunk, reason="size")

            if queue:
                last_enq = last_enqueue_at.get(group_key, now)
                idle_for = now - last_enq
                if idle_for >= max_wait_seconds:
                    chunk = list(queue)
                    queue.clear()
                    _submit_chunk(group_key, chunk, reason="max_wait")

    def _submit_chunk(group_key: tuple[str, str], chunk: list[_PendingNodeCall], *, reason: str) -> None:
        provider_name, model = group_key
        requests: list[dict[str, Any]] = []
        custom_ids: list[str] = []
        for call in chunk:
            custom_id = secrets.token_hex(8)
            inflight_calls[custom_id] = call
            custom_ids.append(custom_id)
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": call.prompt}],
            }
            if call.model_config.temperature is not None:
                body["temperature"] = call.model_config.temperature
            if call.model_config.max_tokens is not None:
                body["max_tokens"] = call.model_config.max_tokens
            if call.model_config.extra_params:
                body.update(call.model_config.extra_params)
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )

        started = time.monotonic()
        logger.info(
            "Submitting OpenAI batch (%s): model=%s requests=%s reason=%s",
            provider_name,
            model,
            len(chunk),
            reason,
        )
        input_file_id = openai_batch_provider._upload_batch_file(requests)
        batch_job = openai_batch_provider._client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = getattr(batch_job, "id", None)
        if not batch_id:
            raise ProviderError("OpenAI batch creation returned no id")
        logger.info(
            "OpenAI batch created (%s): id=%s model=%s requests=%s",
            provider_name,
            batch_id,
            model,
            len(chunk),
        )
        inflight_jobs[batch_id] = _BatchJobInfo(
            batch_id=batch_id,
            provider=provider_name,
            model=model,
            request_count=len(chunk),
            submitted_at=started,
            custom_ids=custom_ids,
        )

    while states or inflight_jobs or any(ready_by_group.values()):
        now = time.monotonic()
        before_jobs = len(inflight_jobs)
        _submit_ready(now)
        submitted_any = len(inflight_jobs) > before_jobs
        while pending_results:
            yield pending_results.pop()

        finished_batch_ids: list[str] = []
        for batch_id, job_info in list(inflight_jobs.items()):
            batch_job = openai_batch_provider._client.batches.retrieve(batch_id)
            status = getattr(batch_job, "status", None)
            if status not in {"completed", "failed", "canceled"}:
                continue
            finished_batch_ids.append(batch_id)

            duration = time.monotonic() - job_info.submitted_at
            logger.info(
                "OpenAI batch completed (%s): id=%s status=%s model=%s requests=%s duration=%.1fs",
                job_info.provider,
                batch_id,
                status,
                job_info.model,
                job_info.request_count,
                duration,
            )

            output_file_id = getattr(batch_job, "output_file_id", None)
            output_lines: list[dict[str, Any]] = []
            if status == "completed" and output_file_id:
                output_lines = openai_batch_provider._download_batch_output(output_file_id)
            responses = {entry.get("custom_id"): entry for entry in output_lines if entry.get("custom_id")}

            for custom_id in job_info.custom_ids:
                call = inflight_calls.pop(custom_id, None)
                if call is None:
                    continue
                state_id = call.state_id
                if state_id not in states:
                    continue
                state = states[state_id]
                node = call.node
                prompt = call.prompt
                model_config = call.model_config

                entry = responses.get(custom_id)
                if status != "completed" or entry is None:
                    if state.current_node_attempt <= node.max_retries:
                        _enqueue_current_node(state_id)
                        continue
                    _fail_state_with_record(
                        state_id,
                        node=node,
                        prompt=prompt,
                        model_config=model_config,
                        error_message=f"OpenAI batch request failed (batch_status={status})",
                        raw_output=None,
                    )
                    continue

                response_obj = entry.get("response") or {}
                status_code = response_obj.get("status_code")
                if isinstance(status_code, int) and status_code >= 400:
                    body = response_obj.get("body")
                    error_text = json.dumps(body, ensure_ascii=False) if body is not None else str(body)
                    if state.current_node_attempt <= node.max_retries:
                        _enqueue_current_node(state_id)
                        continue
                    _fail_state_with_record(
                        state_id,
                        node=node,
                        prompt=prompt,
                        model_config=model_config,
                        error_message=f"OpenAI request failed: {error_text}",
                        raw_output=None,
                    )
                    continue

                try:
                    llm_result = _batch_entry_to_result(entry)
                except Exception as exc:
                    if state.current_node_attempt <= node.max_retries:
                        _enqueue_current_node(state_id)
                        continue
                    _fail_state_with_record(
                        state_id,
                        node=node,
                        prompt=prompt,
                        model_config=model_config,
                        error_message=str(exc),
                        raw_output=None,
                    )
                    continue

                raw_text = llm_result.raw_text
                try:
                    structured_output = _handle_structured_output(node, raw_text)
                except RecoverableExecutionError as exc:
                    if state.current_node_attempt <= node.max_retries:
                        _enqueue_current_node(state_id)
                        continue
                    _fail_state_with_record(
                        state_id,
                        node=node,
                        prompt=prompt,
                        model_config=model_config,
                        error_message=str(exc),
                        raw_output=raw_text,
                    )
                    continue

                record = NodeRunRecord(node_id=node.id)
                record.prompt = prompt
                record.raw_output = raw_text
                record.provider = model_config.provider
                record.model = model_config.model
                if structured_output is not None:
                    record.parsed_output = structured_output
                record.prompt_tokens = llm_result.prompt_tokens
                record.completion_tokens = llm_result.completion_tokens
                record.total_tokens = llm_result.total_tokens
                record.provider_metadata = llm_result.provider_metadata
                record.duration_ms = duration * 1000.0
                state.records.append(record)
                state.run_logger.log_node(record)
                state.context.set_node_result(
                    node.id,
                    raw_output=raw_text,
                    parsed_output=structured_output,
                    provider=record.provider,
                    model=record.model,
                    tokens={
                        "prompt": record.prompt_tokens,
                        "completion": record.completion_tokens,
                        "total": record.total_tokens,
                    },
                    provider_metadata=record.provider_metadata or None,
                )
                if record.model:
                    state.models_used.add(record.model)

                state.current_node_attempt = 0
                loop_decision = state.engine._handle_loop_control(node, state.context)
                if getattr(loop_decision, "exhausted", False):
                    next_node = state.engine._select_next(
                        node.routing.on_loop_exhausted, structured_output, state.context
                    )
                    if next_node is None:
                        _finalize_state(
                            state_id,
                            status="failure",
                            error_message=f"Loop exhausted for node '{node.id}' without an on_loop_exhausted route",
                            final_output=None,
                        )
                        continue
                    state.current_node_id = next_node
                    _enqueue_current_node(state_id)
                    continue
                next_node_id = getattr(loop_decision, "next_node_id", None)
                if next_node_id:
                    state.current_node_id = next_node_id
                    _enqueue_current_node(state_id)
                    continue
                if node.terminal:
                    try:
                        final_output = state.engine._build_system_output(state.context)
                        _finalize_state(state_id, status="success", error_message=None, final_output=final_output)
                    except ExecutionError as exc:
                        _finalize_state(state_id, status="failure", error_message=str(exc), final_output=None)
                    continue
                next_node = state.engine._select_next(node.routing.on_success, structured_output, state.context)
                if next_node is None:
                    _finalize_state(
                        state_id,
                        status="failure",
                        error_message=f"No routing rule matched after node '{node.id}'",
                        final_output=None,
                    )
                    continue
                state.current_node_id = next_node
                _enqueue_current_node(state_id)
                while pending_results:
                    yield pending_results.pop()

        for batch_id in finished_batch_ids:
            inflight_jobs.pop(batch_id, None)

        while pending_results:
            yield pending_results.pop()

        if not finished_batch_ids and not submitted_any:
            if inflight_jobs:
                time.sleep(poll_interval_seconds)
            elif any(ready_by_group.values()):
                next_flush_in = poll_interval_seconds
                for group_key, queue in ready_by_group.items():
                    if not queue:
                        continue
                    last_enq = last_enqueue_at.get(group_key, now)
                    remaining = max_wait_seconds - (now - last_enq)
                    if remaining > 0:
                        next_flush_in = min(next_flush_in, remaining)
                time.sleep(max(0.01, min(poll_interval_seconds, next_flush_in)))

    while pending_results:
        yield pending_results.pop()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"legal", "true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"illegal", "false", "f", "0", "no", "n"}:
            return False
    raise ValueError("expected a boolean-like label (legal/illegal, true/false, 1/0)")


def _load_system(path: Path) -> SystemDef:
    return load_system_definition(path)


def _infer_move_notation(*, system_def: SystemDef, system_path: Path, default: str) -> str:
    try:
        output_fields = set(system_def.system_output.fields.keys())
    except Exception:
        output_fields = set()
    if "move_uci" in output_fields and "move_san" not in output_fields:
        return "uci"
    if "move_san" in output_fields and "move_uci" not in output_fields:
        return "san"
    haystack = f"{system_def.id} {system_path.name}".lower()
    if "uci" in haystack:
        return "uci"
    if "san" in haystack:
        return "san"
    return default


def _infer_move_output_field(*, system_def: SystemDef, preferred_field: str) -> str:
    fields = set(system_def.system_output.fields.keys())
    if preferred_field in fields:
        return preferred_field
    for candidate in ("move", "move_san", "move_uci"):
        if candidate in fields:
            return candidate
    move_like = sorted(name for name in fields if name.startswith("move"))
    if len(move_like) == 1:
        return move_like[0]
    return preferred_field


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-position chess benchmark suite (SAN-first). "
        "Generates N random positions and runs each once per system, with full per-node run logs."
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of random positions (games) to generate")
    parser.add_argument("--plies", type=int, default=8, help="Random legal plies to play before sampling a position")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic RNG seed for position generation")
    parser.add_argument(
        "--illegal-ratio",
        type=float,
        default=0.5,
        help="Fraction of validator-baseline cases to label illegal (0-1)",
    )
    parser.add_argument(
        "--generator-system",
        type=Path,
        action="append",
        default=[],
        help="Move generator system YAML/JSON (repeatable).",
    )
    parser.add_argument(
        "--generator-output-field",
        type=str,
        default="move_san",
        help="System output field containing the generated move (SAN or UCI)",
    )
    parser.add_argument(
        "--validator-system",
        type=Path,
        action="append",
        default=[],
        help="Legality-check system YAML/JSON (repeatable).",
    )
    parser.add_argument(
        "--validator-output-field",
        type=str,
        default="is_legal",
        help="Validator system output field containing the legality boolean",
    )
    parser.add_argument(
        "--pipeline-max-attempts",
        type=int,
        default=3,
        help="Max generator attempts per position when using generator+validator pipeline",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/benchmarks"),
        help="Root directory to store benchmark artifacts",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="auto",
        help='Optional log file path, or "auto" to write into the benchmark directory',
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log progress every N positions (0 disables)",
    )
    parser.add_argument(
        "--openai-batch",
        action="store_true",
        help="Use OpenAI Batch API for model calls (requires OpenAI provider).",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=None,
        help="Requests per OpenAI batch job (default: 2).",
    )
    parser.add_argument(
        "--openai-batch-max-wait",
        type=float,
        default=0.5,
        help="Max seconds to wait after the last queued request before flushing an underfilled batch.",
    )
    parser.add_argument(
        "--openai-batch-poll-interval",
        type=float,
        default=1.0,
        help="Poll interval (seconds) while waiting for a batch job to complete.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Worker threads when --openai-batch is enabled (0=auto).",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.plies < 0:
        raise SystemExit("--plies must be >= 0")
    if args.pipeline_max_attempts < 1:
        raise SystemExit("--pipeline-max-attempts must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if not (0.0 <= args.illegal_ratio <= 1.0):
        raise SystemExit("--illegal-ratio must be between 0 and 1")
    if not args.generator_system and not args.validator_system:
        raise SystemExit("Provide at least one of --generator-system or --validator-system")
    if args.openai_batch_size is not None and args.openai_batch_size < 2:
        raise SystemExit("--openai-batch-size must be >= 2 (omit to use default)")
    if args.openai_batch_max_wait <= 0:
        raise SystemExit("--openai-batch-max-wait must be > 0")
    if args.openai_batch_poll_interval <= 0:
        raise SystemExit("--openai-batch-poll-interval must be > 0")
    if args.max_workers < 0:
        raise SystemExit("--max-workers must be >= 0")

    run_root: Path = args.run_dir.expanduser().resolve()
    benchmark_id = _generate_benchmark_id()
    benchmark_dir = run_root / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=False)

    log_file: Path | None = None
    if args.log_file:
        if args.log_file.strip().lower() == "auto":
            log_file = benchmark_dir / "benchmark.log"
        else:
            log_file = Path(args.log_file).expanduser().resolve()
    logger = _setup_logging(level=args.log_level, log_file=log_file)

    generator_systems: list[GeneratorSystem] = []
    for path in args.generator_system:
        resolved = path.expanduser().resolve()
        system_def = _load_system(resolved)
        notation = _infer_move_notation(system_def=system_def, system_path=resolved, default="san")
        move_output_field = _infer_move_output_field(system_def=system_def, preferred_field=args.generator_output_field)
        generator_systems.append(
            GeneratorSystem(
                system_path=resolved,
                system_def=system_def,
                move_notation=notation,
                move_output_field=move_output_field,
            )
        )

    validator_systems: list[ValidatorSystem] = []
    for path in args.validator_system:
        resolved = path.expanduser().resolve()
        system_def = _load_system(resolved)
        notation = _infer_move_notation(system_def=system_def, system_path=resolved, default="san")
        validator_systems.append(
            ValidatorSystem(system_path=resolved, system_def=system_def, move_notation=notation)
        )

    pipeline_enabled = (
        len(generator_systems) == 1
        and len(validator_systems) == 1
        and generator_systems[0].move_notation == validator_systems[0].move_notation
    )

    openai_batch_provider: OpenAIBatchProvider | None = None
    openai_batch_size: int | None = None
    max_workers = 1
    if args.openai_batch:
        openai_batch_size = int(args.openai_batch_size or 2)
        entry_model_groups: set[tuple[str, str]] = set()
        for system in [*generator_systems, *validator_systems]:
            try:
                entry = system.system_def.get_node(system.system_def.entry_node_id)
            except Exception:
                continue
            provider_name = getattr(entry, "provider", None) or "openai"
            model_name = getattr(entry, "model", None)
            if model_name:
                entry_model_groups.add((provider_name, model_name))
        parallel_model_groups = max(1, len(entry_model_groups))
        concurrent_batches = 2
        tasks_per_position = 1 if pipeline_enabled else max(1, len(generator_systems) + len(validator_systems))
        total_tasks = args.iterations * tasks_per_position
        inferred_workers = max(openai_batch_size, 4, openai_batch_size * concurrent_batches * parallel_model_groups)
        inferred_workers = min(inferred_workers, total_tasks, 2048)
        max_workers = int(args.max_workers) if args.max_workers else int(inferred_workers)
        openai_batch_provider = OpenAIBatchProvider(
            batch_size=openai_batch_size,
            logger=logger,
            max_wait_seconds=float(args.openai_batch_max_wait),
            poll_interval_seconds=float(args.openai_batch_poll_interval),
        )

    def _provider_resolver(name: str) -> Any:
        if openai_batch_provider is not None and name == "openai":
            return openai_batch_provider
        return get_provider(name)

    manifest = {
        "benchmark_id": benchmark_id,
        "task": "chess_random_suite",
        "generated_positions": args.iterations,
        "plies": args.plies,
        "seed": args.seed,
        "illegal_ratio": args.illegal_ratio,
        "pipeline_max_attempts": args.pipeline_max_attempts,
        "generator_systems": [
            {
                "path": str(s.system_path),
                "id": s.system_def.id,
                "label": s.system_def.label,
                "move_notation": s.move_notation,
                "move_output_field": s.move_output_field,
            }
            for s in generator_systems
        ],
        "validator_systems": [
            {
                "path": str(s.system_path),
                "id": s.system_def.id,
                "label": s.system_def.label,
                "move_notation": s.move_notation,
            }
            for s in validator_systems
        ],
        "openai_batch": {
            "enabled": bool(openai_batch_provider is not None),
            "batch_size": openai_batch_size,
            "max_wait_seconds": float(args.openai_batch_max_wait),
            "poll_interval_seconds": float(args.openai_batch_poll_interval),
            "max_workers": max_workers,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "positions": "positions.jsonl",
            "results": "results.jsonl",
            "summary": "summary.json",
            "psf_runs": "psf_runs/",
            "log": str(log_file) if log_file else None,
        },
    }
    _write_json(benchmark_dir / "manifest.json", manifest)

    positions_path = benchmark_dir / "positions.jsonl"
    results_path = benchmark_dir / "results.jsonl"
    summary_path = benchmark_dir / "summary.json"

    logger.info("Benchmark: %s", benchmark_id)
    logger.info("Positions: %s | plies=%s | seed=%s", args.iterations, args.plies, args.seed)
    if generator_systems:
        logger.info(
            "Generators: %s",
            ", ".join(f"{s.system_def.id}({s.move_notation},{s.move_output_field})" for s in generator_systems),
        )
    if validator_systems:
        logger.info(
            "Validators: %s",
            ", ".join(f"{s.system_def.id}({s.move_notation})" for s in validator_systems),
        )
    if openai_batch_provider is not None:
        logger.info(
            "OpenAI batch enabled: batch_size=%s max_workers=%s max_wait=%.2fs poll_interval=%.2fs",
            openai_batch_size,
            max_workers,
            float(args.openai_batch_max_wait),
            float(args.openai_batch_poll_interval),
        )
    logger.info("Artifacts: %s", benchmark_dir)

    pos_rng = random.Random(args.seed)
    case_rng = random.Random(args.seed + 1)

    positions: list[GeneratedPosition] = []
    baseline_candidates: dict[int, tuple[str, str]] = {}
    for index in range(1, args.iterations + 1):
        position = play_random_plies(plies=args.plies, rng=pos_rng)
        positions.append(position)
        _append_jsonl(
            positions_path,
            {
                "index": index,
                "position_fen": position.fen,
                "move_history_san": position.move_history_san,
                "move_history_uci": position.move_history_uci,
            },
        )
        if validator_systems:
            board = chess.Board(position.fen)
            want_illegal = case_rng.random() < args.illegal_ratio
            if want_illegal:
                candidate_san = generate_illegal_san(board, case_rng)
                candidate_uci = generate_illegal_uci(board, case_rng)
            else:
                legal_moves = list(board.legal_moves)
                chosen = case_rng.choice(legal_moves)
                candidate_san = board.san(chosen)
                candidate_uci = chosen.uci()
            baseline_candidates[index] = (candidate_san, candidate_uci)

    # Aggregates
    validator_baseline: dict[str, dict[str, int]] = {
        s.system_def.id: {"total": 0, "correct": 0, "errors": 0} for s in validator_systems
    }
    validator_on_generator: dict[str, dict[str, int]] = {}
    generator: dict[str, dict[str, int]] = {
        s.system_def.id: {
            "total": 0,
            "errors": 0,
            "missing_output": 0,
            "invalid_move": 0,
            "oracle_parseable": 0,
            "oracle_legal": 0,
        }
        for s in generator_systems
    }
    pipeline_total = 0
    pipeline_accepts = 0
    pipeline_rejects = 0
    pipeline_false_accepts = 0
    pipeline_oracle_legal_accepts = 0
    pipeline_errors = 0
    pipeline_total_attempts = 0

    token_totals: dict[str, int] = {}
    if generator_systems and validator_systems and not pipeline_enabled:
        if len(generator_systems) != 1 or len(validator_systems) != 1:
            logger.info("Pipeline mode disabled (requires exactly 1 generator and 1 validator).")
        else:
            logger.info(
                "Pipeline mode disabled (generator notation=%s, validator notation=%s).",
                generator_systems[0].move_notation,
                validator_systems[0].move_notation,
            )

    def _oracle_legality(*, fen: str, move: str, notation: str) -> tuple[bool, Optional[str]]:
        if notation == "uci":
            return _oracle_uci_legality(fen=fen, move_uci=move)
        return _oracle_san_legality(fen=fen, move_san=move)

    def _merge_nested_counts(target: dict[str, dict[str, int]], inc: dict[str, dict[str, int]]) -> None:
        for system_id, stats in inc.items():
            bucket = target.setdefault(system_id, {})
            for key, value in stats.items():
                bucket[key] = bucket.get(key, 0) + int(value)

    def _merge_token_totals(target: dict[str, int], inc: dict[str, int]) -> None:
        for system_id, value in inc.items():
            target[system_id] = target.get(system_id, 0) + int(value)

    def _evaluate_position(index: int, position: GeneratedPosition) -> PositionOutcome:
        local_token_totals: dict[str, int] = {}
        local_validator_baseline: dict[str, dict[str, int]] = {}
        local_generator: dict[str, dict[str, int]] = {}
        local_validator_on_generator: dict[str, dict[str, int]] = {}
        local_pipeline: dict[str, int] = {
            "total": 0,
            "accepts": 0,
            "rejects": 0,
            "errors": 0,
            "false_accepts": 0,
            "oracle_legal_accepts": 0,
            "total_attempts": 0,
        }

        record: dict[str, Any] = {
            "index": index,
            "position_fen": position.fen,
            "meta": {
                "move_history_san": position.move_history_san,
                "move_history_uci": position.move_history_uci,
            },
        }

        # --- Validator baseline on random labeled moves (optional) ---
        if validator_systems:
            candidate_san, candidate_uci = baseline_candidates[index]
            record["validator_baseline"] = {}
            for validator in validator_systems:
                candidate_move = candidate_uci if validator.move_notation == "uci" else candidate_san
                expected_is_legal, expected_error = _oracle_legality(
                    fen=position.fen, move=candidate_move, notation=validator.move_notation
                )
                expected_legality = "legal" if expected_is_legal else "illegal"

                local_validator_baseline.setdefault(validator.system_def.id, {"total": 0, "correct": 0, "errors": 0})
                local_validator_baseline[validator.system_def.id]["total"] += 1
                payload = {
                    "position_fen": position.fen,
                    "candidate_move": candidate_move,
                    "move_notation": validator.move_notation,
                    "expected_legality": expected_legality,
                }
                run_id = f"pos{index:04d}_validator_baseline"
                artifacts = _execute_with_run_logger(
                    system_def=validator.system_def,
                    system_path=validator.system_path,
                    payload=payload,
                    runs_root=benchmark_dir / "psf_runs" / validator.system_def.id,
                    run_id=run_id,
                    provider_resolver=_provider_resolver,
                )
                local_token_totals[validator.system_def.id] = local_token_totals.get(validator.system_def.id, 0) + int(
                    (artifacts.metrics or {}).get("total_tokens") or 0
                )
                baseline_entry: dict[str, Any] = {
                    "candidate_move": candidate_move,
                    "move_notation": validator.move_notation,
                    "expected_is_legal": expected_is_legal,
                    "expected_error": expected_error,
                    "run_dir": str(artifacts.run_dir),
                    "status": artifacts.status,
                    "error": artifacts.error,
                }
                predicted_is_legal: Optional[bool] = None
                if artifacts.status == "success" and artifacts.final_output is not None:
                    try:
                        predicted_is_legal = _coerce_bool(artifacts.final_output.get(args.validator_output_field))
                    except Exception as exc:
                        baseline_entry["parse_error"] = str(exc)
                if predicted_is_legal is None:
                    local_validator_baseline[validator.system_def.id]["errors"] += 1
                    baseline_entry["predicted_is_legal"] = None
                    baseline_entry["correct"] = False
                else:
                    baseline_entry["predicted_is_legal"] = predicted_is_legal
                    correct = predicted_is_legal == expected_is_legal
                    baseline_entry["correct"] = correct
                    if correct:
                        local_validator_baseline[validator.system_def.id]["correct"] += 1
                record["validator_baseline"][validator.system_def.id] = baseline_entry

        # --- Generator run (optional) ---
        if generator_systems:
            record["generator"] = {}
        for generator_system in generator_systems:
            local_generator.setdefault(
                generator_system.system_def.id,
                {
                    "total": 0,
                    "errors": 0,
                    "missing_output": 0,
                    "invalid_move": 0,
                    "oracle_parseable": 0,
                    "oracle_legal": 0,
                },
            )
            local_generator[generator_system.system_def.id]["total"] += 1
            payload = {
                "position_fen": position.fen,
                "move_notation": generator_system.move_notation,
                "illegal_moves": [],
                "attempt": 1,
            }
            run_id = f"pos{index:04d}_generator"
            artifacts = _execute_with_run_logger(
                system_def=generator_system.system_def,
                system_path=generator_system.system_path,
                payload=payload,
                runs_root=benchmark_dir / "psf_runs" / generator_system.system_def.id,
                run_id=run_id,
                provider_resolver=_provider_resolver,
            )
            local_token_totals[generator_system.system_def.id] = local_token_totals.get(
                generator_system.system_def.id, 0
            ) + int((artifacts.metrics or {}).get("total_tokens") or 0)

            gen_entry: dict[str, Any] = {
                "run_dir": str(artifacts.run_dir),
                "status": artifacts.status,
                "error": artifacts.error,
                "move_notation": generator_system.move_notation,
                "move_output_field": generator_system.move_output_field,
            }
            if artifacts.status != "success" or artifacts.final_output is None:
                local_generator[generator_system.system_def.id]["errors"] += 1
                gen_entry["move"] = None
                record["generator"][generator_system.system_def.id] = gen_entry
                continue
            if generator_system.move_output_field not in artifacts.final_output:
                local_generator[generator_system.system_def.id]["missing_output"] += 1
                gen_entry["move"] = None
                gen_entry["missing_output_field"] = generator_system.move_output_field
                record["generator"][generator_system.system_def.id] = gen_entry
                continue
            move_value = artifacts.final_output.get(generator_system.move_output_field)
            if not isinstance(move_value, str) or not move_value.strip():
                local_generator[generator_system.system_def.id]["invalid_move"] += 1
                gen_entry["move"] = None
                gen_entry["invalid_output"] = move_value
                record["generator"][generator_system.system_def.id] = gen_entry
                continue
            move = move_value.strip()
            oracle_is_legal, oracle_error = _oracle_legality(
                fen=position.fen, move=move, notation=generator_system.move_notation
            )
            if oracle_error is None:
                local_generator[generator_system.system_def.id]["oracle_parseable"] += 1
            if oracle_is_legal:
                local_generator[generator_system.system_def.id]["oracle_legal"] += 1
            gen_entry["move"] = move
            if generator_system.move_notation == "uci":
                gen_entry["move_uci"] = move
            else:
                gen_entry["move_san"] = move
            gen_entry["oracle_is_legal"] = bool(oracle_is_legal)
            gen_entry["oracle_error"] = oracle_error
            record["generator"][generator_system.system_def.id] = gen_entry

        # --- Validator on generator move + pipeline loop (optional) ---
        if pipeline_enabled:
            pipeline_notation = generator_systems[0].move_notation
            pipeline_move_output_field = generator_systems[0].move_output_field
            generator_def = generator_systems[0].system_def
            generator_path = generator_systems[0].system_path
            validator_def = validator_systems[0].system_def
            validator_path = validator_systems[0].system_path

            local_validator_on_generator.setdefault(validator_def.id, {"total": 0, "correct": 0, "errors": 0})
            local_pipeline["total"] += 1

            generator_move: Optional[str] = None
            generator_oracle_is_legal: Optional[bool] = None
            generator_oracle_error: Optional[str] = None
            generator_run_dir: Optional[str] = None
            if record.get("generator"):
                only_gen_id = generator_def.id
                gen_entry = record["generator"].get(only_gen_id, {})
                generator_move = gen_entry.get("move")
                generator_oracle_is_legal = gen_entry.get("oracle_is_legal")
                generator_oracle_error = gen_entry.get("oracle_error")
                generator_run_dir = gen_entry.get("run_dir")

            if generator_move is None:
                local_pipeline["errors"] += 1
                local_pipeline["total_attempts"] += 1
                pipeline_entry: dict[str, Any] = {
                    "max_attempts": args.pipeline_max_attempts,
                    "status": "error",
                    "error": "generator did not produce a usable move on attempt 1",
                    "accepted": False,
                    "move_notation": pipeline_notation,
                    "accepted_move": None,
                    "accepted_oracle_is_legal": None,
                    "accepted_oracle_error": None,
                    "attempts": [
                        {
                            "attempt": 1,
                            "generator_run_dir": generator_run_dir,
                            "move": None,
                        }
                    ],
                }
                if pipeline_notation == "uci":
                    pipeline_entry["accepted_move_uci"] = None
                else:
                    pipeline_entry["accepted_move_san"] = None
                record["pipeline"] = pipeline_entry
            else:
                # Validator on generator attempt 1.
                local_validator_on_generator[validator_def.id]["total"] += 1
                payload = {
                    "position_fen": position.fen,
                    "candidate_move": generator_move,
                    "move_notation": pipeline_notation,
                }
                run_id = f"pos{index:04d}_validator_on_generator_attempt01"
                v_artifacts = _execute_with_run_logger(
                    system_def=validator_def,
                    system_path=validator_path,
                    payload=payload,
                    runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                    run_id=run_id,
                    provider_resolver=_provider_resolver,
                )
                local_token_totals[validator_def.id] = local_token_totals.get(validator_def.id, 0) + int(
                    (v_artifacts.metrics or {}).get("total_tokens") or 0
                )
                v_entry: dict[str, Any] = {
                    "candidate_move": generator_move,
                    "move_notation": pipeline_notation,
                    "oracle_is_legal": generator_oracle_is_legal,
                    "oracle_error": generator_oracle_error,
                    "run_dir": str(v_artifacts.run_dir),
                    "status": v_artifacts.status,
                    "error": v_artifacts.error,
                }
                predicted_is_legal: Optional[bool] = None
                if v_artifacts.status == "success" and v_artifacts.final_output is not None:
                    try:
                        predicted_is_legal = _coerce_bool(v_artifacts.final_output.get(args.validator_output_field))
                    except Exception as exc:
                        v_entry["parse_error"] = str(exc)
                if predicted_is_legal is None or generator_oracle_is_legal is None:
                    local_validator_on_generator[validator_def.id]["errors"] += 1
                    v_entry["predicted_is_legal"] = None
                    v_entry["correct"] = False
                else:
                    v_entry["predicted_is_legal"] = predicted_is_legal
                    correct = predicted_is_legal == generator_oracle_is_legal
                    v_entry["correct"] = correct
                    if correct:
                        local_validator_on_generator[validator_def.id]["correct"] += 1
                record["validator_on_generator"] = v_entry

                # Pipeline: retry generator until validator accepts (or attempts exhausted).
                pipeline_attempts: list[dict[str, Any]] = []
                illegal_moves: list[str] = []
                accepted_move: Optional[str] = None
                accepted_oracle_is_legal: Optional[bool] = None
                accepted_oracle_error: Optional[str] = None
                pipeline_error: Optional[str] = None

                pipeline_attempts.append(
                    {
                        "attempt": 1,
                        "generator_run_dir": generator_run_dir,
                        "validator_run_dir": str(v_artifacts.run_dir),
                        "move": generator_move,
                        "move_notation": pipeline_notation,
                        "validator_is_legal": v_entry.get("predicted_is_legal"),
                        "oracle_is_legal": generator_oracle_is_legal,
                        "oracle_error": generator_oracle_error,
                    }
                )
                local_pipeline["total_attempts"] += 1

                first_verdict = v_entry.get("predicted_is_legal")
                if first_verdict is None:
                    local_pipeline["errors"] += 1
                    pipeline_error = "validator did not produce a usable verdict on attempt 1"
                elif first_verdict is True:
                    accepted_move = generator_move
                    accepted_oracle_is_legal = generator_oracle_is_legal
                    accepted_oracle_error = generator_oracle_error
                else:
                    illegal_moves.append(generator_move)
                    for attempt in range(2, args.pipeline_max_attempts + 1):
                        payload = {
                            "position_fen": position.fen,
                            "move_notation": pipeline_notation,
                            "illegal_moves": illegal_moves,
                            "attempt": attempt,
                        }
                        run_id = f"pos{index:04d}_pipeline_generator_attempt{attempt:02d}"
                        gen_artifacts = _execute_with_run_logger(
                            system_def=generator_def,
                            system_path=generator_path,
                            payload=payload,
                            runs_root=benchmark_dir / "psf_runs" / generator_def.id,
                            run_id=run_id,
                            provider_resolver=_provider_resolver,
                        )
                        local_token_totals[generator_def.id] = local_token_totals.get(generator_def.id, 0) + int(
                            (gen_artifacts.metrics or {}).get("total_tokens") or 0
                        )
                        move: Optional[str] = None
                        oracle_is_legal: Optional[bool] = None
                        oracle_error: Optional[str] = None
                        gen_error: Optional[str] = None
                        if gen_artifacts.status != "success" or gen_artifacts.final_output is None:
                            gen_error = gen_artifacts.error or "generator failed"
                        else:
                            move_value = gen_artifacts.final_output.get(pipeline_move_output_field)
                            if not isinstance(move_value, str) or not move_value.strip():
                                gen_error = f"invalid generator output '{pipeline_move_output_field}'"
                            else:
                                move = move_value.strip()
                                oracle_is_legal, oracle_error = _oracle_legality(
                                    fen=position.fen, move=move, notation=pipeline_notation
                                )

                        # Validator call for this attempt (only if we got a move).
                        v_run_dir: Optional[str] = None
                        v_status = "skipped"
                        v_error: Optional[str] = None
                        v_pred: Optional[bool] = None
                        if move is not None:
                            v_payload = {
                                "position_fen": position.fen,
                                "candidate_move": move,
                                "move_notation": pipeline_notation,
                            }
                            v_run_id = f"pos{index:04d}_pipeline_validator_attempt{attempt:02d}"
                            v_art = _execute_with_run_logger(
                                system_def=validator_def,
                                system_path=validator_path,
                                payload=v_payload,
                                runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                                run_id=v_run_id,
                                provider_resolver=_provider_resolver,
                            )
                            local_token_totals[validator_def.id] = local_token_totals.get(validator_def.id, 0) + int(
                                (v_art.metrics or {}).get("total_tokens") or 0
                            )
                            v_run_dir = str(v_art.run_dir)
                            v_status = v_art.status
                            v_error = v_art.error
                            if v_art.status == "success" and v_art.final_output is not None:
                                try:
                                    v_pred = _coerce_bool(v_art.final_output.get(args.validator_output_field))
                                except Exception as exc:
                                    v_error = str(exc)

                        attempt_entry: dict[str, Any] = {
                            "attempt": attempt,
                            "generator_run_dir": str(gen_artifacts.run_dir),
                            "generator_status": gen_artifacts.status,
                            "generator_error": gen_error,
                            "validator_run_dir": v_run_dir,
                            "validator_status": v_status,
                            "validator_error": v_error,
                            "move": move,
                            "move_notation": pipeline_notation,
                            "validator_is_legal": v_pred,
                            "oracle_is_legal": oracle_is_legal,
                            "oracle_error": oracle_error,
                        }
                        if pipeline_notation == "uci":
                            attempt_entry["move_uci"] = move
                        else:
                            attempt_entry["move_san"] = move
                        pipeline_attempts.append(attempt_entry)
                        local_pipeline["total_attempts"] += 1

                        if gen_error is not None:
                            local_pipeline["errors"] += 1
                            pipeline_error = gen_error
                            break
                        if v_pred is None:
                            local_pipeline["errors"] += 1
                            pipeline_error = v_error or "validator did not produce a usable verdict"
                            break
                        if v_pred is True:
                            accepted_move = move
                            accepted_oracle_is_legal = bool(oracle_is_legal)
                            accepted_oracle_error = oracle_error
                            break
                        if move is not None:
                            illegal_moves.append(move)

                status = "accepted" if accepted_move is not None else ("error" if pipeline_error else "rejected")
                pipeline_entry = {
                    "max_attempts": args.pipeline_max_attempts,
                    "status": status,
                    "error": pipeline_error,
                    "accepted": accepted_move is not None,
                    "move_notation": pipeline_notation,
                    "accepted_move": accepted_move,
                    "accepted_oracle_is_legal": accepted_oracle_is_legal,
                    "accepted_oracle_error": accepted_oracle_error,
                    "attempts": pipeline_attempts,
                }
                if pipeline_notation == "uci":
                    pipeline_entry["accepted_move_uci"] = accepted_move
                else:
                    pipeline_entry["accepted_move_san"] = accepted_move
                if status == "accepted":
                    local_pipeline["accepts"] += 1
                    if accepted_oracle_is_legal is True:
                        local_pipeline["oracle_legal_accepts"] += 1
                    elif accepted_oracle_is_legal is False:
                        local_pipeline["false_accepts"] += 1
                elif status == "rejected":
                    local_pipeline["rejects"] += 1
                record["pipeline"] = pipeline_entry

        return PositionOutcome(
            index=index,
            record=record,
            token_totals=local_token_totals,
            validator_baseline=local_validator_baseline,
            generator=local_generator,
            validator_on_generator=local_validator_on_generator,
            pipeline=local_pipeline,
        )

    def _consume_outcome(outcome: PositionOutcome) -> None:
        _merge_token_totals(token_totals, outcome.token_totals)
        _merge_nested_counts(validator_baseline, outcome.validator_baseline)
        _merge_nested_counts(generator, outcome.generator)
        _merge_nested_counts(validator_on_generator, outcome.validator_on_generator)
        nonlocal pipeline_total, pipeline_accepts, pipeline_rejects, pipeline_false_accepts, pipeline_oracle_legal_accepts, pipeline_errors, pipeline_total_attempts
        pipeline_total += int(outcome.pipeline.get("total", 0))
        pipeline_accepts += int(outcome.pipeline.get("accepts", 0))
        pipeline_rejects += int(outcome.pipeline.get("rejects", 0))
        pipeline_false_accepts += int(outcome.pipeline.get("false_accepts", 0))
        pipeline_oracle_legal_accepts += int(outcome.pipeline.get("oracle_legal_accepts", 0))
        pipeline_errors += int(outcome.pipeline.get("errors", 0))
        pipeline_total_attempts += int(outcome.pipeline.get("total_attempts", 0))
        _append_jsonl(results_path, outcome.record)

    def _consume_task_outcome(outcome: SystemTaskOutcome, record_by_index: dict[int, dict[str, Any]]) -> None:
        _merge_token_totals(token_totals, outcome.token_totals)
        if outcome.kind == "validator_baseline":
            _merge_nested_counts(validator_baseline, outcome.counts)
            record_by_index[outcome.index]["validator_baseline"][outcome.system_id] = outcome.entry
        elif outcome.kind == "generator":
            _merge_nested_counts(generator, outcome.counts)
            record_by_index[outcome.index]["generator"][outcome.system_id] = outcome.entry
        else:
            raise ValueError(f"Unknown task kind '{outcome.kind}'")

    def _run_validator_baseline_task(
        index: int,
        position: GeneratedPosition,
        validator: ValidatorSystem,
    ) -> SystemTaskOutcome:
        candidate_san, candidate_uci = baseline_candidates[index]
        candidate_move = candidate_uci if validator.move_notation == "uci" else candidate_san
        expected_is_legal, expected_error = _oracle_legality(
            fen=position.fen,
            move=candidate_move,
            notation=validator.move_notation,
        )
        expected_legality = "legal" if expected_is_legal else "illegal"

        payload = {
            "position_fen": position.fen,
            "candidate_move": candidate_move,
            "move_notation": validator.move_notation,
            "expected_legality": expected_legality,
        }
        run_id = f"pos{index:04d}_validator_baseline"
        artifacts = _execute_with_run_logger(
            system_def=validator.system_def,
            system_path=validator.system_path,
            payload=payload,
            runs_root=benchmark_dir / "psf_runs" / validator.system_def.id,
            run_id=run_id,
            provider_resolver=_provider_resolver,
        )
        token_delta = {
            validator.system_def.id: int((artifacts.metrics or {}).get("total_tokens") or 0),
        }

        baseline_entry: dict[str, Any] = {
            "candidate_move": candidate_move,
            "move_notation": validator.move_notation,
            "expected_is_legal": expected_is_legal,
            "expected_error": expected_error,
            "run_dir": str(artifacts.run_dir),
            "status": artifacts.status,
            "error": artifacts.error,
        }

        predicted_is_legal: Optional[bool] = None
        correct_count = 0
        error_count = 0
        if artifacts.status == "success" and artifacts.final_output is not None:
            try:
                predicted_is_legal = _coerce_bool(artifacts.final_output.get(args.validator_output_field))
            except Exception as exc:
                baseline_entry["parse_error"] = str(exc)
        if predicted_is_legal is None:
            error_count = 1
            baseline_entry["predicted_is_legal"] = None
            baseline_entry["correct"] = False
        else:
            baseline_entry["predicted_is_legal"] = predicted_is_legal
            is_correct = predicted_is_legal == expected_is_legal
            baseline_entry["correct"] = is_correct
            if is_correct:
                correct_count = 1

        counts = {
            validator.system_def.id: {
                "total": 1,
                "correct": correct_count,
                "errors": error_count,
            }
        }
        return SystemTaskOutcome(
            index=index,
            kind="validator_baseline",
            system_id=validator.system_def.id,
            entry=baseline_entry,
            token_totals=token_delta,
            counts=counts,
        )

    def _run_generator_task(
        index: int,
        position: GeneratedPosition,
        generator_system: GeneratorSystem,
    ) -> SystemTaskOutcome:
        payload = {
            "position_fen": position.fen,
            "move_notation": generator_system.move_notation,
            "illegal_moves": [],
            "attempt": 1,
        }
        run_id = f"pos{index:04d}_generator"
        artifacts = _execute_with_run_logger(
            system_def=generator_system.system_def,
            system_path=generator_system.system_path,
            payload=payload,
            runs_root=benchmark_dir / "psf_runs" / generator_system.system_def.id,
            run_id=run_id,
            provider_resolver=_provider_resolver,
        )

        system_id = generator_system.system_def.id
        token_delta = {
            system_id: int((artifacts.metrics or {}).get("total_tokens") or 0),
        }
        counts = {
            system_id: {
                "total": 1,
                "errors": 0,
                "missing_output": 0,
                "invalid_move": 0,
                "oracle_parseable": 0,
                "oracle_legal": 0,
            }
        }

        gen_entry: dict[str, Any] = {
            "run_dir": str(artifacts.run_dir),
            "status": artifacts.status,
            "error": artifacts.error,
            "move_notation": generator_system.move_notation,
            "move_output_field": generator_system.move_output_field,
        }
        if artifacts.status != "success" or artifacts.final_output is None:
            counts[system_id]["errors"] = 1
            gen_entry["move"] = None
            return SystemTaskOutcome(
                index=index,
                kind="generator",
                system_id=system_id,
                entry=gen_entry,
                token_totals=token_delta,
                counts=counts,
            )
        if generator_system.move_output_field not in artifacts.final_output:
            counts[system_id]["missing_output"] = 1
            gen_entry["move"] = None
            gen_entry["missing_output_field"] = generator_system.move_output_field
            return SystemTaskOutcome(
                index=index,
                kind="generator",
                system_id=system_id,
                entry=gen_entry,
                token_totals=token_delta,
                counts=counts,
            )
        move_value = artifacts.final_output.get(generator_system.move_output_field)
        if not isinstance(move_value, str) or not move_value.strip():
            counts[system_id]["invalid_move"] = 1
            gen_entry["move"] = None
            gen_entry["invalid_output"] = move_value
            return SystemTaskOutcome(
                index=index,
                kind="generator",
                system_id=system_id,
                entry=gen_entry,
                token_totals=token_delta,
                counts=counts,
            )
        move = move_value.strip()
        oracle_is_legal, oracle_error = _oracle_legality(
            fen=position.fen,
            move=move,
            notation=generator_system.move_notation,
        )
        if oracle_error is None:
            counts[system_id]["oracle_parseable"] = 1
        if oracle_is_legal:
            counts[system_id]["oracle_legal"] = 1

        gen_entry["move"] = move
        if generator_system.move_notation == "uci":
            gen_entry["move_uci"] = move
        else:
            gen_entry["move_san"] = move
        gen_entry["oracle_is_legal"] = bool(oracle_is_legal)
        gen_entry["oracle_error"] = oracle_error

        return SystemTaskOutcome(
            index=index,
            kind="generator",
            system_id=system_id,
            entry=gen_entry,
            token_totals=token_delta,
            counts=counts,
        )

    has_system_nodes = any(
        getattr(node, "use_system", None)
        for system in [*generator_systems, *validator_systems]
        for node in system.system_def.nodes
    )
    use_openai_batch_scheduler = openai_batch_provider is not None and not pipeline_enabled and not has_system_nodes
    if openai_batch_provider is not None and has_system_nodes:
        logger.info("OpenAI batch scheduler disabled (system nodes detected); using threaded execution instead.")
    if use_openai_batch_scheduler:
        tasks_per_position = len(generator_systems) + len(validator_systems)
        logger.info(
            "Batch scheduler: stage-based; tasks=%s (positions=%s × systems=%s) batch_size=%s",
            args.iterations * tasks_per_position,
            args.iterations,
            tasks_per_position,
            openai_batch_size,
        )
        record_by_index: dict[int, dict[str, Any]] = {}
        remaining_tasks: dict[int, int] = {}
        for idx, position in enumerate(positions, start=1):
            record: dict[str, Any] = {
                "index": idx,
                "position_fen": position.fen,
                "meta": {
                    "move_history_san": position.move_history_san,
                    "move_history_uci": position.move_history_uci,
                },
            }
            if validator_systems:
                record["validator_baseline"] = {}
            if generator_systems:
                record["generator"] = {}
            record_by_index[idx] = record
            remaining_tasks[idx] = tasks_per_position

        task_specs: list[_BatchTaskSpec] = []
        task_id = 0
        for idx, position in enumerate(positions, start=1):
            if validator_systems:
                candidate_san, candidate_uci = baseline_candidates[idx]
                for validator in validator_systems:
                    candidate_move = candidate_uci if validator.move_notation == "uci" else candidate_san
                    expected_is_legal, expected_error = _oracle_legality(
                        fen=position.fen,
                        move=candidate_move,
                        notation=validator.move_notation,
                    )
                    expected_legality = "legal" if expected_is_legal else "illegal"
                    payload = {
                        "position_fen": position.fen,
                        "candidate_move": candidate_move,
                        "move_notation": validator.move_notation,
                        "expected_legality": expected_legality,
                    }
                    task_id += 1
                    task_specs.append(
                        _BatchTaskSpec(
                            task_id=task_id,
                            kind="validator_baseline",
                            index=idx,
                            system_def=validator.system_def,
                            system_path=validator.system_path,
                            runs_root=benchmark_dir / "psf_runs" / validator.system_def.id,
                            run_id=f"pos{idx:04d}_validator_baseline",
                            payload=payload,
                            metadata={
                                "candidate_move": candidate_move,
                                "move_notation": validator.move_notation,
                                "expected_is_legal": expected_is_legal,
                                "expected_error": expected_error,
                            },
                        )
                    )
            for generator_system in generator_systems:
                payload = {
                    "position_fen": position.fen,
                    "move_notation": generator_system.move_notation,
                    "illegal_moves": [],
                    "attempt": 1,
                }
                task_id += 1
                task_specs.append(
                    _BatchTaskSpec(
                        task_id=task_id,
                        kind="generator",
                        index=idx,
                        system_def=generator_system.system_def,
                        system_path=generator_system.system_path,
                        runs_root=benchmark_dir / "psf_runs" / generator_system.system_def.id,
                        run_id=f"pos{idx:04d}_generator",
                        payload=payload,
                        metadata={
                            "move_notation": generator_system.move_notation,
                            "move_output_field": generator_system.move_output_field,
                        },
                    )
                )

        ready_to_write: dict[int, dict[str, Any]] = {}
        next_to_write = 1
        written = 0

        def _outcome_from_run(*, spec: _BatchTaskSpec, artifacts: RunArtifacts) -> SystemTaskOutcome:
            if spec.kind == "validator_baseline":
                system_id = spec.system_def.id
                token_delta = {system_id: int((artifacts.metrics or {}).get("total_tokens") or 0)}
                candidate_move = spec.metadata.get("candidate_move")
                move_notation = spec.metadata.get("move_notation")
                expected_is_legal = bool(spec.metadata.get("expected_is_legal"))
                expected_error = spec.metadata.get("expected_error")
                baseline_entry: dict[str, Any] = {
                    "candidate_move": candidate_move,
                    "move_notation": move_notation,
                    "expected_is_legal": expected_is_legal,
                    "expected_error": expected_error,
                    "run_dir": str(artifacts.run_dir),
                    "status": artifacts.status,
                    "error": artifacts.error,
                }
                predicted_is_legal: Optional[bool] = None
                correct_count = 0
                error_count = 0
                if artifacts.status == "success" and artifacts.final_output is not None:
                    try:
                        predicted_is_legal = _coerce_bool(artifacts.final_output.get(args.validator_output_field))
                    except Exception as exc:
                        baseline_entry["parse_error"] = str(exc)
                if predicted_is_legal is None:
                    error_count = 1
                    baseline_entry["predicted_is_legal"] = None
                    baseline_entry["correct"] = False
                else:
                    baseline_entry["predicted_is_legal"] = predicted_is_legal
                    is_correct = predicted_is_legal == expected_is_legal
                    baseline_entry["correct"] = is_correct
                    if is_correct:
                        correct_count = 1
                counts = {system_id: {"total": 1, "correct": correct_count, "errors": error_count}}
                return SystemTaskOutcome(
                    index=spec.index,
                    kind="validator_baseline",
                    system_id=system_id,
                    entry=baseline_entry,
                    token_totals=token_delta,
                    counts=counts,
                )

            if spec.kind == "generator":
                system_id = spec.system_def.id
                token_delta = {system_id: int((artifacts.metrics or {}).get("total_tokens") or 0)}
                move_notation = spec.metadata.get("move_notation")
                move_output_field = spec.metadata.get("move_output_field")
                counts = {
                    system_id: {
                        "total": 1,
                        "errors": 0,
                        "missing_output": 0,
                        "invalid_move": 0,
                        "oracle_parseable": 0,
                        "oracle_legal": 0,
                    }
                }
                gen_entry: dict[str, Any] = {
                    "run_dir": str(artifacts.run_dir),
                    "status": artifacts.status,
                    "error": artifacts.error,
                    "move_notation": move_notation,
                    "move_output_field": move_output_field,
                }
                if artifacts.status != "success" or artifacts.final_output is None:
                    counts[system_id]["errors"] = 1
                    gen_entry["move"] = None
                    return SystemTaskOutcome(
                        index=spec.index,
                        kind="generator",
                        system_id=system_id,
                        entry=gen_entry,
                        token_totals=token_delta,
                        counts=counts,
                    )
                if move_output_field not in artifacts.final_output:
                    counts[system_id]["missing_output"] = 1
                    gen_entry["move"] = None
                    gen_entry["missing_output_field"] = move_output_field
                    return SystemTaskOutcome(
                        index=spec.index,
                        kind="generator",
                        system_id=system_id,
                        entry=gen_entry,
                        token_totals=token_delta,
                        counts=counts,
                    )
                move_value = artifacts.final_output.get(move_output_field)
                if not isinstance(move_value, str) or not move_value.strip():
                    counts[system_id]["invalid_move"] = 1
                    gen_entry["move"] = None
                    gen_entry["invalid_output"] = move_value
                    return SystemTaskOutcome(
                        index=spec.index,
                        kind="generator",
                        system_id=system_id,
                        entry=gen_entry,
                        token_totals=token_delta,
                        counts=counts,
                    )
                move = move_value.strip()
                fen = spec.payload.get("position_fen")
                oracle_is_legal, oracle_error = _oracle_legality(
                    fen=str(fen),
                    move=move,
                    notation=str(move_notation),
                )
                if oracle_error is None:
                    counts[system_id]["oracle_parseable"] = 1
                if oracle_is_legal:
                    counts[system_id]["oracle_legal"] = 1
                gen_entry["move"] = move
                if move_notation == "uci":
                    gen_entry["move_uci"] = move
                else:
                    gen_entry["move_san"] = move
                gen_entry["oracle_is_legal"] = bool(oracle_is_legal)
                gen_entry["oracle_error"] = oracle_error
                return SystemTaskOutcome(
                    index=spec.index,
                    kind="generator",
                    system_id=system_id,
                    entry=gen_entry,
                    token_totals=token_delta,
                    counts=counts,
                )

            raise ValueError(f"Unknown task kind '{spec.kind}'")

        if openai_batch_size is None:
            raise RuntimeError("openai_batch_size is required when OpenAI batch scheduler is enabled")
        for spec, artifacts in _execute_task_specs_openai_batch(
            task_specs=task_specs,
            openai_batch_provider=openai_batch_provider,
            batch_size=openai_batch_size,
            max_wait_seconds=float(args.openai_batch_max_wait),
            poll_interval_seconds=float(args.openai_batch_poll_interval),
            logger=logger,
        ):
            outcome = _outcome_from_run(spec=spec, artifacts=artifacts)
            _consume_task_outcome(outcome, record_by_index)
            remaining_tasks[outcome.index] -= 1
            if remaining_tasks[outcome.index] == 0:
                ready_to_write[outcome.index] = record_by_index[outcome.index]
            while next_to_write in ready_to_write:
                _append_jsonl(results_path, ready_to_write.pop(next_to_write))
                written += 1
                if args.progress_every > 0 and written % args.progress_every == 0:
                    logger.info("Progress: %s/%s positions", written, args.iterations)
                next_to_write += 1
        while next_to_write in ready_to_write:
            _append_jsonl(results_path, ready_to_write.pop(next_to_write))
            written += 1
            if args.progress_every > 0 and written % args.progress_every == 0:
                logger.info("Progress: %s/%s positions", written, args.iterations)
            next_to_write += 1
    elif max_workers > 1:
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, position in enumerate(positions, start=1):
                futures.append(executor.submit(_evaluate_position, idx, position))
            pending: dict[int, PositionOutcome] = {}
            next_to_write = 1
            completed = 0
            for future in as_completed(futures):
                outcome = future.result()
                pending[outcome.index] = outcome
                completed += 1
                while next_to_write in pending:
                    _consume_outcome(pending.pop(next_to_write))
                    next_to_write += 1
                if args.progress_every > 0 and completed % args.progress_every == 0:
                    logger.info("Progress: %s/%s positions", completed, args.iterations)
    else:
        for idx, position in enumerate(positions, start=1):
            _consume_outcome(_evaluate_position(idx, position))
            if args.progress_every > 0 and idx % args.progress_every == 0:
                logger.info("Progress: %s/%s positions", idx, args.iterations)

    finished_at = datetime.now(timezone.utc).isoformat()
    generator_summaries: dict[str, Any] = {}
    for system_id, stats in generator.items():
        total = stats["total"]
        generator_summaries[system_id] = {
            "system_id": system_id,
            "total": total,
            "errors": stats["errors"],
            "missing_output": stats["missing_output"],
            "invalid_move": stats["invalid_move"],
            "oracle_parseable": stats["oracle_parseable"],
            "oracle_legal": stats["oracle_legal"],
            "oracle_parse_rate": (stats["oracle_parseable"] / total) if total else 0.0,
            "oracle_legal_rate": (stats["oracle_legal"] / total) if total else 0.0,
        }

    validator_summaries: dict[str, Any] = {}
    for system_id, stats in validator_baseline.items():
        total = stats["total"]
        validator_summaries[system_id] = {
            "system_id": system_id,
            "total": total,
            "correct": stats["correct"],
            "errors": stats["errors"],
            "accuracy": (stats["correct"] / total) if total else 0.0,
        }

    summary = {
        "benchmark_id": benchmark_id,
        "task": "chess_random_suite",
        "positions": args.iterations,
        "plies": args.plies,
        "seed": args.seed,
        "illegal_ratio": args.illegal_ratio,
        "generators": generator_summaries,
        "validators": validator_summaries,
        "validator_on_generator": {
            "enabled": bool(validator_on_generator),
            "by_system": {
                system_id: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "errors": stats["errors"],
                    "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
                }
                for system_id, stats in validator_on_generator.items()
            },
        },
        "pipeline": {
            "enabled": pipeline_enabled,
            "max_attempts": args.pipeline_max_attempts,
            "total": pipeline_total,
            "accepts": pipeline_accepts,
            "oracle_legal_accepts": pipeline_oracle_legal_accepts,
            "rejects": pipeline_rejects,
            "errors": pipeline_errors,
            "false_accepts": pipeline_false_accepts,
            "accept_rate": (pipeline_accepts / pipeline_total) if pipeline_total else 0.0,
            "oracle_legal_rate": (pipeline_oracle_legal_accepts / pipeline_total) if pipeline_total else 0.0,
            "avg_attempts": (pipeline_total_attempts / pipeline_total) if pipeline_total else 0.0,
        },
        "token_totals": token_totals,
        "artifacts": {
            "benchmark_dir": str(benchmark_dir),
            "positions": str(positions_path),
            "results": str(results_path),
            "summary": str(summary_path),
        },
        "finished_at": finished_at,
    }
    if len(generator_summaries) == 1:
        system_id, stats = next(iter(generator_summaries.items()))
        label = next((s.system_def.label for s in generator_systems if s.system_def.id == system_id), None)
        summary["generator"] = {
            "system_id": system_id,
            "system_label": label,
            "total": stats["total"],
            "errors": stats["errors"],
            "missing_output": stats["missing_output"],
            "invalid_move": stats["invalid_move"],
            "oracle_parseable": stats["oracle_parseable"],
            "oracle_legal": stats["oracle_legal"],
            "oracle_parse_rate": stats["oracle_parse_rate"],
            "oracle_legal_rate": stats["oracle_legal_rate"],
        }
    if len(validator_summaries) == 1:
        system_id, stats = next(iter(validator_summaries.items()))
        label = next((s.system_def.label for s in validator_systems if s.system_def.id == system_id), None)
        summary["validator_baseline"] = {
            "system_id": system_id,
            "system_label": label,
            "total": stats["total"],
            "correct": stats["correct"],
            "errors": stats["errors"],
            "accuracy": stats["accuracy"],
        }
    _write_json(summary_path, summary)

    logger.info("Done.")
    if summary["generators"]:
        logger.info(
            "Generators: %s",
            ", ".join(
                f"{sid} legal={stats['oracle_legal_rate']*100:.1f}% parse={stats['oracle_parse_rate']*100:.1f}%"
                for sid, stats in summary["generators"].items()
            ),
        )
    if validator_summaries:
        logger.info(
            "Validators: %s",
            ", ".join(
                f"{sid} acc={stats['accuracy']*100:.1f}%"
                for sid, stats in summary["validators"].items()
            ),
        )
    if validator_on_generator:
        for system_id, stats in summary["validator_on_generator"]["by_system"].items():
            logger.info("Validator-on-generator (%s) accuracy: %.1f%%", system_id, stats["accuracy"] * 100)
    if pipeline_total:
        logger.info(
            "Pipeline accept rate: %.1f%% | oracle legal rate: %.1f%% (avg attempts %.2f, false accepts=%s)",
            summary["pipeline"]["accept_rate"] * 100,
            summary["pipeline"]["oracle_legal_rate"] * 100,
            summary["pipeline"]["avg_attempts"],
            summary["pipeline"]["false_accepts"],
        )
    logger.info("Artifacts: %s", benchmark_dir)


if __name__ == "__main__":
    main()
