from __future__ import annotations

import copy
import json
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

import typer
import yaml

from .config_loader import load_system_definition
from .engine import ExecutionEngine, ExecutionResult
from .exceptions import ConfigError, ExecutionError
from .logging import RunLogger
from .models import SystemDef
from .providers import get_provider
from .providers.openai_provider import OpenAIBatchProvider

app = typer.Typer(help="Prompt System Framework CLI")
DEFAULT_RUN_DIR = "runs"
_TRUE_LABELS = {"legal", "true", "t", "1", "yes", "y"}
_FALSE_LABELS = {"illegal", "false", "f", "0", "no", "n"}


@dataclass
class BenchmarkScenario:
    name: str
    path: Path
    input_payload: dict[str, Any]
    match_mode: str
    expected_value: Any
    expected_label: str


@dataclass
class BenchmarkIterationRecord:
    iteration: int
    run_dir: Optional[str]
    expected_label: str
    actual_label: Optional[str]
    status: str
    detail: Optional[str]


@dataclass
class BenchmarkResult:
    scenario: BenchmarkScenario
    total_runs: int
    successes: int
    iterations: List[BenchmarkIterationRecord]

    @property
    def failures(self) -> int:
        return self.total_runs - self.successes

    @property
    def mismatches(self) -> List[BenchmarkIterationRecord]:
        return [record for record in self.iterations if record.status != "success"]


@dataclass
class SystemBenchmarkSummary:
    system_id: str
    system_label: Optional[str]
    system_version: Optional[str]
    system_path: Path
    benchmark_root: Path
    summary_path: Path
    scenario_results: List[BenchmarkResult]

    @property
    def overall_accuracy(self) -> float:
        total_runs = sum(result.total_runs for result in self.scenario_results)
        total_successes = sum(result.successes for result in self.scenario_results)
        return (total_successes / total_runs) if total_runs else 0.0

def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise typer.BadParameter(f"Path not found: {p}")
    return p


@app.command()
def validate(system: str = typer.Option(..., "--system", help="Path to system YAML/JSON definition")) -> None:
    """Validate a system definition file."""
    system_path = _resolve_path(system)
    _load_system_cli(system_path)
    typer.secho("System definition is valid.", fg=typer.colors.GREEN)


@app.command()
def inspect(system: str = typer.Option(..., "--system", help="Path to system definition")) -> None:
    """Inspect the graph structure for a system."""
    system_path = _resolve_path(system)
    system_def = _load_system_cli(system_path)
    typer.echo(f"System: {system_def.id} (version={system_def.version or 'unversioned'})")
    typer.echo(f"Entry node: {system_def.entry_node_id}")
    typer.echo("")
    typer.echo("Nodes:")
    for node in system_def.nodes:
        if getattr(node, "use_system", None):
            typer.echo(
                f"- {node.id} | label={node.label or '-'} | kind=system | use_system={node.use_system}"
            )
        else:
            typer.echo(
                f"- {node.id} | label={node.label or '-'} | kind=llm | provider={node.provider or '<default>'} | model={node.model or '<default>'}"
            )
        if node.loop_control:
            typer.echo(
                f"  loop -> target={node.loop_control.loop_target_node} max={node.loop_control.max_iterations} exit_on_success={node.loop_control.exit_on_success}"
            )
        if node.routing.on_success:
            next_targets = ", ".join(rule.next for rule in node.routing.on_success)
            typer.echo(f"  on_success -> {next_targets}")
        if node.routing.on_loop_exhausted:
            next_targets = ", ".join(rule.next for rule in node.routing.on_loop_exhausted)
            typer.echo(f"  on_loop_exhausted -> {next_targets}")
        if node.terminal:
            typer.echo("  terminal node")
    typer.echo("")


@app.command()
def schema(
    out: Optional[str] = typer.Option(None, "--out", help="Write JSON schema to a file (defaults to stdout)"),
) -> None:
    """Print the JSON schema for the canonical SystemDef model."""
    from .models import SystemDef

    payload = SystemDef.model_json_schema()
    if out:
        out_path = Path(out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        typer.echo(f"Wrote schema to: {out_path}")
        return
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def graph(
    system: str = typer.Option(..., "--system", help="Path to system definition"),
    out: Optional[str] = typer.Option(None, "--out", help="Write graph JSON to a file (defaults to stdout)"),
) -> None:
    """Export a machine-readable graph for a system (includes resolved use_system)."""
    system_path = _resolve_path(system)
    from .graph import export_resolved_graph

    payload = export_resolved_graph(system_path)
    if out:
        out_path = Path(out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        typer.echo(f"Wrote graph to: {out_path}")
        return
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def run(
    system: str = typer.Option(..., "--system", help="Path to system definition"),
    input: Optional[str] = typer.Option(None, "--input", help="Path to input JSON/YAML"),
    stdin: bool = typer.Option(False, "--stdin", help="Read JSON input from stdin"),
    run_dir: str = typer.Option(DEFAULT_RUN_DIR, "--run-dir", help="Directory to store run artifacts"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and input without executing nodes"),
) -> None:
    """Execute a system definition."""
    system_path = _resolve_path(system)
    system_def = _load_system_cli(system_path)
    payload = _load_input_payload(input_path=input, use_stdin=stdin)
    if dry_run:
        typer.secho("Dry run complete – system and input look good.", fg=typer.colors.GREEN)
        return
    engine = ExecutionEngine(system_def)
    run_root = Path(run_dir).expanduser().resolve()
    run_logger = RunLogger(run_root, system_def)
    run_logger.snapshot_system()
    run_logger.snapshot_input(payload)
    result = _execute_system(engine, payload)
    for record in result.node_results:
        run_logger.log_node(record)
    run_logger.finalize(result, config_path=system_path)
    _print_execution_result(result, run_logger.run_dir)


@app.command()
def benchmark(
    systems: List[str] = typer.Option(..., "--system", "-s", help="Path to system definition (repeat for multiple systems)"),
    inputs_dir: Optional[str] = typer.Option(None, "--inputs-dir", help="Directory containing benchmark input files"),
    input: Optional[List[str]] = typer.Option(None, "--input", "-i", help="Specific input file (repeatable)"),
    iterations: int = typer.Option(5, "--iterations", min=1, help="Number of runs per scenario"),
    batch_size: int = typer.Option(1, "--batch-size", min=1, help="Requests per OpenAI batch job (OpenAI only)"),
    expect_field: str = typer.Option(
        "expected",
        "--expect-field",
        help="Scenario field containing the expected answer/label (supports dot paths like expected.value)",
    ),
    output_field: str = typer.Option(
        "answer",
        "--output-field",
        help="System output field to score (supports dot paths like answer.value)",
    ),
    match_mode: str = typer.Option(
        "auto",
        "--match",
        help="Scoring mode: exact | one_of | auto (bool is legacy)",
    ),
    run_dir: str = typer.Option(DEFAULT_RUN_DIR, "--run-dir", help="Directory to store per-iteration artifacts"),
) -> None:
    """Run multiple labeled inputs repeatedly across one or more systems and report accuracy statistics."""
    if not systems:
        raise typer.BadParameter("At least one --system argument is required.")
    scenario_paths = _collect_scenario_paths(inputs_dir, input)
    if not scenario_paths:
        raise typer.BadParameter("No benchmark inputs provided. Use --inputs-dir and/or --input.")
    scenarios = _load_benchmark_scenarios(scenario_paths, expect_field, match_mode=match_mode)
    run_root = Path(run_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    benchmark_id = _generate_benchmark_id()
    suite_root = run_root / "benchmarks" / benchmark_id
    suite_root.mkdir(parents=True, exist_ok=False)
    system_summaries: list[SystemBenchmarkSummary] = []
    # Run systems in parallel so benchmarks don't block on each other.
    with ThreadPoolExecutor(max_workers=max(1, len(systems))) as executor:
        future_map = {}
        for system_arg in systems:
            system_path = _resolve_path(system_arg)
            system_def = _load_system_cli(system_path)
            typer.echo(f"\n=== Benchmarking system: {system_def.id} ({system_def.label or 'unnamed'}) ===")
            future = executor.submit(
                _run_benchmark_for_system,
                system_def=system_def,
                system_path=system_path,
                scenarios=scenarios,
                iterations=iterations,
                batch_size=batch_size,
                output_field=output_field,
                match_mode=match_mode,
                suite_root=suite_root,
                benchmark_id=benchmark_id,
                expect_field=expect_field,
            )
            future_map[future] = system_def
        for future in future_map:
            summary = future.result()
            system_summaries.append(summary)
            _print_benchmark_summary(summary.scenario_results, summary.summary_path)
            typer.echo(f"Artifacts stored under: {summary.benchmark_root}")
    suite_summary_path = suite_root / "suite_summary.json"
    _write_suite_summary(
        suite_summary_path,
        benchmark_id=benchmark_id,
        system_summaries=system_summaries,
        expect_field=expect_field,
        output_field=output_field,
        match_mode=match_mode,
        iterations_per_scenario=iterations,
    )
    _print_system_comparison_table(system_summaries, suite_summary_path)
    any_failures = any(result.failures for summary in system_summaries for result in summary.scenario_results)
    if any_failures:
        raise typer.Exit(code=1)


@app.command()
def replay(
    run: str = typer.Option(..., "--run", help="Path to an existing run directory or run.json"),
    run_dir: str = typer.Option(DEFAULT_RUN_DIR, "--run-dir", help="Directory to store the replayed run"),
) -> None:
    """Replay a prior run using its captured system snapshot and input."""
    run_reference = _resolve_path(run)
    run_directory = run_reference if run_reference.is_dir() else run_reference.parent
    system_snapshot = run_directory / "system.yaml"
    input_snapshot = run_directory / "input.json"
    if not system_snapshot.exists() or not input_snapshot.exists():
        raise typer.BadParameter("Run directory must contain system.yaml and input.json snapshots")
    system_def = _load_system_cli(system_snapshot)
    payload = _read_structured_file(input_snapshot)
    engine = ExecutionEngine(system_def)
    run_root = Path(run_dir).expanduser().resolve()
    run_logger = RunLogger(run_root, system_def)
    run_logger.snapshot_system()
    run_logger.snapshot_input(payload)
    result = _execute_system(engine, payload)
    for record in result.node_results:
        run_logger.log_node(record)
    run_logger.finalize(result)
    _print_execution_result(result, run_logger.run_dir)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface for the UI server"),
    port: int = typer.Option(8321, "--port", help="Port for the UI server"),
) -> None:
    """Run the local PSF UI server."""
    from .ui.server import serve

    typer.echo(f"Starting PSF UI at http://{host}:{port}")
    serve(host=host, port=port)


def _execute_system(engine: ExecutionEngine, payload: dict[str, Any]) -> ExecutionResult:
    try:
        return engine.run(payload)
    except ExecutionError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def _load_system_cli(path: Path) -> SystemDef:
    try:
        return load_system_definition(path)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def _load_input_payload(*, input_path: Optional[str], use_stdin: bool) -> dict[str, Any]:
    if input_path and use_stdin:
        raise typer.BadParameter("Specify either --input or --stdin, not both.")
    if input_path:
        data = _read_structured_file(_resolve_path(input_path))
    elif use_stdin:
        raw = sys.stdin.read()
        data = _read_structured_text(raw, "<stdin>")
    else:
        data = {}
    if not isinstance(data, dict):
        raise typer.BadParameter("System input must be a JSON/YAML object.")
    return data


def _read_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return _read_structured_text(text, str(path))


def _read_structured_text(text: str, source: str) -> Any:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"Failed to parse {source}: {exc}") from exc


def _print_execution_result(result: ExecutionResult, run_dir: Path) -> None:
    typer.echo(f"Run artifacts: {run_dir}")
    if result.status == "success":
        typer.secho("Run succeeded.", fg=typer.colors.GREEN)
        if result.final_output is not None:
            typer.echo(json.dumps(result.final_output, indent=2, ensure_ascii=False))
    else:
        typer.secho(f"Run failed: {result.error}", fg=typer.colors.RED)


def _collect_scenario_paths(inputs_dir: Optional[str], explicit_inputs: Optional[List[str]]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    skip_names = {"manifest.json", "manifest.yaml", "manifest.yml"}

    def _record(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        paths.append(path)

    if inputs_dir:
        directory = _resolve_path(inputs_dir)
        if not directory.is_dir():
            raise typer.BadParameter(f"--inputs-dir must be a directory: {directory}")
        for pattern in ("*.json", "*.yaml", "*.yml"):
            for candidate in sorted(directory.glob(pattern)):
                if candidate.is_file():
                    if candidate.name.lower() in skip_names:
                        continue
                    _record(candidate.resolve())

    if explicit_inputs:
        for entry in explicit_inputs:
            path = _resolve_path(entry)
            if path.is_dir():
                for pattern in ("*.json", "*.yaml", "*.yml"):
                    for candidate in sorted(path.glob(pattern)):
                        if candidate.is_file():
                            if candidate.name.lower() in skip_names:
                                continue
                            _record(candidate.resolve())
            elif path.is_file():
                _record(path.resolve())
            else:
                raise typer.BadParameter(f"Input path is neither file nor directory: {path}")

    return paths


def _normalize_match_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"bool", "exact", "one_of", "auto"}:
        raise typer.BadParameter("--match must be one of: bool, exact, one_of, auto")
    return normalized


def _short_value(value: Any, *, max_chars: int = 64) -> str:
    if isinstance(value, str):
        rendered = value.strip()
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            rendered = str(value)
    rendered = rendered.replace("\n", "\\n")
    if len(rendered) > max_chars:
        return rendered[: max_chars - 3] + "..."
    return rendered


def _normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return ("true" if value else "false").strip()
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True).strip()
    except TypeError:
        return str(value).strip()


_MISSING = object()


def _get_by_path(container: Any, path: str) -> Any:
    current = container
    for part in path.split("."):
        if not part:
            return _MISSING
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return _MISSING
    return current


def _infer_scenario_expectation(value: Any, *, match_mode: str, path: Path) -> tuple[str, Any, str]:
    mode = _normalize_match_mode(match_mode)
    if mode == "auto":
        mode = "one_of" if isinstance(value, list) else "exact"

    if mode == "bool":
        try:
            expected_bool = _coerce_bool(value)
        except ValueError as exc:
            raise typer.BadParameter(f"{path}: {exc}") from exc
        return mode, expected_bool, ("true" if expected_bool else "false")

    if mode == "one_of":
        if not isinstance(value, list):
            raise typer.BadParameter(f"{path}: expected '{match_mode}' values to be a list for --match one_of")
        if not value:
            raise typer.BadParameter(f"{path}: expected list for '{path}' must not be empty")
        normalized = [_normalize_answer(item) for item in value]
        return mode, normalized, "one_of"

    normalized = _normalize_answer(value)
    return "exact", normalized, _short_value(normalized)


def _load_benchmark_scenarios(
    paths: list[Path],
    expect_field: str,
    *,
    match_mode: str,
) -> list[BenchmarkScenario]:
    scenarios: list[BenchmarkScenario] = []
    for path in paths:
        data = _read_structured_file(path)
        if not isinstance(data, dict):
            raise typer.BadParameter(f"Benchmark input must be an object: {path}")
        expected = _get_by_path(data, expect_field)
        if expected is _MISSING:
            raise typer.BadParameter(f"Missing '{expect_field}' in {path}")

        input_payload: dict[str, Any]
        if "input" in data:
            raw_input = data.get("input")
            if not isinstance(raw_input, dict):
                raise typer.BadParameter(f"{path}: 'input' must be an object")
            input_payload = raw_input
        else:
            input_payload = dict(data)
            input_payload.pop("input", None)
            input_payload.pop("expected", None)
            input_payload.pop("meta", None)
            input_payload.pop("id", None)
            if "." not in expect_field:
                input_payload.pop(expect_field, None)

        scenario_mode, expected_value, expected_label = _infer_scenario_expectation(
            expected,
            match_mode=match_mode,
            path=path,
        )
        scenarios.append(
            BenchmarkScenario(
                name=path.stem,
                path=path,
                input_payload=input_payload,
                match_mode=scenario_mode,
                expected_value=expected_value,
                expected_label=expected_label,
            )
        )
    return scenarios


def _run_benchmark_for_system(
    *,
    system_def: SystemDef,
    system_path: Path,
    scenarios: List[BenchmarkScenario],
    iterations: int,
    batch_size: int,
    output_field: str,
    match_mode: str,
    suite_root: Path,
    benchmark_id: str,
    expect_field: str,
) -> SystemBenchmarkSummary:
    benchmark_root = suite_root / system_def.id
    benchmark_root.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    batch_size = max(1, batch_size)
    use_batch = batch_size > 1
    batch_provider: OpenAIBatchProvider | None = None
    if use_batch:
        batch_provider = OpenAIBatchProvider(batch_size=batch_size)

    def _provider_resolver(name: str):
        if use_batch and name == "openai" and batch_provider is not None:
            return batch_provider
        return get_provider(name)

    iteration_results: dict[str, list[BenchmarkIterationRecord]] = defaultdict(list)
    successes: dict[str, int] = defaultdict(int)
    futures = []

    def _execute_iteration(scenario: BenchmarkScenario, iteration_index: int) -> tuple[str, BenchmarkIterationRecord]:
        payload = copy.deepcopy(scenario.input_payload)
        scenario_slug = _sanitize_name(scenario.name)
        run_id = f"{benchmark_id}-{scenario_slug}-{iteration_index:03d}"
        run_logger = RunLogger(benchmark_root, system_def, run_id=run_id, use_system_subdir=False)
        run_logger.snapshot_system()
        run_logger.snapshot_input(payload)
        try:
            local_engine = ExecutionEngine(system_def, provider_resolver=_provider_resolver)
            result = local_engine.run(payload)
        except ExecutionError as exc:  # pragma: no cover - network/provider errors
            result = ExecutionResult(
                status="failure",
                final_output=None,
                context={"runtime": {}},
                node_results=[],
                metrics={},
                error=str(exc),
            )
        for record in result.node_results:
            run_logger.log_node(record)
        run_logger.finalize(result, config_path=system_path)

        iteration_status = "success"
        actual_label: Optional[str] = None
        detail: Optional[str] = None
        if result.status != "success" or result.final_output is None:
            iteration_status = "error"
            actual_label = "error"
            detail = result.error or "run failed without final output"
        else:
            final_output = result.final_output
            actual_raw = _get_by_path(final_output, output_field)
            if actual_raw is _MISSING:
                iteration_status = "missing_output"
                actual_label = "missing"
                detail = f"final output missing '{output_field}'"
            else:
                if scenario.match_mode == "bool":
                    try:
                        actual_bool = _coerce_bool(actual_raw)
                    except ValueError as exc:
                        iteration_status = "invalid_output"
                        actual_label = "invalid"
                        detail = str(exc)
                    else:
                        actual_label = "true" if actual_bool else "false"
                        expected_bool = bool(scenario.expected_value)
                        if actual_bool == expected_bool:
                            detail = final_output.get("reason")
                        else:
                            iteration_status = "mismatch"
                            detail = final_output.get("reason") or "mismatched label"
                elif scenario.match_mode == "one_of":
                    if not isinstance(scenario.expected_value, list):
                        iteration_status = "invalid_expected"
                        actual_label = "invalid_expected"
                        detail = "scenario expected_value is not a list"
                    else:
                        actual_value = _normalize_answer(actual_raw)
                        actual_label = _short_value(actual_value, max_chars=48)
                        expected_values = [_normalize_answer(item) for item in scenario.expected_value]
                        if actual_value in expected_values:
                            detail = final_output.get("reason")
                        else:
                            iteration_status = "mismatch"
                            sample = ", ".join(_short_value(v, max_chars=12) for v in expected_values[:5])
                            detail = f"value not in expected list (n={len(expected_values)}; sample={sample})"
                else:
                    expected_value = _normalize_answer(scenario.expected_value)
                    actual_value = _normalize_answer(actual_raw)
                    actual_label = _short_value(actual_value, max_chars=48)
                    if actual_value == expected_value:
                        detail = final_output.get("reason")
                    else:
                        iteration_status = "mismatch"
                        detail = f"expected {_short_value(expected_value)} got {_short_value(actual_value)}"
        return scenario.name, BenchmarkIterationRecord(
            iteration=iteration_index,
            run_dir=str(run_logger.run_dir),
            expected_label=scenario.expected_label,
            actual_label=actual_label,
            status=iteration_status,
            detail=detail,
        )

    max_workers = max(batch_size, 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for scenario in scenarios:
            for idx in range(iterations):
                futures.append(executor.submit(_execute_iteration, scenario, idx + 1))
        for future in as_completed(futures):
            scenario_name, record = future.result()
            iteration_results[scenario_name].append(record)
            if record.status == "success":
                successes[scenario_name] += 1

    scenario_results: list[BenchmarkResult] = []
    for scenario in scenarios:
        records = iteration_results.get(scenario.name, [])
        records.sort(key=lambda r: r.iteration)
        scenario_results.append(
            BenchmarkResult(
                scenario=scenario,
                total_runs=iterations,
                successes=successes.get(scenario.name, 0),
                iterations=records,
            )
        )
    finished_at = datetime.now(timezone.utc)
    summary_path = benchmark_root / "benchmark_summary.json"
    _write_benchmark_summary(
        summary_path=summary_path,
        system=system_def,
        benchmark_id=benchmark_id,
        results=scenario_results,
        started_at=started_at,
        finished_at=finished_at,
        expect_field=expect_field,
        output_field=output_field,
        match_mode=match_mode,
        iterations_per_scenario=iterations,
    )
    return SystemBenchmarkSummary(
        system_id=system_def.id,
        system_label=system_def.label,
        system_version=system_def.version,
        system_path=system_path,
        benchmark_root=benchmark_root,
        summary_path=summary_path,
        scenario_results=scenario_results,
    )


def _print_benchmark_summary(results: list[BenchmarkResult], summary_path: Optional[Path] = None) -> None:
    if not results:
        typer.echo("No benchmark scenarios executed.")
        return
    typer.echo("")
    typer.echo("Benchmark summary")
    header = f"{'Scenario':35} {'Expected':>9} {'Runs':>6} {'Correct':>8} {'Accuracy':>9}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for result in results:
        accuracy = (result.successes / result.total_runs) * 100 if result.total_runs else 0.0
        typer.echo(
            f"{result.scenario.name:35} "
            f"{result.scenario.expected_label:>9} "
            f"{result.total_runs:>6} "
            f"{result.successes:>8} "
            f"{accuracy:>8.1f}%"
        )
    failures = [res for res in results if res.failures]
    if not failures:
        typer.secho("\nAll scenarios matched the expected outputs.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n{len(failures)} scenario(s) had mismatches:", fg=typer.colors.YELLOW)
        for result in failures:
            typer.secho(f"- {result.scenario.name}: {result.failures}/{result.total_runs} incorrect", fg=typer.colors.YELLOW)
            for mismatch in result.mismatches:
                typer.echo(
                    f"    iteration {mismatch.iteration}: predicted {mismatch.actual_label} "
                    f"(expected {result.scenario.expected_label}) -> {mismatch.detail}"
                )
    if summary_path:
        typer.echo(f"\nBenchmark summary saved to: {summary_path}")


def _print_system_comparison_table(system_summaries: List[SystemBenchmarkSummary], suite_summary_path: Path) -> None:
    if not system_summaries:
        typer.echo("No systems benchmarked.")
        return
    if len(system_summaries) == 1:
        typer.echo(f"\nSuite summary saved to: {suite_summary_path}")
        return
    reference_results = system_summaries[0].scenario_results
    label_width = max(12, max(len(summary.system_label or summary.system_id) for summary in system_summaries))
    typer.echo("\nCross-system comparison")
    header = f"{'Scenario':35} {'Expected':>9}"
    for summary in system_summaries:
        label = (summary.system_label or summary.system_id)[:label_width]
        header += f" {label:>{label_width}}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for reference_result in reference_results:
        scenario_name = reference_result.scenario.name
        expected_label = reference_result.scenario.expected_label
        row = f"{scenario_name:35} {expected_label:>9}"
        for summary in system_summaries:
            scenario_result = _get_scenario_result(summary, scenario_name)
            accuracy = (scenario_result.successes / scenario_result.total_runs) * 100 if scenario_result.total_runs else 0.0
            row += f" {accuracy:>{label_width - 1}.1f}%"
        typer.echo(row)
    typer.echo("-" * len(header))
    overall_row = f"{'Overall':35} {'-':>9}"
    for summary in system_summaries:
        overall_pct = summary.overall_accuracy * 100
        overall_row += f" {overall_pct:>{label_width - 1}.1f}%"
    typer.echo(overall_row)
    typer.echo(f"\nSuite summary saved to: {suite_summary_path}")


def _get_scenario_result(summary: SystemBenchmarkSummary, scenario_name: str) -> BenchmarkResult:
    for result in summary.scenario_results:
        if result.scenario.name == scenario_name:
            return result
    raise RuntimeError(f"Scenario '{scenario_name}' not found in results for system '{summary.system_id}'")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_LABELS:
            return True
        if normalized in _FALSE_LABELS:
            return False
    raise ValueError("expected a boolean-like label (legal/illegal, true/false, 1/0)")


def _sanitize_name(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "scenario"


def _generate_benchmark_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"benchmark_{timestamp}_{suffix}"


def _write_benchmark_summary(
    summary_path: Path,
    *,
    system: SystemDef,
    benchmark_id: str,
    results: list[BenchmarkResult],
    started_at: datetime,
    finished_at: datetime,
    expect_field: str,
    output_field: str,
    match_mode: str,
    iterations_per_scenario: int,
) -> None:
    summary: dict[str, Any] = {
        "system_id": system.id,
        "system_label": system.label,
        "system_version": system.version,
        "benchmark_id": benchmark_id,
        "benchmark_dir": str(summary_path.parent),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "expect_field": expect_field,
        "output_field": output_field,
        "match_mode": match_mode,
        "iterations_per_scenario": iterations_per_scenario,
        "scenario_count": len(results),
        "scenarios": [],
    }
    for result in results:
        accuracy = (result.successes / result.total_runs) if result.total_runs else 0.0
        scenario_entry = {
            "name": result.scenario.name,
            "path": str(result.scenario.path),
            "expected_label": result.scenario.expected_label,
            "total_runs": result.total_runs,
            "successes": result.successes,
            "failures": result.failures,
            "accuracy": accuracy,
            "runs": [
                {
                    "iteration": record.iteration,
                    "run_dir": record.run_dir,
                    "status": record.status,
                    "expected_label": record.expected_label,
                    "actual_label": record.actual_label,
                    "detail": record.detail,
                }
                for record in result.iterations
            ],
        }
        summary["scenarios"].append(scenario_entry)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def _write_suite_summary(
    summary_path: Path,
    *,
    benchmark_id: str,
    system_summaries: List[SystemBenchmarkSummary],
    expect_field: str,
    output_field: str,
    match_mode: str,
    iterations_per_scenario: int,
) -> None:
    data: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_count": len(system_summaries),
        "expect_field": expect_field,
        "output_field": output_field,
        "match_mode": match_mode,
        "iterations_per_scenario": iterations_per_scenario,
        "systems": [],
        "scenarios": [],
    }
    for summary in system_summaries:
        data["systems"].append(
            {
                "system_id": summary.system_id,
                "system_label": summary.system_label,
                "system_version": summary.system_version,
                "system_path": str(summary.system_path),
                "benchmark_root": str(summary.benchmark_root),
                "summary_path": str(summary.summary_path),
                "overall_accuracy": summary.overall_accuracy,
            }
        )
    if system_summaries:
        reference_results = system_summaries[0].scenario_results
        for reference in reference_results:
            scenario_entry = {
                "name": reference.scenario.name,
                "path": str(reference.scenario.path),
                "expected_label": reference.scenario.expected_label,
                "systems": [],
            }
            for summary in system_summaries:
                scenario_result = _get_scenario_result(summary, reference.scenario.name)
                accuracy = (scenario_result.successes / scenario_result.total_runs) if scenario_result.total_runs else 0.0
                scenario_entry["systems"].append(
                    {
                        "system_id": summary.system_id,
                        "system_label": summary.system_label,
                        "accuracy": accuracy,
                        "successes": scenario_result.successes,
                        "failures": scenario_result.failures,
                        "total_runs": scenario_result.total_runs,
                    }
                )
            data["scenarios"].append(scenario_entry)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover
    app()
