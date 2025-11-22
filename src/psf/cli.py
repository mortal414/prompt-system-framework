from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import typer
import yaml

from .config_loader import load_system_definition
from .engine import ExecutionEngine, ExecutionResult
from .exceptions import ConfigError, ExecutionError
from .logging import RunLogger
from .models import SystemDef

app = typer.Typer(help="Prompt System Framework CLI")
DEFAULT_RUN_DIR = "runs"
_TRUE_LABELS = {"legal", "true", "t", "1", "yes", "y"}
_FALSE_LABELS = {"illegal", "false", "f", "0", "no", "n"}


@dataclass
class BenchmarkScenario:
    name: str
    path: Path
    payload: dict[str, Any]
    expected_bool: bool

    @property
    def expected_label(self) -> str:
        return "legal" if self.expected_bool else "illegal"


@dataclass
class BenchmarkMismatch:
    iteration: int
    actual_label: str
    detail: str


@dataclass
class BenchmarkResult:
    scenario: BenchmarkScenario
    total_runs: int
    successes: int
    mismatches: List[BenchmarkMismatch]

    @property
    def failures(self) -> int:
        return self.total_runs - self.successes


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
        typer.echo(f"- {node.id} | label={node.label or '-'} | provider={node.provider or '<default>'} | model={node.model or '<default>'}")
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
    system: str = typer.Option(..., "--system", help="Path to system definition"),
    inputs_dir: Optional[str] = typer.Option(None, "--inputs-dir", help="Directory containing benchmark input files"),
    input: Optional[List[str]] = typer.Option(None, "--input", "-i", help="Specific input file (repeatable)"),
    iterations: int = typer.Option(5, "--iterations", min=1, help="Number of runs per scenario"),
    expect_field: str = typer.Option("expected_legality", "--expect-field", help="Input field containing the expected legality label"),
    output_field: str = typer.Option("is_legal", "--output-field", help="System output field to compare"),
) -> None:
    """Run multiple labeled inputs repeatedly and report accuracy statistics."""
    system_path = _resolve_path(system)
    system_def = _load_system_cli(system_path)
    scenario_paths = _collect_scenario_paths(inputs_dir, input)
    if not scenario_paths:
        raise typer.BadParameter("No benchmark inputs provided. Use --inputs-dir and/or --input.")
    scenarios = _load_benchmark_scenarios(scenario_paths, expect_field)
    engine = ExecutionEngine(system_def)
    results: list[BenchmarkResult] = []
    for scenario in scenarios:
        result = _run_benchmark_for_scenario(engine, scenario, iterations, output_field)
        results.append(result)
    _print_benchmark_summary(results)
    if any(result.failures for result in results):
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
                    _record(candidate.resolve())

    if explicit_inputs:
        for entry in explicit_inputs:
            path = _resolve_path(entry)
            if path.is_dir():
                for pattern in ("*.json", "*.yaml", "*.yml"):
                    for candidate in sorted(path.glob(pattern)):
                        if candidate.is_file():
                            _record(candidate.resolve())
            elif path.is_file():
                _record(path.resolve())
            else:
                raise typer.BadParameter(f"Input path is neither file nor directory: {path}")

    return paths


def _load_benchmark_scenarios(paths: list[Path], expect_field: str) -> list[BenchmarkScenario]:
    scenarios: list[BenchmarkScenario] = []
    for path in paths:
        data = _read_structured_file(path)
        if not isinstance(data, dict):
            raise typer.BadParameter(f"Benchmark input must be an object: {path}")
        if expect_field not in data:
            raise typer.BadParameter(f"Missing '{expect_field}' in {path}")
        try:
            expected_bool = _coerce_bool(data[expect_field])
        except ValueError as exc:
            raise typer.BadParameter(f"{path}: {exc}") from exc
        scenarios.append(
            BenchmarkScenario(
                name=path.stem,
                path=path,
                payload=data,
                expected_bool=expected_bool,
            )
        )
    return scenarios


def _run_benchmark_for_scenario(
    engine: ExecutionEngine,
    scenario: BenchmarkScenario,
    iterations: int,
    output_field: str,
) -> BenchmarkResult:
    mismatches: list[BenchmarkMismatch] = []
    successes = 0
    for idx in range(iterations):
        payload = copy.deepcopy(scenario.payload)
        try:
            result = engine.run(payload)
        except ExecutionError as exc:  # pragma: no cover - network/provider errors
            mismatches.append(
                BenchmarkMismatch(iteration=idx + 1, actual_label="error", detail=str(exc))
            )
            continue
        if result.status != "success" or result.final_output is None:
            detail = result.error or "run failed without final output"
            mismatches.append(BenchmarkMismatch(iteration=idx + 1, actual_label="error", detail=detail))
            continue
        final_output = result.final_output
        if output_field not in final_output:
            mismatches.append(
                BenchmarkMismatch(
                    iteration=idx + 1,
                    actual_label="missing",
                    detail=f"final output missing '{output_field}'",
                )
            )
            continue
        try:
            actual_bool = _coerce_bool(final_output[output_field])
        except ValueError as exc:
            mismatches.append(
                BenchmarkMismatch(
                    iteration=idx + 1,
                    actual_label="invalid",
                    detail=f"{exc}",
                )
            )
            continue
        if actual_bool == scenario.expected_bool:
            successes += 1
        else:
            reason = final_output.get("reason") or "mismatched legality"
            mismatches.append(
                BenchmarkMismatch(
                    iteration=idx + 1,
                    actual_label="legal" if actual_bool else "illegal",
                    detail=reason,
                )
            )
    return BenchmarkResult(
        scenario=scenario,
        total_runs=iterations,
        successes=successes,
        mismatches=mismatches,
    )


def _print_benchmark_summary(results: list[BenchmarkResult]) -> None:
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
        typer.secho("\nAll scenarios matched the expected legality labels.", fg=typer.colors.GREEN)
        return
    typer.secho(f"\n{len(failures)} scenario(s) had mismatches:", fg=typer.colors.YELLOW)
    for result in failures:
        typer.secho(f"- {result.scenario.name}: {result.failures}/{result.total_runs} incorrect", fg=typer.colors.YELLOW)
        for mismatch in result.mismatches:
            typer.echo(
                f"    iteration {mismatch.iteration}: predicted {mismatch.actual_label} "
                f"(expected {result.scenario.expected_label}) -> {mismatch.detail}"
            )


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


if __name__ == "__main__":  # pragma: no cover
    app()
