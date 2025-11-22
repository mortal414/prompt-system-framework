from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from .config_loader import load_system_definition
from .engine import ExecutionEngine, ExecutionResult
from .exceptions import ConfigError, ExecutionError
from .logging import RunLogger
from .models import SystemDef

app = typer.Typer(help="Prompt System Framework CLI")
DEFAULT_RUN_DIR = "runs"


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


if __name__ == "__main__":  # pragma: no cover
    app()
