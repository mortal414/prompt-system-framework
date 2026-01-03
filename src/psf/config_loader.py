from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import ValidationError

from .exceptions import ConfigError
from .models import SystemDef

SUPPORTED_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json"}


def load_system_definition(path: str | Path) -> SystemDef:
    cache: dict[Path, SystemDef] = {}
    return _load_system_definition_internal(Path(path), cache=cache, stack=[])


def load_system_tree(path: str | Path) -> tuple[SystemDef, dict[Path, SystemDef]]:
    """Load a system definition and all transitive `use_system` dependencies.

    Returns the root `SystemDef` and a `{Path: SystemDef}` cache keyed by resolved absolute paths.
    """

    cache: dict[Path, SystemDef] = {}
    root = _load_system_definition_internal(Path(path), cache=cache, stack=[])
    return root, cache


def _load_system_definition_internal(
    path: Path,
    *,
    cache: dict[Path, SystemDef],
    stack: list[Path],
) -> SystemDef:
    path = path.expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise ConfigError(f"System definition not found: {path}")

    path = path.resolve()
    if path in stack:
        cycle = " -> ".join([*map(str, stack), str(path)])
        raise ConfigError(f"Detected system include cycle: {cycle}")
    cached = cache.get(path)
    if cached is not None:
        return cached

    raw_config = _load_raw_config(path)
    _resolve_use_system_paths(raw_config, base_dir=path.parent)
    try:
        system = SystemDef.model_validate(raw_config)
    except ValidationError as exc:  # pragma: no cover - pydantic formatting is complex
        raise ConfigError(f"Invalid system definition: {exc}") from exc
    _validate_cross_references(system)
    cache[path] = system
    _validate_nested_system_nodes(system, system_path=path, cache=cache, stack=[*stack, path])
    return system


def _load_raw_config(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_CONFIG_EXTENSIONS:
        raise ConfigError(
            f"Unsupported config format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_CONFIG_EXTENSIONS))}"
        )
    text = path.read_text()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        raise ConfigError(f"System definition is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError("Top-level system definition must be an object/dict")
    return data


def _resolve_use_system_paths(raw_config: dict[str, Any], *, base_dir: Path) -> None:
    nodes = raw_config.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        use_system = node.get("use_system")
        if use_system is None:
            continue
        if not isinstance(use_system, str) or not use_system.strip():
            raise ConfigError("use_system must be a non-empty path string")
        expanded = os.path.expandvars(use_system.strip())
        p = Path(expanded).expanduser()
        if not p.is_absolute():
            candidates = [
                (base_dir / p),
                (base_dir.parent / p),
                (Path.cwd() / p),
            ]
            chosen: Path | None = None
            for candidate in candidates:
                if candidate.exists():
                    chosen = candidate
                    break
            p = (chosen or candidates[0]).resolve()
        else:
            p = p.resolve()
        node["use_system"] = str(p)


def _validate_nested_system_nodes(
    system: SystemDef,
    *,
    system_path: Path,
    cache: dict[Path, SystemDef],
    stack: list[Path],
) -> None:
    for node in system.nodes:
        if not node.use_system:
            continue
        include_path = Path(node.use_system).expanduser().resolve()
        if not include_path.exists():
            raise ConfigError(f"node '{node.id}' use_system not found: {include_path}")
        included = _load_system_definition_internal(include_path, cache=cache, stack=stack)
        required_inputs = {name for name, field in included.system_input.fields.items() if field.required}
        missing = sorted(required_inputs - set(node.input_bindings.keys()))
        if missing:
            raise ConfigError(
                f"node '{node.id}' missing bindings for required subsystem inputs: {', '.join(missing)}"
            )

        subsystem_outputs = set(included.system_output.fields.keys())
        prefix = f"node.{node.id}.output."
        for parent_node in system.nodes:
            for binding_name, binding in parent_node.input_bindings.items():
                if not binding.from_path:
                    continue
                if binding.from_path.startswith(prefix):
                    field = binding.from_path[len(prefix) :].split(".", 1)[0]
                    if field and field not in subsystem_outputs:
                        raise ConfigError(
                            f"node '{parent_node.id}' input '{binding_name}' references '{binding.from_path}', "
                            f"but subsystem '{included.id}' does not define system_output field '{field}'"
                        )
        for output_name, output in system.system_output.fields.items():
            if output.from_path.startswith(prefix):
                field = output.from_path[len(prefix) :].split(".", 1)[0]
                if field and field not in subsystem_outputs:
                    raise ConfigError(
                        f"system output '{output_name}' references '{output.from_path}', "
                        f"but subsystem '{included.id}' does not define system_output field '{field}'"
                    )
        for rules, rule_context in (
            (node.routing.on_success, "on_success"),
            (node.routing.on_failure, "on_failure"),
            (node.routing.on_loop_exhausted, "on_loop_exhausted"),
        ):
            for rule in rules:
                if not rule.condition:
                    continue
                for field in _extract_output_fields(rule.condition):
                    if field not in subsystem_outputs:
                        raise ConfigError(
                            f"node '{node.id}' {rule_context} condition references output.{field}, "
                            f"but subsystem '{included.id}' does not define system_output field '{field}'"
                        )


def _validate_cross_references(system: SystemDef) -> None:
    node_lookup = system.node_index()
    node_ids = set(node_lookup.keys())
    system_input_fields = set(system.system_input.fields.keys())
    for node in system.nodes:
        for binding_name, binding in node.input_bindings.items():
            if binding.from_path:
                _validate_context_path(
                    binding.from_path,
                    node_ids=node_ids,
                    system_input_fields=system_input_fields,
                    context=f"node '{node.id}' input '{binding_name}'",
                )
                _validate_structured_output_reference(
                    binding.from_path,
                    node_lookup=node_lookup,
                    context=f"node '{node.id}' input '{binding_name}'",
                )
        _validate_routing(node.routing.on_success, node_ids, f"node '{node.id}' on_success")
        _validate_routing(node.routing.on_failure, node_ids, f"node '{node.id}' on_failure")
        _validate_routing(node.routing.on_loop_exhausted, node_ids, f"node '{node.id}' on_loop_exhausted")
        if node.loop_control and node.loop_control.loop_target_node not in node_ids:
            raise ConfigError(
                f"node '{node.id}' loop_control references unknown node '{node.loop_control.loop_target_node}'"
            )
    for output_name, output in system.system_output.fields.items():
        _validate_context_path(
            output.from_path,
            node_ids=node_ids,
            system_input_fields=system_input_fields,
            context=f"system output '{output_name}'",
        )
        _validate_structured_output_reference(
            output.from_path,
            node_lookup=node_lookup,
            context=f"system output '{output_name}'",
        )


def _validate_routing(rules: Iterable, node_ids: set[str], context: str) -> None:
    for rule in rules:
        if rule.next not in node_ids:
            raise ConfigError(f"{context} references unknown node '{rule.next}'")
        if rule.condition is not None:
            _validate_condition_expression(rule.condition, context=f"{context} condition")


def _validate_context_path(
    path: str,
    *,
    node_ids: set[str],
    system_input_fields: set[str],
    context: str,
) -> None:
    if not path:
        raise ConfigError(f"{context}: context path must not be empty")
    head, sep, tail = path.partition(".")
    if not sep:
        raise ConfigError(f"{context}: invalid context path '{path}'")
    if head == "system":
        field = tail.split(".")[0]
        if field not in system_input_fields:
            raise ConfigError(f"{context}: unknown system input '{field}'")
    elif head == "node":
        parts = tail.split(".")
        if len(parts) < 2:
            raise ConfigError(f"{context}: node path must include output segment (got '{path}')")
        node_id = parts[0]
        if node_id not in node_ids:
            raise ConfigError(f"{context}: unknown node '{node_id}'")
    elif head in {"runtime", "const"}:
        # runtime and const namespaces are reserved for engine-managed data.
        return
    else:
        raise ConfigError(f"{context}: unsupported context prefix '{head}'")


def _validate_structured_output_reference(
    path: str,
    *,
    node_lookup: dict[str, Any],
    context: str,
) -> None:
    if not path.startswith("node."):
        return
    parts = path.split(".")
    if len(parts) < 3:
        return
    node_id = parts[1]
    segment = parts[2]
    if segment != "output":
        return
    ref = node_lookup.get(node_id)
    if ref is None:
        return
    if getattr(ref, "use_system", None):
        return
    if getattr(ref, "output_mode", None) == "raw":
        raise ConfigError(
            f"{context}: context path '{path}' references structured output for node '{node_id}', "
            "but that node has output_mode='raw'"
        )


def _validate_condition_expression(expression: str, *, context: str) -> None:
    expr = expression.strip()
    if not expr:
        raise ConfigError(f"{context}: condition must not be empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ConfigError(f"{context}: invalid condition '{expression}': {exc}") from exc

    allowed_names = {"output", "context", "true", "false", "null"}
    allowed_nodes: tuple[type[ast.AST], ...] = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Attribute,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.Gt,
        ast.LtE,
        ast.GtE,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ConfigError(f"{context}: unsupported expression element '{node.__class__.__name__}'")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ConfigError(f"{context}: unsupported name '{node.id}' (allowed: {', '.join(sorted(allowed_names))})")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.Not):
            raise ConfigError(f"{context}: only 'not' unary operator is supported")
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, (ast.And, ast.Or)):
            raise ConfigError(f"{context}: only 'and'/'or' boolean operators are supported")
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
                    raise ConfigError(f"{context}: unsupported comparison operator '{op.__class__.__name__}'")


def _extract_output_fields(expression: str) -> set[str]:
    fields: set[str] = set()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return fields
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "output":
            fields.add(node.attr)
    return fields
