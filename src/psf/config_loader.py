from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import ValidationError

from .exceptions import ConfigError
from .models import SystemDef

SUPPORTED_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json"}


def load_system_definition(path: str | Path) -> SystemDef:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"System definition not found: {path}")
    raw_config = _load_raw_config(path)
    try:
        system = SystemDef.model_validate(raw_config)
    except ValidationError as exc:  # pragma: no cover - pydantic formatting is complex
        raise ConfigError(f"Invalid system definition: {exc}") from exc
    _validate_cross_references(system)
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


def _validate_cross_references(system: SystemDef) -> None:
    node_ids = {node.id for node in system.nodes}
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


def _validate_routing(rules: Iterable, node_ids: set[str], context: str) -> None:
    for rule in rules:
        if rule.next not in node_ids:
            raise ConfigError(f"{context} references unknown node '{rule.next}'")


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
