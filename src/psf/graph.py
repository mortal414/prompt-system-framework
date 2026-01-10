from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import load_system_tree
from .models import InputBinding, NodeDef, SystemDef


def export_resolved_graph(path: str | Path) -> dict[str, Any]:
    """Export a machine-readable graph representation for a system.

    The export is intended for deterministic introspection (nodes, edges, nested subsystems) and
    includes resolved `use_system` references.
    """

    root_path = Path(path).expanduser().resolve()
    root, systems = load_system_tree(root_path)
    systems_payload: dict[str, Any] = {}
    for system_path, system_def in sorted(systems.items(), key=lambda item: str(item[0])):
        systems_payload[str(system_path)] = _export_system(system_def, system_path=system_path, systems=systems)
    return {
        "format": "psf_graph_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_system": {
            "path": str(root_path),
            "id": root.id,
            "label": root.label,
            "version": root.version,
        },
        "systems": systems_payload,
    }


def _export_system(system: SystemDef, *, system_path: Path, systems: dict[Path, SystemDef]) -> dict[str, Any]:
    nodes = [_export_node(node, systems=systems) for node in system.nodes]
    control_edges: list[dict[str, Any]] = []
    data_edges: list[dict[str, Any]] = []
    subsystem_edges: list[dict[str, Any]] = []

    for node in system.nodes:
        control_edges.extend(_export_routing_edges(node))
        if node.loop_control:
            control_edges.append(
                {
                    "kind": "loop_target",
                    "from": node.id,
                    "to": node.loop_control.loop_target_node,
                    "counter_key": node.loop_control.counter_key,
                    "max_iterations": node.loop_control.max_iterations,
                    "exit_on_success": node.loop_control.exit_on_success,
                }
            )
        data_edges.extend(_export_binding_edges(node))
        if node.use_system:
            subsystem_edges.append(
                {
                    "kind": "use_system",
                    "node_id": node.id,
                    "subsystem_path": node.use_system,
                }
            )

    for output_name, output_field in system.system_output.fields.items():
        data_edges.append(
            {
                "kind": "system_output",
                "to_output": output_name,
                "from_path": output_field.from_path,
            }
        )

    return {
        "path": str(system_path),
        "id": system.id,
        "label": system.label,
        "version": system.version,
        "entry_node_id": system.entry_node_id,
        "system_input": system.system_input.model_dump(exclude_none=True),
        "system_output": system.system_output.model_dump(by_alias=True, exclude_none=True),
        "nodes": nodes,
        "control_edges": control_edges,
        "data_edges": data_edges,
        "subsystem_edges": subsystem_edges,
    }


def _export_node(node: NodeDef, *, systems: dict[Path, SystemDef]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": node.id,
        "label": node.label,
        "terminal": node.terminal,
        "max_retries": node.max_retries,
        "input_bindings": _dump_input_bindings(node.input_bindings),
        "routing": node.routing.model_dump(exclude_none=True),
        "loop_control": node.loop_control.model_dump(exclude_none=True) if node.loop_control else None,
    }
    if node.use_system:
        data["kind"] = "system"
        data["use_system"] = node.use_system
        subsystem = systems.get(Path(node.use_system).expanduser().resolve())
        if subsystem is not None:
            data["resolved_use_system"] = {
                "path": node.use_system,
                "id": subsystem.id,
                "label": subsystem.label,
                "version": subsystem.version,
            }
    else:
        data["kind"] = "llm"
        data["provider"] = node.provider
        data["model"] = node.model
        data["temperature"] = node.temperature
        data["max_tokens"] = node.max_tokens
        data["extra_params"] = dict(node.extra_params or {})
        data["instructions"] = node.instructions
        data["prompt_template"] = node.prompt_template
        data["output_mode"] = node.output_mode
        data["output_schema"] = node.output_schema
    return data


def _dump_input_bindings(bindings: dict[str, InputBinding]) -> dict[str, Any]:
    return {
        name: binding.model_dump(by_alias=True, exclude_none=True)
        for name, binding in bindings.items()
    }


def _export_binding_edges(node: NodeDef) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for input_name, binding in node.input_bindings.items():
        if not binding.from_path:
            continue
        edges.append(
            {
                "kind": "input_binding",
                "to_node": node.id,
                "to_input": input_name,
                "from_path": binding.from_path,
            }
        )
    return edges


def _export_routing_edges(node: NodeDef) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for kind, rules in (
        ("on_success", node.routing.on_success),
        ("on_failure", node.routing.on_failure),
        ("on_loop_exhausted", node.routing.on_loop_exhausted),
    ):
        for rule in rules:
            edges.append(
                {
                    "kind": kind,
                    "from": node.id,
                    "to": rule.next,
                    "condition": rule.condition,
                }
            )
    return edges
