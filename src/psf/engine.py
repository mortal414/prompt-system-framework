from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from jinja2 import Environment, StrictUndefined, TemplateError

from .exceptions import ExecutionError, NodeExecutionError, ProviderError, RecoverableExecutionError
from .models import InputBinding, NodeDef, RoutingRule, SystemDef
from .providers import LLMProvider, LLMResult, get_provider

_JINJA_ENV = Environment(autoescape=False, undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)


@dataclass
class NodeRunRecord:
    node_id: str
    prompt: str = ""
    raw_output: Optional[str] = None
    parsed_output: Optional[dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    status: str
    final_output: Optional[dict[str, Any]]
    context: dict[str, Any]
    node_results: List[NodeRunRecord]
    metrics: dict[str, Any]
    error: Optional[str] = None


class ContextStore:
    def __init__(self, system_input: dict[str, Any]) -> None:
        self._data: dict[str, Any] = {
            "system": system_input,
            "node": {},
            "runtime": {
                "loop_counters": {},
                "node_history": [],
            },
        }

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, path: str) -> Any:
        parts = path.split(".")
        current: Any = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ExecutionError(f"Context value not found for path '{path}'")
        return current

    def set_node_result(
        self,
        node_id: str,
        *,
        raw_output: Optional[str],
        parsed_output: Optional[dict[str, Any]],
        provider: Optional[str],
        model: Optional[str],
        tokens: dict[str, Optional[int]],
        provider_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        bucket = self._data.setdefault("node", {}).setdefault(node_id, {})
        bucket["raw_output"] = raw_output
        if parsed_output is not None:
            bucket["output"] = parsed_output
        bucket["provider"] = provider
        bucket["model"] = model
        bucket["tokens"] = tokens
        if provider_metadata:
            bucket["provider_metadata"] = provider_metadata
        self._data["runtime"].setdefault("node_history", []).append(node_id)
        self._data["runtime"]["last_node_id"] = node_id

    def increment_loop_counter(self, key: str) -> int:
        counters = self._data["runtime"].setdefault("loop_counters", {})
        counters[key] = counters.get(key, 0) + 1
        return counters[key]

    def set_runtime(self, key: str, value: Any) -> None:
        self._data["runtime"][key] = value


@dataclass
class LoopDecision:
    next_node_id: Optional[str] = None
    exhausted: bool = False


class ExecutionEngine:
    def __init__(
        self,
        system: SystemDef,
        *,
        provider_resolver: Callable[[str], LLMProvider] | None = None,
    ) -> None:
        self.system = system
        self._provider_resolver = provider_resolver or get_provider
        self._node_index = system.node_index()

    def run(self, system_input: Optional[dict[str, Any]] = None) -> ExecutionResult:
        normalized_input = _normalize_system_input(self.system, system_input or {})
        context = ContextStore(normalized_input)
        started_at = datetime.now(timezone.utc)
        context.set_runtime("started_at", started_at.isoformat())
        records: list[NodeRunRecord] = []
        models_used: set[str] = set()
        status = "success"
        error_message: Optional[str] = None
        final_output: Optional[dict[str, Any]] = None
        current_node_id = self.system.entry_node_id

        while True:
            node = self._node_index[current_node_id]
            record, structured_output = self._execute_node(node, context)
            records.append(record)
            if record.error:
                status = "failure"
                error_message = record.error
                break
            if record.model:
                models_used.add(record.model)
            loop_decision = self._handle_loop_control(node, context)
            if loop_decision.exhausted:
                next_node = self._select_next(node.routing.on_loop_exhausted, structured_output, context)
                if next_node is None:
                    status = "failure"
                    error_message = (
                        f"Loop exhausted for node '{node.id}' without an on_loop_exhausted route"
                    )
                    break
                current_node_id = next_node
                continue
            if loop_decision.next_node_id:
                current_node_id = loop_decision.next_node_id
                continue
            if node.terminal:
                try:
                    final_output = self._build_system_output(context)
                except ExecutionError as exc:
                    status = "failure"
                    error_message = str(exc)
                break
            next_node = self._select_next(node.routing.on_success, structured_output, context)
            if next_node is None:
                status = "failure"
                error_message = f"No routing rule matched after node '{node.id}'"
                break
            current_node_id = next_node

        finished_at = datetime.now(timezone.utc)
        context.set_runtime("finished_at", finished_at.isoformat())
        metrics = _build_metrics(records, models_used)
        context.set_runtime("status", status)
        return ExecutionResult(
            status=status,
            final_output=final_output,
            context=context.data,
            node_results=records,
            metrics=metrics,
            error=error_message,
        )

    def _execute_node(
        self,
        node: NodeDef,
        context: ContextStore,
    ) -> tuple[NodeRunRecord, Optional[dict[str, Any]]]:
        record = NodeRunRecord(node_id=node.id)
        start_time = time.perf_counter()
        try:
            inputs = self._resolve_inputs(node, context)
            prompt = self._render_prompt(node, inputs)
            record.prompt = prompt
            model_config = self._build_model_config(node)
            record.provider = model_config.provider
            record.model = model_config.model
            llm_result, structured_output = self._invoke_node_model(node, model_config, prompt)
            record.raw_output = llm_result.raw_text if llm_result else None
            if structured_output is not None:
                record.parsed_output = structured_output
            if llm_result:
                record.prompt_tokens = llm_result.prompt_tokens
                record.completion_tokens = llm_result.completion_tokens
                record.total_tokens = llm_result.total_tokens
                record.provider_metadata = llm_result.provider_metadata
            context.set_node_result(
                node.id,
                raw_output=record.raw_output,
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
            return record, structured_output
        except NodeExecutionError as exc:
            if exc.raw_output is not None:
                record.raw_output = exc.raw_output
            record.error = str(exc)
            return record, None
        except ExecutionError as exc:
            record.error = str(exc)
            return record, None
        finally:
            record.duration_ms = (time.perf_counter() - start_time) * 1000

    def _resolve_inputs(self, node: NodeDef, context: ContextStore) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for name, binding in node.input_bindings.items():
            resolved[name] = _resolve_binding(name, binding, context)
        return resolved

    def _render_prompt(self, node: NodeDef, values: dict[str, Any]) -> str:
        instruction_block = _render_template(node.instructions, values)
        prompt_block = _render_template(node.prompt_template, values)
        if instruction_block.strip():
            return f"{instruction_block.strip()}\n\n{prompt_block.strip()}".strip()
        return prompt_block.strip()

    def _build_model_config(self, node: NodeDef) -> "ModelConfig":
        from .models import ModelConfig

        provider_name = node.provider or "openai"
        model = node.model
        if not model:
            raise ExecutionError(f"Node '{node.id}' is missing a model configuration")
        temperature = node.temperature
        max_tokens = node.max_tokens
        extra_params = dict(node.extra_params or {})
        return ModelConfig(
            provider=provider_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )

    def _invoke_node_model(
        self,
        node: NodeDef,
        model_config: "ModelConfig",
        prompt: str,
    ) -> tuple[Optional[LLMResult], Optional[dict[str, Any]]]:
        llm_result: Optional[LLMResult] = None
        last_raw_output: Optional[str] = None
        structured_output: Optional[dict[str, Any]] = None
        attempts = node.max_retries + 1
        provider = self._provider_resolver(model_config.provider)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                llm_result = provider.complete(prompt, model_config)
                last_raw_output = llm_result.raw_text
                structured_output = _handle_structured_output(node, llm_result.raw_text)
                return llm_result, structured_output
            except RecoverableExecutionError as exc:
                last_error = exc
            except ProviderError as exc:
                last_error = exc
            if attempt >= attempts:
                raise NodeExecutionError(
                    str(last_error or "Node execution failed"), raw_output=last_raw_output
                ) from last_error
        raise NodeExecutionError(f"Node '{node.id}' failed to complete", raw_output=last_raw_output)

    def _handle_loop_control(self, node: NodeDef, context: ContextStore) -> LoopDecision:
        if not node.loop_control:
            return LoopDecision()
        loop = node.loop_control
        counter = context.increment_loop_counter(loop.counter_key)
        if loop.exit_on_success:
            return LoopDecision()
        if counter < loop.max_iterations:
            return LoopDecision(next_node_id=loop.loop_target_node)
        return LoopDecision(exhausted=True)

    def _select_next(
        self,
        rules: Iterable[RoutingRule],
        output: Optional[dict[str, Any]],
        context: ContextStore,
    ) -> Optional[str]:
        for rule in rules:
            if rule.condition is None:
                return rule.next
            if _evaluate_condition(rule.condition, output or {}, context.data):
                return rule.next
        return None

    def _build_system_output(self, context: ContextStore) -> dict[str, Any]:
        final: dict[str, Any] = {}
        for name, mapping in self.system.system_output.fields.items():
            final[name] = context.get(mapping.from_path)
        return final


def _handle_structured_output(node: NodeDef, raw_text: str) -> Optional[dict[str, Any]]:
    if node.output_mode == "raw":
        return None
    candidate = _extract_json_candidate(raw_text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RecoverableExecutionError(
            f"Node '{node.id}' produced invalid JSON output: {exc}"
        ) from exc
    if node.output_schema:
        _validate_schema(parsed, node.output_schema, path=f"node.{node.id}.output")
    return parsed


def _extract_json_candidate(text: str) -> str:
    trimmed = text.strip()
    match = _CODE_BLOCK_RE.search(trimmed)
    if match:
        return match.group(1).strip()
    return trimmed


def _resolve_binding(name: str, binding: InputBinding, context: ContextStore) -> Any:
    if binding.from_path is not None:
        try:
            value = context.get(binding.from_path)
        except ExecutionError:
            if not binding.required:
                return binding.default
            raise
    else:
        value = binding.value
    if value is None and binding.required and binding.default is None:
        source = binding.from_path or "literal value"
        raise ExecutionError(f"Required input '{name}' (from {source}) is missing")
    return value if value is not None else binding.default


def _render_template(template_str: str, values: dict[str, Any]) -> str:
    try:
        return _JINJA_ENV.from_string(template_str).render(**values)
    except TemplateError as exc:
        raise ExecutionError(f"Failed to render template: {exc}") from exc


def _normalize_system_input(system: SystemDef, data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    for name, field in system.system_input.fields.items():
        if field.required and name not in normalized:
            raise ExecutionError(f"Missing required system input '{name}'")
        if name in normalized:
            value = normalized[name]
            if value is None:
                if field.required:
                    raise ExecutionError(f"System input '{name}' must not be null")
                continue
            if not _matches_type(value, field.type):
                raise ExecutionError(
                    f"System input '{name}' expected type {field.type} but got {type(value).__name__}"
                )
    return normalized


def _matches_type(value: Any, type_name: str) -> bool:
    match type_name:
        case "string":
            return isinstance(value, str)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case _:
            return False


def _validate_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise RecoverableExecutionError(f"{path} expected object but got {type(value).__name__}")
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in value:
                raise RecoverableExecutionError(f"{path} missing required field '{field}'")
        for key, sub_value in value.items():
            if key in properties:
                _validate_schema(sub_value, properties[key], path=f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise RecoverableExecutionError(f"{path} expected array but got {type(value).__name__}")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                _validate_schema(item, item_schema, path=f"{path}[{idx}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise RecoverableExecutionError(f"{path} expected string but got {type(value).__name__}")
        enum_values = schema.get("enum")
        if enum_values and value not in enum_values:
            raise RecoverableExecutionError(f"{path} must be one of {enum_values}")
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RecoverableExecutionError(f"{path} expected number but got {type(value).__name__}")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise RecoverableExecutionError(f"{path} expected boolean but got {type(value).__name__}")


def _build_metrics(records: list[NodeRunRecord], models_used: set[str]) -> dict[str, Any]:
    total_prompt = sum((r.prompt_tokens or 0) for r in records)
    total_completion = sum((r.completion_tokens or 0) for r in records)
    total_tokens = sum((r.total_tokens or 0) for r in records)
    per_node = {
        record.node_id: {
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "duration_ms": record.duration_ms,
        }
        for record in records
    }
    return {
        "total_nodes_executed": len(records),
        "total_prompt_tokens": total_prompt or 0,
        "total_completion_tokens": total_completion or 0,
        "total_tokens": total_tokens or 0,
        "models_used": sorted(models_used),
        "nodes": per_node,
    }


def _evaluate_condition(expression: str, output: dict[str, Any], context: dict[str, Any]) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - syntax errors are rare
        raise ExecutionError(f"Invalid routing condition '{expression}': {exc}") from exc
    namespace = {
        "output": _to_attr_dict(output),
        "context": _to_attr_dict(context),
        "true": True,
        "false": False,
        "null": None,
    }
    return bool(_eval_ast(tree.body, namespace))


def _eval_ast(node: ast.AST, namespace: dict[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast(value, namespace) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_ast(node.operand, namespace)
    elif isinstance(node, ast.Compare):
        left = _eval_ast(node.left, namespace)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator, namespace)
            if isinstance(op, ast.Eq) and not left == right:
                return False
            elif isinstance(op, ast.NotEq) and not left != right:
                return False
            elif isinstance(op, ast.Lt) and not left < right:
                return False
            elif isinstance(op, ast.Gt) and not left > right:
                return False
            elif isinstance(op, ast.LtE) and not left <= right:
                return False
            elif isinstance(op, ast.GtE) and not left >= right:
                return False
            left = right
        return True
    elif isinstance(node, ast.Attribute):
        value = _eval_ast(node.value, namespace)
        if isinstance(value, AttrDict):
            return value.get_attr(node.attr)
        if hasattr(value, node.attr):
            return getattr(value, node.attr)
        raise ExecutionError(f"Unknown attribute '{node.attr}' in routing condition")
    elif isinstance(node, ast.Name):
        if node.id in namespace:
            return namespace[node.id]
        raise ExecutionError(f"Unknown name '{node.id}' in routing condition")
    elif isinstance(node, ast.Constant):
        return node.value
    else:  # pragma: no cover - unsupported nodes are guarded here
        raise ExecutionError("Unsupported expression in routing condition")


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


class AttrDict(dict):
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough helper
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return value

    def get_attr(self, item: str) -> Any:
        value = self.__getattr__(item)
        if isinstance(value, dict):
            return AttrDict(value)
        return value
