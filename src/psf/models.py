from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

OutputMode = Literal["raw", "structured", "both"]


class ModelConfig(BaseModel):
    provider: Optional[str] = None
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _ensure_provider(self) -> "ModelConfig":
        if not self.provider:
            self.provider = "openai"
        return self


class InputField(BaseModel):
    type: Literal["string", "number", "boolean", "object", "array"]
    required: bool = False
    description: Optional[str] = None


class SystemInputConfig(BaseModel):
    fields: Dict[str, InputField] = Field(default_factory=dict)


class SystemOutputField(BaseModel):
    from_path: str = Field(alias="from")
    description: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class SystemOutputConfig(BaseModel):
    fields: Dict[str, SystemOutputField]

    @model_validator(mode="after")
    def _ensure_fields(self) -> "SystemOutputConfig":
        if not self.fields:
            raise ValueError("system_output.fields must not be empty")
        return self


class InputBinding(BaseModel):
    from_path: Optional[str] = Field(default=None, alias="from")
    value: Optional[Any] = None
    required: bool = True
    default: Optional[Any] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _validate_source(self) -> "InputBinding":
        if self.from_path is None and self.value is None:
            raise ValueError("input binding must define either 'from' or 'value'")
        return self


class RoutingRule(BaseModel):
    next: str
    condition: Optional[str] = None


class RoutingConfig(BaseModel):
    on_success: List[RoutingRule] = Field(default_factory=list)
    on_failure: List[RoutingRule] = Field(default_factory=list)
    on_loop_exhausted: List[RoutingRule] = Field(default_factory=list)


class LoopControl(BaseModel):
    counter_key: str
    max_iterations: int = Field(gt=0)
    loop_target_node: str
    exit_on_success: bool = True


class NodeDef(BaseModel):
    id: str
    label: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    instructions: str
    prompt_template: str
    input_bindings: Dict[str, InputBinding] = Field(default_factory=dict)
    output_mode: OutputMode = "raw"
    output_schema: Optional[Dict[str, Any]] = None
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    loop_control: Optional[LoopControl] = None
    terminal: bool = False
    max_retries: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_output_schema(self) -> "NodeDef":
        if self.output_mode in {"structured", "both"} and not self.output_schema:
            raise ValueError("Structured output nodes must define output_schema")
        return self

    @model_validator(mode="after")
    def _trim_prompts(self) -> "NodeDef":
        if not self.instructions.strip():
            raise ValueError("instructions must not be empty")
        if not self.prompt_template.strip():
            raise ValueError("prompt_template must not be empty")
        return self


class SystemDef(BaseModel):
    id: str
    label: Optional[str] = None
    version: Optional[str] = None
    entry_node_id: str
    provider_defaults: Dict[str, ModelConfig] = Field(default_factory=dict)
    system_input: SystemInputConfig = Field(default_factory=SystemInputConfig)
    system_output: SystemOutputConfig
    nodes: List[NodeDef]

    @model_validator(mode="after")
    def _validate_structure(self) -> "SystemDef":
        if not self.nodes:
            raise ValueError("System must define at least one node")
        node_ids = [n.id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")
        if self.entry_node_id not in node_ids:
            raise ValueError("entry_node_id must reference an existing node")
        return self

    def node_index(self) -> Dict[str, NodeDef]:
        return {node.id: node for node in self.nodes}

    def get_node(self, node_id: str) -> NodeDef:
        node_lookup = self.node_index()
        if node_id not in node_lookup:
            raise KeyError(f"Unknown node id '{node_id}'")
        return node_lookup[node_id]
