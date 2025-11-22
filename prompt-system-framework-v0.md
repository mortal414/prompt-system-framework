# Prompt System Framework – v0 Implementation Plan (Updated)

This document defines the first implementable version of the **Prompt System Framework**: a config-driven, Python-based engine for running **systems of prompts** (graphs of LLM calls), with a CLI entrypoint.

Focus for v0:

- **Config-first** (YAML/JSON) definitions of systems and nodes.  
- **Python library + CLI** runtime.  
- **Single provider: OpenAI** (but provider-agnostic abstraction).  
- **Explicit configuration**, minimal magic.  
- **Logging + token usage tracking** from day one.  
- **No UI yet** – but system representation is UI-ready (graph model).

---

## 1. Scope & Non-Goals (v0)

### 1.1 In Scope

- Represent a **System** as a graph of **Nodes** with:
  - A required **`entry_node_id`** (no implicit entry).
  - Explicit **system input** schema (fields from external caller).
  - Explicit **system output** mapping (from context → final result).

- Each **Node**:
  - Has instructions and a prompt template.
  - Pulls **inputs** from:
    - System input.
    - Outputs of other nodes.
    - Constants.
  - Produces:
    - **Raw output** (text) and/or
    - **Structured output** (JSON-like dict validated against a schema).
  - Defines routing behavior:
    - Next node(s).
    - Loop behavior via `max_iterations`.
    - Simple condition expressions on outputs.

- **Execution engine**:
  - Runs nodes sequentially, with branching and loops.
  - Maintains a shared context.
  - Stops on:
    - Reaching a terminal node, or
    - A hard failure.

- **LLM integration**:
  - OpenAI client with configurable model, temperature, etc.
  - Token usage and model info tracked per node execution.

- **Logging & runs**:
  - Per-run metadata (`run.json`).
  - Per-node logs (`nodes.jsonl`).
  - Token usage and latency tracking.

- **CLI**:
  - Run systems.
  - Validate system definitions.
  - Inspect graph structure.
  - Replay runs.

- **Example system**:
  - Chess move proposal + validation pipeline.

### 1.2 Out of Scope (v0)

- UI / visual editor (planned for later).  
- Advanced testing framework (node-level & system-level tests will come later; for now, manual testing with small examples).  
- Multiple providers (only OpenAI, but abstraction designed for extension).  
- Rich routing expression languages (start with a small DSL; JMESPath or similar can be added later).  
- Complex error routing (for now: fail fast on unrecoverable node failures).

---

## 2. Architecture Overview

### 2.1 High-Level Components

- **Core domain models**
  - `SystemDef` – entire system definition (nodes, I/O, entry node).
  - `NodeDef` – node definition (instructions, prompt, I/O, routing, loops).
  - `RoutingRule` – condition + target node.
  - `LoopControl` – max iterations, loop counters, exit conditions.
  - `ModelConfig` – provider + model parameters.

- **Config layer**
  - Loader for YAML/JSON → `SystemDef`.
  - Validation of schema and wiring (explicit configuration, minimal defaults).

- **Execution engine**
  - Context store for system input, node outputs, runtime metadata.
  - Node scheduling + routing + loops based on conditions/loop control.
  - Error handling and run termination.

- **Provider layer**
  - `LLMProvider` interface.
  - `OpenAIProvider` implementation (tokens + model metadata extracted).

- **Logging & run store**
  - Per-run directory with:
    - System snapshot.
    - Input snapshot.
    - Node-level logs.
    - Run metadata (including token usage).

- **CLI**
  - `psf run`, `psf validate`, `psf inspect`, `psf replay`.

### 2.2 Data Flow (v0)

1. CLI command (e.g., `psf run`) is invoked with a **system config path** and **input**.  
2. Config loader reads and validates the YAML/JSON → `SystemDef`.  
3. Engine initializes **Context**:
   - `system.*` fields from input.
   - Runtime metadata fields.  
4. Engine locates `entry_node_id` and begins execution.  
5. For each node:
   - Resolve `input_bindings` from `Context`.
   - Render prompt template.
   - Call provider (OpenAI).
   - Record token usage + latency.
   - Parse output (raw and/or structured).
   - Update `Context`.
   - Evaluate routing (conditions + loop control).  
6. On reaching a terminal node or a failure:
   - Engine writes run artifacts.
   - Returns final result (or error) to caller.

---

## 3. Configuration Model

v0 uses **explicit, verbose configuration** to keep behavior transparent and predictable.

### 3.1 System Definition Schema (Conceptual)

Top-level structure:

```yaml
id: "chess_move_system"
label: "Chess Move Proposal + Validation"
version: "0.1.0"

entry_node_id: "propose_move"

provider_defaults:
  openai:
    model: "gpt-4o-mini"
    temperature: 0.2
    max_tokens: 512

system_input:
  fields:
    position_fen:
      type: string
      required: true
    game_context:
      type: string
      required: false

system_output:
  fields:
    move_san:
      from: "node.parse_proposed_move.output.move_san"
    rationale:
      from: "node.parse_proposed_move.output.rationale"
    final_message:
      from: "node.final_response.raw_output"

nodes:
  - # NodeDef objects (see below)
```

Key decisions reflected:

- **`entry_node_id` is required.**  
- **System output** is a mapping from context paths (e.g. `node.X.output.field`) to named fields in the final result.  
- System input has explicit type and `required` flags for clarity and future validation.

### 3.2 Node Definition Schema (Conceptual)

A node has:

```yaml
- id: "validate_move"
  label: "Validate proposed move"

  provider: "openai"        # optional; defaults from provider_defaults
  model: "gpt-4o-mini"      # optional override
  temperature: 0.0          # optional override
  max_tokens: 256           # optional override

  instructions: |
    You validate whether the proposed SAN move is legal in the given FEN position.

  prompt_template: |
    Position (FEN): {{ position_fen }}
    Proposed move (SAN): {{ move_san }}

    Respond with JSON:
    {
      "is_valid": true or false,
      "reason": "<short explanation>"
    }

  input_bindings:
    position_fen:
      from: "system.position_fen"
    move_san:
      from: "node.parse_proposed_move.output.move_san"

  output_mode: "structured"  # "raw" | "structured" | "both"

  output_schema:
    type: object
    properties:
      is_valid: { type: boolean }
      reason: { type: string }
    required: [is_valid]

  routing:
    on_success:
      - condition: "output.is_valid == true"
        next: "final_response"
      - condition: "output.is_valid == false"
        next: "handle_invalid_move"

  loop_control: null        # see section 6.3

  terminal: false           # true for terminal nodes
```

Notes:

- `input_bindings` explicitly map node-local variable names to context paths.  
- `output_mode` controls whether we parse response as JSON and/or retain raw text.  
- `routing` uses a small DSL (see next section).  
- `loop_control` is optional; when present, this node participates in a loop.

---

## 4. Routing DSL

### 4.1 v0 DSL

Routing conditions are small, explicit expressions evaluated in a **restricted environment**:

- Available names:
  - `output` – the node’s structured output (if any).
  - `context` – read-only view of global context (optional in v0).
- Supported operations (v0):
  - Equality / inequality: `==`, `!=`
  - Boolean ops: `and`, `or`, `not`
  - Comparison for numbers: `<`, `>`, `<=`, `>=`
  - Attribute access via dot notation:
    - `output.is_valid`
    - `output.score`
- Examples:
  - `"output.is_valid == true"`
  - `"output.score >= 0.7"`
  - `"output.label == 'ACCEPT'"`

For unconditional routing:

```yaml
routing:
  on_success:
    - next: "some_node"  # condition omitted → always true
```

### 4.2 JMESPath (Future)

- v0 does **not** require JMESPath or similar.  
- Implementation should keep the routing evaluation layer isolated so that later we can:
  - Swap to or augment with JMESPath / JSONPath.
  - Support more complex expressions without changing node configs too much.

---

## 5. Type System & Schemas

### 5.1 Supported Types

We use a **minimal JSON-Schema-inspired** subset:

- **Primitive types**: `string`, `number`, `boolean`  
- **Composite**: `object`, `array`  
- Optional support for `enum` (string enums) in v0 if it’s easy.

Example:

```yaml
output_schema:
  type: object
  properties:
    decision:
      type: string
      enum: ["ACCEPT", "REJECT", "RETRY"]
    confidence:
      type: number
  required: [decision]
```

Validation in v0:

- Parse output JSON.  
- Ensure:
  - All `required` fields exist.
  - Value types roughly match (`isinstance`-style checks).
  - If `enum` present, value is in set.

### 5.2 JSON vs Non-JSON Outputs

- **Structured outputs**:
  - Node prompt must elicit JSON.
  - Engine parses and validates against `output_schema`.
- **Raw outputs**:
  - Node response treated as opaque string.
  - Can be consumed by downstream nodes or parser nodes.

Pattern encouraged for better accuracy:

- Node A: **task-focused, raw** output.  
- Node B: **parser** node that converts Node A’s output into structured JSON.

---

## 6. Execution Engine

### 6.1 Context Store

Global `Context` map with namespaced keys:

- `system.position_fen`  
- `system.game_context`  
- `node.propose_move.raw_output`  
- `node.propose_move.output.move_san`  
- `runtime.loop_counters.invalid_move_attempts`  
- `runtime.started_at`, etc.

Nodes only see the **values** they bind via `input_bindings`, but the engine has full context for routing and logging.

### 6.2 Node Execution Lifecycle

For each node:

1. **Input resolution**
   - For each entry in `input_bindings`, fetch the value from `Context`.
   - If missing:
     - Node fails (for v0, run fails).

2. **Prompt rendering**
   - Render `prompt_template` with a simple templating engine.
   - Template variables come from resolved inputs (e.g., `position_fen`, `game_context`).

3. **LLM call**
   - Create `ModelConfig` by merging:
     - Node overrides.
     - Provider/system defaults.
   - Call `provider.complete(prompt, config)`.

4. **Token usage & metadata**
   - Capture:
     - `prompt_tokens`, `completion_tokens`, `total_tokens` (if available).
     - Request/response IDs.

5. **Output handling**
   - Save `raw_output`.
   - If `output_mode` includes `"structured"`:
     - Parse the output as JSON.
     - Validate against `output_schema`.
     - If parsing/validation fails:
       - Apply retry policy (if any), else treat as node failure).

6. **Context update**
   - Update `Context` with:
     - `node.<id>.raw_output`
     - `node.<id>.output.*` fields

7. **Routing & loops**
   - Evaluate routing conditions in defined order.
   - Apply `loop_control` if present (see below).
   - Decide next node or terminate.

### 6.3 Loops & Loop Control

v0 loop semantics:

- Loop is modeled via **`loop_control`** on a node (typically an “error handling” or “retry” node):

```yaml
loop_control:
  counter_key: "invalid_move_attempts"
  max_iterations: 3
  loop_target_node: "propose_move"
  exit_on_success: true
```

- Engine maintains `runtime.loop_counters.counter_key` in `Context`.  
- On node success:
  - If `exit_on_success: true`, engine **does not** continue looping – it follows normal `routing.on_success`.
  - Otherwise, if `counter < max_iterations`, engine routes to `loop_target_node`.
- When `counter >= max_iterations`:
  - Loop for this node is considered **exhausted**.
  - v0 behavior: either:
    - **Fail run**, or
    - Follow a special `on_loop_exhausted` route if defined:

```yaml
routing:
  on_success:
    - next: "propose_move"
  on_loop_exhausted:
    - next: "final_response"  # e.g., "we failed to find a legal move"
```

Future extension (not implemented in v0 but planned):

- Allow nodes to signal loop termination due to **failure** (e.g., explicit `stop_loop` flag in output).

### 6.4 Error Handling (v0)

Types of errors:

- Missing inputs.  
- LLM failures (network, rate limit, etc.).  
- JSON parse / validation errors.  
- No routing rule matches.

v0 behavior:

- Optional **simple retry** for transient LLM errors (per-node or global config).  
- If retries exhausted or non-transient error:
  - Run **fails**.
  - No `on_error` routing in v0 (kept simpler).
- Failure is recorded in `run.json` and `nodes.jsonl`.

---

## 7. LLM Provider Integration (OpenAI v0)

### 7.1 Provider Interface

```python
class LLMProvider(Protocol):
    def complete(self, prompt: str, config: ModelConfig) -> LLMResult:
        ...
```

Where:

```python
@dataclass
class ModelConfig:
    provider: str
    model: str
    temperature: float | None
    max_tokens: int | None
    extra_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResult:
    raw_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    model: str | None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
```

### 7.2 OpenAI Implementation

- `OpenAIProvider`:
  - Uses the official OpenAI Python client.
  - Maps `ModelConfig` → `client.responses.create(...)` or similar.
  - Extracts token usage from response.

- At the System level:
  - `provider_defaults.openai` gives base `ModelConfig`.  
- At the Node level:
  - Node `provider`, `model`, `temperature`, etc., override defaults.

---

## 8. Logging, Metrics, Runs & Replay

### 8.1 Run Artifacts

Per run, we create a directory like:

```text
runs/
  chess_move_system/
    2025-11-21T12-34-56Z_abcdef/
      run.json
      nodes.jsonl
      system.yaml
      input.json
```

**`run.json`** structure (conceptual):

```json
{
  "system_id": "chess_move_system",
  "system_version": "0.1.0",
  "run_id": "2025-11-21T12-34-56Z_abcdef",
  "started_at": "2025-11-21T12:34:56Z",
  "finished_at": "2025-11-21T12:35:02Z",
  "status": "success",
  "entry_node_id": "propose_move",
  "final_output": {
    "move_san": "Nf3",
    "rationale": "Develops a piece and controls the center.",
    "final_message": "I recommend Nf3..."
  },
  "metrics": {
    "total_nodes_executed": 4,
    "total_prompt_tokens": 320,
    "total_completion_tokens": 220,
    "total_tokens": 540,
    "nodes": {
      "propose_move": {
        "prompt_tokens": 100,
        "completion_tokens": 80,
        "total_tokens": 180
      },
      "parse_proposed_move": {
        "prompt_tokens": 60,
        "completion_tokens": 40,
        "total_tokens": 100
      }
    }
  },
  "models_used": ["gpt-4o-mini"]
}
```

**`nodes.jsonl`** entries:

```json
{
  "timestamp": "2025-11-21T12:34:57Z",
  "node_id": "validate_move",
  "prompt": "Position (FEN): ...",
  "raw_output": "{"is_valid": true, "reason": "..."}",
  "parsed_output": { "is_valid": true, "reason": "..." },
  "duration_ms": 340,
  "provider": "openai",
  "model": "gpt-4o-mini",
  "prompt_tokens": 60,
  "completion_tokens": 40,
  "total_tokens": 100,
  "error": null
}
```

v0 includes:

- Per-node and per-run **token usage**.  
- Basic timing and model identification.

### 8.2 Replay

`psf replay`:

- Takes a `run.json` or run directory.  
- Reloads `system.yaml` and `input.json`.  
- Executes system again (non-deterministic due to LLM, but same config).

---

## 9. CLI Design

Assume a modern CLI library (e.g., `typer`) on **Python ≥ 3.11**.

### 9.1 Commands

```bash
# Run a system
psf run   --system systems/chess_move_system.yaml   --input examples/input/chess_position.json   --run-dir runs/

# Validate system definition
psf validate --system systems/chess_move_system.yaml

# Inspect system graph
psf inspect --system systems/chess_move_system.yaml

# Replay a previous run
psf replay --run runs/chess_move_system/2025-11-21T12-34-56Z_abcdef/run.json
```

#### `psf run`

- Required:
  - `--system PATH`
  - `--input PATH` (JSON/YAML) or `--stdin` for piped JSON.
- Optional:
  - `--run-dir PATH` (defaults to `runs/`).
  - `--dry-run` (validate only; don’t call LLM).

#### `psf validate`

- Checks:
  - Top-level structure.
  - `entry_node_id` exists and is valid.
  - Node IDs are unique.
  - `input_bindings` refer to valid context paths.
  - `routing.next` references existing nodes.
  - `loop_control` references valid `loop_target_node` and positive `max_iterations`.
  - Basic `output_schema` sanity.

#### `psf inspect`

- Prints:
  - Node list with providers/models.
  - Adjacency (routing graph).
  - Highlight entry and terminal nodes.

#### `psf replay`

- Reads `run.json` and associated `system.yaml` and `input.json`.  
- Re-executes the system; optionally prints diff vs original run.

---

## 10. Recommended Project Layout

Suggested lightweight layout for a repo:

```text
prompt-system-framework/
  pyproject.toml
  src/
    psf/
      __init__.py
      cli.py
      config_loader.py
      models.py        # SystemDef, NodeDef, RoutingRule, etc.
      engine.py        # Execution engine
      providers/
        __init__.py
        openai_provider.py
      logging/
        __init__.py
        run_logger.py
  systems/
    chess_move_system.yaml
  examples/
    input/
      chess_position_1.json
  runs/
    # created by psf run
  tests/
    test_config_loader.py
    test_engine_basic.py
```

- Users define their own systems under `systems/`.  
- `runs/` is the default run artifact root.  
- `examples/` holds initial demo inputs and systems.

---

## 11. Initial Implementation Milestones

1. **Core models & validation**
   - Implement `SystemDef`, `NodeDef`, etc. using dataclasses or Pydantic.
   - Implement `config_loader` → YAML/JSON → `SystemDef`.
   - Implement `psf validate`.

2. **Provider layer**
   - Implement `OpenAIProvider.complete()`.
   - Include token usage extraction.

3. **Execution engine**
   - Context store + node execution cycle.
   - Routing DSL evaluator.
   - Loop control with `max_iterations` and `exit_on_success`.

4. **Logging & runs**
   - `run_logger` to write `run.json` and `nodes.jsonl`.
   - Include token metrics.

5. **CLI**
   - `run`, `validate`, `inspect`, `replay`.

6. **Chess example**
   - Implement `chess_move_system.yaml` using the patterns above.
   - Add example inputs and manual test instructions.

---

## 12. Remaining Design Notes & Future Questions

These don’t block v0, but are worth tracking as future decisions:

- How rich should the routing DSL become before we adopt a standard like JMESPath?  
- When adding providers beyond OpenAI:
  - How should we configure provider selection per node?
  - How do we normalize token metrics across providers?  
- When we add testing:
  - Best test authoring format (YAML vs Python).
  - Snapshot storage shape and integration with runs.

For now, v0 is intentionally **small, explicit, and debuggable**, with enough structure to grow into a UI-driven, multi-provider, test-heavy framework later.
