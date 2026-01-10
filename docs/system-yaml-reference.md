# System YAML reference (canonical schema)

This is the stable config surface area for PSF. The canonical schema is defined by the Pydantic models:

- `src/psf/models.py` (`SystemDef`, `NodeDef`, `InputBinding`, …)

Machine-readable JSON schema:

- `psf schema` (prints to stdout) or `psf schema --out schema.json`

Machine-readable graph export (includes resolved `use_system`):

- `psf graph --system <sys.yaml> --out graph.json`

Systems can be authored as YAML or JSON. YAML examples below use the exact field names PSF validates.

## Top-level: `SystemDef`

Required:

- `id` (string): Stable system identifier (also used in `runs/<system_id>/...`).
- `entry_node_id` (string): Node id where execution starts.
- `system_output` (object): Declares the returned payload.
- `nodes` (array): Node definitions.

Optional:

- `label` (string)
- `version` (string)
- `system_input` (object, default `{fields: {}}`): Declares expected inputs.

## `system_input`

```yaml
system_input:
  fields:
    <name>:
      type: string|number|boolean|object|array
      required: true|false   # default: false
      description: "..."
```

PSF validates required inputs and basic types at runtime before executing the first node.

## `system_output`

```yaml
system_output:
  fields:
    <name>:
      from: "system.<field>" | "node.<node_id>.<path...>" | "runtime.<path...>"
      description: "..."     # optional
```

The `from` key is a context path; PSF resolves it at the terminal node to build `final_output`.

## Nodes: `nodes[]` (`NodeDef`)

A node is either:

- an **LLM node** (normal node), or
- a **system node** (composition) when `use_system` is set.

Common fields:

```yaml
- id: "..."
  label: "..."          # optional
  terminal: true|false  # default: false
  max_retries: 0        # default: 0 (extra attempts)

  input_bindings:
    <var_name>: {from: "system.position_fen"}
    <var_name>: {value: "literal"}
```

### `input_bindings` (`InputBinding`)

Each binding must provide either:

- `from: "<context path>"`, or
- `value: <literal>`

Optional binding controls:

- `required` (bool, default `true`)
- `default` (any)

If a `from` path cannot be resolved at runtime:

- when `required: false`, the binding becomes `default`
- otherwise, execution fails with an actionable error

### LLM node fields

```yaml
- id: "..."
  provider: "openai"         # default: openai
  model: "gpt-4o-mini"       # required for LLM nodes
  temperature: 0.2           # optional
  max_tokens: 256            # optional
  extra_params: {}           # optional (provider-specific)

  instructions: "..."        # required, non-empty
  prompt_template: "..."     # required, non-empty (Jinja2, StrictUndefined)

  output_mode: raw|structured|both     # default: raw
  output_schema: { ... }               # required when output_mode != raw

  routing:
    on_success:
      - next: "<node_id>"
        condition: "output.is_ok == true"   # optional
```

#### `output_mode` and `output_schema`

- `raw`: No JSON parsing. `output` is not available in routing conditions (it is `{}`).
- `structured` / `both`: PSF extracts a JSON candidate from the model text (supports plain JSON or a ```json code block), parses it, and validates it against `output_schema`.

`output_schema` supports a small JSON-schema-like subset:

- `type: object` with `properties` and `required`
- `type: array` with `items`
- `type: string` with optional `enum`
- `type: number`
- `type: boolean`

Schema validation failures are treated as retryable errors for `max_retries`.

### Routing (`RoutingConfig`)

```yaml
routing:
  on_success: []
  on_failure: []
  on_loop_exhausted: []
```

Each list contains rules:

```yaml
- next: "<node_id>"
  condition: "..."   # optional
```

Conditions are a small expression language (Python-like) evaluated with:

- `output`: the node’s parsed structured output (dict-like), or `{}` when none
- `context`: full execution context (`system`, `node`, `runtime`)
- literals: `true`, `false`, `null`

#### Routing + loop evaluation order

After each node execution, PSF evaluates control flow in this order:

1. If the node failed, try `routing.on_failure` (first matching rule wins).
2. Otherwise apply `loop_control` (increment counter; either loop, exhaust to `routing.on_loop_exhausted`, or continue).
3. If the node is `terminal: true`, build `final_output` from `system_output` and stop.
4. Otherwise, try `routing.on_success` (first matching rule wins).

### Loop control (`LoopControl`)

```yaml
loop_control:
  counter_key: "illegal_move_attempts"
  max_iterations: 5
  loop_target_node: "propose_move"
  exit_on_success: false
```

Loop counters are stored under `runtime.loop_counters.<counter_key>`.

### System nodes (`use_system`)

```yaml
- id: "validate_legality"
  use_system: "systems/chess_uci_legal_check_yn_system.yaml"
  input_bindings:
    position_fen: {from: "system.position_fen"}
    candidate_move: {from: "node.propose_move.output.move_uci"}
```

System nodes execute the referenced system and expose the subsystem’s `final_output` under:

- `node.<id>.output`
- `output` (for routing conditions in that node)

See `docs/composition.md`.
