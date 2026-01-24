# Composition (`use_system`)

PSF supports **system nodes**: a node can execute another complete system defined in a separate YAML/JSON file. This lets you reuse common building blocks (parsers, validators, formatters) without copying prompts.

## How it works (contract)

- A node with `use_system: <path>` loads the referenced system and executes it as a **subsystem**.
- The node's `input_bindings` become the subsystem's `system_input` payload.
- The node's output is the subsystem's `final_output` (built from the subsystem's `system_output` mappings).
- In the parent system, the subsystem output is available as:
  - `node.<system_node_id>.output.<field>` (in bindings and system_output mappings)
  - `output.<field>` (inside that node's routing conditions)

## Constraints (validated)

When `use_system` is set, the node is a **system node** and must not define LLM-facing fields:

- Must not set: `provider`, `model`, `temperature`, `max_tokens`, `extra_params`
- Must not set: `instructions`, `prompt_template`
- Must not set: `output_mode` (system node output is always the subsystem `final_output`)
- Must not set: `output_schema`

`max_retries` is allowed and controls how many times PSF will re-run the subsystem if it fails.

## Path resolution

`use_system` supports:

- Absolute paths
- Relative paths (resolved relative to the parent system file, with repo-root style paths like `systems/...` supported when running from the repo root)
- `~` and environment variable expansion

## Example

```yaml
nodes:
  - id: "validate_legality"
    label: "Reuse an existing legality-check system"
    use_system: "systems/chess_uci_legal_check_yn_system.yaml"
    input_bindings:
      position_fen: {from: "system.position_fen"}
      candidate_move: {from: "node.generate_move.output.move_uci"}
      move_notation: {value: "uci"}
    routing:
      on_success:
        - condition: "output.is_legal == true"
          next: "final_response"
        - condition: "output.is_legal == false"
          next: "retry_move"
```

