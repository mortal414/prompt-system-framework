# System composition (`use_system`)

PSF supports **system nodes**: a node can execute another complete system defined in a separate YAML/JSON file.

This lets you reuse common building blocks (e.g. a legality checker, a parser, a formatter) without duplicating prompts.

## How it works

- A node with `use_system: ...` runs the referenced system as a **subsystem**.
- The node’s `input_bindings` become the **subsystem `system_input`** payload.
- The node’s output becomes the subsystem’s **`final_output`** (built from the subsystem’s `system_output` mappings).
- In the parent system, the output is available at:
  - `node.<your_node_id>.output.<field>`
  - and can be used in routing conditions via `output.<field>` for that node.

## Example

```yaml
nodes:
  - id: "validate_legality"
    label: "Reuse an existing legality-check system"
    use_system: "systems/syste/chess_uci_legal_check_yn_system_1.yaml"
    input_bindings:
      position_fen:
        from: "system.position_fen"
      candidate_move:
        from: "node.generate_move.output.move_uci"
      move_notation:
        value: "uci"
    routing:
      on_success:
        - condition: "output.is_legal == true"
          next: "final_response"
        - condition: "output.is_legal == false"
          next: "retry_move"
```

## Notes / constraints

- `use_system` can be absolute or relative; repo-root style paths like `systems/...` are supported.
- System nodes **must not** set `provider`, `model`, `instructions`, `prompt_template`, `output_mode`, or `output_schema`.
- Subsystem token usage and models are aggregated into the parent run metrics (the system node reports the subsystem totals).

