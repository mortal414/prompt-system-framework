# Prompt System Framework (v0)

Config-first framework for running **systems of prompts**: graphs of LLM-powered nodes with explicit inputs, outputs, and routing. Instead of stuffing everything into one mega-prompt, you wire together small, typed steps, reuse intermediate outputs, and loop or branch until they pass validation (see `Prompt-System.md` for the primer).

## Why a system of prompts?

- Break complex tasks into **small, typed subtasks** so each prompt stays focused and accurate.  
- **Wire nodes together**: later prompts consume earlier outputs; routing rules decide the next hop.  
- Add **control flow**: retries, branches, and loops (e.g., re-prompt until a validator accepts a move).  
- **Validate** structured JSON at every step and map the final fields into a clean system output.

## Architecture at a glance

- **System definition (YAML/JSON)** — declares system inputs/outputs, nodes, routing rules, loop controls, and per-node model settings.  
- **Execution engine** (`src/psf/engine.py`) — renders Jinja2 prompts, calls providers, enforces schemas, resolves routing/loops, and builds the final output.  
- **Providers** (`src/psf/providers/`) — OpenAI implementation today; extensible via the registry.  
- **CLI** (`psf`) — validate configs, inspect graphs, run/replay systems, and benchmark labeled scenarios.  
- **Run logging** (`src/psf/logging/`) — snapshots configs/inputs and emits per-node traces plus metrics under `runs/`.

## System definition anatomy

A system is a DAG (with optional loops) of nodes. Key pieces you will touch when authoring configs:

- `system_input` — typed fields required from the caller.  
- `nodes[]` — each node has instructions, a prompt template (Jinja2), input bindings (from system or prior nodes), output mode/schema, routing rules, and optional loop control.  
- `routing` — `on_success`, `on_failure`, and `on_loop_exhausted` rules choose the next node; conditions can reference `output.*` or `context.*`.  
- `loop_control` — increment counters, rerun a target node, and optionally exit when a node succeeds.  
- `system_output` — maps final fields from context paths such as `node.<id>.output.*` into the returned payload.  
- Per-node model settings (`provider`, `model`, temperature, tokens, extra params) keep multi-model graphs explicit.

Minimal shape (see `systems/chess_move_system.yaml` for a fuller example):

```yaml
id: "my_system"
entry_node_id: "first_node"
benchmark_batch_size: 4
system_input:
  fields:
    task: {type: string, required: true}
system_output:
  fields:
    result: {from: "node.final.output.answer"}
nodes:
  - id: "first_node"
    instructions: "...role/setup..."
    prompt_template: "...rendered with {{ task }}..."
    provider: "openai"
    model: "gpt-4o-mini"
    input_bindings:
      task: {from: "system.task"}
    output_mode: "structured"
    output_schema: {type: object, properties: {answer: {type: string}}, required: [answer]}
    routing:
      on_success: [{next: "final"}]
  - id: "final"
    terminal: true
    instructions: "Package the final reply"
    prompt_template: "Answer: {{ answer }}"
    provider: "openai"
    model: "gpt-4o-mini"
    input_bindings:
      answer: {from: "node.first_node.output.answer"}
    output_mode: "structured"
    output_schema: {type: object, properties: {answer: {type: string}}, required: [answer]}
```

## Working with the CLI

1. **Install & configure**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   export OPENAI_API_KEY=sk-...
   ```
2. **Validate a system** — schema + wiring checks only:
   ```bash
   psf validate --system systems/chess_move_system.yaml
   ```
3. **Inspect the graph** — list nodes, providers, routing, and loops:
   ```bash
   psf inspect --system systems/chess_move_system.yaml
   ```
4. **Run a system** — execute, log artifacts, and print final output:
   ```bash
   psf run \
     --system systems/chess_move_system.yaml \
     --input examples/input/chess_position_1.json
   ```
   Artifacts land under `runs/chess_move_system/<run_id>/` (`system.yaml`, `input.json`, `run.json`, `nodes.jsonl`).
5. **Replay a prior run** — rerun using saved snapshots:
   ```bash
   psf replay --run runs/chess_move_system/<run_id>/run.json
   ```
6. **Benchmark labeled scenarios** — repeat runs and score accuracy:
   ```bash
   psf benchmark \
     --system systems/move_validator_system.yaml \
     --inputs-dir examples/input/move_validator \
     --iterations 20 \
     --expect-field expected_legality \
     --output-field is_legal
   ```
   Each benchmark run is grouped under `runs/benchmarks/<benchmark_id>/` with per-system artifacts in subfolders (e.g., `runs/benchmarks/<benchmark_id>/move_validator_system/`) and a cross-system suite report at `runs/benchmarks/<benchmark_id>/suite_summary.json`. Benchmark iterations run in batches per system (size is controlled by `benchmark_batch_size` in the system config) and use the OpenAI Batch API when the provider is OpenAI.

## Included systems and samples

- `systems/chess_move_system.yaml` — multi-node pipeline: propose move → parse → validate → loop on invalid → final reply.  
- `systems/move_validator_system.yaml` — default SAN legality checker used by the chess pipeline.  
- `systems/move_validator_simple_system.yaml` — single-pass validator baseline.  
- `systems/move_validator_consensus_system.yaml` — dual-path validator that seeks agreement.  
- `examples/input/` — ready-to-run payloads (FEN + moves) with legality labels for quick regression runs or benchmarks.

## Project layout

```
pyproject.toml
src/psf/
  cli.py            # Typer CLI entrypoint
  config_loader.py  # YAML/JSON → SystemDef with validation
  engine.py         # Execution engine, routing DSL, schema enforcement
  providers/        # Provider registry + OpenAI implementation
  logging/          # Run logger writing run.json + nodes.jsonl
systems/            # System definitions (YAML)
examples/input/     # Sample inputs (JSON)
runs/               # Created per psf run/benchmark
Prompt-System.md    # Concept primer for systems of prompts
```

## Tips for authoring new systems

- Keep prompts focused; lean on **structured outputs** and schema validation instead of long natural-language instructions.  
- Specify **provider/model per node** (no global defaults) so batching and multi-model systems stay explicit.  
- Use **input bindings** to pass only what each node needs, and prefer system inputs over re-querying the model.  
- Add **routing conditions** for happy/sad paths and **loop controls** for controlled retries.  
- Populate **descriptions** in system inputs/outputs to make `inspect` easier for others.  
- Capture a **baseline benchmark** whenever you change prompts to detect regressions.

## Next steps
- Batches!  
- System within System!  
- Expand provider layer (multi-provider support, custom params).  
- Introduce richer routing (JMESPath/JSONPath) and better testing utilities.  
- Build the planned UI atop these explicit configs once the engine stabilizes.
