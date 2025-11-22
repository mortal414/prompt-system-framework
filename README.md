# Prompt System Framework (v0)

Config-first framework for authoring and running systems of prompts – graphs of LLM-powered nodes with explicit inputs, outputs, and routing flows. The long-term goal is still the UI described originally, but v0 focuses on a Python engine and CLI so we can define, execute, and observe systems immediately.

## What’s in v0

- **Config-driven systems** defined in YAML/JSON with explicit system inputs/outputs, node graphs, routing rules, and loop controls.  
- **Execution engine** that renders prompts with Jinja2, calls OpenAI via a provider abstraction, validates structured JSON outputs, tracks context, tokens, and routing decisions.  
- **CLI (`psf`)** with `run`, `validate`, `inspect`, and `replay` commands plus logging of every run (system snapshot, input snapshot, per-node logs, aggregated metrics).  
- **Chess example** (`systems/chess_move_system.yaml`) showing a multi-node workflow with parsing, validation, and retry loops.  
- **Run artifacts** stored under `runs/<system_id>/<run_id>/` containing `run.json`, `nodes.jsonl`, `system.yaml`, and `input.json` for replay/debugging.

## Quick start

1. **Install dependencies**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```
2. **Configure OpenAI credentials**
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. **Validate the sample system**
   ```bash
   psf validate --system systems/chess_move_system.yaml
   ```
4. **Run the chess pipeline**
   ```bash
   psf run \
     --system systems/chess_move_system.yaml \
     --input examples/input/chess_position_1.json
   ```
   Artifacts land under `runs/chess_move_system/<run_id>/`.

## CLI overview

| Command | Description |
| --- | --- |
| `psf run --system <path> --input <path>` | Execute a system with structured input, log artifacts, and print the final mapped output. Add `--stdin` to pipe JSON or `--dry-run` to only validate config+input. |
| `psf validate --system <path>` | Schema + wiring validation (node IDs, bindings, routing, loop targets, system input/output definition). |
| `psf inspect --system <path>` | Textual graph inspection listing providers, models, routing edges, and loop metadata. |
| `psf replay --run <run.json or dir>` | Re-run a stored snapshot (`system.yaml` + `input.json`) and capture a fresh run in the current runs directory. |

## Project layout

```
pyproject.toml
src/psf/
  cli.py            # Typer CLI entrypoint
  config_loader.py  # YAML/JSON → SystemDef with wiring validation
  engine.py         # Execution engine, routing DSL, schema validation
  providers/        # Provider registry + OpenAI implementation
  logging/          # Run logger writing run.json + nodes.jsonl
systems/
  chess_move_system.yaml
examples/input/
  chess_position_1.json
runs/
  ... created per psf run
prompt-system-framework-v0.md  # Spec/roadmap for future work
```

## Next steps

- Expand provider layer (multi-provider support, custom params).  
- Introduce richer routing (JMESPath/JSONPath) and better testing utilities.  
- Build the planned UI atop these explicit configs once the engine stabilizes.
- Use System within another system: allow user to start small and build simple systems (move validator) and then directly import them into larger more complex systems (chess move system)