# Prompt System Framework (v0)

Config-first framework for authoring and running systems of prompts – graphs of LLM-powered nodes with explicit inputs, outputs, and routing flows. The long-term goal is still the UI described originally, but v0 focuses on a Python engine and CLI so we can define, execute, and observe systems immediately.

## What’s in v0

- **Config-driven systems** defined in YAML/JSON with explicit system inputs/outputs, node graphs, routing rules, and loop controls.  
- **Execution engine** that renders prompts with Jinja2, calls OpenAI via a provider abstraction, validates structured JSON outputs, tracks context, tokens, and routing decisions.  
- **CLI (`psf`)** with `run`, `validate`, `inspect`, and `replay` commands plus logging of every run (system snapshot, input snapshot, per-node logs, aggregated metrics).  
- **Chess example** (`systems/chess_move_system.yaml`) showing a multi-node workflow with parsing, validation, and retry loops.  
- **Move validator examples** ranging from a single-pass arbiter (`systems/move_validator_simple_system.yaml`) to the default validator (`systems/move_validator_system.yaml`) and a dual-path consensus checker (`systems/move_validator_consensus_system.yaml`).  
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
5. **Probe the move validator against known labels**
   ```bash
   psf run \
     --system systems/move_validator_system.yaml \
     --input examples/input/move_validator/legal_opening_knight.json
   ```
   Swap the input file with any other JSON in `examples/input/move_validator/` (legal or illegal) to observe how the validator reacts.
6. **Benchmark the validator over every labeled scenario (optionally across multiple systems)**
   ```bash
   psf benchmark \
     --system systems/move_validator_system.yaml \
     --system systems/move_validator_simple_system.yaml \
     --system systems/move_validator_consensus_system.yaml \
     --inputs-dir examples/input/move_validator \
     --iterations 20
   ```
   Use `--input` repeatedly to target a subset (e.g., just the late-endgame cases) while keeping the iteration count. Repeat `--system` to compare the single-pass, default, and consensus validators side-by-side. Every iteration is logged like a normal `psf run`, per-system artifacts land under `runs/<system_id>/benchmarks/<benchmark_id>/`, and a suite-level report (with cross-system accuracy) is written to `runs/benchmarks/<benchmark_id>/suite_summary.json`.

## Move validator samples

`examples/input/move_validator/` collects FEN + move pairs with a known legality label so you can quickly regression-test the validator graph or feed its output back into the move generator loop.

- `legal_opening_knight.json` – standard `Nf3` development from the initial position.  
- `legal_middlegame_capture.json` – Ruy Lopez bishop capture with check in a crowded middlegame.  
- `legal_endgame_king_walk.json` – calm king centralization in a king-and-pawn ending.  
- `illegal_starting_queen_ray.json` – early `Qh5` attempt blocked by the pawn wall on the first move.  
- `illegal_middlegame_queen_capture.json` – premature `Qxf7#` when own pawn still blocks the queen’s diagonal.  
- `illegal_endgame_king_into_check.json` – king tries to step onto d3 while a black rook on d4 covers the square.

## Benchmark automation

Run structured accuracy sweeps on any labeled system by comparing an expected input label with a specific final output field.

```bash
psf benchmark \
  --system systems/move_validator_system.yaml \
  --system systems/move_validator_simple_system.yaml \
  --system systems/move_validator_consensus_system.yaml \
  --inputs-dir examples/input/move_validator \
  --iterations 20 \
  --expect-field expected_legality \
  --output-field is_legal \
  --run-dir runs
```

- Repeat `--system` to benchmark multiple definitions at once (each must accept the same input schema).  
- `--inputs-dir` scans a directory for JSON/YAML inputs (use `--input` repeatedly to target individual files).  
- `--iterations` controls how many times each scenario is executed, which is handy for spotting stochastic model errors.  
- Per-iteration artifacts (including prompts and responses) land under `runs/<system_id>/benchmarks/<benchmark_id>/<benchmark_id>-<scenario>-<iteration>/`.  
- Each system receives its own `benchmark_summary.json`, and a cross-system `suite_summary.json` is saved at `runs/benchmarks/<benchmark_id>/` so you can diff accuracy tables later. The CLI still prints per-scenario accuracy plus mismatch logs and exits non-zero when any system disagrees with the expected label.

## CLI overview

| Command | Description |
| --- | --- |
| `psf run --system <path> --input <path>` | Execute a system with structured input, log artifacts, and print the final mapped output. Add `--stdin` to pipe JSON or `--dry-run` to only validate config+input. |
| `psf validate --system <path>` | Schema + wiring validation (node IDs, bindings, routing, loop targets, system input/output definition). |
| `psf inspect --system <path>` | Textual graph inspection listing providers, models, routing edges, and loop metadata. |
| `psf benchmark --system <path> --inputs-dir <dir>` | Re-run labeled scenarios multiple times, log each run, persist per-system and suite summaries, and compare accuracies across systems. |
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
  move_validator_simple_system.yaml
  move_validator_system.yaml
  move_validator_consensus_system.yaml
examples/input/
  chess_position_1.json
  move_validator/
    legal_opening_knight.json
    legal_middlegame_capture.json
    legal_endgame_king_walk.json
    illegal_starting_queen_ray.json
    illegal_middlegame_queen_capture.json
    illegal_endgame_king_into_check.json
runs/
  ... created per psf run
prompt-system-framework-v0.md  # Spec/roadmap for future work
```

## Next steps
- Expand provider layer (multi-provider support, custom params).  
- Introduce richer routing (JMESPath/JSONPath) and better testing utilities.  
- Build the planned UI atop these explicit configs once the engine stabilizes.
