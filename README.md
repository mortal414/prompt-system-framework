# Prompt System Framework (PSF)

Config-first framework for running **systems of prompts**: typed graphs of LLM-powered nodes with explicit inputs, structured outputs, routing, and loops. Systems are authored as YAML/JSON and executed by a small engine that produces stable run artifacts under `runs/`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=sk-...
```

## CLI (stable surface)

- Validate: `psf validate --system systems/chess_move_system.yaml`
- Inspect: `psf inspect --system systems/chess_move_system.yaml`
- Export schema: `psf schema --out schema.json`
- Export resolved graph: `psf graph --system systems/chess_move_system.yaml --out graph.json`
- Run: `psf run --system systems/chess_move_system.yaml --input <input.json>`
- Replay: `psf replay --run runs/<system_id>/<run_id>/run.json`
- Benchmark (labeled scenarios): `psf benchmark --system <sys.yaml> --inputs-dir <dir> --iterations 5 --expect-field expected_legality --output-field is_legal`
- UI: `psf ui` (launches the local UI server)

Artifacts contract: `docs/runs-and-artifacts.md`.

## Canonical config + docs

- System YAML reference (matches Pydantic models): `docs/system-yaml-reference.md`
- Composition via `use_system`: `docs/composition.md`
- Primer / motivation: `Prompt-System.md`
- Chess example benchmarks and scripts: `docs/chess-benchmarks.md`

## Repository layout

```
src/psf/          # Engine + schema + CLI (product)
systems/          # Curated example systems (safe defaults)
systems/_scratch/ # Experiments / variants (not part of the stable surface)
experiments/      # Notebooks / one-offs (not part of the stable surface)
scripts/          # Example utilities (not part of the stable surface)
docs/             # Contract + guides
```
