# Prompt System Framework (PSF)

Config-first framework for running **systems of prompts**: typed graphs of LLM-powered nodes with explicit inputs, structured outputs, routing, and loops. Systems are authored as YAML/JSON and executed by a small engine that produces stable run artifacts under `runs/`.

## Quickstart

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=sk-...
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
$env:OPENAI_API_KEY="sk-..."
```

For chess datasets/utilities:

```bash
pip install -e ".[chess]"
```

## CLI (stable surface)

- Validate: `psf validate --system systems/chess_move_system.yaml`
- Inspect: `psf inspect --system systems/chess_move_system.yaml`
- Export schema: `psf schema --out schema.json`
- Export resolved graph: `psf graph --system systems/chess_move_system.yaml --out graph.json`
- Run: `psf run --system systems/chess_move_system.yaml --input <input.json>`
- Replay: `psf replay --run runs/<system_id>/<run_id>/run.json`
- Benchmark (scenario suites): `psf benchmark --system <sys.yaml> --inputs-dir <dir> --iterations 5`
- UI: `psf ui` (launches the local UI server)

Artifacts contract: `docs/runs-and-artifacts.md`.

## Docs

Start here: [docs/README.md](docs/README.md).
Browse: [docs/](docs/).

- System YAML reference (matches Pydantic models): [docs/system-yaml-reference.md](docs/system-yaml-reference.md)
- Benchmarking format + runner: [docs/benchmarking.md](docs/benchmarking.md)
- Composition via `use_system`: [docs/composition.md](docs/composition.md)
- Primer / motivation: [Prompt-System.md](Prompt-System.md)
- Chess guide (data generation + benchmarking): [docs/chess.md](docs/chess.md)

## Repository layout

```
src/psf/          # Engine + schema + CLI (product)
systems/          # Curated example systems (safe defaults)
systems/_scratch/ # Experiments / variants (not part of the stable surface)
experiments/      # Notebooks / one-offs (not part of the stable surface)
scripts/          # Example utilities (not part of the stable surface)
docs/             # Contract + guides
```
