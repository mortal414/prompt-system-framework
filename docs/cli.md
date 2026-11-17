# CLI and setup

[Project overview](../README.md)

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

## CLI

- Validate: `psf validate --system systems/chess_move_system.yaml`
- Inspect: `psf inspect --system systems/chess_move_system.yaml`
- Export schema: `psf schema --out schema.json`
- Export resolved graph: `psf graph --system systems/chess_move_system.yaml --out graph.json`
- Run: `psf run --system systems/chess_move_system.yaml --input <input.json>`
- Replay: `psf replay --run runs/<system_id>/<run_id>/run.json`
- Benchmark (scenario suites): `psf benchmark --system <sys.yaml> --inputs-dir <dir> --iterations 5`
- UI: `psf ui` (launches the local UI server)

Artifacts contract: `docs/runs-and-artifacts.md`.

