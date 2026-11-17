# Prompt System Framework

An experimental Python runner for YAML/JSON prompt graphs with typed inputs, structured outputs, routing, and loops.

**Inactive hobby project.**

## Install

```sh
python -m venv .venv
# Activate .venv for your shell.
pip install -e .
```

Set `OPENAI_API_KEY` in your environment for API runs. Install `pip install -e ".[chess]"` for the chess utilities.

## Use

```sh
psf validate --system systems/chess_move_system.yaml
psf inspect --system systems/chess_move_system.yaml
psf run --system systems/chess_move_system.yaml --input input.json
```

Supply a JSON input matching the chosen system. API runs may incur charges. Outputs are stored under `runs/`.

- [CLI commands and shell setup](docs/cli.md)
- [System YAML reference](docs/system-yaml-reference.md)
- [Run files](docs/runs-and-artifacts.md)
- [Benchmarking](docs/benchmarking.md)
- [Composition](docs/composition.md)
- [Chess examples](docs/chess.md)

The engine lives in `src/psf/`; example graphs are in `systems/`. Scratch systems, notebooks, and scripts are experiments rather than supported interfaces.
