# Runs and artifacts (contract)

PSF writes run artifacts to make executions inspectable and replayable. The layouts below are part of the stable contract for the backend and for future UI consumption.

## `psf run`

`psf run --system <path> [--input <json|yaml>]` writes a run directory:

```
runs/<system_id>/<run_id>/
  system.yaml
  input.json
  nodes.jsonl
  run.json
```

### `system.yaml`

Snapshot of the validated `SystemDef` used for the run (with `from` aliases preserved).

### `input.json`

Snapshot of the input payload passed to the system.

### `nodes.jsonl`

One JSON object per executed node, in execution order, with:

- `node_id`
- `prompt`
- `raw_output`
- `parsed_output` (when structured)
- `provider`, `model`
- token counts: `prompt_tokens`, `completion_tokens`, `total_tokens`
- `duration_ms`
- `error` (string when the node failed)
- `provider_metadata` (provider- and subsystem-specific metadata)

### `run.json`

Top-level run summary, including:

- `system_id`, `system_version`, `system_label`, `entry_node_id`
- `run_id`
- `status` (`success` or `failure`) and `error`
- `started_at`, `finished_at`
- `final_output` (the system’s returned payload when successful)
- `metrics` (token totals, per-node totals, `models_used`)
- `artifacts` (paths to the files above)
- `source_config` (when invoked via CLI with a source `--system` path)

## `psf benchmark`

`psf benchmark --system <path> --inputs-dir <dir> --iterations N ...` writes a suite directory:

```
runs/benchmarks/<benchmark_id>/
  suite_summary.json
  <system_id>/
    benchmark_summary.json
    <run_id>/
      system.yaml
      input.json
      nodes.jsonl
      run.json
```

Notes:

- Benchmark runs reuse the same per-run artifact schema as `psf run`.
- `benchmark_summary.json` is per-system; `suite_summary.json` compares systems across the same scenarios.

## Non-contract artifacts

The `scripts/` directory contains example/experimental benchmark utilities (not part of the stable artifact contract). They may write additional files under `runs/benchmarks/...` for their own summaries while still embedding standard PSF run directories for per-run inspection.

