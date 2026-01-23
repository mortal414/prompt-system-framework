# Benchmarking (general-purpose)

PSF benchmarking is intentionally domain-agnostic: you provide a directory of scenarios (inputs + ground truth), and one or more systems to run against them.

## Scenario format (recommended): `psf_benchmark_scenario_v1`

Each scenario is a JSON/YAML file with:

```json
{
  "input": { "question": "...", "context": "..." },
  "expected": "the correct answer (string) OR a list of acceptable strings",
  "meta": { "optional": "anything" }
}
```

- `input` is the **only** object passed into the system.
- `expected` is used only for scoring (it is not passed to the system).
- `meta` is optional and ignored by the runner.

### What systems should output

By default, benchmarks score `final_output.answer` against `scenario.expected`. If your system uses a different output field, override it with `--output-field`.

### Supported expected types

Benchmarks are scored by **string matching** (after trimming whitespace):

- If `expected` is a string: correct when `actual == expected`
- If `expected` is a list of strings: correct when `actual` matches **any** entry

`psf benchmark --match auto` chooses:

- `one_of` when `expected` is a list
- `exact` otherwise

(`--match bool` exists for legacy boolean-style benchmarks, but the recommended format is string matching.)

## Run a benchmark

```bash
psf benchmark \
  --system systems/my_system.yaml \
  --system systems/my_other_system.yaml \
  --inputs-dir examples/input/my_task \
  --iterations 5
```

To customize field names:

```bash
psf benchmark \
  --system systems/my_system.yaml \
  --inputs-dir examples/input/my_task \
  --iterations 5 \
  --expect-field expected.value \
  --output-field answer.value
```

## Artifacts

Benchmarks are written under `runs/benchmarks/<benchmark_id>/`:

- `suite_summary.json`: cross-system comparison table + metadata
- `<system_id>/benchmark_summary.json`: per-scenario results for one system
- `<system_id>/<run_id>/...`: standard PSF per-run artifacts (`system.yaml`, `input.json`, `nodes.jsonl`, `run.json`)
