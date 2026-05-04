# Contributing

Thanks for your interest in improving Agent Trajectory Eval. The project is pure-Python (3.10+) and uses only the standard library at runtime — please keep new contributions inside that boundary unless there's a strong reason otherwise.

## Development setup

```bash
git clone <fork-url>
cd trace-agent
python -m unittest discover -v
```

There are no runtime dependencies. `requirements-dev.txt` documents optional tools used while developing.

## Running the CLI locally

```bash
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/example
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/example --format json
```

See `README.md` for the full command set.

## Running tests

```bash
python -m unittest discover -v
```

Single module:

```bash
python -m unittest tests.test_classifier -v
```

## Adding a new trajectory format adapter

1. Subclass `BaseAdapter` from `adapters/base.py`.
2. Implement `detect(data)` (cheap structural sniff) and `transform(data, source_path)` (returns a `Trajectory`).
3. Register the adapter in `adapters/__init__.py`'s `_ADAPTERS` list.
4. Add tests under `tests/` covering both `detect` and `transform`, plus an end-to-end `evaluate_file` test.

## Adding a new suspicious-step pattern

1. Add a `Pattern` entry to the `PATTERNS` registry in `patterns.py`. Each entry needs `name`, `description`, `score_weight`, `error_type`, and `repair_hint`.
2. Add a rule block to `score_suspicious_steps` in `analyzer.py` that calls `_apply(step, "<pattern_name>", "<reason>")` when the rule matches.
3. Add a test case in `tests/test_analyzer.py` that constructs the matching trajectory and asserts the score and reason are populated.

## Coding standards

- No third-party runtime dependencies.
- Stick to the dataclass model in `models.py` — extend it rather than passing dicts around.
- Prefer pure functions; only `analyzer.py` and `evaluator.py` orchestrate state across modules.
- All new public functions should have unit tests.
- Use Markdown tables in reports cautiously: route any user-supplied text through `_md_table_cell` in `report.py`.

## Commit style

We use short, conventional-style messages:

- `feat:` new behavior
- `fix:` bug fix
- `test:` test-only change
- `docs:` documentation
- `refactor:` no behavior change
