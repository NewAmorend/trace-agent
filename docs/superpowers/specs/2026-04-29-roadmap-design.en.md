# Codex Trajectory Analyzer — Roadmap Design

## Goal

Transform from a single-format CLI tool into an extensible, open-source trajectory analyzer focused on fault diagnosis for coding agents.

## Principles

- Pure Python standard library for core; optional dependencies (pytest, LLM providers) only for dev/extensions
- Adapter pattern for multi-format support; hooks for future extensibility
- CLI-first, no web UI or server component

## Priority Order

### 1. Adapter Layer Refactor

**Problem**: `parser.py` directly parses a single JSON format. Adding new formats requires modifying core code.

**Design**:

```
adapters/
├── __init__.py        # Registry + get_adapter()
├── base.py            # BaseAdapter ABC
├── codex_adapter.py   # OpenAI Codex trajectory format
└── internal.py        # Current internal JSON format
```

**Interface**:

```python
class BaseAdapter(ABC):
    @abstractmethod
    def detect(self, data: dict) -> bool:
        """Return True if this adapter handles the given JSON structure."""

    @abstractmethod
    def transform(self, data: dict) -> tuple[str, str, list[Step]]:
        """Convert to unified (task, final_status, steps)."""
```

**Flow**: `main.py` loads JSON, iterates registered adapters, calls `detect()` then `transform()`. Current format moves to `internal.py` with no behavior change.

**Codex adapter**: Parse Codex session format, mapping its fields to `Step` dataclass. Exact field mapping TBD once we have sample Codex traces.

### 2. Project Engineering

- `pyproject.toml` — package config, support `pip install` then `trace-analyzer` CLI command
- `requirements-dev.txt` — pytest and dev tools
- `LICENSE` — MIT license
- `README.md` — installation, usage examples, contribution guide
- `CONTRIBUTING.md` — how to add adapters, patterns, and run tests

### 3. Test Suite (pytest)

```
tests/
├── conftest.py           # Shared fixtures
├── test_adapters.py      # Adapter detection + transform
├── test_classifier.py    # Action classification, stage logic
├── test_tree.py          # Tree building and rendering
├── test_analyzer.py      # Suspicious scoring, failure location
├── test_report.py        # Output file generation
└── fixtures/
    ├── sample_internal.json
    └── sample_codex.json
```

- Full coverage of existing modules before adding new features
- Fixture files for both internal and Codex formats
- pytest as dev dependency

### 4. Classifier LLM Hook

**Problem**: Rule-based classification is limited; users may want LLM-based judgment.

**Design**: Add a hook point in `classifier.py`:

```python
class StepClassifier:
    def __init__(self, judge=None):
        # judge=None uses default rule-based classification
        # judge=<callable> delegates to external classifier
        self._judge = judge

    def classify(self, step: Step) -> NormalizedStep:
        if self._judge:
            return self._judge(step)
        return self._rule_based_classify(step)
```

- Default behavior unchanged (rule-based)
- Hook accepts any callable that takes `Step` and returns `NormalizedStep`
- No provider-specific code; users wire in their own LLM call

### 5. Diagnosis Enhancement

**Pattern library**: Extract hardcoded suspicious patterns from `analyzer.py` into a config file (`patterns.yaml` or `patterns.json`):

```yaml
patterns:
  - name: test_file_manipulation
    description: "Agent modifies test files during execution"
    indicators:
      - action_type: edit_file
        path_pattern: "*/test_*"
    score_weight: 0.8
```

**Confidence levels**: Each diagnosis carries a confidence (high/medium/low) based on how many patterns match and their weights.

**Repair suggestions**: Diagnosis output includes suggested fix actions based on matched patterns.

### 6. Batch Analysis

- `--input` accepts a directory path in addition to single file
- Scans all `.json` files, runs analysis on each
- Generates summary report: success/failure rates, common failure pattern ranking, time distribution
- Individual trace analysis unchanged

### 7. CI Integration

- Exit codes: 0 = no issues found, 1 = diagnosis found problems, 2 = tool error
- `--format json` flag: structured output for CI pipeline consumption
- `--quiet` flag: suppress progress output, only print conclusions

## Non-Goals

- Web UI or server mode
- Real-time monitoring
- Multi-agent orchestration
- Database storage
