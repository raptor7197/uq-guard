# UQ-Guard Code Audit Findings

## Overview

**Date**: 2026-07-07
**Scope**: Full codebase audit (Phase 0-1 implementation)
**Verdict**: Early-stage project with solid specification but significant implementation gaps

---

## Critical Issues

### 1. README Advertises Non-Existent API
**Priority**: P0
**Files**: `README.md`, `uqguard/__init__.py`

The README shows an API that doesn't exist:
```python
from uqguard import Guard, policies
guard = Guard(scorers=[...], policy=policies.Conformal(alpha=0.1), ...)
```

Reality - `__init__.py` only exports:
```python
from uqguard.capture import CaptureMiddleware
from uqguard.fallback import ModelFallbackMiddleware
from uqguard.step import AgentStep, CandidateAction, Gate
from uqguard.trace import TraceWriter, read_trace
```

**Impact**: Any user following the README gets `ImportError`.

---

### 2. Synchronous K-Sampling
**Priority**: P0
**File**: `uqguard/capture.py:65`

```python
responses = [handler(request) for _ in range(self.k)]  # Sequential!
```

If k=5 and each call takes 2s, that's 10s latency. SPEC.md says "k async calls" but implementation is synchronous.

---

### 3. Overbroad Exception Handling
**Priority**: P0
**File**: `uqguard/fallback.py:28-35`

```python
except Exception as e:  # Catches KeyboardInterrupt, SystemExit, MemoryError!
    print(f"uqguard: primary model failed ({type(e).__name__}), falling back")
```

Catches **all** exceptions including programming bugs. Comment acknowledges this: "tighten...if this ever masks a real bug".

---

### 4. No Logging Infrastructure
**Priority**: P1
**Files**: `fallback.py`, `capture.py`, `examples/demo_agent.py`

Uses `print()` instead of `logging`. No log levels, no file output, no observability integration.

---

## High Priority

### 5. Minimal Test Coverage
**Files**: `tests/`

| Module | Status |
|--------|--------|
| `step.py` | Basic tests only |
| `capture.py` | Basic tests only |
| `fallback.py` | Basic tests only |
| `trace.py` | ❌ No tests |

Missing: concurrent access tests, error conditions, edge cases, integration tests.

---

### 6. Thread Safety
**File**: `uqguard/capture.py`

`CaptureMiddleware._pending` state is not thread-safe for concurrent requests.

---

### 7. Type Hints Missing
**File**: `uqguard/capture.py:63`

```python
def wrap_model_call(self, request, handler):  # No type hints
```

---

## Medium Priority

### 8. Unused Dependencies
**File**: `pyproject.toml`

```toml
langchain-google-genai>=4.2.7  # Listed but never imported
```

---

### 9. No Observability
- No metrics (Prometheus/statsd)
- No distributed tracing (LangSmith not integrated)
- No error tracking (Sentry)
- SPEC mentions W&B but not implemented

---

### 10. Configuration Management
- Model selection hardcoded via env vars
- No `config.yaml` or `settings.py`
- No runtime reconfiguration

---

## Low Priority

### 11. Project Hygiene
- Single commit, no conventional commits
- No pre-commit hooks
- No mypy type checking
- No Dependabot/safety scanning
- `__version__ = "0.0.1"` but no git tags/releases

---

## Implementation Gap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold + toy agent | ✅ Done |
| 1 | Capture layer (k-sampling) | ✅ Done |
| 2 | First scorer + working gate | ❌ Not implemented |
| 3 | Full scorer suite | ❌ Not implemented |
| 4 | Fusion + conformal calibration | ❌ Not implemented |
| 5 | Real eval harness | ❌ Not implemented |
| 6 | Audit UI (Streamlit) | ❌ Not implemented |

---

## Recommendations

### Must Fix Before Production
1. Update README to match actual implementation
2. Implement async k-sampling with `asyncio.gather()`
3. Replace broad `except Exception` with specific API error handling
4. Add logging infrastructure

### Should Fix
5. Add tests for `trace.py` and error conditions
6. Add thread-safety to `CaptureMiddleware`
7. Add type hints to all public methods
8. Remove unused `langchain-google-genai` dependency

### Nice to Have
9. Add pre-commit hooks (ruff, mypy, isort)
10. Set up Dependabot for dependency updates
11. Implement W&B integration as specified in SPEC.md
