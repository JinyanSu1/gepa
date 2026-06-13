"""Offline benchmark for gskill reflection record distillation.

This benchmark needs no Docker, no SWE-smith images, and no model API calls. It
replays saved or synthetic rollout artifacts through the distiller and reports:

- reflection payload size by mode
- reduction from full raw-trace records
- preservation of expected debugging signals

Input JSONL rows may contain:
``instance_id``, ``problem``, ``patch``, ``agent_trace``, ``test_output``,
``status``, ``score``, ``agent_metrics``, and an ``expected`` object with
``failure_mode``, ``changed_files``, ``commands``, and ``test_terms``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gepa.gskill.gskill.trace_distillation import (
    DistillationConfig,
    build_reflection_side_info,
    estimate_record_chars,
)

MODES = ("full", "hybrid", "summary")


@dataclass(frozen=True)
class ReplayCase:
    instance_id: str
    problem: str
    patch: str
    agent_trace: str
    test_output: str
    status: str
    score: float
    agent_metrics: dict[str, Any]
    expected: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkCaseResult:
    instance_id: str
    sizes: dict[str, int]
    reductions: dict[str, float]
    summary_signal: dict[str, bool]


@dataclass(frozen=True)
class BenchmarkReport:
    num_cases: int
    avg_sizes: dict[str, float]
    avg_reductions: dict[str, float]
    signal_preservation_rate: float
    cases: list[BenchmarkCaseResult]


def _noise(label: str, count: int) -> str:
    return "\n".join(f"{label} noisy line {i}: {'x' * 72}" for i in range(count))


def default_cases() -> list[ReplayCase]:
    return [
        ReplayCase(
            instance_id="math-utils__zero-division",
            problem="Fix safe_divide so division by zero returns 0 instead of crashing.",
            patch="""diff --git a/src/pkg/math_utils.py b/src/pkg/math_utils.py
index 1111111..2222222 100644
--- a/src/pkg/math_utils.py
+++ b/src/pkg/math_utils.py
@@ -10,7 +10,8 @@ def safe_divide(left, right):
-    return left / right
+    if right == 0:
+        return 0
+    return left / right
""",
            agent_trace=f"""[ASSISTANT]
I need to inspect the failing function.
```bash
rg "safe_divide" -n src tests
sed -n '1,120p' src/pkg/math_utils.py
python -m pytest tests/test_math_utils.py -q
```
{_noise("trace", 140)}
$ python -m pytest tests/test_math_utils.py -q
E   ZeroDivisionError: division by zero
""",
            test_output=f"""FAILED tests/test_math_utils.py::test_safe_divide_zero - ZeroDivisionError: division by zero
Traceback (most recent call last):
  File "tests/test_math_utils.py", line 3, in test_safe_divide_zero
E   ZeroDivisionError: division by zero
{_noise("pytest", 80)}
""",
            status="f2p_failed",
            score=0.0,
            agent_metrics={"steps": 7, "estimated_tokens": 4200, "num_messages": 13},
            expected={
                "failure_mode": "test_failure",
                "changed_files": ["src/pkg/math_utils.py"],
                "commands": ['rg "safe_divide" -n src tests', "python -m pytest tests/test_math_utils.py -q"],
                "test_terms": ["ZeroDivisionError", "FAILED tests/test_math_utils.py::test_safe_divide_zero"],
            },
        ),
        ReplayCase(
            instance_id="parser__patch-apply",
            problem="Fix parser handling for empty tokens.",
            patch="""diff --git a/src/parser.py b/src/parser.py
index 1111111..2222222 100644
--- a/src/parser.py
+++ b/src/parser.py
@@ -22,7 +22,7 @@ def parse_token(token):
-    return token.value.strip()
+    return token.value.strip() if token else ""
""",
            agent_trace=f"""[ASSISTANT]
```bash
rg "parse_token" -n .
python -m pytest tests/test_parser.py -q
```
{_noise("trace", 95)}
""",
            test_output=f"""Patch Apply Failed: patch does not apply cleanly
error: patch failed: src/parser.py:22
{_noise("apply", 60)}
""",
            status="f2p_failed",
            score=0.0,
            agent_metrics={"steps": 5, "estimated_tokens": 2900, "num_messages": 9},
            expected={
                "failure_mode": "patch_apply_failed",
                "changed_files": ["src/parser.py"],
                "commands": ['rg "parse_token" -n .', "python -m pytest tests/test_parser.py -q"],
                "test_terms": ["Patch Apply Failed"],
            },
        ),
        ReplayCase(
            instance_id="api__regression",
            problem="Preserve legacy API behavior while fixing bad status codes.",
            patch="""diff --git a/src/api.py b/src/api.py
index 1111111..2222222 100644
--- a/src/api.py
+++ b/src/api.py
@@ -40,7 +40,8 @@ def status_code(response):
-    return response.status
+    if response.status is None:
+        return 500
+    return response.status
""",
            agent_trace=f"""[ASSISTANT]
```bash
rg "status_code" -n src tests
pytest -q tests/test_api.py
```
{_noise("trace", 110)}
""",
            test_output=f"""FAILED tests/test_api.py::test_legacy_none_status_remains_none - AssertionError: assert 500 is None
=========================== short test summary info ===========================
FAILED tests/test_api.py::test_legacy_none_status_remains_none
{_noise("pytest", 70)}
""",
            status="p2p_regression",
            score=0.0,
            agent_metrics={"steps": 6, "estimated_tokens": 3500, "num_messages": 11},
            expected={
                "failure_mode": "regression",
                "changed_files": ["src/api.py"],
                "commands": ['rg "status_code" -n src tests', "pytest -q tests/test_api.py"],
                "test_terms": ["AssertionError", "FAILED tests/test_api.py::test_legacy_none_status_remains_none"],
            },
        ),
        ReplayCase(
            instance_id="cli__no-patch",
            problem="Improve CLI handling for --version.",
            patch="",
            agent_trace=f"""[ASSISTANT]
I inspected the repository but did not find the right entrypoint.
```bash
ls
find . -maxdepth 2 -type f
rg "--version" -n .
```
{_noise("trace", 90)}
""",
            test_output="No patch to test.",
            status="no_patch",
            score=0.0,
            agent_metrics={"steps": 4, "estimated_tokens": 2500, "num_messages": 8},
            expected={
                "failure_mode": "no_patch",
                "changed_files": [],
                "commands": ["find . -maxdepth 2 -type f", 'rg "--version" -n .'],
                "test_terms": ["No patch"],
            },
        ),
    ]


def load_cases(path: Path | None) -> list[ReplayCase]:
    if path is None:
        return default_cases()

    cases = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                cases.append(
                    ReplayCase(
                        instance_id=data["instance_id"],
                        problem=data["problem"],
                        patch=data.get("patch", ""),
                        agent_trace=data.get("agent_trace", ""),
                        test_output=data.get("test_output", ""),
                        status=data.get("status", "unknown"),
                        score=float(data.get("score", 0.0)),
                        agent_metrics=dict(data.get("agent_metrics", {})),
                        expected=dict(data.get("expected", {})),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing required field {exc} on {path}:{line_num}") from exc
    return cases


def _build_record(case: ReplayCase, mode: str) -> dict[str, Any]:
    return build_reflection_side_info(
        task={"instance_id": case.instance_id},
        problem=case.problem,
        patch=case.patch,
        agent_trace=case.agent_trace,
        agent_metrics=case.agent_metrics,
        status=case.status,
        test_output=case.test_output,
        score=case.score,
        config=DistillationConfig(mode=mode),
    )


def _summary_signal(record: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    diagnostics = record["Generated Outputs"]["Diagnostics"]
    feedback = record["Feedback"]
    changed_files = diagnostics["patch_stats"]["changed_files"]
    commands = diagnostics["recent_commands"]
    test_signal = "\n".join(feedback["Test Signal"] + [feedback.get("Test Output", "")])

    return {
        "failure_mode": diagnostics["failure_mode"] == expected.get("failure_mode"),
        "changed_files": all(path in changed_files for path in expected.get("changed_files", [])),
        "commands": all(command in commands for command in expected.get("commands", [])),
        "test_terms": all(term in test_signal for term in expected.get("test_terms", [])),
    }


def run_benchmark(cases: list[ReplayCase]) -> BenchmarkReport:
    case_results = []
    for case in cases:
        records = {mode: _build_record(case, mode) for mode in MODES}
        sizes = {mode: estimate_record_chars(record) for mode, record in records.items()}
        full_size = sizes["full"]
        reductions = {mode: 0.0 if mode == "full" else 1.0 - (sizes[mode] / full_size) for mode in MODES}
        case_results.append(
            BenchmarkCaseResult(
                instance_id=case.instance_id,
                sizes=sizes,
                reductions=reductions,
                summary_signal=_summary_signal(records["summary"], case.expected),
            )
        )

    avg_sizes = {
        mode: sum(result.sizes[mode] for result in case_results) / len(case_results)
        for mode in MODES
    }
    avg_reductions = {
        mode: sum(result.reductions[mode] for result in case_results) / len(case_results)
        for mode in MODES
    }
    signal_checks = [
        ok
        for result in case_results
        for ok in result.summary_signal.values()
    ]
    return BenchmarkReport(
        num_cases=len(case_results),
        avg_sizes=avg_sizes,
        avg_reductions=avg_reductions,
        signal_preservation_rate=sum(signal_checks) / len(signal_checks) if signal_checks else 0.0,
        cases=case_results,
    )


def format_report(report: BenchmarkReport) -> str:
    lines = [
        "gskill trace distillation benchmark",
        f"cases: {report.num_cases}",
        "",
        "mode      avg_chars   reduction_vs_full",
    ]
    for mode in MODES:
        lines.append(f"{mode:<8} {report.avg_sizes[mode]:>9.0f}   {report.avg_reductions[mode]:>6.1%}")
    lines.extend(
        [
            "",
            f"summary_signal_preservation: {report.signal_preservation_rate:.1%}",
            "",
            "case details:",
        ]
    )
    for result in report.cases:
        checks = ", ".join(f"{name}={'yes' if ok else 'no'}" for name, ok in result.summary_signal.items())
        lines.append(
            f"- {result.instance_id}: summary={result.sizes['summary']} chars, "
            f"full={result.sizes['full']} chars, reduction={result.reductions['summary']:.1%}, {checks}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline gskill trace-distillation benchmark.")
    parser.add_argument("--input", type=Path, default=None, help="Optional JSONL replay cases. Uses built-in cases by default.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    report = run_benchmark(load_cases(args.input))
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report.signal_preservation_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
