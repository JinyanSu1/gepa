import json

from gepa.gskill.gskill.trace_distillation import (
    DistillationConfig,
    build_reflection_side_info,
    classify_failure,
    estimate_record_chars,
    extract_changed_files,
    extract_commands,
    extract_test_signal,
    patch_stats,
)


def _sample_patch() -> str:
    return """diff --git a/src/pkg/math_utils.py b/src/pkg/math_utils.py
index 1111111..2222222 100644
--- a/src/pkg/math_utils.py
+++ b/src/pkg/math_utils.py
@@ -10,7 +10,8 @@ def safe_divide(left, right):
-    return left / right
+    if right == 0:
+        return 0
+    return left / right
diff --git a/tests/test_math_utils.py b/tests/test_math_utils.py
index 3333333..4444444 100644
--- a/tests/test_math_utils.py
+++ b/tests/test_math_utils.py
@@ -1,3 +1,4 @@
+from pkg.math_utils import safe_divide
 def test_safe_divide_zero():
     assert safe_divide(10, 0) == 0
"""


def _sample_trace() -> str:
    repeated_noise = "\n".join(f"[TOOL]\nlarge irrelevant listing line {i}" for i in range(240))
    return f"""[ASSISTANT]
I will inspect the failing function.
```bash
rg "safe_divide" -n src tests
sed -n '1,120p' src/pkg/math_utils.py
pytest -q tests/test_math_utils.py
```

[USER]
The command failed.

{repeated_noise}

$ python -m pytest tests/test_math_utils.py -q
E   ZeroDivisionError: division by zero
"""


def _sample_test_output() -> str:
    return """============================= test session starts =============================
FAILED tests/test_math_utils.py::test_safe_divide_zero - ZeroDivisionError: division by zero
Traceback (most recent call last):
  File "tests/test_math_utils.py", line 3, in test_safe_divide_zero
    assert safe_divide(10, 0) == 0
E   ZeroDivisionError: division by zero
=========================== short test summary info ===========================
FAILED tests/test_math_utils.py::test_safe_divide_zero
""" + "\n".join(f"extra noisy pytest line {i}" for i in range(160))


def _build_record(mode: str) -> dict:
    return build_reflection_side_info(
        task={"instance_id": "repo__issue-123"},
        problem="Fix safe_divide so it handles division by zero without crashing.",
        patch=_sample_patch(),
        agent_trace=_sample_trace(),
        agent_metrics={"steps": 7, "estimated_tokens": 4200, "num_messages": 13},
        status="f2p_failed",
        test_output=_sample_test_output(),
        score=0.0,
        config=DistillationConfig(mode=mode),
    )


def test_patch_stats_extracts_changed_files_and_line_counts():
    stats = patch_stats(_sample_patch())

    assert stats["changed_files"] == ["src/pkg/math_utils.py", "tests/test_math_utils.py"]
    assert stats["num_changed_files"] == 2
    assert stats["additions"] == 4
    assert stats["deletions"] == 1
    assert stats["hunks"] == 2


def test_trace_extractors_keep_commands_and_failure_signal():
    commands = extract_commands(_sample_trace())
    test_signal = extract_test_signal(_sample_test_output())

    assert 'rg "safe_divide" -n src tests' in commands
    assert "python -m pytest tests/test_math_utils.py -q" in commands
    assert any("ZeroDivisionError" in line for line in test_signal)
    assert any("FAILED tests/test_math_utils.py::test_safe_divide_zero" in line for line in test_signal)


def test_failure_classification_prioritizes_regression_and_patch_apply():
    assert classify_failure("p2p_regression", "", _sample_patch()) == "regression"
    assert classify_failure("f2p_failed", "Patch Apply Failed", _sample_patch()) == "patch_apply_failed"
    assert classify_failure("f2p_failed", "AssertionError: bad value", _sample_patch()) == "test_failure"
    assert classify_failure("no_patch", "", "") == "no_patch"


def test_summary_record_preserves_debugging_signal_with_smaller_payload():
    full = _build_record("full")
    hybrid = _build_record("hybrid")
    summary = _build_record("summary")

    full_chars = estimate_record_chars(full)
    hybrid_chars = estimate_record_chars(hybrid)
    summary_chars = estimate_record_chars(summary)

    assert summary_chars < full_chars * 0.35
    assert hybrid_chars < full_chars
    assert "Agent Trace" in full["Generated Outputs"]
    assert "Agent Trace Excerpt" in hybrid["Generated Outputs"]
    assert "Trace Summary" in summary["Generated Outputs"]

    diagnostics = summary["Generated Outputs"]["Diagnostics"]
    assert diagnostics["failure_mode"] == "test_failure"
    assert diagnostics["patch_stats"]["changed_files"] == extract_changed_files(_sample_patch())
    assert "python -m pytest tests/test_math_utils.py -q" in diagnostics["recent_commands"]
    assert any("ZeroDivisionError" in line for line in summary["Feedback"]["Test Signal"])


def test_record_is_json_serializable_for_proposer_logs():
    record = _build_record("summary")

    encoded = json.dumps(record, sort_keys=True)

    assert "repo__issue-123" in encoded
    assert "safe_divide" in encoded
