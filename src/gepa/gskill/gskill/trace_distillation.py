"""Distill coding-agent rollouts into compact reflection records.

The SWE harness can produce very long conversation traces and test logs. GEPA's
reflection step usually needs the actionable debugging signal, not every token
of terminal output. This module turns raw agent artifacts into a stable,
structured record that is cheaper to send to a reflection model and easier to
analyze across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

ReflectionRecordMode = Literal["summary", "hybrid", "full"]


@dataclass(frozen=True)
class DistillationConfig:
    mode: ReflectionRecordMode = "summary"
    problem_chars: int = 600
    patch_chars: int = 1200
    trace_chars: int = 1200
    test_output_chars: int = 1600
    max_commands: int = 8
    max_test_lines: int = 16


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep the beginning and end of long text with a clear truncation marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n... [truncated] ...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


def extract_changed_files(patch: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch, flags=re.MULTILINE):
        path = match.group(2)
        if path not in files:
            files.append(path)
    return files


def patch_stats(patch: str) -> dict[str, Any]:
    additions = 0
    deletions = 0
    hunks = 0
    for line in patch.splitlines():
        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    changed_files = extract_changed_files(patch)
    return {
        "changed_files": changed_files,
        "num_changed_files": len(changed_files),
        "additions": additions,
        "deletions": deletions,
        "hunks": hunks,
        "has_patch": bool(patch.strip()),
    }


def extract_commands(agent_trace: str, *, limit: int = 8) -> list[str]:
    """Extract likely shell commands from common agent trace formats."""
    commands: list[str] = []

    for fenced in re.finditer(r"```(?:bash|sh|shell)?\n(.*?)```", agent_trace, flags=re.DOTALL | re.IGNORECASE):
        for line in fenced.group(1).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)

    for line in agent_trace.splitlines():
        stripped = line.strip()
        if stripped.startswith(("$ ", "> ")):
            commands.append(stripped[2:].strip())
        elif re.match(r"^(pytest|python -m pytest|rg |grep |sed |cat |python |git |ls |find )", stripped):
            commands.append(stripped)

    deduped: list[str] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped[-limit:]


def extract_test_signal(test_output: str, *, limit: int = 16) -> list[str]:
    """Extract failure-oriented lines from test output."""
    signal_patterns = (
        "FAILED",
        "ERROR",
        "AssertionError",
        "Traceback",
        "Patch Apply Failed",
        "timeout",
        "E   ",
        "F   ",
        "short test summary",
    )
    lines = []
    for line in test_output.splitlines():
        if any(pattern.lower() in line.lower() for pattern in signal_patterns):
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines[-limit:]


def classify_failure(status: str, test_output: str, patch: str) -> str:
    status = status or "unknown"
    output_lower = test_output.lower()
    if status in {"all_passed", "f2p_passed"}:
        return "passed"
    if status == "no_patch" or not patch.strip():
        return "no_patch"
    if status == "setup_error":
        return "setup_error"
    if status == "p2p_regression":
        return "regression"
    if "patch apply failed" in output_lower:
        return "patch_apply_failed"
    if "timeout" in output_lower or "timed out" in output_lower:
        return "timeout"
    if "assertionerror" in output_lower or "failed" in output_lower:
        return "test_failure"
    if "traceback" in output_lower or "error" in output_lower:
        return "runtime_error"
    return status


def build_trace_diagnostics(
    *,
    status: str,
    patch: str,
    agent_trace: str,
    test_output: str,
    agent_metrics: dict[str, Any] | None = None,
    config: DistillationConfig | None = None,
) -> dict[str, Any]:
    config = config or DistillationConfig()
    agent_metrics = agent_metrics or {}
    stats = patch_stats(patch)
    return {
        "status": status,
        "failure_mode": classify_failure(status, test_output, patch),
        "patch_stats": stats,
        "recent_commands": extract_commands(agent_trace, limit=config.max_commands),
        "test_signal": extract_test_signal(test_output, limit=config.max_test_lines),
        "agent_steps": agent_metrics.get("steps", 0),
        "estimated_tokens": agent_metrics.get("estimated_tokens", 0),
        "num_messages": agent_metrics.get("num_messages", 0),
        "raw_trace_chars": len(agent_trace),
        "raw_test_output_chars": len(test_output),
    }


def build_reflection_side_info(
    *,
    task: dict[str, Any],
    problem: str,
    patch: str,
    agent_trace: str,
    agent_metrics: dict[str, Any],
    status: str,
    test_output: str,
    score: float,
    config: DistillationConfig | None = None,
) -> dict[str, Any]:
    config = config or DistillationConfig()
    instance_id = task.get("instance_id", "unknown")[:50]
    diagnostics = build_trace_diagnostics(
        status=status,
        patch=patch,
        agent_trace=agent_trace,
        test_output=test_output,
        agent_metrics=agent_metrics,
        config=config,
    )

    generated_outputs: dict[str, Any] = {
        "Patch": truncate_middle(patch, config.patch_chars),
        "Diagnostics": diagnostics,
    }
    if config.mode == "full":
        generated_outputs["Agent Trace"] = agent_trace
    elif config.mode == "hybrid":
        generated_outputs["Agent Trace Excerpt"] = truncate_middle(agent_trace, config.trace_chars)
    elif config.mode == "summary":
        generated_outputs["Trace Summary"] = {
            "recent_commands": diagnostics["recent_commands"],
            "changed_files": diagnostics["patch_stats"]["changed_files"],
            "failure_mode": diagnostics["failure_mode"],
        }
    else:
        raise ValueError(f"Unknown reflection record mode: {config.mode}")

    side_info = {
        "Input": {
            "Task ID": instance_id,
            "Problem": truncate_middle(problem, config.problem_chars),
        },
        "Generated Outputs": generated_outputs,
        "Feedback": {
            "Status": status,
            "Failure Mode": diagnostics["failure_mode"],
            "Test Signal": diagnostics["test_signal"],
            "Test Output": truncate_middle(test_output, config.test_output_chars),
        },
        "scores": {
            "correctness": score,
        },
    }
    return side_info


def estimate_record_chars(record: dict[str, Any]) -> int:
    """Deterministic size proxy for reflection payload experiments."""
    import json

    return len(json.dumps(record, sort_keys=True, default=str))
