import json

from gepa.gskill.gskill.trace_benchmark import (
    default_cases,
    format_report,
    load_cases,
    run_benchmark,
)


def test_default_benchmark_cases_cover_core_failure_modes():
    cases = default_cases()
    failure_modes = {case.expected["failure_mode"] for case in cases}

    assert failure_modes == {"test_failure", "patch_apply_failed", "regression", "no_patch"}


def test_default_benchmark_reports_payload_reduction_and_signal_preservation():
    report = run_benchmark(default_cases())

    assert report.num_cases == 4
    assert report.avg_reductions["summary"] > 0.60
    assert report.avg_reductions["hybrid"] > 0.50
    assert report.signal_preservation_rate == 1.0

    rendered = format_report(report)
    assert "summary_signal_preservation: 100.0%" in rendered
    assert "math-utils__zero-division" in rendered


def test_load_cases_accepts_jsonl_replay_file(tmp_path):
    source_case = default_cases()[0]
    replay_file = tmp_path / "cases.jsonl"
    replay_file.write_text(
        json.dumps(
            {
                "instance_id": source_case.instance_id,
                "problem": source_case.problem,
                "patch": source_case.patch,
                "agent_trace": source_case.agent_trace,
                "test_output": source_case.test_output,
                "status": source_case.status,
                "score": source_case.score,
                "agent_metrics": source_case.agent_metrics,
                "expected": source_case.expected,
            }
        )
        + "\n"
    )

    cases = load_cases(replay_file)
    report = run_benchmark(cases)

    assert len(cases) == 1
    assert report.signal_preservation_rate == 1.0
    assert report.cases[0].instance_id == "math-utils__zero-division"
