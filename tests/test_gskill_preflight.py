import importlib.util

from gskill.preflight import check_package, format_report, run_preflight


def test_gskill_entrypoint_specs_resolve_without_importing_optional_deps():
    assert importlib.util.find_spec("gskill.preflight") is not None
    assert importlib.util.find_spec("gskill.train_optimize_anything") is not None
    assert importlib.util.find_spec("gskill.evaluate.mini_swe_agent") is not None


def test_gepa_gskill_shorthand_specs_resolve_without_importing_optional_deps():
    assert importlib.util.find_spec("gepa.gskill.preflight") is not None
    assert importlib.util.find_spec("gepa.gskill.train_optimize_anything") is not None
    assert importlib.util.find_spec("gepa.gskill.evaluate.mini_swe_agent") is not None


def test_check_package_reports_missing_dependency_with_install_hint():
    check = check_package("definitely_missing_gskill_dependency", "missing-package")

    assert not check.ok
    assert check.required
    assert check.level == "fail"
    assert "pip install missing-package" in check.message


def test_run_preflight_allows_optional_openai_key_warning(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    report = run_preflight(packages=[("sys", "python")], check_docker=False, require_openai_key=False)

    assert report.ok
    env_check = next(check for check in report.checks if check.name == "env:OPENAI_API_KEY")
    assert not env_check.ok
    assert not env_check.required
    assert env_check.level == "warn"
    assert "[WARN]" in format_report(report)
