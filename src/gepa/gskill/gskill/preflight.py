"""Preflight checks for the gskill SWE-smith agent harness."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    message: str
    required: bool = True

    @property
    def level(self) -> str:
        if self.ok:
            return "ok"
        return "fail" if self.required else "warn"


@dataclass(frozen=True)
class PreflightReport:
    checks: list[PreflightCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(check) | {"level": check.level} for check in self.checks]}


DEFAULT_REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("yaml", "pyyaml"),
    ("docker", "docker"),
    ("minisweagent", "mini-swe-agent"),
    ("swebench", "swebench"),
    ("swesmith", "swesmith"),
    ("datasets", "datasets"),
    ("dotenv", "python-dotenv"),
    ("litellm", "litellm"),
)


def check_python_version(min_version: tuple[int, int] = (3, 10)) -> PreflightCheck:
    version = sys.version_info
    ok = (version.major, version.minor) >= min_version
    current = f"{version.major}.{version.minor}.{version.micro}"
    required = f"{min_version[0]}.{min_version[1]}+"
    message = f"Python {current} detected; requires {required}."
    return PreflightCheck(name="python", ok=ok, message=message)


def check_package(module_name: str, install_name: str | None = None) -> PreflightCheck:
    install_name = install_name or module_name
    ok = importlib.util.find_spec(module_name) is not None
    if ok:
        message = f"Found Python module '{module_name}'."
    else:
        message = f"Missing Python module '{module_name}'. Install with: pip install {install_name}"
    return PreflightCheck(name=f"package:{module_name}", ok=ok, message=message)


def check_env_var(name: str, *, required: bool = False, hint: str | None = None) -> PreflightCheck:
    ok = bool(os.getenv(name))
    if ok:
        message = f"{name} is set."
    else:
        message = f"{name} is not set."
        if hint:
            message = f"{message} {hint}"
    return PreflightCheck(name=f"env:{name}", ok=ok, message=message, required=required)


def check_docker_daemon() -> PreflightCheck:
    if importlib.util.find_spec("docker") is None:
        return PreflightCheck(
            name="docker:daemon",
            ok=False,
            message="Cannot check Docker daemon because the 'docker' Python package is missing.",
        )

    try:
        docker = importlib.import_module("docker")
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        return PreflightCheck(
            name="docker:daemon",
            ok=False,
            message=(
                f"Cannot connect to Docker: {exc}. Start Docker or set DOCKER_HOST "
                "for rootless Docker, for example: export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock"
            ),
        )

    return PreflightCheck(name="docker:daemon", ok=True, message="Docker daemon is reachable.")


def run_preflight(
    *,
    packages: Iterable[tuple[str, str]] = DEFAULT_REQUIRED_PACKAGES,
    check_docker: bool = True,
    require_openai_key: bool = False,
) -> PreflightReport:
    checks = [check_python_version()]
    checks.extend(check_package(module_name, install_name) for module_name, install_name in packages)
    checks.append(
        check_env_var(
            "OPENAI_API_KEY",
            required=require_openai_key,
            hint="Set it when using OpenAI-backed agent or reflection models.",
        )
    )
    if check_docker:
        checks.append(check_docker_daemon())
    return PreflightReport(checks=checks)


def format_report(report: PreflightReport) -> str:
    lines = ["gskill preflight"]
    for check in report.checks:
        marker = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[check.level]
        lines.append(f"[{marker}] {check.message}")
    lines.append("")
    lines.append("Ready to run gskill." if report.ok else "Fix failed checks before running the SWE harness.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the gskill SWE-smith agent harness environment.")
    parser.add_argument("--skip-docker", action="store_true", help="Skip the Docker daemon connectivity check.")
    parser.add_argument(
        "--require-openai-key",
        action="store_true",
        help="Treat OPENAI_API_KEY as required instead of a warning.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    report = run_preflight(check_docker=not args.skip_docker, require_openai_key=args.require_openai_key)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
