<h1 align="center">gskill: Learning Repository-Specific Skills for Coding Agents</h1>

<p align="center">
  <em>Given any GitHub repository, gskill creates important agent skill files for coding agents.</em>
</p>

## Overview

gskill learns repository-specific skills for coding agents. On a GitHub repository it automatically discovers the common patterns, structures, and debugging strategies that matter for that repo, then produces a skill file that makes agents better at fixing bugs in it.

It combines [SWE-smith](https://swesmith.com) for task generation and [GEPA's `optimize_anything`](https://gepa-ai.github.io/gepa/) for skill optimization:

1. **Generate tasks.** SWE-smith mines real commits from the target repo, introduces bugs, and produces hundreds of verifiable task instances, each with a problem statement, a Docker environment, and tests.

2. **Optimize skills.** GEPA starts with empty skills and runs the agent (mini-SWE-agent + gpt-5-mini on default) on batches of tasks in parallel Docker containers. Pass/fail results, agent traces, and test output go to a reflection model that proposes better skills. Repeat until budget is exhausted.

3. **Deploy.** The output is `best_skills.txt`, injected into the agent's system prompt. Skills transfer directly to other agents without retraining.

<p align="center">
  <img src="assets/architecture.png" alt="gskill architecture" width="800">
</p>


## Installation

```bash
pip install gepa[full]
pip install swesmith mini-swe-agent docker python-dotenv
```

Set up API keys and Docker:

```bash
export OPENAI_API_KEY=<your-key>

# Docker must be running
docker ps

# Download SWE-smith images for target repo
python -m swesmith.build_repo.download_images
```

## Using gskill

### Training

```bash
# Smoke test
python -m gepa.gskill.train_optimize_anything \
  --smoke-test --model "gemini/gemini-2.0-flash-exp"

# Full run
python -m gepa.gskill.train_optimize_anything \
  --repo pygments__pygments \
  --train-size 200 --val-size 50 --test-size 100 \
  --model gpt-5-mini --reflection-model gpt-5.2-pro \
  --workers 6 --max-metric-calls 600 \
  --proposer loop --wandb

# Resume from a previous run
python -m gepa.gskill.train_optimize_anything \
  --resume gepa_results/logs/run_XXXXXXXX

# Pre/post optimization test comparison
python -m gepa.gskill.train_optimize_anything \
  --run-testset --repo pygments__pygments --model gpt-5-mini
```

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | `pygments__pygments` | Target repository in SWE-smith |
| `--model` | `gpt-5-mini` | Agent model (runs in Docker) |
| `--reflection-model` | `gpt-5.2-pro` | Model for reflection/skill proposal |
| `--workers` | 6 | Parallel Docker containers |
| `--max-metric-calls` | 600 | Total rollout budget |
| `--proposer` | `batch` | `batch` or `loop` (one-at-a-time then merge) |
| `--reflection-record-mode` | `summary` | Reflection payload mode: `summary`, `hybrid`, or `full` |
| `--run-testset` | off | Evaluate before AND after optimization |
| `--resume` | None | Resume from previous run directory |
| `--smoke-test` | off | Quick validation with 3 tasks |
| `--wandb` | off | Enable Weights & Biases tracking |

### Reflection Record Distillation

By default, gskill sends compact rollout diagnostics to GEPA's reflection model
instead of the full raw agent transcript. Each reflection record includes:

- failure mode (`no_patch`, `test_failure`, `regression`, `patch_apply_failed`, etc.)
- changed files and patch line counts
- recent shell commands extracted from the agent trace
- high-signal test failure lines
- step, token, trace-size, and test-output-size metadata

Use `--reflection-record-mode full` to preserve the previous raw-trace behavior
for ablations, or `--reflection-record-mode hybrid` to include diagnostics plus
a bounded trace excerpt. The deterministic fixture in
`tests/test_gskill_trace_distillation.py` verifies that `summary` mode preserves
the changed file, command, failure-mode, and pytest-failure signal while cutting
the reflection payload to less than 35% of the full record. On the deterministic
test fixture, the serialized reflection record drops from 14,254 chars (`full`)
to 5,018 chars (`hybrid`) or 4,034 chars (`summary`), a 71.7% reduction for the
default summary mode.

### No-Docker Reflection Benchmark

You can benchmark the distiller without Docker, SWE-smith images, or model API
calls:

```bash
python -m gepa.gskill.gskill.trace_benchmark
```

The built-in replay cases cover four common rollout outcomes:
`test_failure`, `patch_apply_failed`, `regression`, and `no_patch`. The benchmark
checks whether `summary` mode preserves the expected failure mode, changed files,
recent commands, and test-failure terms while measuring serialized reflection
payload size.

Latest deterministic result:

| Mode | Avg serialized chars | Reduction vs full |
|------|---------------------:|------------------:|
| `full` | 12,918 | 0.0% |
| `hybrid` | 3,662 | 71.9% |
| `summary` | 2,599 | 80.6% |

`summary` mode preserved 100.0% of the expected debugging signals across the
four replay cases. To replay your own saved artifacts, write one JSON object per
line with `instance_id`, `problem`, `patch`, `agent_trace`, `test_output`,
`status`, `score`, `agent_metrics`, and an `expected` object:

```bash
python -m gepa.gskill.gskill.trace_benchmark --input traces.jsonl --json
```

### Evaluation

After training, evaluate learned skills on the held-out test set.

```bash
# Mini-SWE-agent: with skills vs without (runs both conditions)
python -m src.evaluate.mini_swe_agent \
  --config gepa_results/logs/run_xxx/config.json \
  --workers 16

# Claude Code: baseline (no skills)
python -m src.evaluate.claude_code \
  --config gepa_results/logs/run_xxx/config.json \
  --model haiku --workers 4

# Claude Code: with skills (copies best_skills.txt as CLAUDE.md)
python -m src.evaluate.claude_code \
  --config gepa_results/logs/run_xxx/config.json \
  --model haiku --workers 4 --use-skills

# Claude Code: with proper Claude Code Skills (.claude/skills/<repo>/SKILL.md)
python -m src.evaluate.claude_code_skills \
  --config gepa_results/logs/run_xxx/config.json \
  --model sonnet --workers 4 --use-skills
```

`evaluate/claude_code.py` builds a Docker image with Node.js + Claude Code on top of the SWE-smith base, runs `claude --print` with the problem statement, and verifies the patch with the same two-stage harness. `evaluate/claude_code_skills.py` is identical except it installs skills as a proper Claude Code skill (`.claude/skills/<repo>/SKILL.md` with YAML frontmatter) instead of a flat `CLAUDE.md`.

### Output

Results saved to `gepa_results/logs/run_<timestamp>_<id>/`:

- `best_skills.txt` - Learned skills
- `config.json` - Experiment configuration
- `iterations.jsonl` - Per-evaluation batch metrics
- `proposer_calls/` - Full proposer call logs
- `prompts/` - Each unique prompt version by hash
- `cost_summary.txt` - Cost breakdown (agent vs reflection)
- `gepa_state.bin` - State for resumption

## References

- [SWE-smith](https://swesmith.com)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
