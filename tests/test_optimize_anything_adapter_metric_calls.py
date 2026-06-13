from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.optimize_anything import RefinerConfig


def test_memory_cache_hits_do_not_count_as_metric_calls():
    calls = {"count": 0}

    def evaluator(candidate, example=None, **kwargs):
        calls["count"] += 1
        return 1.0, {"prompt": candidate["prompt"], "example": example}, {"example": example}

    adapter = OptimizeAnythingAdapter(evaluator=evaluator, parallel=False, cache_mode="memory")
    candidate = {"prompt": "cached"}
    batch = [{"id": 1}, {"id": 2}]

    first_eval = adapter.evaluate(batch, candidate)
    second_eval = adapter.evaluate(batch, candidate)

    assert first_eval.num_metric_calls == 2
    assert second_eval.num_metric_calls == 0
    assert calls["count"] == 2
    assert second_eval.scores == first_eval.scores
    assert second_eval.trajectories == first_eval.trajectories


def test_disk_cache_hits_do_not_count_as_metric_calls(tmp_path):
    calls = {"count": 0}

    def evaluator(candidate, example=None, **kwargs):
        calls["count"] += 1
        return 1.0, {"prompt": candidate["prompt"], "example": example}, {"example": example}

    candidate = {"prompt": "cached"}
    batch = [{"id": 1}]

    first_adapter = OptimizeAnythingAdapter(
        evaluator=evaluator,
        parallel=False,
        cache_mode="disk",
        cache_dir=tmp_path,
    )
    first_eval = first_adapter.evaluate(batch, candidate)

    second_adapter = OptimizeAnythingAdapter(
        evaluator=evaluator,
        parallel=False,
        cache_mode="disk",
        cache_dir=tmp_path,
    )
    second_eval = second_adapter.evaluate(batch, candidate)

    assert first_eval.num_metric_calls == 1
    assert second_eval.num_metric_calls == 0
    assert calls["count"] == 1


def test_refiner_cache_hits_do_not_count_as_metric_calls():
    calls = {"count": 0}

    def evaluator(candidate, example=None, **kwargs):
        calls["count"] += 1
        score = 1.0 if candidate["number"] == "42" else 0.0
        return score, {"number": candidate["number"]}, {"number": candidate["number"]}

    def refiner_lm(_prompt):
        return '{"number": "42"}'

    adapter = OptimizeAnythingAdapter(
        evaluator=evaluator,
        parallel=False,
        refiner_config=RefinerConfig(refiner_lm=refiner_lm, max_refinements=1),
        cache_mode="memory",
    )

    candidate = {
        "number": "42",
        "refiner_prompt": "Return the same number as JSON.",
    }

    eval_batch = adapter.evaluate([{"id": 1}], candidate)

    attempts = eval_batch.trajectories[0]["refiner_prompt_specific_info"]["Attempts"]
    metric_counts = [attempt["metric_call_count"] for attempt in attempts if "side_info" in attempt]
    assert metric_counts == [1, 0]
    assert eval_batch.num_metric_calls == 1
    assert calls["count"] == 1
