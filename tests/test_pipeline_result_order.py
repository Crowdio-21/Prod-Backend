from types import SimpleNamespace

from foreman.core.job_manager import _task_input_index


def test_pipeline_results_follow_input_order_past_ten_inputs():
    ids = [f"job1_task_2_{i}" for i in range(25)]
    shuffled = sorted(ids)  # string order: _0, _1, _10, _11, ..., _2, ...
    ordered = sorted((SimpleNamespace(id=i) for i in shuffled), key=_task_input_index)
    assert [t.id for t in ordered] == ids
