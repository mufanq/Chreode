"""Task registry for unified experiment entrypoints."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    module: str
    description: str


TASKS: dict[str, TaskSpec] = {
    "benchmark": TaskSpec(
        name="benchmark",
        module="cellworldmodel.script.run_benchmark",
        description="Train/evaluate endpoint benchmark with baselines.",
    ),
    "intermediate": TaskSpec(
        name="intermediate",
        module="cellworldmodel.script.run_intermediate_eval",
        description="Train/evaluate timepoint intermediate metrics.",
    ),
    "zesta": TaskSpec(
        name="zesta",
        module="cellworldmodel.script.run_zesta_time_eval",
        description="Train/evaluate ZESTA control-time task.",
    ),
    "fate": TaskSpec(
        name="fate",
        module="scripts.prescient.eval_fate_pearson",
        description="Evaluate Weinreb fate Pearson from a checkpoint.",
    ),
    "foundation": TaskSpec(
        name="foundation",
        module="cellworldmodel.script.run_foundation",
        description="Foundation-model workflow utilities.",
    ),
}


def task_names() -> list[str]:
    return sorted(TASKS)


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task={name!r}; expected one of {task_names()}") from exc
