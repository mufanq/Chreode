"""Unified entrypoint for CellWorldModel experiment tasks.

Examples:
  python -m cellworldmodel.script.run_experiment --list-tasks
  python -m cellworldmodel.script.run_experiment intermediate --experiment g2a_m10_md_adamw --dataset weinreb_scvi
  python -m cellworldmodel.script.run_experiment zesta --experiment g2a_m10_md_adamw --output-dir output/foo
"""
from __future__ import annotations

import argparse
import runpy
import sys

from cellworldmodel.benchmark.task_registry import TASKS, get_task, task_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch to a registered CellWorldModel experiment task.",
        add_help=True,
    )
    parser.add_argument("--list-tasks", action="store_true", help="List registered tasks and exit.")
    parser.add_argument("task", nargs="?", choices=task_names())
    parser.add_argument("task_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the task.")
    args = parser.parse_args()

    if args.list_tasks:
        for name in task_names():
            spec = TASKS[name]
            print(f"{name}\t{spec.module}\t{spec.description}")
        return
    if args.task is None:
        parser.error("task is required unless --list-tasks is used")

    spec = get_task(args.task)
    forwarded = args.task_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = [spec.module, *forwarded]
    runpy.run_module(spec.module, run_name="__main__")


if __name__ == "__main__":
    main()
