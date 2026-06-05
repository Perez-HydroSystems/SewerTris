"""Helpers for running SewerTris project siblings as ensemble members."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import time

from .project import SewerTrisProject


def _progress_writer(progress_path: str | Path | None):
    if progress_path is None:
        return None
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def write_progress(event: dict[str, Any]) -> None:
        progress_path.write_text(
            json.dumps(
                {
                    **event,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    return write_progress


def run_project_sibling(
    spec: dict[str, Any],
    *,
    parent_project_file: str | Path,
    scenario_name: str = "bwf_gwi_rdii",
    run_flow_components: bool = True,
    stop_after_step: str | int | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Clone and rerun one project sibling from a parent project file.

    This function is intentionally importable from a normal Python module so it
    can be called from subprocess workers. PySWMM cannot run multiple
    simulations inside one Python process, so subprocess isolation is the safer
    option for ensemble execution.
    """
    parent = SewerTrisProject.load(parent_project_file)
    sibling = parent.clone_sibling(
        spec["output_dir"],
        name=spec.get("name"),
        changes=spec.get("changes", {}),
    )
    sibling.rerun_from_parent_parameters(
        parent,
        scenario_name=scenario_name,
        run_flow_components=run_flow_components,
        rerun_from=spec.get("rerun_from"),
        stop_after_step=stop_after_step,
        progress_callback=_progress_writer(progress_path),
    )
    scenario = (
        sibling.load_run(scenario_name)
        if scenario_name in sibling.metadata.get("scenarios", {})
        else None
    )
    return {
        "ensemble": spec.get("ensemble"),
        "realization": spec.get("realization"),
        "project_file": str(sibling.project_file),
        "flows_path": str(scenario.flows_path) if scenario else None,
        "rerun_from": sibling.metadata.get("lineage", {}).get("rerun_from"),
        "stop_after_step": stop_after_step,
    }


def run_project_simulation(
    spec: dict[str, Any],
    *,
    parent_project_file: str | Path,
    scenario_name: str = "bwf_gwi_rdii",
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run SWMM inputs, simulation, and flow extraction for one prepared sibling."""
    parent = SewerTrisProject.load(parent_project_file)
    project = SewerTrisProject.load(spec["project_file"])
    project.rerun_from_parent_parameters(
        parent,
        scenario_name=scenario_name,
        run_flow_components=True,
        rerun_from=10,
        progress_callback=_progress_writer(progress_path),
    )
    scenario = project.load_run(scenario_name)
    return {
        "ensemble": spec.get("ensemble"),
        "realization": spec.get("realization"),
        "project_file": str(project.project_file),
        "scenario_input": str(scenario.swmm_inp_path),
        "flows_path": str(scenario.flows_path),
        "rerun_from": "10_dynamic_flow_input_definition_base_model",
    }


def run_project_sibling_from_file(
    payload_path: str | Path,
    *,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one sibling described by a JSON payload file."""
    payload_path = Path(payload_path)
    payload = json.loads(payload_path.read_text())
    if payload.get("mode") == "simulation":
        result = run_project_simulation(
            payload["spec"],
            parent_project_file=payload["parent_project_file"],
            scenario_name=payload.get("scenario_name", "bwf_gwi_rdii"),
            progress_path=payload.get("progress_path"),
        )
    else:
        result = run_project_sibling(
            payload["spec"],
            parent_project_file=payload["parent_project_file"],
            scenario_name=payload.get("scenario_name", "bwf_gwi_rdii"),
            run_flow_components=payload.get("run_flow_components", True),
            stop_after_step=payload.get("stop_after_step"),
            progress_path=payload.get("progress_path"),
        )
    if result_path is not None:
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by notebook subprocess workers."""
    parser = argparse.ArgumentParser(
        description="Run one SewerTris project sibling from a JSON payload."
    )
    parser.add_argument("payload_path")
    parser.add_argument("--result-file", dest="result_path")
    args = parser.parse_args(argv)

    result = run_project_sibling_from_file(
        args.payload_path,
        result_path=args.result_path,
    )
    if args.result_path is None:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_project_sibling",
    "run_project_simulation",
    "run_project_sibling_from_file",
]
