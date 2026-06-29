"""End-to-end integration smoke test for the full SewerTris pipeline.

Runs all 12 steps (domain -> ... -> flow decomposition) on a small domain via
the shared session-scoped fixture chain, including a real EPA-SWMM simulation,
and asserts that every stage produced its artifact and the final outlet flow
decomposes into physically-sane components. Mirrors
``tests/Comp_time_Test.ipynb`` as an automated regression guard.

Marked ``slow`` (requires pyswmm); skip with ``pytest -m "not slow"``.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_full_pipeline_produces_all_artifacts_and_flow_components(flow_components):
    scenario, df = flow_components
    project = scenario.project

    # Every pipeline stage left its artifact on disk.
    expected_artifacts = [
        project.domain_mask_path,          # 1  domain
        project.layout_blocks_path,        # 3  tetris layout
        project.road_centerlines_path,     # 4  roads
        project.road_polygons_path,        # 4  roads
        project.blocks_path,               # 5  land use
        project.dem_path,                  # 6  DEM
        project.manholes_path,             # 7  network
        project.pipes_path,                # 7  network
        project.subcatchments_path,        # 8  predesign
        project.swmm_inp_path,             # 10 base SWMM model
        scenario.swmm_inp_path,            # 11 scenario SWMM model
    ]
    missing = [str(p) for p in expected_artifacts if not p.exists()]
    assert not missing, f"missing artifacts: {missing}"

    # Final decomposition is valid end-to-end.
    assert len(df) > 0
    assert df["Flow_model_units"].sum() > 0
    assert df["DWF"].sum() > 0
    assert df["GWI"].sum() > 0
    assert (df["RDII_runoff"].astype(float) >= -1e-6).all()
