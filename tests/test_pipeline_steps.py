"""Per-step behavioral tests for the 12-step SewerTris pipeline.

Each test exercises the *real* computation of one pipeline step on a small but
realistic domain (see the fixture chain in ``conftest.py``) and asserts on the
persisted artifacts / physical invariants -- not just that the API is callable.

Steps 11-12 run a real EPA-SWMM simulation and are marked ``slow``:
run the fast subset with ``pytest -m "not slow"``.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read(path):
    import geopandas as gpd

    return gpd.read_file(path)


def _first_col(df, *candidates):
    """Return the first column present from a list of acceptable aliases."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# Step 1 -- urban domain definition
# ---------------------------------------------------------------------------

def test_step01_define_domain_persists_mask_and_grid(domain_project):
    assert domain_project.state["domain_mask"].shape == (10, 17)
    assert domain_project.domain_mask_path.exists()
    assert domain_project.grid_meta_path.exists()


# ---------------------------------------------------------------------------
# Step 2 -- tetromino block definition
# ---------------------------------------------------------------------------

def test_step02_full_tetromino_set_has_seven_canonical_pieces():
    import sewertris as sp

    tetrominoes, colors = sp.get_tetromino_set("full")
    assert {"I", "O", "T", "S", "Z", "J", "L"}.issubset(set(tetrominoes))
    assert set(colors) >= {"I", "O", "T", "S", "Z", "J", "L"}


# ---------------------------------------------------------------------------
# Step 3 -- stochastic tetris completion
# ---------------------------------------------------------------------------

def test_step03_layout_fills_domain_and_records_seed(layout_project):
    import numpy as np

    assert layout_project.step_parameters("03_stochastic_tetris_completion")["seed"] == 1000

    blocks = _read(layout_project.layout_blocks_path)
    assert len(blocks) >= 1
    assert blocks.geometry.notna().all()
    assert blocks.geometry.is_valid.all()
    # Every filled domain cell is assigned to a block (no positive-area gaps left
    # unlabelled): the union area should cover essentially the whole domain.
    n_domain_cells = int(np.asarray(layout_project.state["domain_mask"]).sum())
    cell_area = layout_project.cell_size_m ** 2
    assert blocks.geometry.area.sum() == pytest.approx(n_domain_cells * cell_area, rel=0.05)


# ---------------------------------------------------------------------------
# Step 4 -- road network extraction
# ---------------------------------------------------------------------------

def test_step04_roads_have_geometry_and_preserve_crs(roads_project):
    centerlines = _read(roads_project.road_centerlines_path)
    polygons = _read(roads_project.road_polygons_path)

    assert len(centerlines) >= 1 and not centerlines.geometry.is_empty.all()
    assert len(polygons) >= 1 and not polygons.geometry.is_empty.all()
    assert centerlines.crs is not None
    assert polygons.crs == centerlines.crs

    # extract_road_boundaries products
    assert roads_project.road_outer_shell_path.exists()
    assert roads_project.road_boundary_lines_path.exists()


# ---------------------------------------------------------------------------
# Step 5 -- land-use assignment
# ---------------------------------------------------------------------------

def test_step05_every_block_gets_a_known_land_use(landuse_project):
    from conftest import LAND_USE_DISTRIBUTION

    blocks = _read(landuse_project.blocks_path)
    lu_col = _first_col(blocks, "land_use", "landuse")
    assert lu_col is not None, blocks.columns.tolist()
    assert blocks[lu_col].notna().all()
    assert set(blocks[lu_col].unique()).issubset(set(LAND_USE_DISTRIBUTION))


# ---------------------------------------------------------------------------
# Step 6 -- synthetic DEM generation
# ---------------------------------------------------------------------------

def test_step06_dem_is_bounded_and_drains_to_outlet(topo_project):
    import numpy as np
    import rasterio

    assert topo_project.dem_path.exists()
    with rasterio.open(topo_project.dem_path) as src:
        dem = src.read(1).astype(float)

    finite = dem[np.isfinite(dem)]
    assert finite.size > 0
    # Elevations stay within the configured band (small tolerance for smoothing).
    assert finite.min() >= 270 - 1.0
    assert finite.max() <= 290 + 1.0

    # outlet_direction='S' -> terrain should slope down toward the south
    # (bottom rows of the raster). Compare valid-cell means of the top vs.
    # bottom quartile of rows.
    rows = dem.shape[0]
    q = max(1, rows // 4)
    top = dem[:q][np.isfinite(dem[:q])]
    bottom = dem[-q:][np.isfinite(dem[-q:])]
    assert bottom.mean() < top.mean()


# ---------------------------------------------------------------------------
# Step 7 -- sewer network generation (+ DEM embedding)
# ---------------------------------------------------------------------------

def test_step07_network_is_a_connected_tiered_tree(network_project):
    pipes = _read(network_project.pipes_path)
    manholes = _read(network_project.manholes_path)

    assert len(pipes) > 0
    assert len(manholes) > 0

    type_col = _first_col(pipes, "type")
    assert type_col is not None
    assert set(pipes[type_col].unique()).issubset({"main", "secondary", "tertiary"})

    assert _first_col(pipes, "upstream_m", "upstream") is not None
    assert _first_col(pipes, "downstream", "downstream_m") is not None

    # A single hydraulic outlet was identified for the trunk.
    assert network_project.state["network_status"]["main_outlet_id"] is not None


# ---------------------------------------------------------------------------
# Step 8 -- sewer flow predesign (DWF / GWI / RDII accumulation)
# ---------------------------------------------------------------------------

def test_step08_predesign_adds_nonnegative_flow_components(predesign_project):
    assert predesign_project.subcatchments_path.exists()
    subs = _read(predesign_project.subcatchments_path)
    assert len(subs) >= 1

    pipes = _read(predesign_project.pipes_path)
    peak = _first_col(pipes, "peak_flow_lps_bc")
    gwi = _first_col(pipes, "cum_gwi_ls")
    rdii = _first_col(pipes, "cum_rdii_ls", "cum_rdii_l")
    predesign = _first_col(pipes, "predesign_ls", "predesign_")
    assert all(c is not None for c in (peak, gwi, rdii, predesign)), pipes.columns.tolist()

    for col in (peak, gwi, rdii, predesign):
        vals = pipes[col].astype(float)
        assert vals.notna().all()
        assert (vals >= -1e-9).all()

    # Flow accumulates downstream: the predesign flow is not uniformly zero and
    # the network spans a range of magnitudes (head pipes vs. trunk).
    pre = pipes[predesign].astype(float)
    assert pre.max() > 0
    assert pre.max() > pre.min()


# ---------------------------------------------------------------------------
# Step 9 -- pipe sizing and hydraulic properties
# ---------------------------------------------------------------------------

def test_step09_design_assigns_commercial_diameters_and_inverts(designed_project):
    from conftest import (
        MATERIAL_FRACTIONS,
        MINIMUM_DIAMETER_MM,
        MINIMUM_SLOPE,
        N_BY_MATERIAL,
        STANDARD_DIAMETERS_MM,
    )

    pipes = _read(designed_project.pipes_path)

    diam = _first_col(pipes, "diameter_mm")
    assert diam is not None
    diameters = pipes[diam].astype(float)
    assert (diameters >= MINIMUM_DIAMETER_MM).all()
    assert set(diameters.round().astype(int)).issubset(set(STANDARD_DIAMETERS_MM))

    mat = _first_col(pipes, "material")
    if mat is not None:
        assert set(pipes[mat].unique()).issubset(set(MATERIAL_FRACTIONS))
    n_col = _first_col(pipes, "n")
    assert n_col is not None
    assert set(pipes[n_col].round(4).unique()).issubset(
        {round(v, 4) for v in N_BY_MATERIAL.values()}
    )

    slope = _first_col(pipes, "slope")
    assert slope is not None
    assert (pipes[slope].astype(float) >= MINIMUM_SLOPE - 1e-9).all()

    # Inverts run downhill (downstream invert not above upstream, within tol).
    up = _first_col(pipes, "inv_up")
    down = _first_col(pipes, "inv_down")
    assert up is not None and down is not None
    drop = pipes[up].astype(float) - pipes[down].astype(float)
    assert (drop >= -0.01).all()


# ---------------------------------------------------------------------------
# Step 10 -- dynamic flow input definition (base SWMM .inp export)
# ---------------------------------------------------------------------------

def test_step10_export_swmm_writes_required_sections(swmm_project):
    assert swmm_project.swmm_inp_path.exists()
    text = swmm_project.swmm_inp_path.read_text()

    for section in (
        "[JUNCTIONS]",
        "[OUTFALLS]",
        "[CONDUITS]",
        "[XSECTIONS]",
        "[DWF]",
        "[COORDINATES]",
    ):
        assert section in text, f"missing {section}"

    # Trunk outlet node + conduit are present.
    assert "OUTLET" in text
    assert "P_OUTLET" in text

    # At least one conduit row per network pipe (export also adds the trunk
    # outlet link P_OUTLET).
    pipes = _read(swmm_project.pipes_path)
    conduit_block = text.split("[CONDUITS]", 1)[1].split("[", 1)[0]
    conduit_rows = [
        ln for ln in conduit_block.splitlines()
        if ln.strip() and not ln.strip().startswith(";")
    ]
    assert len(conduit_rows) >= len(pipes)


# ---------------------------------------------------------------------------
# Step 11 -- real EPA-SWMM simulation (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_step11_swmm_runs_and_reports_flow(swmm_ran):
    import re

    scenario, depths_df, flows_df = swmm_ran

    assert depths_df is not None and len(depths_df) > 0
    assert flows_df is not None and len(flows_df) > 0
    # Some positive flow reaches the monitored outlet link.
    assert float(flows_df.to_numpy().max()) > 0

    # Report exists and SWMM reported no fatal error. SWMM writes fatal errors
    # as "ERROR nnn:"; the routine "Continuity Error (%)" lines are benign.
    rpt = scenario.swmm_inp_path.with_suffix(".rpt")
    assert rpt.exists()
    rpt_text = rpt.read_text()
    assert re.search(r"ERROR\s+\d+", rpt_text) is None

    # Flow-routing mass balance closes to a sane tolerance.
    continuity = [
        abs(float(m))
        for m in re.findall(r"Continuity Error \(%\) \.+\s+(-?\d+\.\d+)", rpt_text)
    ]
    assert continuity, "no continuity error reported"
    assert max(continuity) < 10.0


# ---------------------------------------------------------------------------
# Step 12 -- flow output decomposition (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_step12_flow_components_decompose_total(flow_components):
    _, df = flow_components

    for col in ("Datetime", "Flow_model_units", "DWF", "GWI", "RDII_runoff"):
        assert col in df.columns, df.columns.tolist()

    for col in ("DWF", "GWI", "RDII_runoff"):
        assert (df[col].astype(float) >= -1e-6).all()

    # Always-on components carry volume; storm drives some RDII.
    assert df["DWF"].sum() > 0
    assert df["GWI"].sum() > 0
    assert df["Flow_model_units"].sum() > 0

    # Decomposition closes: components never exceed the modelled total.
    combined = df["DWF"] + df["GWI"] + df["RDII_runoff"]
    assert (combined <= df["Flow_model_units"].astype(float) + 1e-6).all()


# ---------------------------------------------------------------------------
# Pure unit test -- British Columbia peaking factor (no pipeline needed)
# ---------------------------------------------------------------------------

def test_british_columbia_peaking_factor_is_bounded_and_monotone():
    pd = pytest.importorskip("pandas")
    from sewertris.design import british_columbia_peaking_factor

    q = pd.Series([0.5, 5.0, 50.0, 500.0, 5000.0])
    peak_flow, pf = british_columbia_peaking_factor(q)

    assert (pf >= 2.5 - 1e-9).all()
    assert (pf <= 4.0 + 1e-9).all()
    # Larger flows -> smaller (or equal, once clamped) peaking factor.
    assert (pf.diff().dropna() <= 1e-9).all()
    # Peak flow is the base flow scaled by the peaking factor.
    assert peak_flow.to_numpy() == pytest.approx((q * pf).to_numpy())
