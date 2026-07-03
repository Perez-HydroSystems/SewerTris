from __future__ import annotations

import os

import pytest


os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# Shared pipeline fixtures
#
# The 12-step SewerTris pipeline is sequential: each step consumes the previous
# step's artifacts. To get per-step test coverage without re-running the whole
# (expensive) pipeline once per test, we build it ONCE through a chain of
# session-scoped fixtures. Each fixture depends on the previous one, runs a
# single step on a small but realistic domain, and returns the shared
# ``SewerTrisProject`` (which carries both on-disk artifacts and in-memory
# ``state``). Per-step tests then assert on the persisted outputs.
#
# A failure in one step only fails that fixture and the tests that depend on it;
# earlier steps still report independently.
#
# The domain mask below is the known-good example domain from
# ``Examples/example_02_Stillwater_sewertris_project.ipynb`` (17x10 cells at
# 100 m), which reliably yields a non-degenerate sewer network.
# ---------------------------------------------------------------------------

CELL_SIZE_M = 100

LAND_USE_DISTRIBUTION = {
    "RESIDENTIAL": 0.6,
    "COMMERCIAL": 0.2,
    "INDUSTRIAL": 0.1,
    "PUBLIC": 0.05,
    "RECREATIONAL": 0.05,
}

LAND_USE_INFO = {
    "RESIDENTIAL": {"density": 60, "demand": 100},
    "COMMERCIAL": {"density": 50, "demand": 60},
    "INDUSTRIAL": {"density": 25, "demand": 150},
    "PUBLIC": {"density": 20, "demand": 100},
    "RECREATIONAL": {"density": 10, "demand": 40},
}

STANDARD_DIAMETERS_MM = [
    200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
]
MATERIAL_FRACTIONS = {"PVC": 0.6, "CONCRETE": 0.3, "HDPE": 0.1}
N_BY_MATERIAL = {"PVC": 0.011, "CONCRETE": 0.013, "HDPE": 0.012}
MINIMUM_SLOPE = 0.005
MINIMUM_DIAMETER_MM = 200

# Short simulation window that still covers the first rain pulse (1/2/2025) so
# all three flow components (DWF, GWI, RDII) are exercised at the outlet.
SWMM_OPTIONS = {
    "FLOW_UNITS": "LPS",
    "INFILTRATION": "CURVE_NUMBER",
    "FLOW_ROUTING": "KINWAVE",
    "LINK_OFFSETS": "DEPTH",
    "MIN_SLOPE": "0",
    "ALLOW_PONDING": "NO",
    "SKIP_STEADY_STATE": "NO",
    "START_DATE": "01/01/2025",
    "START_TIME": "00:00:00",
    "REPORT_START_DATE": "01/01/2025",
    "REPORT_START_TIME": "00:00:00",
    "END_DATE": "01/04/2025",
    "END_TIME": "00:00:00",
    "SWEEP_START": "01/01",
    "SWEEP_END": "12/31",
    "DRY_DAYS": "0",
    "REPORT_STEP": "00:15:00",
    "WET_STEP": "00:01:00",
    "DRY_STEP": "00:01:00",
    "ROUTING_STEP": "0:00:60",
    "RULE_STEP": "00:00:00",
    "INERTIAL_DAMPING": "PARTIAL",
    "NORMAL_FLOW_LIMITED": "BOTH",
    "FORCE_MAIN_EQUATION": "D-W",
    "VARIABLE_STEP": "0.75",
    "LENGTHENING_STEP": "0",
    "MIN_SURFAREA": "1.167",
    "MAX_TRIALS": "8",
    "HEAD_TOLERANCE": "0.0015",
    "SYS_FLOW_TOL": "5",
    "LAT_FLOW_TOL": "5",
    "MINIMUM_STEP": "0.5",
    "THREADS": "1",
}

# Hourly / daily / monthly / weekend DWF multipliers (from example_02).
DWF_HOURLY = [
    0.5631, 0.4163, 0.3181, 0.3178, 0.4838, 0.8014,
    1.1337, 1.3239, 1.3321, 1.2512, 1.1900, 1.1703,
    1.1489, 1.1063, 1.0762, 1.0943, 1.1545, 1.2270,
    1.2931, 1.3362, 1.3175, 1.1978, 0.9890, 0.7576,
]
DWF_DAILY = [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2]
DWF_MONTHLY = [0.90, 0.95, 1.00, 1.05, 1.10, 1.10, 1.05, 1.00, 0.95, 0.90, 0.90, 0.90]
DWF_WEEKEND = [
    0.5631, 0.4163, 0.3181, 0.3178, 0.3568, 0.4068,
    0.6068, 0.7068, 0.8014, 1.1337, 1.3239, 1.3321,
    1.1489, 1.1063, 1.0762, 1.0943, 1.1545, 1.2270,
    1.2931, 1.1362, 1.0175, 0.8978, 0.8090, 0.7576,
]
GWI_COEFFICIENT = 0.00005

# A single 1/2/2025 storm (30-min interval) is enough to drive RDII runoff
# within the short simulation window.
RDII_RAINFALL = [
    ("1/2/2025", "00:00", 0.0), ("1/2/2025", "00:30", 0.2205),
    ("1/2/2025", "01:00", 0.2205), ("1/2/2025", "01:30", 0.3855),
    ("1/2/2025", "02:00", 0.3855), ("1/2/2025", "02:30", 0.447),
    ("1/2/2025", "03:00", 0.447), ("1/2/2025", "03:30", 0.447),
    ("1/2/2025", "04:00", 0.447), ("1/2/2025", "04:30", 0.4155),
    ("1/2/2025", "05:00", 0.4155), ("1/2/2025", "05:30", 0.3705),
    ("1/2/2025", "06:00", 0.3705), ("1/2/2025", "06:30", 0.3235),
    ("1/2/2025", "07:00", 0.3235), ("1/2/2025", "07:30", 0.2795),
    ("1/2/2025", "08:00", 0.2795), ("1/2/2025", "08:30", 0.242),
    ("1/2/2025", "09:00", 0.242), ("1/2/2025", "09:30", 0.211),
    ("1/2/2025", "10:00", 0.211), ("1/2/2025", "11:00", 0.186),
    ("1/2/2025", "12:00", 0.1655), ("1/2/2025", "13:00", 0.148),
    ("1/2/2025", "14:00", 0.133), ("1/2/2025", "15:00", 0.12),
    ("1/2/2025", "16:00", 0.1075), ("1/2/2025", "17:00", 0.0955),
    ("1/2/2025", "18:00", 0.085), ("1/2/2025", "19:00", 0.0745),
    ("1/2/2025", "20:00", 0.0655), ("1/2/2025", "21:00", 0.0575),
    ("1/2/2025", "22:00", 0.0505), ("1/2/2025", "23:00", 0.045),
    ("1/3/2025", "00:00", 0.0405), ("1/3/2025", "01:00", 0.0),
]


def make_domain_mask():
    import numpy as np

    mask = np.array(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    grid_meta = {
        "crs_out": "EPSG:3857",
        "origin_x": 0,
        "origin_y": 0,
        "cell_w": CELL_SIZE_M,
        "cell_h": CELL_SIZE_M,
        "rows": mask.shape[0],
        "cols": mask.shape[1],
        "flip_y": False,
    }
    return mask, grid_meta


# ---------------------------------------------------------------------------
# Step fixtures (session-scoped: built once, shared across test modules)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def domain_project(tmp_path_factory):
    """Step 1 -- urban domain definition."""
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    import sewertris as st

    out_dir = tmp_path_factory.mktemp("pipeline") / "project"
    project = st.SewerTrisProject(out_dir, cell_size_m=CELL_SIZE_M, autosave=False)
    mask, grid_meta = make_domain_mask()
    project.define_domain(domain_mask=mask, cell_size_m=CELL_SIZE_M, grid_meta=grid_meta)
    return project


@pytest.fixture(scope="session")
def layout_project(domain_project):
    """Steps 2-3 -- tetromino definition + stochastic completion."""
    import sewertris as st

    tetrominoes, colors = st.get_tetromino_set("full")
    domain_project.define_tetrominoes(tetrominoes, colors)
    domain_project.complete_tetris_layout(seed=1000, georeferenced=True)
    return domain_project


@pytest.fixture(scope="session")
def roads_project(layout_project):
    """Step 4 -- road network extraction (+ boundary shells)."""
    layout_project.generate_roads(road_width=10, simplify_tol=0.5)
    layout_project.extract_road_boundaries(keep_holes=False)
    return layout_project


@pytest.fixture(scope="session")
def landuse_project(roads_project):
    """Step 5 -- land-use assignment."""
    roads_project.assign_land_use(
        land_use_distribution=LAND_USE_DISTRIBUTION,
        seed=42,
    )
    return roads_project


@pytest.fixture(scope="session")
def topo_project(landuse_project):
    """Step 6 -- synthetic DEM generation."""
    import sewertris as st

    config = st.TopographyConfig(
        min_elevation=270,
        max_elevation=290,
        cell_size=10,
        outlet_direction="S",
        smoothing_factor=1,
    )
    landuse_project.generate_topography(
        boundary_path=landuse_project.road_outer_shell_path,
        roads_path=landuse_project.road_polygons_path,
        config=config,
    )
    return landuse_project


@pytest.fixture(scope="session")
def network_project(topo_project):
    """Step 7 -- sewer network generation + DEM embedding."""
    topo_project.generate_sewer_network_V2(
        road_width=10,
        block_size=CELL_SIZE_M * 2,
        main_slope_tolerance=-0.01,
        secondary_slope_tolerance=0.0,
        prefer_slope=0.5,
        tertiary_block_size=CELL_SIZE_M * 10,
        neighbor_radius_factor=1.5,
        tertiary_min_pipe_length=1e-3,
        point_on_line_tol=0.01,
        tertiary_adverse_slope_weight=200.0,
        tertiary_mild_adverse_slope=-0.005,
        tertiary_moderate_adverse_slope=-0.01,
        tertiary_severe_adverse_multiplier=8.0,
    )
    topo_project.embed_sewer_network_in_dem(
        upstream_field="upstream_m",
        downstream_field="downstream_m",
        manhole_id_field="id",
        manhole_elev_field="elevation",
        type_field="type",
        tier_order=("main", "secondary", "tertiary"),
        Smin=0.001,
        along_pipe_weight=2,
        idw_power=2.0,
        idw_k=12,
        idw_tile=1024,
        centerline_writeback=True,
        verify_on_raster=True,
    )
    return topo_project


@pytest.fixture(scope="session")
def predesign_project(network_project):
    """Step 8 -- sewer flow predesign (DWF/GWI/RDII accumulation)."""
    network_project.predesign_flows(
        land_use_info=LAND_USE_INFO,
        gwi_factor_ls_per_m=0.0002,
        rdii_factor_ls_per_m2=0.00002,
        target_crs_m="EPSG:3857",
    )
    return network_project


@pytest.fixture(scope="session")
def designed_project(predesign_project):
    """Step 9 -- pipe sizing and hydraulic properties."""
    predesign_project.design_pipes(
        minimum_slope=MINIMUM_SLOPE,
        material_fractions=MATERIAL_FRACTIONS,
        n_by_material=N_BY_MATERIAL,
        standard_diameters_mm=STANDARD_DIAMETERS_MM,
        minimum_diameter_mm=MINIMUM_DIAMETER_MM,
        min_cover=1.4,
        min_slope=MINIMUM_SLOPE,
        manhole_drop=0.05,
    )
    return predesign_project


@pytest.fixture(scope="session")
def swmm_project(designed_project):
    """Step 10 -- dynamic flow input definition (base SWMM .inp export)."""
    designed_project.export_swmm(options_dict=SWMM_OPTIONS)
    return designed_project


@pytest.fixture(scope="session")
def swmm_ran(swmm_project):
    """Step 11 -- real EPA-SWMM run with DWF + GWI + RDII forcing.

    Returns ``(scenario, depths_df, flows_df)``. Requires pyswmm; runs one
    real simulation shared by the slow tests.
    """
    pytest.importorskip("pyswmm")
    scenario = swmm_project.create_run("bwf_gwi_rdii")
    scenario.assign_dwf_patterns(
        hourly_id="1", hourly_values=DWF_HOURLY,
        daily_id="2", daily_values=DWF_DAILY,
        monthly_id="3", monthly_values=DWF_MONTHLY,
        weekend_id="4", weekend_values=DWF_WEEKEND,
    )
    scenario.assign_gwi_from_pipe_length(coefficient=GWI_COEFFICIENT)
    scenario.add_subcatchment_rdii(
        raingage_id="1",
        raingage_coords=(500, 500),
        timeseries=RDII_RAINFALL,
        interval="0:30",
        n_imperv=0.011, n_perv=0.15,
        s_imperv=0.0, s_perv=0.0,
        pct_zero=0, route_to="OUTLET", pct_routed="",
        infiltration_params=(30, 0.5, 7, "", ""),
        imperv_pct=2, width=100, slope=0.005, curblen=0,
    )
    scenario.add_pollutant_tags()
    depths_df, flows_df = scenario.run_swmm(
        monitored_nodes=["OUTLET"], monitored_links=["P_OUTLET"]
    )
    return scenario, depths_df, flows_df


@pytest.fixture(scope="session")
def flow_components(swmm_ran):
    """Step 12 -- decompose outlet flow into DWF/GWI/RDII components."""
    scenario, _, _ = swmm_ran
    df = scenario.get_flow_components(link_id="P_OUTLET")
    return scenario, df
