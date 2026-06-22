from __future__ import annotations


def test_top_level_api_exports_expected_functions():
    import sewertris as sp

    assert callable(sp.build_domain_mask_from_shapefile)
    assert callable(sp.fill_domain_with_tetrominoes_and_blocks)
    assert callable(sp.get_tetromino_set)
    assert callable(sp.generate_road_network_from_blocks)
    assert callable(sp.generate_topography)
    assert callable(sp.export_swmm_inp)
    assert callable(sp.plot_domain_mask)
    assert callable(sp.plot_two_models)
    assert hasattr(sp, "TopographyConfig")
    assert hasattr(sp, "SewerTrisProject")
    assert sp.build_domain_mask_from_shapefile.__module__ == "sewertris.domain"
    assert sp.export_swmm_inp.__module__ == "sewertris.swmm"
    assert sp.plot_domain_mask.__module__ == "sewertris.plots"
    assert sp.plot_two_models.__module__ == "sewertris.plots"


def test_named_tetromino_set_helper_returns_four_piece_sibling():
    import sewertris as sp

    tetrominoes, colors = sp.get_tetromino_set("I_O_T_S_only")

    assert list(tetrominoes) == ["I", "O", "T", "S"]
    assert set(colors) == {"I", "O", "T", "S"}


def test_plot_two_models_compares_project_domain_masks(tmp_path):
    import pytest

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import sewertris as sp

    project1 = sp.SewerTrisProject(tmp_path / "project1", autosave=False)
    project2 = sp.SewerTrisProject(tmp_path / "project2", autosave=False)
    grid_meta = {
        "crs_out": "EPSG:3857",
        "origin_x": 0,
        "origin_y": 0,
        "cell_w": 100,
        "cell_h": 100,
        "rows": 2,
        "cols": 2,
    }
    project1.define_domain(
        domain_mask=np.array([[1, 0], [1, 1]], dtype=np.uint8),
        cell_size_m=100,
        grid_meta=grid_meta,
    )
    project2.define_domain(
        domain_mask=np.array([[1, 1], [0, 1]], dtype=np.uint8),
        cell_size_m=100,
        grid_meta=grid_meta,
    )

    fig = sp.plot_two_models(
        "domain_mask",
        project1,
        project2,
        labels=("Base", "Sibling"),
        show_grid=False,
        show=False,
    )

    assert fig.axes[0].get_title() == "Base"
    assert fig.axes[1].get_title() == "Sibling"
    plt.close(fig)


def test_plot_ensemble_results_summarizes_flow_components(tmp_path):
    import pytest

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import sewertris as sp

    times = pd.date_range("1990-01-01", periods=3, freq="h")
    rows = []
    for ensemble, realization, scale in [
        ("seed_only", 1, 1.0),
        ("seed_only", 2, 2.0),
        ("drainage_shift_west", 1, 3.0),
    ]:
        flow_path = tmp_path / f"{ensemble}_{realization}.csv"
        pd.DataFrame(
            {
                "Datetime": times,
                "Flow_model_units": [10, 12, 11],
                "DWF": [scale, scale, scale],
                "GWI": [2 * scale, 2 * scale, 2 * scale],
                "RDII_runoff": [0, 4 * scale, 0],
            }
        ).to_csv(flow_path, index=False)
        rows.append(
            {
                "base_model": "Stillwater",
                "ensemble": ensemble,
                "realization": realization,
                "flows_path": flow_path,
            }
        )

    fig, summary = sp.plot_ensemble_results(
        pd.DataFrame(rows),
        group_cols=("base_model", "ensemble"),
        show=False,
        return_summary=True,
    )

    assert callable(sp.plot_ensemble_results)
    assert len(fig.axes) == 10
    assert set(summary["component"]) == {"BWF / DWF", "GWI", "RDII"}
    assert len(summary) == 9
    assert summary["volume"].gt(0).all()
    assert summary["peak_flow"].gt(0).all()
    assert "Stillwater | seed_only" in set(summary["group_label"])
    plt.close(fig)


def test_submodule_api_remains_available():
    import sewertris as sp

    assert callable(sp.plots.plot_flow_components_v2)
    assert callable(sp.plots.plot_ensemble_results)
    assert callable(sp.swmm.assign_inflow_from_pipe_length)


def test_complete_tetris_layout_seed_is_reproducible(monkeypatch, tmp_path):
    import numpy as np
    import sewertris as sp
    import sewertris.layout as layout

    monkeypatch.setattr(
        layout,
        "export_individual_figures_to_shapefile",
        lambda **kwargs: None,
    )

    project = sp.SewerTrisProject(tmp_path / "project", cell_size_m=100, autosave=False)
    tetrominoes, _ = sp.get_tetromino_set("I_O_T_S_only")
    original_rotation_ids = {
        key: [id(rotation) for rotation in rotations]
        for key, rotations in tetrominoes.items()
    }
    project.define_domain(domain_mask=np.ones((6, 6), dtype=np.uint8), cell_size_m=100)
    project.define_tetrominoes(tetrominoes)

    board_a, _, _ = project.complete_tetris_layout(seed=123, georeferenced=False)
    board_b, _, _ = project.complete_tetris_layout(seed=123, georeferenced=False)

    assert np.array_equal(board_a, board_b)
    assert project.step_parameters("03_stochastic_tetris_completion")["seed"] == 123
    assert {
        key: [id(rotation) for rotation in rotations]
        for key, rotations in tetrominoes.items()
    } == original_rotation_ids


def test_project_saves_loads_json_metadata(tmp_path):
    import json
    import numpy as np
    import sewertris as sp

    project = sp.SewerTrisProject(
        tmp_path / "project",
        cell_size_m=100,
        name="Example Project",
        autosave=False,
    )

    assert project.blocks_path.name == "city_blocks.gpkg"
    assert project.swmm_inp_path.name == "sewer_model.inp"
    assert project.domain_mask_path.name == "domain_mask.npy"
    assert project.scenarios_dir.exists()

    domain_mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    grid_meta = {
        "crs_out": "EPSG:3857",
        "origin_x": 0,
        "origin_y": 0,
        "cell_w": 100,
        "cell_h": 100,
        "rows": 2,
        "cols": 2,
    }
    project.define_domain(
        domain_mask=domain_mask,
        cell_size_m=100,
        grid_meta=grid_meta,
    )
    project.record_step(
        "test_step",
        parameters={"path": project.pipes_path},
        outputs={"array": {"shape": [2, 3]}},
    )
    project.save()

    data = json.loads(project.project_file.read_text())
    assert data["name"] == "Example Project"
    assert data["cell_size_m"] == 100
    assert data["geometry_format"] == "gpkg"
    assert data["manifest"]["storage"]["vector_format"] == "gpkg"
    assert data["manifest"]["artifacts"]["domain_mask"]["exists"] is True
    assert data["metadata"]["steps"]["test_step"]["parameters"]["path"].endswith(
        "sewer_pipes.gpkg"
    )

    loaded = sp.SewerTrisProject.load(project.output_dir)
    assert loaded.output_dir == project.output_dir
    assert loaded.cell_size_m == 100
    assert "test_step" in loaded.metadata["steps"]
    assert loaded.geometry_format == "gpkg"
    assert loaded.state["domain_mask"].shape == (2, 2)


def test_project_load_resolves_legacy_relative_output_dir_to_project_file_parent(tmp_path):
    import json
    import sewertris as sp

    project_dir = tmp_path / "examples" / "output_example_2_project"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "sewertris_project.json"
    project_file.write_text(
        json.dumps(
            {
                "output_dir": "output_example_2_project",
                "cell_size_m": 100,
                "name": "Legacy Relative Project",
                "metadata": {},
                "state": {},
                "geometry_format": "gpkg",
            }
        )
    )

    loaded = sp.SewerTrisProject.load(project_file)

    assert loaded.output_dir == project_dir
    assert loaded.project_file == project_file
    assert not (tmp_path / "output_example_2_project").exists()


def test_project_creates_scenario_folder_without_base_inp(tmp_path):
    import sewertris as sp

    project = sp.SewerTrisProject(tmp_path / "project", autosave=False)
    scenario = project.create_run("bwf_pattern_a", copy_base_inp=False)

    assert isinstance(scenario, sp.SewerTrisScenario)
    assert scenario.output_dir.exists()
    assert scenario.swmm_inp_path.name == "sewer_model.inp"

    scenario.record_step("assign_dwf_patterns", parameters={"hourly_id": "1"})
    project.save()

    loaded = sp.SewerTrisProject.load(project.project_file)
    assert "bwf_pattern_a" in loaded.metadata["scenarios"]
    assert (
        "assign_dwf_patterns"
        in loaded.metadata["scenarios"]["bwf_pattern_a"]["steps"]
    )


def test_project_clone_sibling_copies_reusable_domain_state(tmp_path):
    import numpy as np
    import sewertris as sp

    base = sp.SewerTrisProject(
        tmp_path / "base",
        cell_size_m=100,
        name="Base Project",
        autosave=False,
    )
    base.define_domain(
        domain_mask=np.array([[1, 1], [0, 1]], dtype=np.uint8),
        cell_size_m=100,
        grid_meta={
            "crs_out": "EPSG:3857",
            "origin_x": 0,
            "origin_y": 0,
            "cell_w": 100,
            "cell_h": 100,
            "rows": 2,
            "cols": 2,
        },
    )
    base.record_step("02_tetris_block_definition", parameters={"tetromino_keys": ["I"]})
    base.save()

    sibling = base.clone_sibling(
        tmp_path / "sibling",
        changes={"tetromino_set": "I_O_T_S_only"},
    )

    assert sibling.geometry_format == "gpkg"
    assert sibling.domain_mask_path.exists()
    assert sibling.grid_meta_path.exists()
    assert not sibling.layout_blocks_path.exists()
    assert sibling.metadata["lineage"]["type"] == "sibling"
    assert sibling.metadata["lineage"]["parent_project_file"] == str(base.project_file)
    assert sibling.metadata["lineage"]["rerun_from"] == "02_tetris_block_definition"
    assert "domain_mask" in sibling.metadata["lineage"]["copied_artifacts"]

    loaded = sp.SewerTrisProject.load(sibling.project_file)
    assert loaded.state["domain_mask"].shape == (2, 2)


def test_sibling_replay_applies_road_width_change(monkeypatch, tmp_path):
    import sewertris as sp

    base = sp.SewerTrisProject(tmp_path / "base", cell_size_m=100, autosave=False)
    base.record_step(
        "04_road_network_extraction",
        parameters={"road_width": 10, "simplify_tol": 0.5},
    )
    base.record_step("05_land_use_assignment", parameters={"seed": 42})
    base.record_step("06_synthetic_dem_generation", parameters={"config": {}})
    base.record_step(
        "07_sewer_network_generation",
        parameters={"road_width": 10, "block_size": 200},
    )
    base.record_step("07_embed_sewer_network_in_dem", parameters={})
    base.record_step(
        "08_sewer_flow_predesign",
        parameters={"land_use_info": {"RESIDENTIAL": {"density": 1, "demand": 1}}},
    )
    base.record_step("09_pipe_sizing_and_hydraulic_properties", parameters={})
    base.record_step(
        "10_dynamic_flow_input_definition_base_model",
        parameters={"options_dict": {"START_DATE": "01/01/1990", "START_TIME": "00:00:00"}},
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "add_subcatchment_rdii",
        parameters={"timeseries": [["1/1/1990", "00:00", 0.0]]},
    )

    calls = {}

    def fake_generate_roads(self, road_width=10, simplify_tol=0.5, **kwargs):
        calls["roads"] = {"road_width": road_width, "simplify_tol": simplify_tol}
        self.state["road_width"] = road_width
        return "road_lines", "road_buffer", "EPSG:3857"

    def fake_extract_road_boundaries(self, **kwargs):
        calls["extract_boundaries"] = kwargs
        return self.road_boundary_lines_path, self.road_outer_shell_path

    def fake_assign_land_use(self, **kwargs):
        calls["land_use"] = kwargs
        return "blocks"

    def fake_generate_topography(self, **kwargs):
        calls["topography"] = kwargs
        return "elevation", "xx", "yy", "mask"

    def fake_generate_sewer_network(self, **kwargs):
        calls["network"] = kwargs
        return "pipes"

    def fake_embed(self, **kwargs):
        calls["embed"] = kwargs
        return self.dem_path

    def fake_predesign(self, **kwargs):
        calls["predesign"] = kwargs
        return "predesign"

    def fake_design(self, **kwargs):
        calls["design"] = kwargs
        return "pipes_clean", "manholes_clean"

    def fake_export_swmm(self, **kwargs):
        calls["swmm"] = kwargs
        return self.swmm_inp_path

    class FakeScenario:
        def __init__(self, project, name):
            self.swmm_inp_path = project.scenarios_dir / name / "sewer_model.inp"
            self.flows_path = project.scenarios_dir / name / "flows.nc"

        def assign_dwf_patterns(self, **kwargs):
            calls["dwf"] = kwargs

        def assign_gwi_from_pipe_length(self, **kwargs):
            calls["gwi"] = kwargs

        def add_subcatchment_rdii(self, **kwargs):
            calls["rdii"] = kwargs

    def fake_create_run(self, name):
        calls["scenario_name"] = name
        return FakeScenario(self, name)

    monkeypatch.setattr(sp.SewerTrisProject, "generate_roads", fake_generate_roads)
    monkeypatch.setattr(sp.SewerTrisProject, "extract_road_boundaries", fake_extract_road_boundaries)
    monkeypatch.setattr(sp.SewerTrisProject, "assign_land_use", fake_assign_land_use)
    monkeypatch.setattr(sp.SewerTrisProject, "generate_topography", fake_generate_topography)
    monkeypatch.setattr(sp.SewerTrisProject, "generate_sewer_network", fake_generate_sewer_network)
    monkeypatch.setattr(sp.SewerTrisProject, "embed_sewer_network_in_dem", fake_embed)
    monkeypatch.setattr(sp.SewerTrisProject, "predesign_flows", fake_predesign)
    monkeypatch.setattr(sp.SewerTrisProject, "design_pipes", fake_design)
    monkeypatch.setattr(sp.SewerTrisProject, "export_swmm", fake_export_swmm)
    monkeypatch.setattr(sp.SewerTrisProject, "create_run", fake_create_run)

    sibling = base.clone_sibling(
        tmp_path / "sibling",
        changes={"road_width": 20},
        copy_artifacts=False,
    )
    sibling.rerun_from_parent_parameters(base, run_flow_components=False)

    assert sibling.metadata["lineage"]["rerun_from"] == "04_road_network_extraction"
    assert calls["roads"]["road_width"] == 20
    assert calls["network"]["road_width"] == 20
    assert sibling.step_parameters("99_rerun_from_parent_parameters")["changes"]["road_width"] == 20


def test_sibling_replay_applies_design_and_scenario_changes(monkeypatch, tmp_path):
    import sewertris as sp

    base = sp.SewerTrisProject(tmp_path / "base", cell_size_m=100, autosave=False)
    base.record_step(
        "09_pipe_sizing_and_hydraulic_properties",
        parameters={
            "material_fractions": {"PVC": 0.6, "CONCRETE": 0.4},
            "standard_diameters_mm": [200, 250, 300],
            "minimum_diameter_mm": 200,
        },
    )
    base.record_step(
        "10_dynamic_flow_input_definition_base_model",
        parameters={
            "options_dict": {
                "FLOW_UNITS": "LPS",
                "START_DATE": "01/01/1990",
                "START_TIME": "00:00:00",
            }
        },
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "assign_dwf_patterns",
        parameters={"hourly_id": "1", "hourly_values": [1.0] * 24},
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "assign_gwi_from_pipe_length",
        parameters={"coefficient": 0.00001},
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "add_subcatchment_rdii",
        parameters={"timeseries": [["1/1/1990", "00:00", 0.0]], "imperv_pct": 5},
    )

    calls = {}

    def fake_design(self, **kwargs):
        calls["design"] = kwargs
        return "pipes_clean", "manholes_clean"

    def fake_export_swmm(self, **kwargs):
        calls["swmm"] = kwargs
        return self.swmm_inp_path

    class FakeScenario:
        def __init__(self, project, name):
            self.swmm_inp_path = project.scenarios_dir / name / "sewer_model.inp"
            self.flows_path = project.scenarios_dir / name / "flows.nc"

        def assign_dwf_patterns(self, **kwargs):
            calls["dwf"] = kwargs

        def assign_gwi_from_pipe_length(self, **kwargs):
            calls["gwi"] = kwargs

        def add_subcatchment_rdii(self, **kwargs):
            calls["rdii"] = kwargs

    def fake_create_run(self, name):
        calls["scenario_name"] = name
        return FakeScenario(self, name)

    monkeypatch.setattr(sp.SewerTrisProject, "design_pipes", fake_design)
    monkeypatch.setattr(sp.SewerTrisProject, "export_swmm", fake_export_swmm)
    monkeypatch.setattr(sp.SewerTrisProject, "create_run", fake_create_run)

    hourly = [1.2] * 24
    rainfall = [["1/1/1990", "00:00", 1.5]]
    sibling = base.clone_sibling(
        tmp_path / "sibling",
        changes={
            "diameters": [200, 400, 600],
            "materials": {"PVC": 1.0},
            "bwf": {"hourly_values": hourly},
            "gwi_coefficient": 0.0003,
            "rdii": {"timeseries": rainfall, "imperv_pct": 15},
            "swmm_options": {"FLOW_UNITS": "CMS"},
        },
        copy_artifacts=False,
    )
    sibling.rerun_from_parent_parameters(base, run_flow_components=False)

    assert sibling.metadata["lineage"]["rerun_from"] == "09_pipe_sizing_and_hydraulic_properties"
    assert calls["design"]["standard_diameters_mm"] == [200, 400, 600]
    assert calls["design"]["material_fractions"] == {"PVC": 1.0}
    assert calls["swmm"]["options_dict"]["FLOW_UNITS"] == "CMS"
    assert calls["swmm"]["options_dict"]["START_DATE"] == "01/01/1990"
    assert calls["dwf"]["hourly_values"] == hourly
    assert calls["gwi"]["coefficient"] == 0.0003
    assert calls["rdii"]["imperv_pct"] == 15
    assert calls["rdii"]["timeseries"] == rainfall


def test_sibling_replay_applies_gwi_and_rdii_rasters(monkeypatch, tmp_path):
    import sewertris as sp

    base = sp.SewerTrisProject(tmp_path / "base", cell_size_m=100, autosave=False)
    base.record_step(
        "10_dynamic_flow_input_definition_base_model",
        parameters={
            "options_dict": {
                "START_DATE": "01/01/1990",
                "START_TIME": "00:00:00",
                "END_DATE": "01/02/1990",
                "END_TIME": "00:00:00",
            }
        },
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "assign_gwi_from_pipe_length",
        parameters={"coefficient": 0.00001},
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "add_subcatchment_rdii",
        parameters={
            "timeseries": [["1/1/1990", "00:00", 0.0]],
            "imperv_pct": 5,
            "infiltration_params": [90, 0.5, 7, "", ""],
        },
    )

    calls = {}

    def fake_export_swmm(self, **kwargs):
        calls["swmm"] = kwargs
        self.swmm_inp_path.write_text("[TITLE]\n")
        return self.swmm_inp_path

    def fake_assign_gwi_from_pipe_length(self, **kwargs):
        calls["gwi_pipe_length"] = kwargs

    def fake_assign_gwi_from_raster(self, **kwargs):
        calls["gwi_raster"] = kwargs

    def fake_add_subcatchment_rdii(self, **kwargs):
        calls["rdii_uniform"] = kwargs

    def fake_add_subcatchment_rdii_raster(self, **kwargs):
        calls["rdii_raster"] = kwargs

    monkeypatch.setattr(sp.SewerTrisProject, "export_swmm", fake_export_swmm)
    monkeypatch.setattr(
        sp.SewerTrisScenario,
        "assign_gwi_from_pipe_length",
        fake_assign_gwi_from_pipe_length,
    )
    monkeypatch.setattr(
        sp.SewerTrisScenario,
        "assign_gwi_from_raster",
        fake_assign_gwi_from_raster,
    )
    monkeypatch.setattr(
        sp.SewerTrisScenario,
        "add_subcatchment_rdii",
        fake_add_subcatchment_rdii,
    )
    monkeypatch.setattr(
        sp.SewerTrisScenario,
        "add_subcatchment_rdii_raster",
        fake_add_subcatchment_rdii_raster,
    )

    gwi_raster = "rasters/gwi.tif"
    rdii_raster = tmp_path / "rdii.tif"
    rdii_raster.write_bytes(b"rdii")

    sibling = base.clone_sibling(
        tmp_path / "sibling",
        changes={
            "gwi_raster": {
                "raster_path": gwi_raster,
                "samples_per_pipe": 7,
            },
            "rdii_raster": {
                "raster_path": rdii_raster,
                "rdii_to_imperv_scale": (0.0, 300.0),
            },
            "rdii": {
                "infiltration_params": (50, 0.5, 7, "", ""),
            },
        },
        copy_artifacts=False,
    )
    sibling.rerun_from_parent_parameters(base, run_flow_components=False)

    assert sibling.metadata["lineage"]["rerun_from"] == "10_dynamic_flow_input_definition_base_model"
    assert "gwi_pipe_length" not in calls
    assert calls["gwi_raster"]["raster_path"] == str(sibling.output_dir / gwi_raster)
    assert calls["gwi_raster"]["samples_per_pipe"] == 7
    assert "rdii_uniform" not in calls
    assert calls["rdii_raster"]["rdii_raster_path"] == str(rdii_raster)
    assert calls["rdii_raster"]["rdii_to_imperv_scale"] == [0.0, 300.0]
    assert "imperv_pct" not in calls["rdii_raster"]
    assert calls["rdii_raster"]["infiltration_params"] == [50, 0.5, 7, "", ""]


def test_sibling_replay_generated_rdii_raster_sets_density_scale(monkeypatch, tmp_path):
    import sewertris as sp

    base = sp.SewerTrisProject(tmp_path / "base", cell_size_m=100, autosave=False)
    base.record_step(
        "10_dynamic_flow_input_definition_base_model",
        parameters={
            "options_dict": {
                "START_DATE": "01/01/1990",
                "START_TIME": "00:00:00",
                "END_DATE": "01/02/1990",
                "END_TIME": "00:00:00",
            }
        },
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "assign_gwi_from_pipe_length",
        parameters={"coefficient": 0.00001},
    )
    base.record_scenario_step(
        "bwf_gwi_rdii",
        "add_subcatchment_rdii",
        parameters={"timeseries": [["1/1/1990", "00:00", 0.0]], "imperv_pct": 5},
    )

    calls = {}

    def fake_export_swmm(self, **kwargs):
        self.swmm_inp_path.write_text("[TITLE]\n")
        return self.swmm_inp_path

    def fake_generate_rdii_density_raster(self, **kwargs):
        calls["generate_rdii"] = kwargs
        return self.rdii_raster_path

    def fake_add_subcatchment_rdii_raster(self, **kwargs):
        calls["rdii_raster"] = kwargs

    monkeypatch.setattr(sp.SewerTrisProject, "export_swmm", fake_export_swmm)
    monkeypatch.setattr(
        sp.SewerTrisProject,
        "generate_rdii_density_raster",
        fake_generate_rdii_density_raster,
    )
    monkeypatch.setattr(sp.SewerTrisScenario, "assign_gwi_from_pipe_length", lambda *a, **k: None)
    monkeypatch.setattr(
        sp.SewerTrisScenario,
        "add_subcatchment_rdii_raster",
        fake_add_subcatchment_rdii_raster,
    )

    sibling = base.clone_sibling(
        tmp_path / "sibling",
        changes={
            "rdii_raster": {
                "min_density": 2.0,
                "max_density": 6.0,
                "n_hills": 2,
                "hill_max_density": 14.0,
            },
        },
        copy_artifacts=False,
    )
    sibling.rerun_from_parent_parameters(base, run_flow_components=False)

    assert calls["generate_rdii"]["min_density"] == 2.0
    assert calls["generate_rdii"]["hill_max_density"] == 14.0
    assert calls["rdii_raster"]["rdii_raster_path"] == str(sibling.rdii_raster_path)
    assert calls["rdii_raster"]["rdii_to_imperv_scale"] == [2.0, 14.0]
    assert "imperv_pct" not in calls["rdii_raster"]


def test_pipe_topology_aliases_support_geopackage_and_shapefile_names():
    import pytest

    pd = pytest.importorskip("pandas")
    from sewertris._deps import ensure_pipe_topology_aliases

    gpkg_style = ensure_pipe_topology_aliases(
        pd.DataFrame({"upstream_m": ["MH001"], "downstream_m": ["MH002"]})
    )
    assert gpkg_style.loc[0, "downstream"] == "MH002"

    shp_style = ensure_pipe_topology_aliases(
        pd.DataFrame({"upstream_m": ["MH001"], "downstream": ["MH002"]})
    )
    assert shp_style.loc[0, "downstream_m"] == "MH002"


def test_project_patches_downstream_alias_before_predesign(monkeypatch, tmp_path):
    import pytest

    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point, Polygon
    import sewertris as sp

    project = sp.SewerTrisProject(tmp_path / "project", autosave=False)
    gpd.GeoDataFrame(
        {"land_use": ["RESIDENTIAL"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    ).to_file(project.blocks_path)
    gpd.GeoDataFrame(
        {"id": ["MH001", "MH002"], "elevation": [10.0, 9.0]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:3857",
    ).to_file(project.manholes_path)
    gpd.GeoDataFrame(
        {
            "pipe_id": ["P00001"],
            "upstream_m": ["MH001"],
            "downstream_m": ["MH002"],
            "type": ["main"],
            "cumulative_flow_lps": [1.0],
            "cum_gwi_ls": [0.1],
            "cum_rdii_ls": [0.2],
        },
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:3857",
    ).to_file(project.pipes_path)
    project.dem_path.write_bytes(b"placeholder")

    calls = {}

    def fake_delineate(**kwargs):
        calls["delineate"] = kwargs
        gpd.GeoDataFrame(
            {"pipe_id": ["P00001"], "base_flow_lps": [1.0]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:3857",
        ).to_file(kwargs["output_path"])

    def fake_assign_flow_to_pipes_fast(**kwargs):
        pipes = gpd.read_file(kwargs["pipes_path"])
        assert "downstream" in pipes.columns
        pipes["cumulative"] = [1.0]
        pipes["peak_flow_"] = [4.0]
        pipes.to_file(kwargs["output_path"])

    def fake_compute_gwi_cumulative(**kwargs):
        pipes = gpd.read_file(kwargs["pipes_path"])
        assert "downstream" in pipes.columns
        pipes["cum_gwi_ls"] = [0.1]
        pipes.to_file(kwargs["out_path"])

    def fake_compute_rdii_and_accumulate(**kwargs):
        pipes = gpd.read_file(kwargs["pipes_path"])
        assert "downstream" in pipes.columns
        pipes["cum_rdii_l"] = [0.2]
        pipes.to_file(kwargs["out_pipes"])

    def fake_add_predesign_flow(pipes_path, out_path):
        pipes = gpd.read_file(pipes_path)
        pipes["predesign_ls"] = [4.3]
        pipes.to_file(out_path)
        return pipes

    monkeypatch.setattr(
        "sewertris.hydrology.delineate_afferent_areas_and_baseflow",
        fake_delineate,
    )
    monkeypatch.setattr(
        "sewertris.hydrology.assign_flow_to_pipes_fast",
        fake_assign_flow_to_pipes_fast,
    )
    monkeypatch.setattr(
        "sewertris.hydrology.compute_gwi_cumulative",
        fake_compute_gwi_cumulative,
    )
    monkeypatch.setattr(
        "sewertris.hydrology.compute_rdii_and_accumulate",
        fake_compute_rdii_and_accumulate,
    )
    monkeypatch.setattr("sewertris.design.add_predesign_flow", fake_add_predesign_flow)

    result = project.predesign_flows(
        land_use_info={"RESIDENTIAL": {"density": 1, "demand": 1}},
    )

    assert "delineate" in calls
    assert "downstream" in gpd.read_file(project.pipes_path).columns
    assert "predesign_ls" in result.columns


def test_add_predesign_flow_accepts_geopackage_full_field_names(tmp_path):
    import pytest

    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString
    from sewertris.design import add_predesign_flow

    pipes_path = tmp_path / "pipes.gpkg"
    gpd.GeoDataFrame(
        {
            "pipe_id": ["P00001"],
            "upstream_m": ["MH001"],
            "downstream_m": ["MH002"],
            "peak_flow_lps_bc": [4.0],
            "cum_gwi_ls": [0.1],
            "cum_rdii_ls": [0.2],
        },
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:3857",
    ).to_file(pipes_path)

    result = add_predesign_flow(pipes_path=pipes_path, out_path=pipes_path)

    assert result.loc[0, "predesign_ls"] == 4.3
    saved = gpd.read_file(pipes_path)
    assert "predesign_" in saved.columns
    assert saved.loc[0, "predesign_ls"] == 4.3


def test_export_swmm_accepts_geopackage_full_flow_and_topology_names(tmp_path):
    import pytest

    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point
    from sewertris.swmm import export_swmm_inp

    pipes_path = tmp_path / "pipes.gpkg"
    manholes_path = tmp_path / "manholes.gpkg"
    output_path = tmp_path / "model.inp"

    gpd.GeoDataFrame(
        {
            "pipe_id": ["P00001"],
            "upstream_m": ["MH001"],
            "downstream_m": ["MH002"],
            "length_m": [10.0],
            "inv_up": [8.0],
            "inv_down": [7.9],
            "n": [0.013],
            "diameter_mm": [200],
            "own_flow_lps": [1.2],
            "peak_flow_lps_bc": [4.0],
        },
        geometry=[LineString([(0, 0), (10, 0)])],
        crs="EPSG:3857",
    ).to_file(pipes_path)
    gpd.GeoDataFrame(
        {
            "id": ["MH001", "MH002"],
            "elevation": [10.0, 9.5],
        },
        geometry=[Point(0, 0), Point(10, 0)],
        crs="EPSG:3857",
    ).to_file(manholes_path)

    export_swmm_inp(
        pipes_path=pipes_path,
        manholes_path=manholes_path,
        output_path=output_path,
    )

    text = output_path.read_text()
    assert "[DWF]" in text
    assert "MH001" in text
    assert "P_OUTLET" in text


def test_swmm_wrappers_convert_path_objects_for_pyswmm(monkeypatch, tmp_path):
    import sys
    import types

    from sewertris.swmm import (
        get_flow_components_from_node_pyswmm,
        run_swmm_and_plot,
    )

    seen_paths = []
    fake_pyswmm = types.ModuleType("pyswmm")

    class FakeSimulation:
        def __init__(self, inputfile):
            assert isinstance(inputfile, str)
            seen_paths.append(inputfile)
            self.current_time = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

    class FakeLinks:
        def __init__(self, sim):
            pass

        def __getitem__(self, item):
            return types.SimpleNamespace(flow=0.0)

    class FakeNodes:
        def __init__(self, sim):
            pass

        def __getitem__(self, item):
            return types.SimpleNamespace(depth=0.0, pollut_quality={})

    fake_pyswmm.Simulation = FakeSimulation
    fake_pyswmm.Links = FakeLinks
    fake_pyswmm.Nodes = FakeNodes
    monkeypatch.setitem(sys.modules, "pyswmm", fake_pyswmm)

    inp_path = tmp_path / "model.inp"
    run_swmm_and_plot(inp_path, monitored_nodes=["OUTLET"], monitored_links=["P_OUTLET"])
    get_flow_components_from_node_pyswmm(inp_path)

    assert seen_paths == [str(inp_path), str(inp_path)]
