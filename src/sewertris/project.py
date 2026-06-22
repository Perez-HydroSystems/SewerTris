"""Project and scenario workflow helpers for SewerTris."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import copy
import json
import shutil


_VECTOR_FORMAT_EXTENSIONS = {
    "gpkg": ".gpkg",
    "geopackage": ".gpkg",
    "shp": ".shp",
    "shapefile": ".shp",
}

_SHAPEFILE_SIDECARS = (
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
    ".fix",
    ".sbn",
    ".sbx",
)

_STEP_ORDER = {
    "01_urban_domain_definition": 1,
    "02_tetris_block_definition": 2,
    "03_stochastic_tetris_completion": 3,
    "04_road_network_extraction": 4,
    "05_land_use_assignment": 5,
    "06_synthetic_dem_generation": 6,
    "07_sewer_network_generation": 7,
    "07_embed_sewer_network_in_dem": 7,
    "08_sewer_flow_predesign": 8,
    "09_pipe_sizing_and_hydraulic_properties": 9,
    "10_dynamic_flow_input_definition_base_model": 10,
    "11_epa_swmm_simulation": 11,
    "12_flow_output_decomposition": 12,
}

_CHANGE_STEP_MAP = {
    "domain": "01_urban_domain_definition",
    "domain_mask": "01_urban_domain_definition",
    "shapefile_path": "01_urban_domain_definition",
    "cell_size_m": "01_urban_domain_definition",
    "tetromino_set": "02_tetris_block_definition",
    "tetrominoes": "02_tetris_block_definition",
    "tetromino_colors": "02_tetris_block_definition",
    "layout": "03_stochastic_tetris_completion",
    "georeferenced": "03_stochastic_tetris_completion",
    "layout_seed": "03_stochastic_tetris_completion",
    "random_seed": "03_stochastic_tetris_completion",
    "seed": "03_stochastic_tetris_completion",
    "road_width": "04_road_network_extraction",
    "roads": "04_road_network_extraction",
    "simplify_tol": "04_road_network_extraction",
    "keep_holes": "04_road_network_extraction",
    "land_use": "05_land_use_assignment",
    "land_use_distribution": "05_land_use_assignment",
    "config": "06_synthetic_dem_generation",
    "topography_config": "06_synthetic_dem_generation",
    "dem": "06_synthetic_dem_generation",
    "embed_dem": "07_sewer_network_generation",
    "dem_embedding": "07_sewer_network_generation",
    "sewer_network": "07_sewer_network_generation",
    "network": "07_sewer_network_generation",
    "block_size": "07_sewer_network_generation",
    "main_slope_tolerance": "07_sewer_network_generation",
    "secondary_slope_tolerance": "07_sewer_network_generation",
    "prefer_slope": "07_sewer_network_generation",
    "tertiary_block_size": "07_sewer_network_generation",
    "neighbor_radius_factor": "07_sewer_network_generation",
    "tertiary_min_pipe_length": "07_sewer_network_generation",
    "point_on_line_tol": "07_sewer_network_generation",
    "max_search_depth": "07_sewer_network_generation",
    "max_outer_iterations": "07_sewer_network_generation",
    "min_pipe_length": "07_sewer_network_generation",
    "network_method": "07_sewer_network_generation",
    "sewer_network_method": "07_sewer_network_generation",
    "use_sewer_network_v2": "07_sewer_network_generation",
    "tertiary_adverse_slope_weight": "07_sewer_network_generation",
    "tertiary_mild_adverse_slope": "07_sewer_network_generation",
    "tertiary_moderate_adverse_slope": "07_sewer_network_generation",
    "tertiary_severe_adverse_multiplier": "07_sewer_network_generation",
    "predesign": "08_sewer_flow_predesign",
    "land_use_info": "08_sewer_flow_predesign",
    "gwi_factor_ls_per_m": "08_sewer_flow_predesign",
    "rdii_factor_ls_per_m2": "08_sewer_flow_predesign",
    "target_crs_m": "08_sewer_flow_predesign",
    "pipe_design": "09_pipe_sizing_and_hydraulic_properties",
    "materials": "09_pipe_sizing_and_hydraulic_properties",
    "material_fractions": "09_pipe_sizing_and_hydraulic_properties",
    "n_by_material": "09_pipe_sizing_and_hydraulic_properties",
    "diameters": "09_pipe_sizing_and_hydraulic_properties",
    "standard_diameters_mm": "09_pipe_sizing_and_hydraulic_properties",
    "minimum_diameter_mm": "09_pipe_sizing_and_hydraulic_properties",
    "minimum_slope": "09_pipe_sizing_and_hydraulic_properties",
    "min_cover": "09_pipe_sizing_and_hydraulic_properties",
    "min_slope": "09_pipe_sizing_and_hydraulic_properties",
    "manhole_drop": "09_pipe_sizing_and_hydraulic_properties",
    "swmm": "10_dynamic_flow_input_definition_base_model",
    "swmm_options": "10_dynamic_flow_input_definition_base_model",
    "options_dict": "10_dynamic_flow_input_definition_base_model",
    "bwf": "10_dynamic_flow_input_definition_base_model",
    "dwf": "10_dynamic_flow_input_definition_base_model",
    "hourly_id": "10_dynamic_flow_input_definition_base_model",
    "hourly_values": "10_dynamic_flow_input_definition_base_model",
    "daily_id": "10_dynamic_flow_input_definition_base_model",
    "daily_values": "10_dynamic_flow_input_definition_base_model",
    "monthly_id": "10_dynamic_flow_input_definition_base_model",
    "monthly_values": "10_dynamic_flow_input_definition_base_model",
    "weekend_id": "10_dynamic_flow_input_definition_base_model",
    "weekend_values": "10_dynamic_flow_input_definition_base_model",
    "gwi": "10_dynamic_flow_input_definition_base_model",
    "gwi_coefficient": "10_dynamic_flow_input_definition_base_model",
    "gwi_raster": "10_dynamic_flow_input_definition_base_model",
    "gwi_inflow_raster": "10_dynamic_flow_input_definition_base_model",
    "rdii": "10_dynamic_flow_input_definition_base_model",
    "rdii_raster": "10_dynamic_flow_input_definition_base_model",
    "rdii_density_raster": "10_dynamic_flow_input_definition_base_model",
    "rainfall": "10_dynamic_flow_input_definition_base_model",
    "scenario": "10_dynamic_flow_input_definition_base_model",
    "scenario_name": "10_dynamic_flow_input_definition_base_model",
    "run_flow_components": "10_dynamic_flow_input_definition_base_model",
}

_DEFAULT_LAND_USE_INFO = {
    "RESIDENTIAL": {"density": 60, "demand": 100},
    "COMMERCIAL": {"density": 50, "demand": 60},
    "INDUSTRIAL": {"density": 25, "demand": 150},
    "PUBLIC": {"density": 20, "demand": 100},
    "RECREATIONAL": {"density": 10, "demand": 40},
}

_DEFAULT_STANDARD_DIAMETERS_MM = [
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vector_suffix(vector_format: str) -> str:
    key = (vector_format or "gpkg").lower().lstrip(".")
    return _VECTOR_FORMAT_EXTENSIONS.get(key, f".{key}")


def _normal_step(step: str | int | None) -> tuple[str, int]:
    if step is None:
        return "01_urban_domain_definition", 1
    if isinstance(step, int):
        for name, order in _STEP_ORDER.items():
            if order == step:
                return name, order
        raise ValueError(f"Unknown project step number: {step}")
    step_text = str(step)
    if step_text in _STEP_ORDER:
        return step_text, _STEP_ORDER[step_text]
    lowered = step_text.lower().strip()
    for name, order in _STEP_ORDER.items():
        if lowered == name.lower() or lowered in name.lower():
            return name, order
    aliases = {
        "domain": "01_urban_domain_definition",
        "tetris": "02_tetris_block_definition",
        "tetrominoes": "02_tetris_block_definition",
        "layout": "03_stochastic_tetris_completion",
        "roads": "04_road_network_extraction",
        "land_use": "05_land_use_assignment",
        "topography": "06_synthetic_dem_generation",
        "dem": "06_synthetic_dem_generation",
        "network": "07_sewer_network_generation",
        "predesign": "08_sewer_flow_predesign",
        "design": "09_pipe_sizing_and_hydraulic_properties",
        "swmm": "10_dynamic_flow_input_definition_base_model",
        "simulation": "11_epa_swmm_simulation",
        "outputs": "12_flow_output_decomposition",
    }
    if lowered in aliases:
        name = aliases[lowered]
        return name, _STEP_ORDER[name]
    raise ValueError(f"Unknown project step: {step}")


def _json_safe(value: Any) -> Any:
    """Convert common workflow values into JSON-safe metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(getattr(value, "shape", ())),
        }
    if hasattr(value, "crs") and hasattr(value, "__len__"):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "crs": str(getattr(value, "crs", "")),
        }
    return repr(value)


def _deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively update a dict without mutating the inputs."""
    result = copy.deepcopy(base or {})
    if not isinstance(updates, dict):
        return result
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_project_path(project: "SewerTrisProject", value: str | Path) -> str:
    """Resolve project-configured paths while preserving existing absolute paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path)
    project_relative = project.output_dir / path
    if project_relative.exists():
        return str(project_relative)
    return str(project_relative)


def _merge_change_groups(
    params: dict[str, Any] | None,
    changes: dict[str, Any],
    *,
    groups: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Merge grouped and direct sibling changes into step parameters."""
    merged = copy.deepcopy(params or {})
    for group in groups:
        value = changes.get(group)
        if isinstance(value, dict):
            merged = _deep_update(merged, value)
    for key in keys:
        if key in changes:
            merged[key] = copy.deepcopy(changes[key])
    return merged


def _looks_like_timeseries(value: Any) -> bool:
    """Return True for SWMM rainfall time series rows like (date, time, value)."""
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return False
    first = value[0]
    return isinstance(first, (list, tuple)) and len(first) >= 3


@dataclass
class SewerTrisScenario:
    """A dynamic SWMM run inside a :class:`SewerTrisProject`.

    Scenarios reuse the project's base physical system and write modified SWMM
    inputs/results into their own folder.
    """

    project: "SewerTrisProject"
    name: str
    output_dir: str | Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def swmm_inp_path(self) -> Path:
        return self.output_dir / "sewer_model.inp"

    @property
    def flows_path(self) -> Path:
        return self.output_dir / "flows.nc"

    def path(self, *parts: str | Path) -> Path:
        return self.output_dir.joinpath(*map(Path, parts))

    def record_step(
        self,
        step: str,
        parameters: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        self.project.record_scenario_step(
            self.name,
            step,
            parameters=parameters,
            outputs=outputs,
        )

    def prepare_swmm_input(self, source_path: str | Path | None = None) -> Path:
        """Copy the base SWMM input into the scenario folder."""
        source = Path(source_path) if source_path is not None else self.project.swmm_inp_path
        if not source.exists():
            raise FileNotFoundError(f"Base SWMM input not found: {source}")
        shutil.copy2(source, self.swmm_inp_path)
        self.record_step(
            "prepare_swmm_input",
            parameters={"source_path": source},
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def assign_dwf_patterns(self, **kwargs: Any) -> Path:
        """Rewrite DWF patterns in this scenario's SWMM input."""
        from .design import assign_all_dwf_patterns

        if not self.swmm_inp_path.exists():
            self.prepare_swmm_input()
        assign_all_dwf_patterns(
            inp_path=self.swmm_inp_path,
            output_path=self.swmm_inp_path,
            **kwargs,
        )
        self.record_step(
            "assign_dwf_patterns",
            parameters=kwargs,
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def assign_gwi_from_pipe_length(self, coefficient: float) -> Path:
        """Assign baseline GWI inflows from pipe length for this scenario."""
        from .swmm import assign_inflow_from_pipe_length

        if not self.swmm_inp_path.exists():
            self.prepare_swmm_input()
        assign_inflow_from_pipe_length(
            inp_path=self.swmm_inp_path,
            output_path=self.swmm_inp_path,
            coefficient=coefficient,
        )
        self.record_step(
            "assign_gwi_from_pipe_length",
            parameters={"coefficient": coefficient},
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def assign_gwi_from_raster(
        self,
        raster_path: str | Path,
        samples_per_pipe: int = 5,
    ) -> Path:
        """Assign baseline GWI inflows by sampling a coefficient raster."""
        from .swmm import assign_inflow_from_raster

        if not self.swmm_inp_path.exists():
            self.prepare_swmm_input()
        assign_inflow_from_raster(
            inp_path=self.swmm_inp_path,
            output_path=self.swmm_inp_path,
            raster_path=raster_path,
            samples_per_pipe=samples_per_pipe,
        )
        self.record_step(
            "assign_gwi_from_raster",
            parameters={
                "raster_path": raster_path,
                "samples_per_pipe": samples_per_pipe,
            },
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def add_subcatchment_rdii(self, **kwargs: Any) -> Path:
        """Add rainfall/RDII subcatchment data to this scenario."""
        from .swmm import add_subcatchment_data_to_inp

        if not self.swmm_inp_path.exists():
            self.prepare_swmm_input()
        kwargs.setdefault("subcatchments_path", self.project.subcatchments_path)
        add_subcatchment_data_to_inp(
            inp_path=self.swmm_inp_path,
            output_path=self.swmm_inp_path,
            **kwargs,
        )
        self.record_step(
            "add_subcatchment_rdii",
            parameters=kwargs,
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def add_subcatchment_rdii_raster(
        self,
        rdii_raster_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Add rainfall/RDII subcatchments using spatially variable RDII raster."""
        from .swmm import add_subcatchment_data_with_rdii_raster

        if not self.swmm_inp_path.exists():
            self.prepare_swmm_input()
        kwargs.setdefault("subcatchments_path", self.project.subcatchments_path)
        add_subcatchment_data_with_rdii_raster(
            inp_path=self.swmm_inp_path,
            output_path=self.swmm_inp_path,
            rdii_raster_path=rdii_raster_path,
            **kwargs,
        )
        self.record_step(
            "add_subcatchment_rdii_raster",
            parameters={"rdii_raster_path": rdii_raster_path, **kwargs},
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def add_pollutant_tags(self) -> Path:
        """Add tracer pollutants used for flow component decomposition."""
        from .swmm import auto_add_pollutants_to_inp_fixed

        if not self.swmm_inp_path.exists():
            self.prepare_swmm_input()
        auto_add_pollutants_to_inp_fixed(
            inp_path=self.swmm_inp_path,
            output_path=self.swmm_inp_path,
        )
        self.record_step(
            "add_pollutant_tags",
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return self.swmm_inp_path

    def run_swmm(self, monitored_nodes=None, monitored_links=None):
        """Run PySWMM and record the monitored node/link output."""
        from .swmm import run_swmm_and_plot

        depths_df, flows_df = run_swmm_and_plot(
            self.swmm_inp_path,
            monitored_nodes=monitored_nodes,
            monitored_links=monitored_links,
        )
        self.record_step(
            "run_swmm",
            parameters={
                "monitored_nodes": monitored_nodes,
                "monitored_links": monitored_links,
            },
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return depths_df, flows_df

    def get_flow_components(self, link_id: str = "P_OUTLET", node_id: str = "OUTLET"):
        """Extract DWF/RDII/GWI flow components from a completed simulation."""
        from .swmm import get_flow_components_from_node_pyswmm

        df = get_flow_components_from_node_pyswmm(
            inp_path=self.swmm_inp_path,
            link_id=link_id,
            node_id=node_id,
        )
        self.record_step(
            "get_flow_components",
            parameters={"link_id": link_id, "node_id": node_id},
        )
        return df

    def save_flow_components(self, df, output_path: str | Path | None = None) -> Path:
        """Save a flow-component DataFrame to NetCDF."""
        output = Path(output_path) if output_path is not None else self.flows_path
        ds = df.set_index("Datetime").to_xarray() if "Datetime" in df.columns else df.to_xarray()
        ds.to_netcdf(output)
        self.record_step(
            "save_flow_components",
            outputs={"flows_path": output},
        )
        return output


@dataclass
class SewerTrisProject:
    """Manage a reproducible SewerTris project and its SWMM scenarios.

    The procedural ``sp.function(...)`` API remains the source of truth. This
    class manages standard output paths, JSON metadata, repeated runs, and
    common workflow methods for notebooks and batch experiments.
    """

    output_dir: str | Path
    cell_size_m: float | None = None
    name: str = "SewerTris Project"
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    autosave: bool = True
    geometry_format: str = "gpkg"

    project_filename: str = "sewertris_project.json"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("schema_version", "1.0")
        self.metadata.setdefault("name", self.name)
        self.metadata.setdefault("created_at", _utc_now())
        self.metadata.setdefault("steps", {})
        self.metadata.setdefault("scenarios", {})
        self.metadata.setdefault("storage", {})
        self.metadata["storage"].setdefault("vector_format", self.geometry_format)
        self.metadata["storage"].setdefault("state_dir", str(self.state_dir))
        self.metadata["storage"].setdefault("inputs_dir", str(self.inputs_dir))
        if self.cell_size_m is not None:
            self.metadata.setdefault("cell_size_m", self.cell_size_m)

    @property
    def project_file(self) -> Path:
        return self.output_dir / self.project_filename

    @property
    def scenarios_dir(self) -> Path:
        return self.output_dir / "scenarios"

    @property
    def inputs_dir(self) -> Path:
        return self.output_dir / "inputs"

    @property
    def state_dir(self) -> Path:
        return self.output_dir / "state"

    @property
    def domain_mask_path(self) -> Path:
        return self.state_dir / "domain_mask.npy"

    @property
    def grid_meta_path(self) -> Path:
        return self.state_dir / "grid_meta.json"

    @property
    def vector_suffix(self) -> str:
        return _vector_suffix(self.geometry_format)

    @property
    def layout_blocks_path(self) -> Path:
        return self.output_dir / f"city_layout{self.vector_suffix}"

    @property
    def blocks_path(self) -> Path:
        return self.output_dir / f"city_blocks{self.vector_suffix}"

    @property
    def road_centerlines_path(self) -> Path:
        return self.output_dir / f"road_centerlines{self.vector_suffix}"

    @property
    def road_polygons_path(self) -> Path:
        return self.output_dir / f"road_polygons{self.vector_suffix}"

    @property
    def road_boundary_lines_path(self) -> Path:
        return self.output_dir / f"road_boundary_lines{self.vector_suffix}"

    @property
    def road_outer_shell_path(self) -> Path:
        return self.output_dir / f"road_outer_shell{self.vector_suffix}"

    @property
    def dem_path(self) -> Path:
        return self.output_dir / "generated_topography.tif"

    @property
    def gwi_raster_path(self) -> Path:
        return self.output_dir / "gwi_inflow_coefficients.tif"

    @property
    def rdii_raster_path(self) -> Path:
        return self.output_dir / "rdii_density.tif"

    @property
    def manholes_path(self) -> Path:
        return self.output_dir / f"manholes{self.vector_suffix}"

    @property
    def pipes_path(self) -> Path:
        return self.output_dir / f"sewer_pipes{self.vector_suffix}"

    @property
    def subcatchments_path(self) -> Path:
        return self.output_dir / f"sewer_subcatchments{self.vector_suffix}"

    @property
    def swmm_inp_path(self) -> Path:
        return self.output_dir / "sewer_model.inp"

    @property
    def flows_path(self) -> Path:
        return self.output_dir / "flows.nc"

    def path(self, *parts: str | Path) -> Path:
        """Build a path inside the project output directory."""
        return self.output_dir.joinpath(*map(Path, parts))

    def paths(self, as_str: bool = False) -> dict[str, str | Path]:
        values = {
            "project_file": self.project_file,
            "inputs_dir": self.inputs_dir,
            "state_dir": self.state_dir,
            "domain_mask_path": self.domain_mask_path,
            "grid_meta_path": self.grid_meta_path,
            "layout_blocks_path": self.layout_blocks_path,
            "blocks_path": self.blocks_path,
            "road_centerlines_path": self.road_centerlines_path,
            "road_polygons_path": self.road_polygons_path,
            "road_boundary_lines_path": self.road_boundary_lines_path,
            "road_outer_shell_path": self.road_outer_shell_path,
            "dem_path": self.dem_path,
            "gwi_raster_path": self.gwi_raster_path,
            "rdii_raster_path": self.rdii_raster_path,
            "manholes_path": self.manholes_path,
            "pipes_path": self.pipes_path,
            "subcatchments_path": self.subcatchments_path,
            "swmm_inp_path": self.swmm_inp_path,
            "flows_path": self.flows_path,
            "scenarios_dir": self.scenarios_dir,
        }
        if as_str:
            return {key: str(value) for key, value in values.items()}
        return values

    def _artifact_entry(self, path: str | Path, kind: str, step: str | None = None) -> dict[str, Any]:
        artifact_path = Path(path)
        entry = {
            "path": str(artifact_path),
            "kind": kind,
            "step": step,
            "exists": artifact_path.exists(),
        }
        if artifact_path.exists() and artifact_path.is_file():
            stat = artifact_path.stat()
            entry["size_bytes"] = stat.st_size
            entry["modified_at"] = datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat(timespec="seconds")
        return entry

    def artifacts(self) -> dict[str, dict[str, Any]]:
        """Return a manifest of files used for reproducibility and siblings."""
        artifacts = {
            "domain_mask": self._artifact_entry(
                self.domain_mask_path,
                kind="state",
                step="01_urban_domain_definition",
            ),
            "grid_meta": self._artifact_entry(
                self.grid_meta_path,
                kind="state",
                step="01_urban_domain_definition",
            ),
            "layout_blocks": self._artifact_entry(
                self.layout_blocks_path,
                kind="vector",
                step="03_stochastic_tetris_completion",
            ),
            "city_blocks": self._artifact_entry(
                self.blocks_path,
                kind="vector",
                step="05_land_use_assignment",
            ),
            "road_centerlines": self._artifact_entry(
                self.road_centerlines_path,
                kind="vector",
                step="04_road_network_extraction",
            ),
            "road_polygons": self._artifact_entry(
                self.road_polygons_path,
                kind="vector",
                step="04_road_network_extraction",
            ),
            "road_boundary_lines": self._artifact_entry(
                self.road_boundary_lines_path,
                kind="vector",
                step="04_road_network_extraction",
            ),
            "road_outer_shell": self._artifact_entry(
                self.road_outer_shell_path,
                kind="vector",
                step="04_road_network_extraction",
            ),
            "dem": self._artifact_entry(
                self.dem_path,
                kind="raster",
                step="06_synthetic_dem_generation",
            ),
            "gwi_raster": self._artifact_entry(
                self.gwi_raster_path,
                kind="raster",
                step="10_dynamic_flow_input_definition_base_model",
            ),
            "rdii_raster": self._artifact_entry(
                self.rdii_raster_path,
                kind="raster",
                step="10_dynamic_flow_input_definition_base_model",
            ),
            "manholes": self._artifact_entry(
                self.manholes_path,
                kind="vector",
                step="07_sewer_network_generation",
            ),
            "pipes": self._artifact_entry(
                self.pipes_path,
                kind="vector",
                step="09_pipe_sizing_and_hydraulic_properties",
            ),
            "subcatchments": self._artifact_entry(
                self.subcatchments_path,
                kind="vector",
                step="08_sewer_flow_predesign",
            ),
            "swmm_input": self._artifact_entry(
                self.swmm_inp_path,
                kind="swmm",
                step="10_dynamic_flow_input_definition_base_model",
            ),
            "flow_components": self._artifact_entry(
                self.flows_path,
                kind="netcdf",
                step="12_flow_output_decomposition",
            ),
        }
        input_domain = self.metadata.get("inputs", {}).get("domain_path")
        if input_domain:
            artifacts["input_domain"] = self._artifact_entry(
                input_domain,
                kind="input_vector",
                step="01_urban_domain_definition",
            )
        for scenario_name in self.metadata.get("scenarios", {}):
            scenario = self.load_run(scenario_name)
            artifacts[f"scenario:{scenario_name}:swmm_input"] = self._artifact_entry(
                scenario.swmm_inp_path,
                kind="scenario_swmm",
                step="10_dynamic_flow_input_definition_base_model",
            )
            artifacts[f"scenario:{scenario_name}:flow_components"] = self._artifact_entry(
                scenario.flows_path,
                kind="scenario_netcdf",
                step="12_flow_output_decomposition",
            )
        return artifacts

    def _persist_domain_state(self, domain_mask=None, grid_meta=None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if domain_mask is not None:
            import numpy as np

            np.save(self.domain_mask_path, domain_mask)
        if grid_meta is not None:
            self.grid_meta_path.write_text(json.dumps(_json_safe(grid_meta), indent=2))
        self.metadata.setdefault("state_files", {})
        if self.domain_mask_path.exists():
            self.metadata["state_files"]["domain_mask"] = str(self.domain_mask_path)
        if self.grid_meta_path.exists():
            self.metadata["state_files"]["grid_meta"] = str(self.grid_meta_path)

    def _load_persisted_state(self) -> None:
        if self.domain_mask_path.exists():
            import numpy as np

            self.state["domain_mask"] = np.load(self.domain_mask_path)
        if self.grid_meta_path.exists():
            self.state["grid_meta"] = json.loads(self.grid_meta_path.read_text())

    def _copy_dataset_file(self, source: str | Path, destination: str | Path) -> Path:
        source_path = Path(source)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() == ".shp":
            copied_main = destination_path.with_suffix(".shp")
            for suffix in _SHAPEFILE_SIDECARS:
                sidecar = source_path.with_suffix(suffix)
                if sidecar.exists():
                    shutil.copy2(sidecar, copied_main.with_suffix(suffix))
            return copied_main
        shutil.copy2(source_path, destination_path)
        return destination_path

    def _copy_input_dataset(self, source: str | Path) -> Path:
        source_path = Path(source)
        destination = self.inputs_dir / source_path.name
        return self._copy_dataset_file(source_path, destination)

    def _ensure_pipe_downstream_alias(self) -> None:
        """Keep full GeoPackage names and legacy shapefile-truncated names usable."""
        if not self.pipes_path.exists():
            return
        import geopandas as gpd

        pipes = gpd.read_file(self.pipes_path)
        changed = False
        if "downstream" not in pipes.columns and "downstream_m" in pipes.columns:
            pipes["downstream"] = pipes["downstream_m"]
            changed = True
        if "downstream_m" not in pipes.columns and "downstream" in pipes.columns:
            pipes["downstream_m"] = pipes["downstream"]
            changed = True
        if changed:
            pipes.to_file(self.pipes_path)

    def remember(self, **items: Any) -> "SewerTrisProject":
        """Store in-memory workflow results on the project object."""
        self.state.update(items)
        return self

    def record_step(
        self,
        step: str,
        parameters: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> "SewerTrisProject":
        self.metadata.setdefault("steps", {})[step] = {
            "updated_at": _utc_now(),
            "parameters": _json_safe(parameters or {}),
            "outputs": _json_safe(outputs or {}),
        }
        if self.autosave:
            self.save()
        return self

    def record_scenario_step(
        self,
        scenario_name: str,
        step: str,
        parameters: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> "SewerTrisProject":
        scenarios = self.metadata.setdefault("scenarios", {})
        scenario = scenarios.setdefault(
            scenario_name,
            {
                "created_at": _utc_now(),
                "output_dir": str(self.scenarios_dir / scenario_name),
                "steps": {},
            },
        )
        scenario.setdefault("steps", {})[step] = {
            "updated_at": _utc_now(),
            "parameters": _json_safe(parameters or {}),
            "outputs": _json_safe(outputs or {}),
        }
        if self.autosave:
            self.save()
        return self

    def to_dict(self) -> dict[str, Any]:
        self.metadata["updated_at"] = _utc_now()
        self.metadata.setdefault("storage", {})
        self.metadata["storage"]["vector_format"] = self.geometry_format
        self.metadata["storage"]["state_dir"] = str(self.state_dir)
        self.metadata["storage"]["inputs_dir"] = str(self.inputs_dir)
        self.metadata["paths"] = self.paths(as_str=True)
        self.metadata["artifacts"] = self.artifacts()
        if self.cell_size_m is not None:
            self.metadata["cell_size_m"] = self.cell_size_m
        manifest = {
            "schema_version": self.metadata.get("schema_version", "1.0"),
            "project_file": str(self.project_file),
            "storage": _json_safe(self.metadata.get("storage", {})),
            "paths": self.paths(as_str=True),
            "artifacts": _json_safe(self.metadata.get("artifacts", {})),
            "lineage": _json_safe(self.metadata.get("lineage", {})),
        }
        return {
            "name": self.name,
            "output_dir": str(self.output_dir),
            "cell_size_m": self.cell_size_m,
            "geometry_format": self.geometry_format,
            "manifest": manifest,
            "metadata": _json_safe(self.metadata),
            "state": _json_safe(self.state),
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Save project metadata to JSON."""
        output = Path(path) if path is not None else self.project_file
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2))
        return output

    @classmethod
    def load(cls, path: str | Path) -> "SewerTrisProject":
        """Load a project from a JSON file or a project directory."""
        input_path = Path(path)
        project_file = input_path if input_path.suffix == ".json" else input_path / cls.project_filename
        data = json.loads(project_file.read_text())
        output_dir = Path(data.get("output_dir", project_file.parent))
        if not output_dir.is_absolute():
            if output_dir.name == project_file.parent.name:
                output_dir = project_file.parent
            else:
                output_dir = project_file.parent / output_dir
        metadata = data.get("metadata", {})
        geometry_format = (
            data.get("geometry_format")
            or metadata.get("storage", {}).get("vector_format")
            or Path(metadata.get("paths", {}).get("blocks_path", "city_blocks.gpkg")).suffix.lstrip(".")
            or "gpkg"
        )
        project = cls(
            output_dir=output_dir,
            cell_size_m=data.get("cell_size_m"),
            name=data.get("name", "SewerTris Project"),
            metadata=metadata,
            state=data.get("state", {}),
            autosave=False,
            geometry_format=geometry_format,
        )
        project.project_filename = project_file.name
        project._load_persisted_state()
        project.autosave = True
        return project

    def create_run(self, name: str, copy_base_inp: bool = True) -> SewerTrisScenario:
        """Create a named SWMM scenario folder."""
        scenario = SewerTrisScenario(
            project=self,
            name=name,
            output_dir=self.scenarios_dir / name,
        )
        self.metadata.setdefault("scenarios", {}).setdefault(
            name,
            {
                "created_at": _utc_now(),
                "output_dir": str(scenario.output_dir),
                "steps": {},
            },
        )
        if copy_base_inp and self.swmm_inp_path.exists():
            scenario.prepare_swmm_input()
        elif self.autosave:
            self.save()
        return scenario

    def load_run(self, name: str) -> SewerTrisScenario:
        """Return a scenario object for an existing scenario folder."""
        return SewerTrisScenario(
            project=self,
            name=name,
            output_dir=self.scenarios_dir / name,
        )

    def step_parameters(self, step: str | int, default: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return recorded parameters for a project step."""
        steps = self.metadata.get("steps", {})
        if isinstance(step, str) and step in steps:
            params = steps.get(step, {}).get("parameters")
            return copy.deepcopy(params if params is not None else (default or {}))
        step_name, _ = _normal_step(step)
        params = steps.get(step_name, {}).get("parameters")
        return copy.deepcopy(params if params is not None else (default or {}))

    def scenario_step_parameters(
        self,
        scenario_name: str,
        step: str,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return recorded parameters for a scenario step."""
        params = (
            self.metadata.get("scenarios", {})
            .get(scenario_name, {})
            .get("steps", {})
            .get(step, {})
            .get("parameters")
        )
        return copy.deepcopy(params if params is not None else (default or {}))

    def define_tetromino_set(self, name: str):
        """Define one of the named built-in tetromino sets."""
        from .layout import get_tetromino_set

        tetrominoes, colors = get_tetromino_set(name)
        self.define_tetrominoes(tetrominoes, colors)
        self.metadata.setdefault("sibling_parameters", {})
        self.metadata["sibling_parameters"]["tetromino_set"] = name
        if self.autosave:
            self.save()
        return tetrominoes, colors

    def _active_sibling_changes(
        self,
        extra_changes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Collect changes stored on a sibling plus optional runtime changes."""
        changes: dict[str, Any] = {}
        for source in (
            self.metadata.get("lineage", {}).get("changes"),
            self.metadata.get("changes"),
            self.metadata.get("sibling_parameters"),
            extra_changes,
        ):
            if isinstance(source, dict):
                changes = _deep_update(changes, source)
        return changes

    def _sibling_rerun_step(
        self,
        changes: dict[str, Any],
        rerun_from: str | int | None = None,
    ) -> tuple[str, int]:
        if rerun_from is not None:
            return _normal_step(rerun_from)
        lineage = self.metadata.get("lineage", {})
        if lineage.get("rerun_from") is not None:
            return _normal_step(lineage["rerun_from"])
        return self._infer_rerun_from_changes(changes)

    def _ensure_tetrominoes_for_replay(
        self,
        parent: "SewerTrisProject",
        changes: dict[str, Any],
    ) -> None:
        if changes.get("tetromino_set"):
            self.define_tetromino_set(changes["tetromino_set"])
            return
        if changes.get("tetrominoes") is not None:
            self.define_tetrominoes(
                changes["tetrominoes"],
                changes.get("tetromino_colors"),
            )
            return
        if "tetrominoes" in self.state:
            return

        parent_tetromino_set = (
            parent.metadata.get("sibling_parameters", {}).get("tetromino_set")
            or parent.metadata.get("changes", {}).get("tetromino_set")
        )
        if parent_tetromino_set:
            self.define_tetromino_set(parent_tetromino_set)
            return

        tetromino_keys = parent.step_parameters(
            "02_tetris_block_definition",
            default={},
        ).get("tetromino_keys")
        if tetromino_keys:
            key_set = set(tetromino_keys)
            if key_set == {"I", "O", "T", "S"}:
                self.define_tetromino_set("I_O_T_S_only")
                return
            if key_set == {"I", "O", "T", "S", "Z", "J", "L", "BO"}:
                self.define_tetromino_set("full")
                return

        raise ValueError(
            "Could not infer tetromino shapes for replay. Provide "
            "changes={'tetromino_set': ...} or define tetrominoes first."
        )

    def rerun_from_parent_parameters(
        self,
        parent_project: "SewerTrisProject | str | Path",
        *,
        tetromino_set: str | None = None,
        scenario_name: str = "bwf_gwi_rdii",
        run_flow_components: bool = True,
        changes: dict[str, Any] | None = None,
        rerun_from: str | int | None = None,
        stop_after_step: str | int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Rerun dependent workflow steps using parent parameters plus changes."""
        parent = (
            type(self).load(parent_project)
            if isinstance(parent_project, (str, Path))
            else parent_project
        )

        active_changes = self._active_sibling_changes(changes)
        if tetromino_set is not None:
            active_changes["tetromino_set"] = tetromino_set

        scenario_change = active_changes.get("scenario")
        if isinstance(scenario_change, dict):
            scenario_name = (
                scenario_change.get("name")
                or scenario_change.get("scenario_name")
                or scenario_name
            )
            if "run_flow_components" in scenario_change:
                run_flow_components = bool(scenario_change["run_flow_components"])
        elif isinstance(scenario_change, str):
            scenario_name = scenario_change
        if "scenario_name" in active_changes:
            scenario_name = active_changes["scenario_name"]
        if "run_flow_components" in active_changes:
            run_flow_components = bool(active_changes["run_flow_components"])

        rerun_step, rerun_order = self._sibling_rerun_step(
            active_changes,
            rerun_from=rerun_from,
        )
        stop_after_name, stop_after_order = (
            _normal_step(stop_after_step)
            if stop_after_step is not None
            else (None, None)
        )

        def report_progress(step: str, status: str = "running") -> None:
            if progress_callback is None:
                return
            step_name, step_order = _normal_step(step)
            progress_callback(
                {
                    "project_file": str(self.project_file),
                    "step": step_name,
                    "step_order": step_order,
                    "status": status,
                }
            )

        road_lines = road_buffer = crs = None
        blocks_gdf = None
        elevation = xx = yy = mask = None
        gdf_pipes = None
        predesign_pipes = None
        pipes_clean = manholes_clean = None
        scenario = None
        flow_components = None

        if rerun_order <= 1:
            report_progress("01_urban_domain_definition")
            domain_params = _merge_change_groups(
                parent.step_parameters("01_urban_domain_definition"),
                active_changes,
                groups=("domain",),
                keys=("shapefile_path", "domain_mask", "cell_size_m", "grid_meta"),
            )
            if domain_params.get("shapefile_path") is not None:
                shapefile_path = domain_params.pop("shapefile_path")
                cell_size_m = domain_params.pop("cell_size_m", self.cell_size_m)
                self.define_domain(
                    shapefile_path=shapefile_path,
                    cell_size_m=cell_size_m,
                    **domain_params,
                )
            elif domain_params.get("domain_mask") is not None:
                domain_mask = domain_params.pop("domain_mask")
                cell_size_m = domain_params.pop("cell_size_m", self.cell_size_m)
                self.define_domain(
                    domain_mask=domain_mask,
                    cell_size_m=cell_size_m,
                    **domain_params,
                )
            else:
                raise ValueError(
                    "Domain siblings must provide shapefile_path or domain_mask."
                )
        else:
            self._load_persisted_state()

        if rerun_order <= 3:
            report_progress("02_tetris_block_definition")
            self._ensure_tetrominoes_for_replay(parent, active_changes)

        if rerun_order <= 3:
            report_progress("03_stochastic_tetris_completion")
            layout_params = _merge_change_groups(
                parent.step_parameters("03_stochastic_tetris_completion"),
                active_changes,
                groups=("layout",),
                keys=(
                    "georeferenced",
                    "cell_size",
                    "crs",
                    "flip_y",
                    "layout_seed",
                    "random_seed",
                    "seed",
                ),
            )
            layout_seed = (
                layout_params.get("layout_seed")
                if layout_params.get("layout_seed") is not None
                else layout_params.get("seed", layout_params.get("random_seed"))
            )
            layout_kwargs = {
                "georeferenced": layout_params.get("georeferenced", True),
                "seed": layout_seed,
            }
            for key in ("cell_size", "crs", "flip_y"):
                if key in layout_params:
                    layout_kwargs[key] = layout_params[key]
            self.complete_tetris_layout(**layout_kwargs)

        road_params = _merge_change_groups(
            parent.step_parameters("04_road_network_extraction"),
            active_changes,
            groups=("roads",),
            keys=("road_width", "simplify_tol"),
        )
        road_width = road_params.get("road_width", self.state.get("road_width", 10))
        if rerun_order <= 4:
            report_progress("04_road_network_extraction")
            road_lines, road_buffer, crs = self.generate_roads(
                road_width=road_width,
                simplify_tol=road_params.get("simplify_tol", 0.5),
            )
            self.extract_road_boundaries(
                keep_holes=road_params.get("keep_holes", False),
            )

        if rerun_order <= 5:
            report_progress("05_land_use_assignment")
            land_use_params = _merge_change_groups(
                parent.step_parameters("05_land_use_assignment"),
                active_changes,
                groups=("land_use",),
                keys=("land_use_distribution", "seed"),
            )
            blocks_gdf = self.assign_land_use(
                land_use_distribution=land_use_params.get("land_use_distribution"),
                seed=land_use_params.get("seed", 42),
            )

        if rerun_order <= 6:
            report_progress("06_synthetic_dem_generation")
            topo_params = parent.step_parameters("06_synthetic_dem_generation")
            config = None
            config_dict = copy.deepcopy(topo_params.get("config", {}) or {})
            for key in ("config", "topography_config", "dem"):
                value = active_changes.get(key)
                if isinstance(value, dict):
                    config_dict = _deep_update(config_dict, value)
                elif value is not None and key in {"config", "topography_config"}:
                    config = value

            if config is None:
                from .topography import TopographyConfig

                config = TopographyConfig(**config_dict) if config_dict else TopographyConfig()
            elevation, xx, yy, mask = self.generate_topography(
                boundary_path=self.road_outer_shell_path,
                roads_path=self.road_polygons_path,
                config=config,
            )

        network_params = _merge_change_groups(
            parent.step_parameters("07_sewer_network_generation"),
            active_changes,
            groups=("sewer_network", "network"),
            keys=(
                "road_width",
                "block_size",
                "main_slope_tolerance",
                "secondary_slope_tolerance",
                "prefer_slope",
                "tertiary_block_size",
                "neighbor_radius_factor",
                "tertiary_min_pipe_length",
                "point_on_line_tol",
                "max_search_depth",
                "max_outer_iterations",
                "min_pipe_length",
                "network_method",
                "sewer_network_method",
                "use_sewer_network_v2",
                "tertiary_adverse_slope_weight",
                "tertiary_mild_adverse_slope",
                "tertiary_moderate_adverse_slope",
                "tertiary_severe_adverse_multiplier",
            ),
        )
        if "road_width" not in network_params:
            network_params["road_width"] = road_width
        if rerun_order <= 7:
            report_progress("07_sewer_network_generation")
            network_method = (
                network_params.get("network_method")
                or network_params.get("sewer_network_method")
                or ""
            )
            use_network_v2 = bool(network_params.get("use_sewer_network_v2", False)) or str(
                network_method
            ).lower() in {"v2", "2", "shortest_path", "shortest-path"}

            network_call_kwargs = {
                "road_width": network_params.get("road_width", road_width),
                "block_size": network_params.get("block_size", (self.cell_size_m or 100) * 2),
                "main_slope_tolerance": network_params.get("main_slope_tolerance", -0.01),
                "secondary_slope_tolerance": network_params.get("secondary_slope_tolerance", 0.0),
                "prefer_slope": network_params.get("prefer_slope", 0.5),
                "tertiary_block_size": network_params.get("tertiary_block_size", (self.cell_size_m or 100) * 10),
                "neighbor_radius_factor": network_params.get("neighbor_radius_factor", 1.5),
                "tertiary_min_pipe_length": network_params.get("tertiary_min_pipe_length", 1e-3),
                "point_on_line_tol": network_params.get("point_on_line_tol", 0.01),
                "min_pipe_length": network_params.get("min_pipe_length", 5.0),
            }

            if use_network_v2:
                gdf_pipes = self.generate_sewer_network_V2(
                    **network_call_kwargs,
                    max_outer_iterations=network_params.get("max_outer_iterations", 10000),
                    tertiary_adverse_slope_weight=network_params.get("tertiary_adverse_slope_weight", 200.0),
                    tertiary_mild_adverse_slope=network_params.get("tertiary_mild_adverse_slope", -0.005),
                    tertiary_moderate_adverse_slope=network_params.get("tertiary_moderate_adverse_slope", -0.01),
                    tertiary_severe_adverse_multiplier=network_params.get("tertiary_severe_adverse_multiplier", 8.0),
                )
            else:
                gdf_pipes = self.generate_sewer_network(
                    **network_call_kwargs,
                    max_search_depth=network_params.get("max_search_depth", 300),
                )

            embed_params = _merge_change_groups(
                parent.step_parameters("07_embed_sewer_network_in_dem"),
                active_changes,
                groups=("embed_dem", "dem_embedding"),
            )
            embed_params = {
                key: value
                for key, value in embed_params.items()
                if key not in {"dem_path", "pipes_path", "manholes_path", "output_path"}
            }
            embed_params.setdefault("upstream_field", "upstream_m")
            embed_params.setdefault("downstream_field", "downstream_m")
            embed_params.setdefault("manhole_id_field", "id")
            embed_params.setdefault("manhole_elev_field", "elevation")
            embed_params.setdefault("type_field", "type")
            embed_params.setdefault("tier_order", ("main", "secondary", "tertiary"))
            embed_params.setdefault("Smin", 0.001)
            self.embed_sewer_network_in_dem(**embed_params)

        if rerun_order <= 8:
            report_progress("08_sewer_flow_predesign")
            predesign_params = _merge_change_groups(
                parent.step_parameters("08_sewer_flow_predesign"),
                active_changes,
                groups=("predesign",),
                keys=(
                    "land_use_info",
                    "gwi_factor_ls_per_m",
                    "rdii_factor_ls_per_m2",
                    "target_crs_m",
                ),
            )
            predesign_pipes = self.predesign_flows(
                land_use_info=predesign_params.get("land_use_info", _DEFAULT_LAND_USE_INFO),
                gwi_factor_ls_per_m=predesign_params.get("gwi_factor_ls_per_m", 0.0002),
                rdii_factor_ls_per_m2=predesign_params.get("rdii_factor_ls_per_m2", 0.00002),
                target_crs_m=predesign_params.get("target_crs_m", "EPSG:3857"),
            )

        if rerun_order <= 9:
            report_progress("09_pipe_sizing_and_hydraulic_properties")
            design_params = _merge_change_groups(
                parent.step_parameters("09_pipe_sizing_and_hydraulic_properties"),
                active_changes,
                groups=("pipe_design",),
                keys=(
                    "minimum_slope",
                    "material_fractions",
                    "n_by_material",
                    "standard_diameters_mm",
                    "minimum_diameter_mm",
                    "min_cover",
                    "min_slope",
                    "manhole_drop",
                ),
            )
            materials_change = active_changes.get("materials")
            if isinstance(materials_change, dict):
                if any(
                    key in materials_change
                    for key in ("material_fractions", "n_by_material")
                ):
                    design_params = _deep_update(design_params, materials_change)
                else:
                    design_params["material_fractions"] = materials_change
            diameters_change = active_changes.get("diameters")
            if isinstance(diameters_change, dict):
                design_params = _deep_update(design_params, diameters_change)
            elif diameters_change is not None:
                design_params["standard_diameters_mm"] = diameters_change

            pipes_clean, manholes_clean = self.design_pipes(
                minimum_slope=design_params.get("minimum_slope", 0.005),
                material_fractions=design_params.get("material_fractions"),
                n_by_material=design_params.get("n_by_material"),
                standard_diameters_mm=design_params.get("standard_diameters_mm", _DEFAULT_STANDARD_DIAMETERS_MM),
                minimum_diameter_mm=design_params.get("minimum_diameter_mm", 200),
                min_cover=design_params.get("min_cover", 1.4),
                min_slope=design_params.get("min_slope", 0.005),
                manhole_drop=design_params.get("manhole_drop", 0.05),
            )

        if rerun_order <= 10 and (
            stop_after_order is None or stop_after_order >= 10
        ):
            report_progress("10_dynamic_flow_input_definition_base_model")
            swmm_params = _merge_change_groups(
                parent.step_parameters("10_dynamic_flow_input_definition_base_model"),
                active_changes,
                groups=("swmm",),
            )
            options_dict = copy.deepcopy(swmm_params.get("options_dict", {}) or {})
            for key in ("swmm_options", "options_dict"):
                value = active_changes.get(key)
                if isinstance(value, dict):
                    options_dict = _deep_update(options_dict, value)
            swmm_kwargs = {
                key: value
                for key, value in swmm_params.items()
                if key != "options_dict"
            }
            self.export_swmm(options_dict=options_dict, **swmm_kwargs)
            scenario = self.create_run(scenario_name)

            dwf_params = parent.scenario_step_parameters(
                scenario_name,
                "assign_dwf_patterns",
            )
            for key in ("dwf", "bwf"):
                value = active_changes.get(key)
                if isinstance(value, dict):
                    dwf_params = _deep_update(dwf_params, value)
                elif value is not None:
                    dwf_params["hourly_values"] = value
            for key in (
                "hourly_id",
                "hourly_values",
                "daily_id",
                "daily_values",
                "monthly_id",
                "monthly_values",
                "weekend_id",
                "weekend_values",
            ):
                if key in active_changes:
                    dwf_params[key] = active_changes[key]
            if dwf_params:
                scenario.assign_dwf_patterns(**dwf_params)

            gwi_params = parent.scenario_step_parameters(
                scenario_name,
                "assign_gwi_from_pipe_length",
            )
            gwi_change = active_changes.get("gwi")
            if isinstance(gwi_change, dict):
                gwi_params = _deep_update(gwi_params, gwi_change)
            elif gwi_change is not None:
                gwi_params["coefficient"] = gwi_change
            if "gwi_coefficient" in active_changes:
                gwi_params["coefficient"] = active_changes["gwi_coefficient"]

            gwi_raster_path = None
            gwi_samples_per_pipe = 5
            gwi_raster_change = (
                active_changes.get("gwi_raster")
                or active_changes.get("gwi_inflow_raster")
            )
            if isinstance(gwi_raster_change, dict):
                gwi_raster_path = (
                    gwi_raster_change.get("raster_path")
                    or gwi_raster_change.get("path")
                )
                gwi_samples_per_pipe = gwi_raster_change.get("samples_per_pipe", 5)
                if gwi_raster_path is None:
                    gwi_raster_path = self.generate_gwi_inflow_raster(
                        min_value=gwi_raster_change.get("min_value", 0.001),
                        max_value=gwi_raster_change.get("max_value", 0.010),
                        random_seed=gwi_raster_change.get("random_seed"),
                        n_hills=gwi_raster_change.get("n_hills", 3),
                        hill_min_value=gwi_raster_change.get("hill_min_value", 0.010),
                        hill_max_value=gwi_raster_change.get("hill_max_value", 0.050),
                        hill_radius_min=gwi_raster_change.get("hill_radius_min", 20),
                        hill_radius_max=gwi_raster_change.get("hill_radius_max", 80),
                        clip_to_range=gwi_raster_change.get("clip_to_range", True),
                    )
                else:
                    gwi_raster_path = _resolve_project_path(self, gwi_raster_path)
            elif gwi_raster_change is not None:
                gwi_raster_path = _resolve_project_path(self, gwi_raster_change)

            if gwi_raster_path is not None:
                scenario.assign_gwi_from_raster(
                    raster_path=str(gwi_raster_path),
                    samples_per_pipe=gwi_samples_per_pipe,
                )
            else:
                scenario.assign_gwi_from_pipe_length(
                    coefficient=gwi_params.get("coefficient", 0.00001),
                )

            rdii_params = parent.scenario_step_parameters(
                scenario_name,
                "add_subcatchment_rdii",
            )
            rdii_change = active_changes.get("rdii")
            if isinstance(rdii_change, dict):
                rdii_params = _deep_update(rdii_params, rdii_change)
            elif rdii_change is not None:
                rdii_params["timeseries"] = rdii_change
            rainfall_change = active_changes.get("rainfall")
            if isinstance(rainfall_change, dict):
                rdii_params = _deep_update(rdii_params, rainfall_change)
            elif rainfall_change is not None:
                rdii_params["timeseries"] = rainfall_change

            rainfall_generator_keys = {
                "start_date",
                "end_date",
                "timestep_minutes",
                "avg_annual_precip_mm",
                "wet_season_months",
                "dry_wet_ratio",
                "storm_prob",
                "storm_duration_range",
                "random_seed",
                "preview_date",
                "regenerate",
            }
            rainfall_generator_params = {}
            for key in tuple(rdii_params):
                if key in rainfall_generator_keys:
                    rainfall_generator_params[key] = rdii_params.pop(key)
            regenerate_rainfall = bool(rainfall_generator_params.pop("regenerate", False))

            rdii_raster_path = None
            rdii_raster_change = (
                active_changes.get("rdii_raster")
                or active_changes.get("rdii_density_raster")
            )
            if isinstance(rdii_raster_change, dict):
                rdii_raster_path = (
                    rdii_raster_change.get("raster_path")
                    or rdii_raster_change.get("path")
                )
                if rdii_raster_path is None:
                    min_density = rdii_raster_change.get("min_density", 0.1)
                    max_density = rdii_raster_change.get("max_density", 3.0)
                    n_hills = rdii_raster_change.get("n_hills", 3)
                    hill_max_density = rdii_raster_change.get("hill_max_density", 10.0)
                    clip_to_range = rdii_raster_change.get("clip_to_range", False)
                    rdii_raster_path = self.generate_rdii_density_raster(
                        min_density=min_density,
                        max_density=max_density,
                        random_seed=rdii_raster_change.get("random_seed"),
                        n_hills=n_hills,
                        hill_min_density=rdii_raster_change.get("hill_min_density", 2.0),
                        hill_max_density=hill_max_density,
                        hill_radius_min=rdii_raster_change.get("hill_radius_min", 20),
                        hill_radius_max=rdii_raster_change.get("hill_radius_max", 80),
                        clip_to_range=clip_to_range,
                    )
                    if "rdii_to_imperv_scale" not in rdii_raster_change:
                        upper_density = max_density
                        if n_hills and not clip_to_range:
                            upper_density = max(upper_density, hill_max_density)
                        if upper_density == min_density:
                            upper_density = min_density + 1.0
                        rdii_params["rdii_to_imperv_scale"] = [
                            min_density,
                            upper_density,
                        ]
                else:
                    rdii_raster_path = _resolve_project_path(self, rdii_raster_path)
                if "rdii_to_imperv_scale" in rdii_raster_change:
                    rdii_params["rdii_to_imperv_scale"] = rdii_raster_change[
                        "rdii_to_imperv_scale"
                    ]
            elif rdii_raster_change is not None:
                rdii_raster_path = _resolve_project_path(self, rdii_raster_change)

            rdii_params.pop("subcatchments_path", None)
            rdii_timeseries = rdii_params.pop("timeseries", None)
            if regenerate_rainfall:
                rdii_timeseries = None
            rdii_common_defaults = {
                "raingage_id": "1",
                "raingage_coords": (500, 500),
                "n_imperv": 0.011,
                "n_perv": 0.15,
                "s_imperv": 0.0,
                "s_perv": 0.0,
                "pct_zero": 0,
                "route_to": "OUTLET",
                "pct_routed": "",
                "infiltration_params": (90, 0.5, 7, "", ""),
                "width": 100,
                "slope": 0.005,
                "curblen": 0,
            }
            if rdii_raster_path is not None:
                rdii_defaults = {
                    **rdii_common_defaults,
                    "rdii_to_imperv_scale": (0.0, 3.0),
                }
                rdii_params.pop("imperv_pct", None)
            else:
                rdii_defaults = {**rdii_common_defaults, "imperv_pct": 5}
            rdii_params = {**rdii_defaults, **rdii_params}
            if not _looks_like_timeseries(rdii_timeseries):
                from .hydrology import generate_clustered_rainfall_timeseries

                start_date = rainfall_generator_params.pop(
                    "start_date",
                    f"{options_dict.get('START_DATE', '01/01/1990')} {options_dict.get('START_TIME', '00:00:00')}",
                )
                end_date = rainfall_generator_params.pop(
                    "end_date",
                    f"{options_dict.get('END_DATE', '01/10/1990')} {options_dict.get('END_TIME', '00:00:00')}",
                )
                rdii_timeseries = generate_clustered_rainfall_timeseries(
                    start_date=start_date,
                    end_date=end_date,
                    timestep_minutes=rainfall_generator_params.pop("timestep_minutes", 15),
                    avg_annual_precip_mm=rainfall_generator_params.pop("avg_annual_precip_mm", 1200),
                    wet_season_months=rainfall_generator_params.pop(
                        "wet_season_months",
                        [4, 5, 6, 9, 10, 11],
                    ),
                    dry_wet_ratio=rainfall_generator_params.pop("dry_wet_ratio", 0.2),
                    storm_prob=rainfall_generator_params.pop("storm_prob", 0.1),
                    storm_duration_range=rainfall_generator_params.pop(
                        "storm_duration_range",
                        (1, 6),
                    ),
                    random_seed=rainfall_generator_params.pop("random_seed", 42),
                    preview_date=rainfall_generator_params.pop(
                        "preview_date",
                        options_dict.get("START_DATE", "01/01/1990"),
                    ),
                )
            rdii_params["timeseries"] = rdii_timeseries
            if rdii_raster_path is not None:
                scenario.add_subcatchment_rdii_raster(
                    rdii_raster_path=str(rdii_raster_path),
                    **rdii_params,
                )
            else:
                scenario.add_subcatchment_rdii(**rdii_params)

            if run_flow_components:
                report_progress("11_epa_swmm_simulation")
                scenario.add_pollutant_tags()
                flow_components = scenario.get_flow_components(link_id="P_OUTLET")
                report_progress("12_flow_output_decomposition")
                scenario.save_flow_components(flow_components)

        self.record_step(
            "99_rerun_from_parent_parameters",
            parameters={
                "parent_project_file": parent.project_file,
                "changes": active_changes,
                "rerun_from": rerun_step,
                "stop_after_step": stop_after_name,
                "scenario_name": scenario_name,
                "run_flow_components": run_flow_components,
            },
            outputs={
                "layout_blocks_path": self.layout_blocks_path,
                "pipes_path": self.pipes_path,
                "swmm_inp_path": self.swmm_inp_path,
                "scenario_input": scenario.swmm_inp_path if scenario else None,
                "flows_path": scenario.flows_path if scenario else None,
            },
        )
        self.save()
        if stop_after_order is not None:
            report_progress(stop_after_name, status="completed")
        else:
            report_progress("12_flow_output_decomposition", status="completed")

        return {
            "road_lines": road_lines,
            "road_buffer": road_buffer,
            "crs": crs,
            "blocks_gdf": blocks_gdf,
            "elevation": elevation,
            "xx": xx,
            "yy": yy,
            "mask": mask,
            "gdf_pipes": gdf_pipes,
            "predesign_pipes": predesign_pipes,
            "pipes_clean": pipes_clean,
            "manholes_clean": manholes_clean,
            "scenario": scenario,
            "flow_components": flow_components,
        }

    def _infer_rerun_from_changes(self, changes: dict[str, Any] | None) -> tuple[str, int]:
        if not changes:
            return _normal_step("10_dynamic_flow_input_definition_base_model")
        affected_steps = []
        for key in changes:
            mapped = _CHANGE_STEP_MAP.get(str(key))
            if mapped is not None:
                affected_steps.append(_normal_step(mapped))
        if not affected_steps:
            return _normal_step("10_dynamic_flow_input_definition_base_model")
        return min(affected_steps, key=lambda item: item[1])

    def _sibling_destination_for_artifact(
        self,
        role: str,
        source_path: Path,
        sibling: "SewerTrisProject",
    ) -> Path | None:
        destinations = {
            "domain_mask": sibling.domain_mask_path,
            "grid_meta": sibling.grid_meta_path,
            "layout_blocks": sibling.layout_blocks_path,
            "city_blocks": sibling.blocks_path,
            "road_centerlines": sibling.road_centerlines_path,
            "road_polygons": sibling.road_polygons_path,
            "road_boundary_lines": sibling.road_boundary_lines_path,
            "road_outer_shell": sibling.road_outer_shell_path,
            "dem": sibling.dem_path,
            "gwi_raster": sibling.gwi_raster_path,
            "rdii_raster": sibling.rdii_raster_path,
            "manholes": sibling.manholes_path,
            "pipes": sibling.pipes_path,
            "subcatchments": sibling.subcatchments_path,
            "swmm_input": sibling.swmm_inp_path,
            "flow_components": sibling.flows_path,
        }
        if role == "input_domain":
            return sibling.inputs_dir / source_path.name
        return destinations.get(role)

    def clone_sibling(
        self,
        output_dir: str | Path,
        changes: dict[str, Any] | None = None,
        name: str | None = None,
        rerun_from: str | int | None = None,
        copy_artifacts: bool = True,
        geometry_format: str | None = None,
    ) -> "SewerTrisProject":
        """Create a reproducible project sibling with dependency-aware reuse.

        The clone copies only artifacts that remain valid before the earliest
        changed step. For example, changing ``tetromino_set`` keeps the domain
        mask/grid metadata but reruns layout, roads, DEM, sewer network, and
        downstream SWMM outputs.
        """
        changes = changes or {}
        rerun_step, rerun_order = (
            _normal_step(rerun_from)
            if rerun_from is not None
            else self._infer_rerun_from_changes(changes)
        )
        sibling = type(self)(
            output_dir=output_dir,
            cell_size_m=self.cell_size_m,
            name=name or f"{self.name} sibling",
            autosave=False,
            geometry_format=geometry_format or self.geometry_format,
        )

        parent_steps = self.metadata.get("steps", {})
        reused_steps = {
            step: copy.deepcopy(details)
            for step, details in parent_steps.items()
            if _STEP_ORDER.get(step, 99) < rerun_order
        }
        sibling.metadata["lineage"] = {
            "type": "sibling",
            "parent_project_file": str(self.project_file),
            "parent_output_dir": str(self.output_dir),
            "created_at": _utc_now(),
            "changes": _json_safe(changes),
            "rerun_from": rerun_step,
            "rerun_from_order": rerun_order,
        }
        sibling.metadata["changes"] = _json_safe(changes)
        sibling.metadata["reused_parent_steps"] = _json_safe(reused_steps)
        sibling.metadata.setdefault("inputs", {})

        copied_artifacts: dict[str, str] = {}
        if copy_artifacts:
            artifact_steps = {
                "input_domain": 1,
                "domain_mask": 1,
                "grid_meta": 1,
                "layout_blocks": 3,
                "road_centerlines": 4,
                "road_polygons": 4,
                "road_boundary_lines": 4,
                "road_outer_shell": 4,
                "city_blocks": 5,
                "dem": 6,
                "manholes": 7,
                "subcatchments": 8,
                "pipes": 9,
                "gwi_raster": 10,
                "rdii_raster": 10,
                "swmm_input": 10,
                "flow_components": 12,
            }
            for role, artifact in self.artifacts().items():
                if role.startswith("scenario:"):
                    continue
                if artifact_steps.get(role, 99) >= rerun_order:
                    continue
                source = Path(artifact["path"])
                if not source.exists() or not source.is_file():
                    continue
                destination = self._sibling_destination_for_artifact(role, source, sibling)
                if destination is None:
                    continue
                copied_path = sibling._copy_dataset_file(source, destination)
                copied_artifacts[role] = str(copied_path)
                if role == "input_domain":
                    sibling.metadata["inputs"]["domain_path"] = str(copied_path)
                    sibling.metadata["inputs"]["domain_source_path"] = str(source)

        if "domain_mask" in self.state and rerun_order > 1:
            sibling.state["domain_mask"] = copy.deepcopy(self.state["domain_mask"])
        if "grid_meta" in self.state and rerun_order > 1:
            sibling.state["grid_meta"] = copy.deepcopy(self.state["grid_meta"])
        if "tetrominoes" in changes:
            sibling.state["tetrominoes"] = changes["tetrominoes"]
        if "tetromino_colors" in changes:
            sibling.state["tetromino_colors"] = changes["tetromino_colors"]

        sibling._load_persisted_state()
        sibling.metadata["lineage"]["copied_artifacts"] = copied_artifacts
        sibling.record_step(
            "00_clone_sibling",
            parameters={
                "parent_project_file": self.project_file,
                "changes": changes,
                "rerun_from": rerun_step,
                "copy_artifacts": copy_artifacts,
            },
            outputs={"copied_artifacts": copied_artifacts},
        )
        sibling.autosave = True
        sibling.save()
        return sibling

    # ------------------------------------------------------------------
    # Step 1: domain
    # ------------------------------------------------------------------
    def define_domain(
        self,
        domain_mask=None,
        shapefile_path: str | Path | None = None,
        cell_size_m: float | None = None,
        **kwargs: Any,
    ):
        """Define the project domain from a mask or shapefile."""
        resolved_cell_size = cell_size_m if cell_size_m is not None else self.cell_size_m
        if shapefile_path is not None:
            if resolved_cell_size is None:
                raise ValueError("Provide cell_size_m or set project.cell_size_m.")
            from .domain import build_domain_mask_from_shapefile

            copied_input_path = self._copy_input_dataset(shapefile_path)
            domain_mask, grid_meta = build_domain_mask_from_shapefile(
                shapefile_path,
                cell_size_m=resolved_cell_size,
                **kwargs,
            )
            self.metadata.setdefault("inputs", {})
            self.metadata["inputs"]["domain_source_path"] = str(shapefile_path)
            self.metadata["inputs"]["domain_path"] = str(copied_input_path)
        else:
            if domain_mask is None:
                raise ValueError("Provide domain_mask or shapefile_path.")
            grid_meta = kwargs.get("grid_meta")

        self.cell_size_m = resolved_cell_size
        self.remember(domain_mask=domain_mask, grid_meta=grid_meta)
        self._persist_domain_state(domain_mask=domain_mask, grid_meta=grid_meta)
        self.record_step(
            "01_urban_domain_definition",
            parameters={
                "shapefile_path": shapefile_path,
                "cell_size_m": resolved_cell_size,
                **kwargs,
            },
            outputs={
                "domain_mask_path": self.domain_mask_path,
                "grid_meta_path": self.grid_meta_path,
                "input_domain_path": self.metadata.get("inputs", {}).get("domain_path"),
            },
        )
        return domain_mask, grid_meta

    def build_domain_mask_from_shapefile(
        self,
        shapefile_path: str | Path,
        cell_size_m: float | None = None,
        **kwargs: Any,
    ):
        return self.define_domain(
            shapefile_path=shapefile_path,
            cell_size_m=cell_size_m,
            **kwargs,
        )

    build_domain = define_domain

    # ------------------------------------------------------------------
    # Steps 2-3: tetrominoes and layout
    # ------------------------------------------------------------------
    def define_tetrominoes(self, tetrominoes, tetromino_colors=None):
        self.remember(tetrominoes=tetrominoes, tetromino_colors=tetromino_colors)
        self.record_step(
            "02_tetris_block_definition",
            parameters={"tetromino_keys": list(tetrominoes.keys())},
        )
        return tetrominoes

    def fill_domain_with_tetrominoes_and_blocks(self, domain_mask=None, tetrominoes=None):
        """Fill a mask with tetrominoes and store the resulting board."""
        from .layout import fill_domain_with_tetrominoes_and_blocks

        domain_mask = domain_mask if domain_mask is not None else self.state.get("domain_mask")
        tetrominoes = tetrominoes if tetrominoes is not None else self.state.get("tetrominoes")
        if domain_mask is None or tetrominoes is None:
            raise ValueError("Define domain_mask and tetrominoes first.")
        filled_board, id_type_map, block_id = fill_domain_with_tetrominoes_and_blocks(
            domain_mask,
            tetrominoes,
        )
        self.remember(
            filled_board=filled_board,
            id_type_map=id_type_map,
            block_id=block_id,
        )
        return filled_board, id_type_map, block_id

    def complete_tetris_layout(
        self,
        domain_mask=None,
        tetrominoes=None,
        output_path: str | Path | None = None,
        cell_size: float | None = None,
        id_to_type_map=None,
        crs: str = "EPSG:3857",
        flip_y: bool = True,
        grid_meta=None,
        georeferenced: bool | None = None,
        seed: int | None = None,
    ):
        """Populate the domain and export the generated city layout."""
        from .layout import (
            export_individual_figures_to_shapefile,
            export_individual_figures_to_shapefile_georeferenced,
        )

        if seed is not None:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)

        filled_board, id_type_map, block_id = self.fill_domain_with_tetrominoes_and_blocks(
            domain_mask=domain_mask,
            tetrominoes=tetrominoes,
        )
        output = Path(output_path) if output_path is not None else self.layout_blocks_path
        resolved_grid_meta = grid_meta if grid_meta is not None else self.state.get("grid_meta")
        use_georeferenced = georeferenced
        if use_georeferenced is None:
            use_georeferenced = resolved_grid_meta is not None

        if use_georeferenced:
            if resolved_grid_meta is None:
                raise ValueError("Provide grid_meta or define a shapefile domain first.")
            export_individual_figures_to_shapefile_georeferenced(
                filled_board,
                output,
                resolved_grid_meta,
                id_to_type_map=id_to_type_map or id_type_map,
            )
            parameters = {"georeferenced": True, "grid_meta": resolved_grid_meta}
        else:
            resolved_cell_size = cell_size if cell_size is not None else self.cell_size_m
            if resolved_cell_size is None:
                raise ValueError("Provide cell_size or set project.cell_size_m.")
            export_individual_figures_to_shapefile(
                filled_board=filled_board,
                cell_size=resolved_cell_size,
                output_path=output,
                id_to_type_map=id_to_type_map or id_type_map,
                crs=crs,
                flip_y=flip_y,
            )
            parameters = {
                "cell_size": resolved_cell_size,
                "crs": crs,
                "flip_y": flip_y,
                "georeferenced": False,
            }
        if seed is not None:
            parameters["seed"] = seed
        self.record_step(
            "03_stochastic_tetris_completion",
            parameters=parameters,
            outputs={"layout_blocks_path": output, "block_id": block_id},
        )
        return filled_board, id_type_map, block_id

    fill_layout = complete_tetris_layout

    # ------------------------------------------------------------------
    # Steps 4-6: roads, land use, DEM
    # ------------------------------------------------------------------
    def generate_roads(
        self,
        blocks_path: str | Path | None = None,
        road_width: float = 10,
        simplify_tol: float = 0.5,
    ):
        """Generate and save road centerlines and polygons."""
        from .roads import generate_road_network_from_blocks

        blocks = Path(blocks_path) if blocks_path is not None else self.layout_blocks_path
        road_lines, road_buffer, crs = generate_road_network_from_blocks(
            blocks_path=blocks,
            road_width=road_width,
            simplify_tol=simplify_tol,
        )

        import geopandas as gpd

        gpd.GeoDataFrame(geometry=[road_lines], crs=crs).to_file(self.road_centerlines_path)
        gpd.GeoDataFrame(geometry=[road_buffer], crs=crs).to_file(self.road_polygons_path)
        self.remember(
            road_lines=road_lines,
            road_buffer=road_buffer,
            road_crs=crs,
            road_width=road_width,
        )
        self.record_step(
            "04_road_network_extraction",
            parameters={
                "blocks_path": blocks,
                "road_width": road_width,
                "simplify_tol": simplify_tol,
            },
            outputs={
                "road_centerlines_path": self.road_centerlines_path,
                "road_polygons_path": self.road_polygons_path,
            },
        )
        return road_lines, road_buffer, crs

    generate_road_network_from_blocks = generate_roads

    def extract_road_boundaries(
        self,
        roads_path: str | Path | None = None,
        keep_holes: bool = False,
    ) -> tuple[Path, Path]:
        from .roads import extract_boundary

        roads = Path(roads_path) if roads_path is not None else self.road_polygons_path
        extract_boundary(
            roads,
            out_boundary_lines=self.road_boundary_lines_path,
            out_outer_shell_polygon=self.road_outer_shell_path,
            keep_holes=keep_holes,
        )
        return self.road_boundary_lines_path, self.road_outer_shell_path

    def assign_land_use(
        self,
        blocks_path: str | Path | None = None,
        roads_path: str | Path | None = None,
        output_path: str | Path | None = None,
        land_use_distribution: dict[str, float] | None = None,
        seed: int = 42,
    ):
        """Cut blocks by roads, assign land use, and save the resulting blocks."""
        from .roads import (
            assign_land_use_compact,
            cut_blocks,
            export_to_shapefile,
            load_blocks_and_roads,
        )

        blocks_source = Path(blocks_path) if blocks_path is not None else self.layout_blocks_path
        roads_source = Path(roads_path) if roads_path is not None else self.road_polygons_path
        output = Path(output_path) if output_path is not None else self.blocks_path
        blocks, road_network, crs = load_blocks_and_roads(blocks_source, roads_source)
        blocks = cut_blocks(blocks, road_network)
        blocks = assign_land_use_compact(
            blocks,
            land_use_distribution=land_use_distribution,
            seed=seed,
        )
        gdf = export_to_shapefile(blocks, crs, output)
        self.record_step(
            "05_land_use_assignment",
            parameters={
                "blocks_path": blocks_source,
                "roads_path": roads_source,
                "land_use_distribution": land_use_distribution,
                "seed": seed,
            },
            outputs={"blocks_path": output},
        )
        return gdf

    def generate_topography(
        self,
        boundary_path: str | Path | None = None,
        roads_path: str | Path | None = None,
        config=None,
        output_path: str | Path | None = None,
    ):
        """Generate topography and save it as a GeoTIFF."""
        from .topography import TopographyConfig, generate_topography

        if boundary_path is None or roads_path is None:
            self.extract_road_boundaries()
        boundary = Path(boundary_path) if boundary_path is not None else self.road_outer_shell_path
        roads = Path(roads_path) if roads_path is not None else self.road_polygons_path
        output = Path(output_path) if output_path is not None else self.dem_path
        config = config if config is not None else TopographyConfig()

        elevation, xx, yy, mask = generate_topography(boundary, roads, config)

        import geopandas as gpd
        import rasterio

        boundary_gdf = gpd.read_file(boundary)
        input_crs = boundary_gdf.crs or "EPSG:32614"
        transform = rasterio.transform.from_bounds(
            west=xx[0, 0],
            south=yy[-1, 0],
            east=xx[0, -1],
            north=yy[0, 0],
            width=elevation.shape[1],
            height=elevation.shape[0],
        )
        with rasterio.open(
            output,
            "w",
            driver="GTiff",
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype=elevation.dtype,
            crs=input_crs,
            transform=transform,
            nodata=float("nan"),
        ) as dst:
            dst.write(elevation, 1)

        self.remember(elevation=elevation, xx=xx, yy=yy, mask=mask, topography_config=config)
        self.record_step(
            "06_synthetic_dem_generation",
            parameters={
                "boundary_path": boundary,
                "roads_path": roads,
                "config": vars(config) if hasattr(config, "__dict__") else config,
            },
            outputs={"dem_path": output},
        )
        return elevation, xx, yy, mask

    # ------------------------------------------------------------------
    # Step 7: sewer network
    # ------------------------------------------------------------------
    def extract_manholes_from_lines(
        self,
        road_axes_path: str | Path | None = None,
        dem_path: str | Path | None = None,
    ):
        from .sewer_network import extract_manholes_from_lines

        roads = Path(road_axes_path) if road_axes_path is not None else self.road_centerlines_path
        dem = Path(dem_path) if dem_path is not None else self.dem_path
        manholes = extract_manholes_from_lines(roads, dem)
        self.remember(manholes=manholes)
        return manholes

    extract_manholes = extract_manholes_from_lines

    def generate_sewer_network(
        self,
        road_axes_path: str | Path | None = None,
        dem_path: str | Path | None = None,
        road_width: float | None = None,
        block_size: float | None = None,
        crs=None,
        **kwargs: Any,
    ):
        """Generate main, secondary, and tertiary pipes and save shapefiles."""
        from .sewer_network import (
            build_current_network_status,
            build_main_attrs_from_path_info,
            export_manholes_to_shapefile,
            export_pipes_to_shapefile_2,
            generate_main_sewer_path_optimized,
            generate_secondary_pipes_optimized,
            generate_tertiary_pipes_backtracking_stop_at_each_manhole,
            remove_secondary_pipes_overlapping_main_optimized,
        )

        import geopandas as gpd

        road_axes = Path(road_axes_path) if road_axes_path is not None else self.road_centerlines_path
        dem = Path(dem_path) if dem_path is not None else self.dem_path
        road_width = road_width if road_width is not None else self.state.get("road_width", 10)
        block_size = block_size if block_size is not None else (self.cell_size_m or 100) * 2
        manholes = self.extract_manholes_from_lines(road_axes, dem)
        crs = crs or gpd.read_file(road_axes).crs
        export_manholes_to_shapefile(manholes, self.manholes_path, crs=crs)

        road_lines = self.state.get("road_lines")
        if road_lines is None:
            road_lines = gpd.read_file(road_axes).geometry.unary_union
        road_buffer = road_lines.buffer(road_width * 0.6)

        segments, path_info, graph_data = generate_main_sewer_path_optimized(
            manholes=manholes,
            road_buffer=road_buffer,
            block_size=block_size,
            slope_tolerance=kwargs.get("main_slope_tolerance", -0.01),
            min_pipe_length=kwargs.get("min_pipe_length", 5.0),
            prefer_slope=kwargs.get("prefer_slope", 0.5),
            return_graph_data=True,
        )
        main_path = path_info["segments"]

        secondary_pipes, secondary_attrs = generate_secondary_pipes_optimized(
            manholes=manholes,
            main_path=main_path,
            road_buffer=road_buffer,
            block_size=block_size,
            slope_tolerance=kwargs.get("secondary_slope_tolerance", 0.0),
            prefer_slope=kwargs.get("prefer_slope", 0.5),
            return_attrs=True,
        )
        secondary_pipes_clean = remove_secondary_pipes_overlapping_main_optimized(
            manholes=manholes,
            secondary_pipes=secondary_pipes,
            main_pipes=main_path,
        )
        network_status = build_current_network_status(
            manholes=manholes,
            main_path=main_path,
            secondary_pipes=secondary_pipes_clean,
        )
        tertiary_pipes, tertiary_unconnected, tertiary_attrs = (
            generate_tertiary_pipes_backtracking_stop_at_each_manhole(
                manholes=manholes,
                main_path=main_path,
                secondary_pipes=secondary_pipes_clean,
                road_buffer=road_buffer,
                city_boundary=self.road_outer_shell_path,
                block_size=kwargs.get("tertiary_block_size", (self.cell_size_m or 100) * 10),
                neighbor_radius_factor=kwargs.get("neighbor_radius_factor", 1.5),
                min_pipe_length=kwargs.get("tertiary_min_pipe_length", 1e-3),
                point_on_line_tol=kwargs.get("point_on_line_tol", 0.01),
                return_attrs=True,
                max_search_depth=kwargs.get("max_search_depth", 300),
            )
        )
        main_attrs = build_main_attrs_from_path_info(path_info)
        gdf_pipes = export_pipes_to_shapefile_2(
            pipes_main=main_path,
            pipes_sec=secondary_pipes_clean,
            pipes_ter=tertiary_pipes,
            manholes=manholes,
            output_path=self.pipes_path,
            crs=crs,
            main_attrs=main_attrs,
            secondary_attrs=secondary_attrs,
            tertiary_attrs=tertiary_attrs,
        )
        self._ensure_pipe_downstream_alias()
        self.remember(
            manholes=manholes,
            road_buffer=road_buffer,
            main_path=main_path,
            secondary_pipes=secondary_pipes_clean,
            tertiary_pipes=tertiary_pipes,
            tertiary_unconnected=tertiary_unconnected,
            network_status=network_status,
        )
        self.record_step(
            "07_sewer_network_generation",
            parameters={"road_width": road_width, "block_size": block_size, **kwargs},
            outputs={
                "manholes_path": self.manholes_path,
                "pipes_path": self.pipes_path,
            },
        )
        return gdf_pipes

    def generate_sewer_network_V2(
        self,
        road_axes_path: str | Path | None = None,
        dem_path: str | Path | None = None,
        road_width: float | None = None,
        block_size: float | None = None,
        crs=None,
        **kwargs: Any,
    ):
        """Generate sewer pipes using shortest-path tertiary routing."""
        from .sewer_network import (
            build_current_network_status,
            build_main_attrs_from_path_info,
            export_manholes_to_shapefile,
            export_pipes_to_shapefile_2,
            generate_main_sewer_path_optimized,
            generate_secondary_pipes_optimized,
            generate_tertiary_pipes_shortest_path_v2,
            remove_secondary_pipes_overlapping_main_optimized,
        )

        import geopandas as gpd

        road_axes = Path(road_axes_path) if road_axes_path is not None else self.road_centerlines_path
        dem = Path(dem_path) if dem_path is not None else self.dem_path
        road_width = road_width if road_width is not None else self.state.get("road_width", 10)
        block_size = block_size if block_size is not None else (self.cell_size_m or 100) * 2
        manholes = self.extract_manholes_from_lines(road_axes, dem)
        crs = crs or gpd.read_file(road_axes).crs
        export_manholes_to_shapefile(manholes, self.manholes_path, crs=crs)

        road_lines = self.state.get("road_lines")
        if road_lines is None:
            road_lines = gpd.read_file(road_axes).geometry.unary_union
        road_buffer = road_lines.buffer(road_width * 0.6)

        segments, path_info, graph_data = generate_main_sewer_path_optimized(
            manholes=manholes,
            road_buffer=road_buffer,
            block_size=block_size,
            slope_tolerance=kwargs.get("main_slope_tolerance", -0.01),
            min_pipe_length=kwargs.get("min_pipe_length", 5.0),
            prefer_slope=kwargs.get("prefer_slope", 0.5),
            return_graph_data=True,
        )
        main_path = path_info["segments"]

        secondary_pipes, secondary_attrs = generate_secondary_pipes_optimized(
            manholes=manholes,
            main_path=main_path,
            road_buffer=road_buffer,
            block_size=block_size,
            slope_tolerance=kwargs.get("secondary_slope_tolerance", 0.0),
            prefer_slope=kwargs.get("prefer_slope", 0.5),
            return_attrs=True,
        )
        secondary_pipes_clean = remove_secondary_pipes_overlapping_main_optimized(
            manholes=manholes,
            secondary_pipes=secondary_pipes,
            main_pipes=main_path,
        )
        network_status = build_current_network_status(
            manholes=manholes,
            main_path=main_path,
            secondary_pipes=secondary_pipes_clean,
        )
        tertiary_pipes, tertiary_unconnected, tertiary_attrs = (
            generate_tertiary_pipes_shortest_path_v2(
                manholes=manholes,
                main_path=main_path,
                secondary_pipes=secondary_pipes_clean,
                road_buffer=road_buffer,
                city_boundary=self.road_outer_shell_path,
                block_size=kwargs.get("tertiary_block_size", (self.cell_size_m or 100) * 10),
                neighbor_radius_factor=kwargs.get("neighbor_radius_factor", 1.5),
                min_pipe_length=kwargs.get("tertiary_min_pipe_length", 1e-3),
                point_on_line_tol=kwargs.get("point_on_line_tol", 0.01),
                return_attrs=True,
                adverse_slope_weight=kwargs.get("tertiary_adverse_slope_weight", 200.0),
                mild_adverse_slope=kwargs.get("tertiary_mild_adverse_slope", -0.005),
                moderate_adverse_slope=kwargs.get("tertiary_moderate_adverse_slope", -0.01),
                severe_adverse_multiplier=kwargs.get("tertiary_severe_adverse_multiplier", 8.0),
                max_outer_iterations=kwargs.get("max_outer_iterations", 10000),
            )
        )
        main_attrs = build_main_attrs_from_path_info(path_info)
        gdf_pipes = export_pipes_to_shapefile_2(
            pipes_main=main_path,
            pipes_sec=secondary_pipes_clean,
            pipes_ter=tertiary_pipes,
            manholes=manholes,
            output_path=self.pipes_path,
            crs=crs,
            main_attrs=main_attrs,
            secondary_attrs=secondary_attrs,
            tertiary_attrs=tertiary_attrs,
        )
        self._ensure_pipe_downstream_alias()
        self.remember(
            manholes=manholes,
            road_buffer=road_buffer,
            main_path=main_path,
            secondary_pipes=secondary_pipes_clean,
            tertiary_pipes=tertiary_pipes,
            tertiary_unconnected=tertiary_unconnected,
            network_status=network_status,
        )
        self.record_step(
            "07_sewer_network_generation",
            parameters={
                "road_width": road_width,
                "block_size": block_size,
                "network_method": "V2",
                **kwargs,
            },
            outputs={
                "manholes_path": self.manholes_path,
                "pipes_path": self.pipes_path,
            },
        )
        return gdf_pipes

    def embed_sewer_network_in_dem(self, **kwargs: Any) -> Path:
        """Modify DEM and manhole elevations to enforce positive pipe slopes."""
        from .topography import (
            build_dem_with_guaranteed_positive_slopes_idw,
            update_manhole_elevations_from_dem,
        )

        kwargs.setdefault("dem_path", self.dem_path)
        kwargs.setdefault("pipes_path", self.pipes_path)
        kwargs.setdefault("manholes_path", self.manholes_path)
        kwargs.setdefault("output_path", self.dem_path)
        self._ensure_pipe_downstream_alias()
        if kwargs.get("downstream_field") == "downstream":
            try:
                import geopandas as gpd

                pipe_columns = set(gpd.read_file(kwargs["pipes_path"]).columns)
                if "downstream" not in pipe_columns and "downstream_m" in pipe_columns:
                    kwargs["downstream_field"] = "downstream_m"
            except Exception:
                pass
        out_path = build_dem_with_guaranteed_positive_slopes_idw(**kwargs)
        update_manhole_elevations_from_dem(
            dem_path=out_path,
            manholes_path=self.manholes_path,
            output_path=None,
            overwrite=True,
            sampling="nearest",
        )
        self.record_step(
            "07_embed_sewer_network_in_dem",
            parameters=kwargs,
            outputs={"dem_path": out_path, "manholes_path": self.manholes_path},
        )
        return Path(out_path)

    # ------------------------------------------------------------------
    # Steps 8-10: flow predesign, pipe design, SWMM inputs
    # ------------------------------------------------------------------
    def predesign_flows(
        self,
        land_use_info: dict[str, Any],
        gwi_factor_ls_per_m: float = 0.0002,
        rdii_factor_ls_per_m2: float = 0.00002,
        target_crs_m: str = "EPSG:3857",
    ):
        """Compute baseflow, peak flow, GWI, RDII, and predesign flow."""
        from .design import add_predesign_flow, british_columbia_peaking_factor
        from .hydrology import (
            assign_flow_to_pipes_fast,
            compute_gwi_cumulative,
            compute_rdii_and_accumulate,
            delineate_afferent_areas_and_baseflow,
        )

        import geopandas as gpd

        self._ensure_pipe_downstream_alias()
        delineate_afferent_areas_and_baseflow(
            blocks_path=self.blocks_path,
            pipes_path=self.pipes_path,
            manholes_path=self.manholes_path,
            topo_path=self.dem_path,
            output_path=self.subcatchments_path,
            land_use_info=land_use_info,
        )
        assign_flow_to_pipes_fast(
            pipes_path=self.pipes_path,
            subcatchments_path=self.subcatchments_path,
            output_path=self.pipes_path,
        )

        pipes = gpd.read_file(self.pipes_path)
        flow_col = next(
            (
                col
                for col in ["cumulative", "cumulative_", "cumulativ", "cumulative_flow_lps"]
                if col in pipes.columns
            ),
            None,
        )
        if flow_col is None:
            raise ValueError("Could not find cumulative flow column after flow assignment.")
        peak_flow, pf = british_columbia_peaking_factor(pipes[flow_col])
        pipes["peaking_factor_bc"] = pf
        pipes["peak_flow_lps_bc"] = peak_flow
        pipes.to_file(self.pipes_path)

        compute_gwi_cumulative(
            pipes_path=self.pipes_path,
            gwi_factor_ls_per_m=gwi_factor_ls_per_m,
            out_path=self.pipes_path,
            id_field="pipe_id",
            up_field="upstream_m",
            down_field="downstream",
            length_field=None,
            target_crs_m=target_crs_m,
        )
        compute_rdii_and_accumulate(
            pipes_path=self.pipes_path,
            subcatch_path=self.subcatchments_path,
            rdii_factor_ls_per_m2=rdii_factor_ls_per_m2,
            pipe_id_field="pipe_id",
            up_field="upstream_m",
            down_field="downstream",
            sub_pipe_field="pipe_id",
            target_crs_m=target_crs_m,
            out_pipes=self.pipes_path,
            out_subcatch=self.subcatchments_path,
        )
        gdf = add_predesign_flow(pipes_path=self.pipes_path, out_path=self.pipes_path)
        self.record_step(
            "08_sewer_flow_predesign",
            parameters={
                "land_use_info": land_use_info,
                "gwi_factor_ls_per_m": gwi_factor_ls_per_m,
                "rdii_factor_ls_per_m2": rdii_factor_ls_per_m2,
                "target_crs_m": target_crs_m,
            },
            outputs={
                "pipes_path": self.pipes_path,
                "subcatchments_path": self.subcatchments_path,
            },
        )
        return gdf

    def design_pipes(
        self,
        minimum_slope: float = 0.005,
        material_fractions: dict[str, float] | None = None,
        n_by_material: dict[str, float] | None = None,
        standard_diameters_mm: list[int] | None = None,
        minimum_diameter_mm: int = 200,
        min_cover: float = 1.4,
        min_slope: float = 0.005,
        manhole_drop: float = 0.05,
    ):
        """Assign slopes, materials, diameters, invert elevations, and clean files."""
        from .design import (
            assign_invert_elevations,
            assign_material_diameter_to_pipes,
            assign_pipe_slopes,
            preprocess_pipes_and_manholes,
        )

        self._ensure_pipe_downstream_alias()
        material_fractions = material_fractions or {"PVC": 0.6, "CONCRETE": 0.3, "HDPE": 0.1}
        n_by_material = n_by_material or {"PVC": 0.011, "CONCRETE": 0.013, "HDPE": 0.012}
        standard_diameters_mm = standard_diameters_mm or [
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
        ]

        assign_pipe_slopes(
            pipes_path=self.pipes_path,
            manholes_path=self.manholes_path,
            output_path=self.pipes_path,
            minimum_slope=minimum_slope,
        )
        assign_material_diameter_to_pipes(
            pipes_path=self.pipes_path,
            output_path=self.pipes_path,
            material_fractions=material_fractions,
            n_by_material=n_by_material,
            standard_diameters_mm=standard_diameters_mm,
            minimum_diameter_mm=minimum_diameter_mm,
        )
        assign_invert_elevations(
            pipes_path=self.pipes_path,
            output_path=self.pipes_path,
            min_cover=min_cover,
            min_slope=min_slope,
            manhole_drop=manhole_drop,
        )
        pipes_clean, manholes_clean = preprocess_pipes_and_manholes(
            pipes_path=self.pipes_path,
            manholes_path=self.manholes_path,
            output_pipes_path=self.pipes_path,
            output_manholes_path=self.manholes_path,
        )
        self.record_step(
            "09_pipe_sizing_and_hydraulic_properties",
            parameters={
                "minimum_slope": minimum_slope,
                "material_fractions": material_fractions,
                "n_by_material": n_by_material,
                "standard_diameters_mm": standard_diameters_mm,
                "minimum_diameter_mm": minimum_diameter_mm,
                "min_cover": min_cover,
                "min_slope": min_slope,
                "manhole_drop": manhole_drop,
            },
            outputs={"pipes_path": self.pipes_path, "manholes_path": self.manholes_path},
        )
        return pipes_clean, manholes_clean

    def generate_gwi_inflow_raster(
        self,
        output_path: str | Path | None = None,
        min_value: float = 0.001,
        max_value: float = 0.010,
        random_seed: int | None = None,
        n_hills: int = 3,
        hill_min_value: float = 0.010,
        hill_max_value: float = 0.050,
        hill_radius_min: float = 20,
        hill_radius_max: float = 80,
        clip_to_range: bool = True,
    ) -> Path:
        """Generate a spatially variable GWI inflow coefficient raster."""
        from .hydrology import generate_random_inflow_raster

        output = Path(output_path) if output_path is not None else self.gwi_raster_path
        generate_random_inflow_raster(
            topo_tif_path=self.dem_path,
            output_tif_path=output,
            min_value=min_value,
            max_value=max_value,
            random_seed=random_seed,
            n_hills=n_hills,
            hill_min_value=hill_min_value,
            hill_max_value=hill_max_value,
            hill_radius_min=hill_radius_min,
            hill_radius_max=hill_radius_max,
            clip_to_range=clip_to_range,
        )
        self.record_step(
            "10_generate_gwi_inflow_raster",
            parameters={
                "topo_tif_path": self.dem_path,
                "min_value": min_value,
                "max_value": max_value,
                "random_seed": random_seed,
                "n_hills": n_hills,
                "hill_min_value": hill_min_value,
                "hill_max_value": hill_max_value,
                "hill_radius_min": hill_radius_min,
                "hill_radius_max": hill_radius_max,
                "clip_to_range": clip_to_range,
            },
            outputs={"gwi_raster_path": output},
        )
        return output

    def generate_rdii_density_raster(
        self,
        output_path: str | Path | None = None,
        min_density: float = 0.0,
        max_density: float = 5.0,
        random_seed: int | None = None,
        n_hills: int = 5,
        hill_min_density: float = 2.0,
        hill_max_density: float = 10.0,
        hill_radius_min: float = 20,
        hill_radius_max: float = 80,
        clip_to_range: bool = False,
    ) -> Path:
        """Generate a spatially variable RDII density raster."""
        from .hydrology import generate_random_rdii_density_raster

        output = Path(output_path) if output_path is not None else self.rdii_raster_path
        generate_random_rdii_density_raster(
            topo_tif_path=self.dem_path,
            output_tif_path=output,
            min_density=min_density,
            max_density=max_density,
            random_seed=random_seed,
            n_hills=n_hills,
            hill_min_density=hill_min_density,
            hill_max_density=hill_max_density,
            hill_radius_min=hill_radius_min,
            hill_radius_max=hill_radius_max,
            clip_to_range=clip_to_range,
        )
        self.record_step(
            "10_generate_rdii_density_raster",
            parameters={
                "topo_tif_path": self.dem_path,
                "min_density": min_density,
                "max_density": max_density,
                "random_seed": random_seed,
                "n_hills": n_hills,
                "hill_min_density": hill_min_density,
                "hill_max_density": hill_max_density,
                "hill_radius_min": hill_radius_min,
                "hill_radius_max": hill_radius_max,
                "clip_to_range": clip_to_range,
            },
            outputs={"rdii_raster_path": output},
        )
        return output

    def export_swmm(self, options_dict: dict[str, Any] | None = None, **kwargs: Any):
        """Export the base physical system to a SWMM input file."""
        from .swmm import export_swmm_inp

        self._ensure_pipe_downstream_alias()
        result = export_swmm_inp(
            pipes_path=self.pipes_path,
            manholes_path=self.manholes_path,
            output_path=self.swmm_inp_path,
            options_dict=options_dict,
            **kwargs,
        )
        self.record_step(
            "10_dynamic_flow_input_definition_base_model",
            parameters={"options_dict": options_dict, **kwargs},
            outputs={"swmm_inp_path": self.swmm_inp_path},
        )
        return result

    def export_swmm_inp(
        self,
        pipes_path: str | Path | None = None,
        manholes_path: str | Path | None = None,
        output_path: str | Path | None = None,
        **kwargs: Any,
    ):
        """Backward-compatible direct SWMM export wrapper."""
        from .swmm import export_swmm_inp

        self._ensure_pipe_downstream_alias()
        output = Path(output_path) if output_path is not None else self.swmm_inp_path
        result = export_swmm_inp(
            pipes_path=Path(pipes_path) if pipes_path is not None else self.pipes_path,
            manholes_path=Path(manholes_path) if manholes_path is not None else self.manholes_path,
            output_path=output,
            **kwargs,
        )
        self.remember(swmm_inp_path=output)
        self.record_step(
            "10_dynamic_flow_input_definition_base_model",
            parameters=kwargs,
            outputs={"swmm_inp_path": output},
        )
        return result

    def assign_dwf_patterns(self, **kwargs: Any) -> Path:
        """Assign DWF/BWF patterns to the base project SWMM model."""
        scenario = SewerTrisScenario(self, "base", self.output_dir)
        return scenario.assign_dwf_patterns(**kwargs)

    def run_swmm(self, monitored_nodes=None, monitored_links=None):
        """Run the base project SWMM model."""
        scenario = SewerTrisScenario(self, "base", self.output_dir)
        return scenario.run_swmm(monitored_nodes=monitored_nodes, monitored_links=monitored_links)

    def decompose_flows(self, link_id: str = "P_OUTLET", node_id: str = "OUTLET", save: bool = True):
        """Extract component flows from the base model and optionally save NetCDF."""
        scenario = SewerTrisScenario(self, "base", self.output_dir)
        df = scenario.get_flow_components(link_id=link_id, node_id=node_id)
        if save:
            scenario.save_flow_components(df, self.flows_path)
        self.record_step(
            "12_flow_output_decomposition",
            parameters={"link_id": link_id, "node_id": node_id, "save": save},
            outputs={"flows_path": self.flows_path if save else None},
        )
        return df


__all__ = ["SewerTrisProject", "SewerTrisScenario"]
