"""Hydrologic loading, subcatchment, rainfall, GWI, and RDII utilities."""

from __future__ import annotations

from ._deps import *

def delineate_afferent_areas_and_baseflow(
    blocks_path,
    pipes_path,
    manholes_path,
    topo_path,
    output_path,
    land_use_info
):
    """
    Delineates sewer sub-catchments for each pipe and calculates dry-weather baseflow.

    Fix: Uses nearest-neighbor join to ensure all polygons receive a pipe assignment.
    """
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram
    from geopandas.tools import sjoin_nearest

    # Load shapefiles
    blocks = gpd.read_file(blocks_path)
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))

    # Normalize land use to uppercase
    blocks['land_use'] = blocks['land_use'].str.upper()

    # Pipe midpoints
    pipes['midpoint'] = pipes.geometry.interpolate(0.5, normalized=True)
    midpoints = gpd.GeoDataFrame(pipes[['pipe_id', 'type']], geometry=pipes['midpoint'], crs=pipes.crs)

    # Voronoi polygons
    points = MultiPoint(midpoints.geometry.tolist())
    envelope = blocks.union_all().envelope.buffer(100)
    vor = voronoi_diagram(points, envelope=envelope)
    voronoi_polygons = gpd.GeoDataFrame(geometry=[p for p in vor.geoms], crs=blocks.crs)

    # Intersect Voronoi and blocks
    intersected = gpd.overlay(voronoi_polygons, blocks, how="intersection")

    # 🔄 Use nearest pipe midpoint instead of intersects
    intersected = sjoin_nearest(
        intersected,
        midpoints[['pipe_id', 'type', 'geometry']],
        how='left',
        distance_col='dist_to_pipe'
    )

    # Area in hectares
    intersected['area_ha'] = intersected.geometry.area / 10_000

    # Apply land use parameters
    def get_density(x): return land_use_info.get(x, {}).get('density', 0)
    def get_demand(x): return land_use_info.get(x, {}).get('demand', 0)

    intersected['density'] = intersected['land_use'].map(get_density)
    intersected['demand'] = intersected['land_use'].map(get_demand)
    intersected['population'] = intersected['area_ha'] * intersected['density']
    intersected['base_flow_lps'] = (intersected['population'] * intersected['demand']) / (24 * 3600)

    # Final output
    output = intersected[[
        'pipe_id', 'type', 'land_use', 'area_ha', 'population', 'base_flow_lps', 'geometry'
    ]]
    save_vector(output, output_path)
    print(f"✅ Sub-catchments saved to: {output_path}")

def assign_flow_to_pipes(pipes_path, subcatchments_path, output_path):
    """
    Assign own and cumulative flow to pipes using downstream routing via manhole IDs.

    Assumes:
        - Each pipe has: pipe_id, upstream_m, downstream
        - Subcatchments are assigned by pipe_id and contain base_flow column.
    """
    import geopandas as gpd
    from collections import defaultdict, deque

    # Load data
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
    subcatchments = gpd.read_file(subcatchments_path)

    # Detect flow column
    possible_cols = ["base_flow_lps", "baseflow_lps", "flow_lps", "base_flow_"]
    flow_col = next((col for col in possible_cols if col in subcatchments.columns), None)
    if flow_col is None:
        raise ValueError("❌ No base flow column found in subcatchments.")

    # Compute own flow per pipe
    own_flows = subcatchments.groupby("pipe_id")[flow_col].sum().reset_index()
    pipes = pipes.merge(own_flows, on="pipe_id", how="left")
    pipes["own_flow_lps"] = pipes[flow_col].fillna(0)
    pipes.drop(columns=[flow_col], inplace=True)

    # Build pipe-to-pipe connectivity using manholes
    pipes = pipes.set_index("pipe_id")
    downstream_links = defaultdict(list)
    upstream_counts = defaultdict(int)

    for pid_a, row_a in pipes.iterrows():
        manhole_a_down = row_a["downstream"]
        for pid_b, row_b in pipes.iterrows():
            if row_b["upstream_m"] == manhole_a_down:
                downstream_links[pid_a].append(pid_b)
                upstream_counts[pid_b] += 1

    # Find source pipes (no upstream connections)
    source_pipes = [pid for pid in pipes.index if upstream_counts[pid] == 0]

    # Traverse the network in topological order
    cumulative_flow = {}
    queue = deque(source_pipes)

    while queue:
        pid = queue.popleft()
        own = pipes.loc[pid, "own_flow_lps"]
        upstream_flow = sum(cumulative_flow.get(up_pid, 0) for up_pid in pipes.index
                            if pid in downstream_links[up_pid])
        cumulative_flow[pid] = own + upstream_flow
        for down_pid in downstream_links[pid]:
            queue.append(down_pid)

    # Assign to DataFrame
    pipes["cumulative_flow_lps"] = pipes.index.map(lambda pid: cumulative_flow.get(pid, 0))
    pipes.reset_index(inplace=True)

    # Save
    save_vector(pipes, output_path)
    print(f"✅ Updated pipe shapefile saved: {output_path}")

def assign_flow_to_pipes_fast(pipes_path, subcatchments_path, output_path):
    """
    Assign own and cumulative flow to pipes using downstream routing via manhole IDs.

    Assumes:
        - pipes has columns: pipe_id, upstream_m, downstream
        - subcatchments has columns: pipe_id and one flow column such as base_flow_lps

    Output:
        - own_flow_lps
        - cumulative_flow_lps
    """
    import geopandas as gpd
    import pandas as pd
    from collections import defaultdict, deque

    # -----------------------------
    # Load data
    # -----------------------------
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
    subcatchments = gpd.read_file(subcatchments_path)

    required_pipe_cols = {"pipe_id", "upstream_m", "downstream"}
    missing = required_pipe_cols - set(pipes.columns)
    if missing:
        raise ValueError(f"❌ Missing required pipe columns: {missing}")

    if "pipe_id" not in subcatchments.columns:
        raise ValueError("❌ 'pipe_id' column not found in subcatchments.")

    # -----------------------------
    # Detect flow column
    # -----------------------------
    possible_cols = ["base_flow_lps", "baseflow_lps", "flow_lps", "base_flow_"]
    flow_col = next((col for col in possible_cols if col in subcatchments.columns), None)
    if flow_col is None:
        raise ValueError("❌ No base flow column found in subcatchments.")

    # -----------------------------
    # Standardize key types
    # -----------------------------
    pipes["pipe_id"] = pipes["pipe_id"].astype(str)
    pipes["upstream_m"] = pipes["upstream_m"].astype(str)
    pipes["downstream"] = pipes["downstream"].astype(str)
    subcatchments["pipe_id"] = subcatchments["pipe_id"].astype(str)

    # -----------------------------
    # Compute own flow per pipe
    # -----------------------------
    own_flows = (
        subcatchments.groupby("pipe_id", as_index=False)[flow_col]
        .sum()
        .rename(columns={flow_col: "own_flow_lps"})
    )

    pipes = pipes.merge(own_flows, on="pipe_id", how="left")
    pipes["own_flow_lps"] = pipes["own_flow_lps"].fillna(0.0)

    # Keep only needed attributes in a fast pandas view
    pipe_df = pipes[["pipe_id", "upstream_m", "downstream", "own_flow_lps"]].copy()

    # -----------------------------
    # Build connectivity efficiently
    # -----------------------------
    # Map: upstream manhole -> list of pipes starting there
    pipes_starting_at = defaultdict(list)
    for row in pipe_df.itertuples(index=False):
        pipes_starting_at[row.upstream_m].append(row.pipe_id)

    # Map: pipe_id -> downstream pipes
    downstream_links = {}
    indegree = {pid: 0 for pid in pipe_df["pipe_id"]}

    for row in pipe_df.itertuples(index=False):
        pid = row.pipe_id
        down_mh = row.downstream
        children = pipes_starting_at.get(down_mh, [])
        downstream_links[pid] = children
        for child in children:
            indegree[child] += 1

    # -----------------------------
    # Topological flow routing
    # -----------------------------
    own_flow = dict(zip(pipe_df["pipe_id"], pipe_df["own_flow_lps"]))

    # cumulative starts with own flow
    cumulative_flow = {pid: own_flow[pid] for pid in own_flow}

    # queue = pipes with no upstream pipe
    queue = deque([pid for pid, deg in indegree.items() if deg == 0])

    processed_count = 0

    while queue:
        pid = queue.popleft()
        processed_count += 1

        flow_here = cumulative_flow[pid]

        for child in downstream_links.get(pid, []):
            cumulative_flow[child] += flow_here
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    # -----------------------------
    # Check for cycles/disconnected routing issues
    # -----------------------------
    n_pipes = len(pipe_df)
    if processed_count != n_pipes:
        unresolved = [pid for pid, deg in indegree.items() if deg > 0]
        raise ValueError(
            "❌ Network could not be fully topologically sorted. "
            "Possible cycle, duplicate downstream logic, or invalid connectivity. "
            f"Unresolved pipes: {unresolved[:20]}"
            + (" ..." if len(unresolved) > 20 else "")
        )

    # -----------------------------
    # Assign back to GeoDataFrame
    # -----------------------------
    pipes["cumulative_flow_lps"] = pipes["pipe_id"].map(cumulative_flow).fillna(0.0)

    # -----------------------------
    # Save
    # -----------------------------
    save_vector(pipes, output_path)
    print(f"✅ Updated pipe file saved: {output_path}")

def compute_gwi_cumulative(
    pipes_path: str,
    gwi_factor_ls_per_m: float,
    out_path: str | None = None,
    *,
    id_field: str = "pipe_id",
    up_field: str = "upstream_m",
    down_field: str = "downstream",
    length_field: str | None = None,       # e.g. "Length (Cartesian)" if you already have it
    target_crs_m: str | None = None,       # e.g. "EPSG:3857" or "EPSG:32614" if geometry is in degrees
    overwrite: bool = True,
):
    """
    Compute own and cumulative GWI (l/s) per pipe.

    Parameters
    ----------
    pipes_path : path to input shapefile/GeoPackage/etc.
    gwi_factor_ls_per_m : GWI factor in l/s per meter.
    out_path : optional path to save result (same format as input extension).
    id_field, up_field, down_field : field names for pipe id, upstream manhole id, downstream manhole id.
    length_field : if provided, use this numeric field for pipe length (meters). Otherwise use geometry length.
    target_crs_m : if input CRS is geographic (degrees), set a projected CRS (meters) to compute lengths.
    overwrite : if False and file exists, raise.

    Returns
    -------
    GeoDataFrame with columns:
      - own_gwi_ls
      - cum_gwi_ls
    """

    import geopandas as gpd
    import networkx as nx
    from pathlib import Path

    gdf = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))

    # Basic field checks
    for col in [id_field, up_field, down_field]:
        if col not in gdf.columns:
            raise ValueError(f"Missing required column '{col}' in input.")

    # Ensure we can compute length in meters
    if length_field is not None:
        if length_field not in gdf.columns:
            raise ValueError(f"length_field '{length_field}' not found.")
        gdf["length_m"] = gdf[length_field].astype(float)
    else:
        if gdf.crs is None:
            raise ValueError("Input has no CRS. Set a CRS or pass length_field.")
        if not gdf.crs.is_projected:
            if not target_crs_m:
                raise ValueError(
                    "CRS is geographic (degrees). Provide target_crs_m (e.g., 'EPSG:32614')."
                )
            gdf = gdf.to_crs(target_crs_m)
        gdf["length_m"] = gdf.geometry.length

    # Own GWI for each pipe
    gdf["own_gwi_ls"] = gdf["length_m"] * float(gwi_factor_ls_per_m)

    # Build a directed graph of manholes (nodes) connected by pipes (edges)
    # Edge key is pipe_id so we can store results per pipe.
    G = nx.DiGraph()
    for _, row in gdf.iterrows():
        u = row[up_field]
        v = row[down_field]
        pid = row[id_field]
        G.add_edge(u, v, pipe_id=pid)

    # Detect cycles (shouldn't exist in a gravity sewer tree)
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        cycles = list(nx.simple_cycles(G))
        raise RuntimeError(
            f"Network contains cycles (example: {cycles[:1]}). "
            "Fix the topology or break cycles before accumulating."
        )

    # Map for quick pipe lookup
    pid_to_idx = {row[id_field]: i for i, row in gdf.iterrows()}

    # Inflow at each node = sum of cumulative flows from incoming pipes
    node_inflow = {n: 0.0 for n in G.nodes}

    # We’ll compute pipe cumulative as inflow_at_upstream_node + own_gwi
    gdf["cum_gwi_ls"] = 0.0

    # Process nodes in topological order (upstream → downstream)
    for node in order:
        # For each outgoing pipe from this node:
        for _, v, data in G.out_edges(node, data=True):
            pid = data["pipe_id"]
            idx = pid_to_idx[pid]
            own = float(gdf.at[idx, "own_gwi_ls"])
            cum = node_inflow[node] + own
            gdf.at[idx, "cum_gwi_ls"] = cum
            # Add this pipe's cumulative to the downstream node's inflow
            node_inflow[v] = node_inflow.get(v, 0.0) + cum

    # Optional: write out
    if out_path:
        out_path = Path(out_path)
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists and overwrite=False.")
        # GeoPandas will infer format from extension (.shp, .gpkg, .geojson, …)
        save_vector(gdf, out_path)

    return gdf

def compute_rdii_and_accumulate(
    pipes_path: str,
    subcatch_path: str,
    rdii_factor_ls_per_m2: float,
    *,
    # Field names
    pipe_id_field: str = "pipe_id",       # pipe id on BOTH layers
    up_field: str = "upstream_m",
    down_field: str = "downstream",
    sub_pipe_field: str = "pipe_id",      # which pipe each subcatchment drains to
    area_field_m2: str | None = None,     # if you already have an area (m²) attribute on subcatchments
    # CRS handling (for area computation if area_field_m2 is None)
    target_crs_m: str | None = None,      # e.g. "EPSG:32614", required if subcatchments are in degrees
    # Output
    out_pipes: str | None = None,         # e.g. "pipes_with_rdii.shp" / ".gpkg"
    out_subcatch: str | None = None,      # e.g. "subcatch_with_rdii.shp"
    overwrite: bool = True,
):
    """
    Compute RDII (L/s) per subcatchment and accumulate to pipes (own + cumulative).

    Returns
    -------
    pipes_gdf, subs_gdf  (GeoDataFrames with new columns: own_rdii_ls, cum_rdii_ls on pipes;
                          rdii_ls, area_m2 on subcatchments)
    """

    import geopandas as gpd
    import networkx as nx
    from pathlib import Path

    # --- Read data
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
    subs  = gpd.read_file(subcatch_path)

    # --- Sanity checks
    for col in [pipe_id_field, up_field, down_field]:
        if col not in pipes.columns:
            raise ValueError(f"Missing '{col}' in pipes.")
    if sub_pipe_field not in subs.columns:
        raise ValueError(f"Missing '{sub_pipe_field}' in subcatchments (pipe mapping).")

    # --- Area in m² for subcatchments
    if area_field_m2 is not None:
        if area_field_m2 not in subs.columns:
            raise ValueError(f"area_field_m2 '{area_field_m2}' not found on subcatchments.")
        subs["area_m2"] = subs[area_field_m2].astype(float)
    else:
        if subs.crs is None:
            raise ValueError("Subcatchments have no CRS. Set a CRS or provide area_field_m2.")
        if not subs.crs.is_projected:
            if not target_crs_m:
                raise ValueError(
                    "Subcatchment CRS is geographic (degrees). Provide target_crs_m (e.g., 'EPSG:32614')."
                )
            subs = subs.to_crs(target_crs_m)
        subs["area_m2"] = subs.geometry.area

    # --- RDII per subcatchment
    subs["rdii_ls"] = subs["area_m2"] * float(rdii_factor_ls_per_m2)

    # --- Aggregate RDII to pipes (own RDII)
    rdii_by_pipe = subs.groupby(sub_pipe_field, dropna=False)["rdii_ls"].sum()
    pipes["own_rdii_ls"] = pipes[pipe_id_field].map(rdii_by_pipe).fillna(0.0)

    # --- Build directed graph of manholes from pipes
    G = nx.DiGraph()
    for _, r in pipes.iterrows():
        G.add_edge(r[up_field], r[down_field], pipe_id=r[pipe_id_field])

    # Detect cycles (should be a DAG / tree for gravity sewer)
    try:
        topo_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        cyc = list(nx.simple_cycles(G))
        raise RuntimeError(
            f"Network contains cycles (example: {cyc[:1]}). Fix topology before accumulation."
        )

    # --- Accumulate downstream
    pid_to_idx = {r[pipe_id_field]: i for i, r in pipes.iterrows()}
    node_inflow = {n: 0.0 for n in G.nodes}
    pipes["cum_rdii_ls"] = 0.0

    for n in topo_nodes:
        for _, v, data in G.out_edges(n, data=True):
            pid = data["pipe_id"]
            i = pid_to_idx[pid]
            own = float(pipes.at[i, "own_rdii_ls"])
            cum = node_inflow[n] + own
            pipes.at[i, "cum_rdii_ls"] = cum
            node_inflow[v] = node_inflow.get(v, 0.0) + cum

    # --- Optional outputs
    def _write(gdf, path):
        if path:
            path = Path(path)
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists and overwrite=False.")
            save_vector(gdf, path)

    _write(pipes, out_pipes)
    _write(subs, out_subcatch)

    return pipes, subs

def generate_random_inflow_raster(
    topo_tif_path,
    output_tif_path,
    min_value=0.001,
    max_value=0.010,
    random_seed=None,
    n_hills=3,
    hill_min_value=0.010,
    hill_max_value=0.050,
    hill_radius_min=20,
    hill_radius_max=80,
    clip_to_range=True,
):
    """
    Generate a random inflow raster with optional concentrated high-value areas
    represented as smooth Gaussian hills.

    Parameters
    ----------
    topo_tif_path : str
        Reference raster used for shape, transform, CRS, and metadata.

    output_tif_path : str
        Output GeoTIFF path.

    min_value, max_value : float
        Background random inflow range.

    random_seed : int or None
        Seed for reproducibility.

    n_hills : int
        Number of concentrated high-value areas.

    hill_min_value, hill_max_value : float
        Range of peak additional inflow values added by each hill.

    hill_radius_min, hill_radius_max : float
        Minimum and maximum hill radius in pixels.

    clip_to_range : bool
        If True, final values are clipped to max_value + hill_max_value.
    """

    import numpy as np
    import rasterio

    rng = np.random.default_rng(random_seed)

    with rasterio.open(topo_tif_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height
        nodata = src.nodata

        topo = src.read(1)

    dtype = np.float32

    # --------------------------------------------------
    # Background random inflow raster
    # --------------------------------------------------
    inflow_array = rng.uniform(
        low=min_value,
        high=max_value,
        size=(height, width)
    ).astype(dtype)

    # --------------------------------------------------
    # Create coordinate grid in pixel space
    # --------------------------------------------------
    yy, xx = np.indices((height, width))

    # --------------------------------------------------
    # Add Gaussian hills / hotspots
    # --------------------------------------------------
    for _ in range(n_hills):

        center_x = rng.uniform(0, width)
        center_y = rng.uniform(0, height)

        peak_value = rng.uniform(hill_min_value, hill_max_value)
        radius = rng.uniform(hill_radius_min, hill_radius_max)

        hill = peak_value * np.exp(
            -(
                (xx - center_x) ** 2 +
                (yy - center_y) ** 2
            ) / (2 * radius ** 2)
        )

        inflow_array += hill.astype(dtype)

    # --------------------------------------------------
    # Optional clipping
    # --------------------------------------------------
    if clip_to_range:
        inflow_array = np.clip(
            inflow_array,
            min_value,
            max_value + hill_max_value
        ).astype(dtype)

    # --------------------------------------------------
    # Preserve nodata areas from reference raster
    # --------------------------------------------------
    if nodata is not None:
        mask = topo == nodata
        inflow_array[mask] = nodata

    # --------------------------------------------------
    # Update output profile
    # --------------------------------------------------
    profile.update(
        dtype=dtype,
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(inflow_array, 1)

    print(f"Random inflow raster with hills saved to: {output_tif_path}")

    return inflow_array

def generate_random_rdii_density_raster(
    topo_tif_path,
    output_tif_path,
    min_density=0.1,
    max_density=3.0,
    random_seed=None,
    n_hills=3,
    hill_min_density=2.0,
    hill_max_density=10.0,
    hill_radius_min=20,
    hill_radius_max=80,
    clip_to_range=False,
):
    """
    Generate a synthetic RDII density raster with optional
    concentrated infiltration/inflow hotspots represented as
    Gaussian hills.

    Parameters
    ----------
    topo_tif_path : str
        Reference raster.

    output_tif_path : str
        Output GeoTIFF.

    min_density, max_density : float
        Background RDII density range.

    random_seed : int, optional
        Seed for reproducibility.

    n_hills : int
        Number of RDII hotspots.

    hill_min_density, hill_max_density : float
        Peak density added by each hotspot.

    hill_radius_min, hill_radius_max : float
        Radius of hotspots in pixels.

    clip_to_range : bool
        If True, clip values to max_density.
        Usually False for hotspot simulations.
    """

    import numpy as np
    import rasterio

    rng = np.random.default_rng(random_seed)

    with rasterio.open(topo_tif_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height
        nodata = src.nodata

        topo = src.read(1)

    dtype = np.float32

    # --------------------------------------------------
    # Background RDII density
    # --------------------------------------------------
    rdii_array = rng.uniform(
        min_density,
        max_density,
        size=(height, width)
    ).astype(dtype)

    # --------------------------------------------------
    # Coordinate grid
    # --------------------------------------------------
    yy, xx = np.indices((height, width))

    # --------------------------------------------------
    # Add RDII hotspots
    # --------------------------------------------------
    for _ in range(n_hills):

        center_x = rng.uniform(0, width)
        center_y = rng.uniform(0, height)

        peak_density = rng.uniform(
            hill_min_density,
            hill_max_density
        )

        radius = rng.uniform(
            hill_radius_min,
            hill_radius_max
        )

        hill = peak_density * np.exp(
            -(
                (xx - center_x) ** 2 +
                (yy - center_y) ** 2
            ) / (2 * radius ** 2)
        )

        rdii_array += hill.astype(dtype)

    # --------------------------------------------------
    # Optional clipping
    # --------------------------------------------------
    if clip_to_range:
        rdii_array = np.clip(
            rdii_array,
            min_density,
            max_density
        )

    # --------------------------------------------------
    # Preserve nodata mask
    # --------------------------------------------------
    if nodata is not None:
        rdii_array[topo == nodata] = nodata

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    profile.update(
        dtype=dtype,
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(rdii_array.astype(dtype), 1)

    print(f"✅ RDII density raster saved to: {output_tif_path}")

    return rdii_array

def generate_clustered_rainfall_timeseries(
    start_date="2000-01-01 00:00",
    end_date="2020-12-31 23:45",
    timestep_minutes=15,
    avg_annual_precip_mm=800,
    wet_season_months=[4, 5, 6, 9, 10, 11],
    dry_wet_ratio=0.2,
    storm_prob=0.1,  # Probability of storm initiation per timestep
    storm_duration_range=(4, 20),  # Storm duration in number of timesteps (e.g., 1–5 hours)
    random_seed=42,
    preview_date="2025-06-19"
):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(random_seed)

    # Generate full time index
    time_index = pd.date_range(start=start_date, end=end_date, freq=f"{timestep_minutes}min")
    df = pd.DataFrame(index=time_index)
    df['date'] = df.index.strftime("%-m/%-d/%Y")  # SWMM-compatible
    df['time'] = df.index.strftime("%H:%M")
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['rain_mm'] = 0.0

    years = df['year'].unique()

    for year in years:
        year_mask = df['year'] == year
        wet_mask = year_mask & df['month'].isin(wet_season_months)
        dry_mask = year_mask & ~df['month'].isin(wet_season_months)

        wet_total = avg_annual_precip_mm * (1 - dry_wet_ratio)
        dry_total = avg_annual_precip_mm * dry_wet_ratio

        for mask, total, scale in [(wet_mask, wet_total, 2.0), (dry_mask, dry_total, 0.5)]:
            times = df.loc[mask].index
            rainfall = np.zeros(len(times))
            i = 0
            while i < len(times):
                if np.random.rand() < storm_prob:
                    storm_duration = np.random.randint(*storm_duration_range)
                    storm_end = min(i + storm_duration, len(times))
                    storm_rain = np.random.exponential(scale=scale, size=storm_end - i)
                    rainfall[i:storm_end] += storm_rain
                    i = storm_end
                else:
                    i += 1
            if rainfall.sum() > 0:
                rainfall *= (total / rainfall.sum())
            df.loc[mask, 'rain_mm'] = rainfall

    df['rain_mm'] = df['rain_mm'].round(2)

    # Preview visualization for a selected date
    df_day = df.loc[preview_date]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_day.index, df_day['rain_mm'], drawstyle='steps-post')
    ax.set_title(f"Synthetic Rainfall Time Series ({preview_date})")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Time")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Return SWMM-compatible list of tuples
    output = list(df[['date', 'time', 'rain_mm']].itertuples(index=False, name=None))
    return output

def download_noaa_coop_15min_range(state_abbr, start_year, end_year, output_folder="noaa_15min_data"):
    """
    Downloads NOAA COOP 15-minute precipitation data for a given U.S. state and year range.

    Parameters:
    - state_abbr (str): Two-letter state abbreviation (e.g., 'OK' for Oklahoma).
    - start_year (int): Starting year (inclusive).
    - end_year (int): Ending year (inclusive).
    - output_folder (str): Local folder to save the downloaded files.

    Returns:
    - pd.DataFrame: Concatenated data from all years as a pandas DataFrame.
    """
    import os
    import ftplib
    import pandas as pd
    from io import StringIO

    os.makedirs(output_folder, exist_ok=True)
    ftp_server = "ftp.ncei.noaa.gov"
    all_dataframes = []

    for year in range(start_year, end_year + 1):
        ftp_path = f"/pub/data/hourly_precip-15min/{year}/{state_abbr.upper()}"
        try:
            with ftplib.FTP(ftp_server) as ftp:
                ftp.login()
                ftp.cwd(ftp_path)
                files = ftp.nlst()

                for file_name in files:
                    local_file_path = os.path.join(output_folder, f"{year}_{file_name}")

                    with open(local_file_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {file_name}", f.write)

                    with open(local_file_path, 'r') as f:
                        lines = f.readlines()

                    lines = [line for line in lines if not line.startswith('#')]
                    content = ''.join(lines)

                    try:
                        df = pd.read_csv(StringIO(content))
                        df["source_file"] = f"{year}_{file_name}"
                        all_dataframes.append(df)
                    except Exception as e:
                        print(f"Error parsing {file_name} for year {year}: {e}")
        except ftplib.all_errors as e:
            print(f"FTP error for year {year}: {e}")

    if all_dataframes:
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        print("No valid data files were parsed.")
        return pd.DataFrame()

def best_projected_crs(geom):
    """Pick a UTM CRS based on geometry centroid."""
    centroid = geom.centroid
    lon = centroid.x
    lat = centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

def create_vcp_density_raster(
    pipes_path: str,
    boundary_path: str,
    out_raster: str,
    resolution: float,
    material_col: str = "MATERIAL",
    target_value: str = "VCP",
):
    import math
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import box
    # 1. Load data
    pipes = gpd.read_file(pipes_path)
    boundary = gpd.read_file(boundary_path)

    if boundary.empty:
        raise ValueError("Boundary shapefile has no features.")
    if pipes.crs is None or boundary.crs is None:
        raise ValueError("One of the shapefiles has no CRS defined.")

    # Single domain geometry
    domain_geom = boundary.union_all()

    # 2. Decide analysis CRS
    if pipes.crs.is_geographic or boundary.crs.is_geographic:
        if boundary.crs.is_geographic:
            geom_for_utm = domain_geom
        else:
            geom_for_utm = boundary.to_crs(pipes.crs).union_all()
        proj_crs = best_projected_crs(geom_for_utm)
        print(f"✔ Using projected CRS (UTM) for analysis: {proj_crs}")
    else:
        proj_crs = boundary.crs
        print(f"✔ Using existing projected CRS: {proj_crs}")

    # 3. Reproject both layers to the analysis CRS
    pipes = pipes.to_crs(proj_crs)
    boundary = boundary.to_crs(proj_crs)
    domain_geom = boundary.union_all()

    # 4. Build raster grid from boundary extent
    minx, miny, maxx, maxy = domain_geom.bounds
    width = math.ceil((maxx - minx) / resolution)
    height = math.ceil((maxy - miny) / resolution)

    maxx = minx + width * resolution
    maxy = miny + height * resolution

    transform = from_origin(minx, maxy, resolution, resolution)

    print("Boundary extent (projected):")
    print(f"minx: {minx:.2f}, miny: {miny:.2f}, maxx: {maxx:.2f}, maxy: {maxy:.2f}")
    print(f"Raster size (rows x cols): {height} x {width}")

    # 5. Prepare pipe data
    pipes[material_col] = pipes[material_col].astype(str)
    pipes_vcp = pipes[pipes[material_col] == target_value]

    print(f"Total pipes: {len(pipes)}")
    print(f"VCP pipes: {len(pipes_vcp)}")
    print(f"Non-VCP pipes: {len(pipes) - len(pipes_vcp)}")

    # 6. Efficient spatial counting
    print("Counting pipes per pixel...")

    # Initialize arrays with zeros (important - this sets default to 0)
    total_count = np.zeros((height, width), dtype=np.int32)
    vcp_count = np.zeros((height, width), dtype=np.int32)

    # Create VCP index set for fast lookup
    vcp_indices = set(pipes_vcp.index)

    # Count pipes per pixel using spatial intersection
    for i in range(height):
        for j in range(width):
            # Get pixel bounds
            x_min = transform[2] + j * transform[0]
            y_max = transform[5] + i * transform[4]
            x_max = x_min + resolution
            y_min = y_max + resolution
            
            # Create pixel polygon
            pixel_bbox = box(x_min, y_min, x_max, y_max)
            
            # Find pipes that intersect this pixel
            intersecting_pipes = pipes[pipes.intersects(pixel_bbox)]
            if len(intersecting_pipes) > 0:
                total_count[i, j] = len(intersecting_pipes)
                vcp_count[i, j] = len(intersecting_pipes[intersecting_pipes.index.isin(vcp_indices)])

    # 7. Calculate density - KEY CHANGE: Initialize with zeros
    density = np.zeros((height, width), dtype=np.float32)  # All pixels start at 0
    
    # Only calculate ratio where there are pipes
    valid_pixels = total_count > 0
    density[valid_pixels] = (vcp_count[valid_pixels] / total_count[valid_pixels]) * 100.0

    print(f"\n=== FINAL DENSITY STATS ===")
    print(f"Total pixels: {height * width}")
    print(f"Pixels with pipes: {np.sum(valid_pixels)}")
    print(f"Pixels without pipes (value = 0): {np.sum(~valid_pixels)}")
    
    if np.sum(valid_pixels) > 0:
        unique_densities = np.unique(density[valid_pixels])
        print(f"Unique density values: {unique_densities}")
        print(f"Density range: {np.min(density[valid_pixels]):.1f}% to {np.max(density[valid_pixels]):.1f}%")
        print(f"Mean density: {np.mean(density[valid_pixels]):.1f}%")
        
        # Show distribution of density values
        for val in [0, 25, 50, 75, 100]:
            count = np.sum(density == val)
            if count > 0:
                print(f"Pixels with {val}% density: {count}")

    # 8. Apply boundary mask - set areas outside boundary to 0
    boundary_mask = rasterize(
        [(domain_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    density[boundary_mask == 0] = 0.0

    # 9. Save WITHOUT NoData - KEY CHANGE: Remove nodata parameter
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": proj_crs,
        "transform": transform,
        # No 'nodata' parameter - all values are valid (0 represents no pipes)
    }

    with rasterio.open(out_raster, "w", **meta) as dst:
        dst.write(density, 1)

    print(f"✔ Raster saved: {out_raster}")
    print("✓ Pixels without pipes are set to 0 (not NoData)")
    return density

def create_building_density_raster(boundary_path, output_raster, resolution=100.0):
    """
    Create a building density raster (buildings per km²) from OpenStreetMap
    data within the given boundary polygon.

    Parameters
    ----------
    boundary_path : str
        Path to boundary polygon shapefile.
    output_raster : str
        Path to output GeoTIFF.
    resolution : float
        Pixel size in meters.
    """

    import math
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.features import rasterize
    from shapely.geometry import box
    import osmnx as ox

    # 1. Load boundary (keep original CRS)
    boundary = gpd.read_file(boundary_path)
    print(f"Boundary CRS: {boundary.crs}")
    print(f"Boundary bounds (original CRS): {boundary.total_bounds}")

    if boundary.empty:
        print("❌ Boundary file has no features.")
        return None

    # 2. Reproject boundary to WGS84 for OSM query
    boundary_wgs84 = boundary.to_crs("EPSG:4326")
    print(f"Boundary bounds (WGS84): {boundary_wgs84.total_bounds}")

    # 3. Download buildings from OSM using polygon (NOT Edmond)
    try:
        boundary_poly_wgs84 = boundary_wgs84.union_all()
        print("Querying OSM for buildings within boundary polygon...")
        buildings = ox.features_from_polygon(
            boundary_poly_wgs84,
            tags={"building": True}
        )
        print(f"✅ Raw OSM features: {len(buildings)}")

        if len(buildings) == 0:
            print("❌ No building features returned by OSM for this area.")
            return None

        # Filter to valid building polygons
        buildings = buildings[buildings.geometry.notna()]
        valid_idx = []
        for idx, geom in buildings.geometry.items():
            if hasattr(geom, "geom_type") and geom.geom_type in ["Polygon", "MultiPolygon"]:
                valid_idx.append(idx)

        buildings = buildings.loc[valid_idx]
        print(f"✅ Valid building polygons: {len(buildings)}")

        if len(buildings) == 0:
            print("❌ No valid Polygon/MultiPolygon building geometries.")
            return None

    except Exception as e:
        print(f"❌ Error downloading OSM data: {e}")
        return None

    # 4. Choose an appropriate projected CRS (auto-estimate UTM from boundary)
    try:
        projected_crs = boundary.estimate_utm_crs()
        print(f"Using estimated UTM CRS: {projected_crs}")
    except Exception as e:
        print(f"Warning estimating UTM CRS, falling back to EPSG:3857. Error: {e}")
        projected_crs = "EPSG:3857"

    buildings_proj = buildings.to_crs(projected_crs)
    boundary_proj = boundary_wgs84.to_crs(projected_crs)

    # 5. Compute raster grid in projected CRS
    minx, miny, maxx, maxy = boundary_proj.total_bounds
    width = math.ceil((maxx - minx) / resolution)
    height = math.ceil((maxy - miny) / resolution)

    print(f"Raster grid: width={width}, height={height}, resolution={resolution} m")

    transform = from_origin(minx, maxy, resolution, resolution)

    # 6. Count buildings per pixel
    building_count = np.zeros((height, width), dtype=np.float32)
    print("Calculating building count per pixel...")

    # Optional: speed-up by spatial index
    if hasattr(buildings_proj, "sindex"):
        sindex = buildings_proj.sindex
    else:
        sindex = None

    for i in range(height):
        if i % 50 == 0:
            print(f"  Processing row {i}/{height}")
        for j in range(width):
            x_min = transform[2] + j * transform[0]
            y_max = transform[5] + i * transform[4]
            x_max = x_min + resolution
            y_min = y_max + resolution

            pixel_bbox = box(x_min, y_min, x_max, y_max)

            if sindex is not None:
                possible_matches_index = list(sindex.intersection(pixel_bbox.bounds))
                if not possible_matches_index:
                    continue
                subset = buildings_proj.iloc[possible_matches_index]
                buildings_in_pixel = subset[subset.intersects(pixel_bbox)]
            else:
                buildings_in_pixel = buildings_proj[buildings_proj.intersects(pixel_bbox)]

            building_count[i, j] = len(buildings_in_pixel)

    # 7. Convert to density (buildings per square kilometer)
    pixel_area_sq_m = resolution ** 2
    pixel_area_sq_km = pixel_area_sq_m / 1_000_000.0
    density = building_count / pixel_area_sq_km

    # 8. Apply boundary mask so values outside are set to 0
    boundary_geom_proj = boundary_proj.union_all()
    boundary_mask = rasterize(
        [(boundary_geom_proj, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    density[boundary_mask == 0] = 0.0

    # 9. Save results as GeoTIFF
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": projected_crs,
        "transform": transform,
    }

    with rasterio.open(output_raster, "w", **meta) as dst:
        dst.write(density, 1)

    # 10. Print statistics
    valid_pixels = density > 0
    total_buildings = float(np.sum(building_count))

    print(f"\n=== BUILDING DENSITY RESULTS ===")
    print(f"Total buildings counted: {total_buildings}")
    print(f"Pixels with buildings: {np.sum(valid_pixels)}")
    if np.any(valid_pixels):
        print(
            f"Max density: {np.max(density[valid_pixels]):.1f} buildings/sq km"
        )
        print(
            f"Mean density: {np.mean(density[valid_pixels]):.1f} buildings/sq km"
        )
    print(f"Raster saved: {output_raster}")

    return density

def adjust_raster_range(input_raster_path, output_raster_path, new_min, new_max, 
                       original_min=None, original_max=None, nodata_value=None):
    """
    Adjust raster values to a new specified range using min-max normalization.
    
    Parameters:
    -----------
    input_raster_path : str
        Path to input raster file (.tiff)
    output_raster_path : str
        Path for output raster file (.tiff)
    new_min : float
        Minimum value for the new range
    new_max : float
        Maximum value for the new range
    original_min : float, optional
        Original minimum value for scaling. If None, uses actual min from raster
    original_max : float, optional
        Original maximum value for scaling. If None, uses actual max from raster
    nodata_value : float, optional
        Nodata value to preserve in output
    
    Returns:
    --------
    dict : Information about the transformation
    """

    import rasterio
    import numpy as np
    
    with rasterio.open(input_raster_path) as src:
        # Read the raster data
        raster_data = src.read()
        profile = src.profile.copy()
        
        # Handle nodata values
        if nodata_value is None:
            nodata_value = src.nodata
        
        # Create mask for valid data (excluding nodata)
        if nodata_value is not None:
            valid_mask = raster_data != nodata_value
        else:
            valid_mask = np.ones_like(raster_data, dtype=bool)
        
        # Calculate original min/max if not provided
        if original_min is None:
            original_min = np.min(raster_data[valid_mask])
        if original_max is None:
            original_max = np.max(raster_data[valid_mask])
        
        print(f"Original range: {original_min} to {original_max}")
        print(f"Target range: {new_min} to {new_max}")
        
        # Initialize output array
        output_data = np.zeros_like(raster_data, dtype=np.float32)
        
        # Apply transformation only to valid data
        for i in range(raster_data.shape[0]):  # For each band
            band_data = raster_data[i]
            band_mask = valid_mask[i] if valid_mask.ndim > 2 else valid_mask
            
            # Perform min-max normalization
            output_data[i][band_mask] = ((band_data[band_mask] - original_min) / 
                                       (original_max - original_min)) * (new_max - new_min) + new_min
            
            # Preserve nodata values
            if nodata_value is not None:
                output_data[i][~band_mask] = nodata_value
        
        # Update profile for output
        profile.update({
            'dtype': np.float32,
            'nodata': nodata_value
        })
        
        # Write output raster
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(output_data)
    
    return {
        'original_min': original_min,
        'original_max': original_max,
        'new_min': new_min,
        'new_max': new_max,
        'input_file': input_raster_path,
        'output_file': output_raster_path
    }

__all__ = [
    "delineate_afferent_areas_and_baseflow",
    "assign_flow_to_pipes",
    "assign_flow_to_pipes_fast",
    "compute_gwi_cumulative",
    "compute_rdii_and_accumulate",
    "generate_random_inflow_raster",
    "generate_random_rdii_density_raster",
    "generate_clustered_rainfall_timeseries",
    "download_noaa_coop_15min_range",
    "best_projected_crs",
    "create_vcp_density_raster",
    "create_building_density_raster",
    "adjust_raster_range"
]