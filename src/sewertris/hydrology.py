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
    envelope = blocks.unary_union.envelope.buffer(100)
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
    output.to_file(output_path)
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
    pipes.to_file(output_path)
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
    pipes.to_file(output_path)
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
        gdf.to_file(out_path)

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
            gdf.to_file(path)

    _write(pipes, out_pipes)
    _write(subs, out_subcatch)

    return pipes, subs

def generate_random_inflow_raster(
    topo_tif_path,
    output_tif_path,
    min_value=0.001,
    max_value=0.010,
    random_seed=None
):

    import numpy as np
    import rasterio
    from rasterio import Affine
    from rasterio.enums import Resampling
    import os    
    
    # Optional reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load topography raster metadata
    with rasterio.open(topo_tif_path) as src:
        profile = src.profile
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        dtype = np.float32  # output as float32
    
    # Generate random raster
    inflow_array = np.random.uniform(low=min_value, high=max_value, size=(height, width)).astype(dtype)

    # Update profile for output
    profile.update(
        dtype=dtype,
        count=1,
        compress='lzw'
    )

    # Write the synthetic inflow raster
    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(inflow_array, 1)

    print(f"Random inflow raster saved to: {output_tif_path}")

def generate_random_rdii_density_raster(
    topo_tif_path,
    output_tif_path,
    min_density=0.1,       
    max_density=3.0,
    random_seed=None
):
    """
    Generates a random RDII density raster aligned to a given topography raster.

    Parameters:
    - topo_tif_path (str): Path to reference topography GeoTIFF.
    - output_tif_path (str): Path to write the output RDII raster.
    - min_density (float): Minimum RDII density value.
    - max_density (float): Maximum RDII density value.
    - random_seed (int, optional): For reproducible results.
    """
    import numpy as np
    import rasterio
    import os

    if random_seed is not None:
        np.random.seed(random_seed)

    # Load metadata from topography raster
    with rasterio.open(topo_tif_path) as src:
        profile = src.profile
        width = src.width
        height = src.height
        transform = src.transform
        dtype = np.float32
        crs = src.crs

    # Generate random RDII density values
    rdii_array = np.random.uniform(low=min_density, high=max_density, size=(height, width)).astype(dtype)

    # Update profile
    profile.update(
        dtype=dtype,
        count=1,
        compress='lzw'
    )

    # Write output raster
    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(rdii_array, 1)

    print(f"✅ RDII density raster saved to: {output_tif_path}")

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
]
