"""Visualization utilities."""

from __future__ import annotations

from ._deps import *

def plot_board(board):
    plt.figure(figsize=(10, 10))
    plt.imshow(board, cmap='tab20', interpolation='none')
    plt.title("Filled Board with Unique IDs")
    plt.colorbar(label="Unique ID")
    plt.axis('off')
    plt.show()

def plot_filled_board_shapefile(shapefile_path, cmap='tab20', figsize=(10, 8)):
    """
    Plots a shapefile generated from a filled tetromino board.
    
    Parameters:
    - shapefile_path (str): Path to the .shp file
    - cmap (str): Matplotlib colormap
    - figsize (tuple): Figure size
    """
    gdf = gpd.read_file(shapefile_path)
    
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(column="tetro_id", ax=ax, cmap=cmap, legend=True, edgecolor='black', linewidth=0.2)
    ax.set_title("Tetris-based Urban Block Layout")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def visualize_results(elevation, slope, flow_dir, mask, boundary_gdf, roads_gdf, intersections, outlet_point, xx, yy):
    """
    Create visualizations of the results
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot elevation with proper coordinates and orientation
    extent = [xx[0,0], xx[0,-1], yy[-1,0], yy[0,0]]  # [left, right, bottom, top]
    
    # Plot elevation
    im1 = ax1.imshow(elevation, extent=extent, cmap='terrain', origin='upper')
    ax1.set_title('Elevation (m)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Elevation (m)')
    
    # Overlay boundary
    boundary_gdf.boundary.plot(ax=ax1, color='black', linewidth=1)
    
    # Overlay roads
    if roads_gdf is not None:
        roads_gdf.boundary.plot(ax=ax1, color='white', linewidth=0.5)
    
    # Plot intersections
    if len(intersections) > 0:
        intersections.plot(ax=ax1, color='red', markersize=5, alpha=0.5)
    
    # Plot outlet point
    ax1.plot(outlet_point.x, outlet_point.y, 'y*', markersize=15, label='Outlet')
    ax1.legend()
    
    # Plot slope with proper coordinates
    im2 = ax2.imshow(slope, extent=extent, cmap='viridis', origin='upper')
    ax2.set_title('Slope (degrees)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Slope (°)')
    
    # Overlay boundary on slope
    boundary_gdf.boundary.plot(ax=ax2, color='black', linewidth=1)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Add labels
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nTerrain Statistics:")
    print(f"Elevation range: {np.nanmin(elevation):.2f}m - {np.nanmax(elevation):.2f}m")
    print(f"Mean elevation: {np.nanmean(elevation):.2f}m")
    print(f"Mean slope: {np.nanmean(slope):.2f}°")
    print(f"Max slope: {np.nanmax(slope):.2f}°")
    
    # Calculate flow statistics
    flow_points = np.sum(flow_dir)
    total_points = np.sum(mask)
    flow_percentage = (flow_points / total_points) * 100 if total_points > 0 else 0
    print(f"\nDrainage Statistics:")
    print(f"Points with proper drainage: {flow_points} out of {total_points} ({flow_percentage:.1f}%)")

def plot_manholes(manholes, color_by_elevation=True):
    """
    Plot manholes with optional coloring by elevation.

    Parameters:
    - manholes (list): List of manhole dicts (with 'location' and 'elevation').
    - color_by_elevation (bool): If True, color points by elevation.
    """
    xs = [mh['location'].x for mh in manholes]
    ys = [mh['location'].y for mh in manholes]
    elevs = [mh['elevation'] for mh in manholes]

    plt.figure(figsize=(10, 6))
    if color_by_elevation:
        sc = plt.scatter(xs, ys, c=elevs, cmap='terrain', s=30, edgecolor='k')
        plt.colorbar(sc, label='Elevation (m)')
    else:
        plt.scatter(xs, ys, color='blue', s=30, edgecolor='k')

    for mh in manholes:
        plt.text(mh['location'].x, mh['location'].y, mh['id'], fontsize=6, ha='center', va='center')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Manhole Locations")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_sewer_network(manholes, segments, path_info, road_buffer=None):
    """
    Visualize the sewer network with color-coded slopes and detailed annotations.
    
    Parameters:
    -----------
    manholes : list[dict]
        Each dict must have: 'id', 'location' (shapely Point), 'elevation'
    segments : list[tuple]
        List of (from_id, to_id) tuples representing pipe segments
    path_info : dict
        Contains at least:
            path_info['slopes'] -> dict keyed by (from_id, to_id) with slope value
            path_info['total_length']
            path_info['cumulative_drop']
    road_buffer : shapely Polygon or MultiPolygon, optional
        Road buffer polygon(s) to show context
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import Polygon, MultiPolygon

    fig, ax = plt.subplots(figsize=(15, 10))

    # ---- 1. Plot road buffer (Polygon or MultiPolygon) ----
    if road_buffer is not None:
        if isinstance(road_buffer, Polygon):
            polys = [road_buffer]
        elif isinstance(road_buffer, MultiPolygon):
            polys = list(road_buffer.geoms)
        else:
            polys = []
        for poly in polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, 'k-', alpha=0.2, linewidth=1)

    # ---- 2. Prepare color map for slopes ----
    slopes = [path_info['slopes'][seg] for seg in segments]
    min_slope = min(slopes)
    max_slope = max(slopes)
    # avoid zero range
    if max_slope == min_slope:
        max_slope = min_slope + 1e-6

    norm = plt.Normalize(min_slope, max_slope)
    cmap = plt.cm.viridis

    # Build lookup for manholes by id
    mh_by_id = {mh['id']: mh for mh in manholes}

    # ---- 3. Plot segments ----
    for i, segment in enumerate(segments):
        start_id, end_id = segment
        start_mh = mh_by_id[start_id]
        end_mh   = mh_by_id[end_id]

        x = [start_mh['location'].x, end_mh['location'].x]
        y = [start_mh['location'].y, end_mh['location'].y]

        slope = path_info['slopes'][segment]
        ax.plot(
            x, y, '-',
            color=cmap(norm(slope)),
            linewidth=2,
            label=f'Slope: {slope:.1%}' if i == 0 else ""
        )

        # midpoint annotation
        mid_x = np.mean(x)
        mid_y = np.mean(y)
        ax.annotate(
            f'{slope:.1%}',
            (mid_x, mid_y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    # ---- 4. Plot manholes ----
    mh_x = [mh['location'].x for mh in manholes]
    mh_y = [mh['location'].y for mh in manholes]
    ax.scatter(mh_x, mh_y, c='red', s=50, zorder=5, label='Manholes')

    for mh in manholes:
        ax.annotate(
            f"{mh['id']}\n({mh['elevation']:.1f}m)",
            (mh['location'].x, mh['location'].y),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    # ---- 5. Colorbar ----
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Slope (%)')

    # ---- 6. Stats box ----
    avg_slope = float(np.mean(list(path_info['slopes'].values())))
    stats_text = (
        f"Network Statistics:\n"
        f"Total length: {path_info['total_length']:.1f} m\n"
        f"Total drop: {path_info['cumulative_drop']:.2f} m\n"
        f"Avg slope: {avg_slope:.1%}\n"
        f"Segments: {len(segments)}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_title("Sewer Network Layout with Slopes")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_sewer_network_all(
    manholes,
    main_pipes=None,
    secondary_pipes=None,
    tertiary_pipes=None,
    unresolved=None,
    road_buffer=None,
    title="Sewer Network (Main + Secondary + Tertiary)"
):
    """
    Plot manholes and all pipe tiers.

    Parameters
    ----------
    manholes : list[dict]
        Each dict must include: 'id', 'location' (shapely Point), 'elevation' (float)
    main_pipes : list[tuple] | None
        List of (from_id, to_id) tuples for main pipes
    secondary_pipes : list[tuple] | None
        List of (from_id, to_id) tuples for secondary pipes
    tertiary_pipes : list[tuple] | None
        List of (from_id, to_id) tuples for tertiary pipes
    unresolved : list | None
        Optional list of unresolved tertiary items (format varies). If it contains
        (from_id, to_id) tuples, they will be drawn as dotted gray lines.
    road_buffer : shapely geometry | None
        Polygon/MultiPolygon for road area; plotted behind everything if provided.
    title : str
        Plot title.
    """
    if manholes is None or len(manholes) == 0:
        raise ValueError("manholes is empty.")

    id_map = {mh["id"]: mh for mh in manholes}

    xs = [mh["location"].x for mh in manholes]
    ys = [mh["location"].y for mh in manholes]
    elevs = [mh.get("elevation", 0.0) for mh in manholes]

    fig, ax = plt.subplots(figsize=(11, 8))

    # Optional road buffer background
    if road_buffer is not None:
        try:
            geoms = list(getattr(road_buffer, "geoms", [road_buffer]))
            for g in geoms:
                x, y = g.exterior.xy
                ax.fill(x, y, alpha=0.15, edgecolor="none", label="Road area")
        except Exception:
            # If buffer is not polygon-like, just skip
            pass

    # Manholes
    sc = ax.scatter(xs, ys, c=elevs, cmap="terrain", s=28, edgecolor="k", linewidth=0.4, label="Manholes")
    plt.colorbar(sc, ax=ax, label="Elevation (m)")

    def _plot_pipe_list(pipe_list, color, lw, ls, label):
        if not pipe_list:
            return
        first = True
        for u, v in pipe_list:
            if u not in id_map or v not in id_map:
                continue
            p1 = id_map[u]["location"]
            p2 = id_map[v]["location"]
            ax.plot(
                [p1.x, p2.x], [p1.y, p2.y],
                color=color, linewidth=lw, linestyle=ls,
                label=label if first else None
            )
            first = False

    # Pipes
    _plot_pipe_list(main_pipes,      color="red",    lw=2.2, ls="-",  label="Main pipes")
    _plot_pipe_list(secondary_pipes, color="orange", lw=1.6, ls="--", label="Secondary pipes")
    _plot_pipe_list(tertiary_pipes,  color="green",  lw=1.2, ls=":",  label="Tertiary pipes")

    # Unresolved (optional) — only if they look like tuples
    if unresolved:
        # handle common cases: list of tuples or list of dicts
        unresolved_edges = []
        for item in unresolved:
            if isinstance(item, tuple) and len(item) == 2:
                unresolved_edges.append(item)
            elif isinstance(item, dict):
                u = item.get("from") or item.get("u") or item.get("from_id")
                v = item.get("to") or item.get("v") or item.get("to_id")
                if u is not None and v is not None:
                    unresolved_edges.append((u, v))

        _plot_pipe_list(unresolved_edges, color="gray", lw=1.0, ls="dashdot", label="Unresolved (attempts)")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Clean legend (unique labels only)
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()

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

def plot_flow_components_v2(df, start=None, end=None, title="Flow Components at P_OUTLET"):
    """
    Plots total flow and its components using actual column names: RDII_lps, DWF_lps, GWI_lps.

    Parameters:
    - df: DataFrame with 'Datetime', 'Flow_lps', 'DWF_lps', 'RDII_lps', 'GWI_lps'
    - start, end: Optional datetime strings or objects to filter time range
    - title: Plot title
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    if not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"])

    if start:
        df = df[df["Datetime"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Datetime"] <= pd.to_datetime(end)]

    plt.figure(figsize=(12, 6))
    plt.plot(df["Datetime"], df["Flow_model_units"], label="Total Flow", linewidth=2)
    plt.plot(df["Datetime"], df["RDII_runoff"], label="RDII (Rainfall I&I)", linestyle='--')
    plt.plot(df["Datetime"], df["DWF"], label="Dry Weather Flow", linestyle='-.')
    plt.plot(df["Datetime"], df["GWI"], label="GWI (Groundwater Infiltration)", linestyle=':')

    plt.xlabel("Time")
    plt.ylabel("Flow [l/s]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def _read_flow_components_table(source):
    """Load a saved flow-component table from a DataFrame or supported file."""
    if pd is None:
        raise ImportError("pandas is required to read flow-component results.")

    if hasattr(source, "copy") and hasattr(source, "columns"):
        df = source.copy()
    else:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in {".nc", ".netcdf"}:
            import xarray as xr

            with xr.open_dataset(path) as ds:
                df = ds.to_dataframe().reset_index()
        elif suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported flow results format '{suffix}' for {path}. "
                "Use NetCDF, CSV, Parquet, or pass a DataFrame."
            )

    if "Datetime" not in df.columns:
        for candidate in ("datetime", "time", "Time", "date", "Date"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Datetime"})
                break
    if "Datetime" not in df.columns:
        if df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "Datetime"})
        else:
            raise ValueError("Flow results must include a Datetime column or index.")

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def _coerce_ensemble_manifest(
    results,
    *,
    flows_path_col="flows_path",
    ensemble_col="ensemble",
    realization_col="realization",
):
    """Normalize supported ensemble-result inputs into a manifest DataFrame."""
    if pd is None:
        raise ImportError("pandas is required to prepare ensemble manifests.")

    if hasattr(results, "copy") and hasattr(results, "columns"):
        manifest = results.copy()
    elif isinstance(results, (str, Path)):
        path = Path(results)
        if path.suffix.lower() == ".csv":
            manifest = pd.read_csv(path)
        else:
            manifest = pd.DataFrame([{flows_path_col: path}])
    elif isinstance(results, dict):
        rows = []
        for group, values in results.items():
            if isinstance(values, (str, Path)) or not hasattr(values, "__iter__"):
                values = [values]
            for i, value in enumerate(values, start=1):
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault(ensemble_col, group)
                    row.setdefault(realization_col, i)
                else:
                    row = {
                        ensemble_col: group,
                        realization_col: i,
                        flows_path_col: value,
                    }
                rows.append(row)
        manifest = pd.DataFrame(rows)
    else:
        values = list(results)
        if values and isinstance(values[0], dict):
            manifest = pd.DataFrame(values)
        else:
            manifest = pd.DataFrame(
                {
                    flows_path_col: values,
                    ensemble_col: "ensemble",
                    realization_col: range(1, len(values) + 1),
                }
            )

    if flows_path_col not in manifest.columns:
        raise ValueError(f"Manifest must include a '{flows_path_col}' column.")
    if ensemble_col not in manifest.columns:
        manifest[ensemble_col] = "ensemble"
    if realization_col not in manifest.columns:
        manifest[realization_col] = range(1, len(manifest) + 1)
    return manifest.reset_index(drop=True)


def _ensemble_group_label(row, group_cols):
    values = [str(row[col]) for col in group_cols if col in row.index]
    return " | ".join(values) if values else "ensemble"


def _simulation_label(row, group_label, realization_col):
    if realization_col in row.index and pd.notna(row[realization_col]):
        return f"{group_label} #{row[realization_col]}"
    return group_label


def _flow_duration_seconds(df):
    times = pd.to_datetime(df["Datetime"])
    if len(times) < 2:
        return None
    return (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)


def _integrate_flow_volume(df, column, conversion_factor, timestep_seconds=None):
    values = df[column].astype(float).to_numpy()
    elapsed = _flow_duration_seconds(df)
    if elapsed is not None and len(elapsed) == len(values):
        return float(np.trapz(values, elapsed) * conversion_factor)
    if timestep_seconds is None:
        return float(np.nansum(values) * conversion_factor)
    return float(np.nansum(values) * timestep_seconds * conversion_factor)


def _plot_group_summary_line(ax, group_df, column, color, label, quantiles):
    grouped = group_df.groupby("Datetime")[column]
    median = grouped.median()
    ax.plot(median.index, median.values, color=color, linewidth=2.2, label=label)
    if quantiles and len(group_df["member_id"].unique()) > 1:
        low = grouped.quantile(quantiles[0])
        high = grouped.quantile(quantiles[1])
        ax.fill_between(
            median.index,
            low.reindex(median.index).to_numpy(dtype=float),
            high.reindex(median.index).to_numpy(dtype=float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )


def _plot_metric_summary(
    ax,
    summary,
    metric_col,
    *,
    component_order,
    group_order,
    color_by_group,
    ylabel,
):
    n_groups = max(len(group_order), 1)
    width = min(0.72 / n_groups, 0.22)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * width
    rng = np.random.default_rng(24)

    for component_index, component in enumerate(component_order):
        for group_index, group_label in enumerate(group_order):
            values = summary.loc[
                (summary["component"] == component)
                & (summary["group_label"] == group_label),
                metric_col,
            ].dropna()
            if values.empty:
                continue

            color = color_by_group[group_label]
            pos = component_index + offsets[group_index]
            values_array = values.to_numpy(dtype=float)

            if len(values_array) > 1:
                ax.boxplot(
                    values_array,
                    positions=[pos],
                    widths=width * 0.75,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={
                        "facecolor": color,
                        "edgecolor": color,
                        "alpha": 0.22,
                    },
                    medianprops={"color": color, "linewidth": 2},
                    whiskerprops={"color": color, "linewidth": 1},
                    capprops={"color": color, "linewidth": 1},
                )
            jitter = (
                rng.uniform(-width * 0.22, width * 0.22, size=len(values_array))
                if len(values_array) > 1
                else np.zeros(len(values_array))
            )
            ax.scatter(
                np.full(len(values_array), pos) + jitter,
                values_array,
                s=22,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
                zorder=3,
            )

    ax.set_xticks(range(len(component_order)))
    ax.set_xticklabels(component_order)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_component_metric_panel(
    ax,
    summary,
    component,
    metric_col,
    *,
    group_order,
    color_by_group,
    ylabel,
):
    values_by_group = [
        summary.loc[
            (summary["component"] == component)
            & (summary["group_label"] == group_label),
            metric_col,
        ].dropna().to_numpy(dtype=float)
        for group_label in group_order
    ]
    rng = np.random.default_rng(24)

    for index, (group_label, values) in enumerate(zip(group_order, values_by_group)):
        if len(values) == 0:
            continue

        color = color_by_group[group_label]
        if len(values) > 1:
            ax.boxplot(
                values,
                positions=[index],
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": color,
                    "edgecolor": color,
                    "alpha": 0.22,
                },
                medianprops={"color": color, "linewidth": 2},
                whiskerprops={"color": color, "linewidth": 1},
                capprops={"color": color, "linewidth": 1},
            )
        jitter = (
            rng.uniform(-0.12, 0.12, size=len(values))
            if len(values) > 1
            else np.zeros(len(values))
        )
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.86,
            zorder=3,
        )

    ax.set_title(component)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def plot_ensemble_results(
    results,
    *,
    flows_path_col="flows_path",
    ensemble_col="ensemble",
    realization_col="realization",
    group_cols=None,
    flow_columns=None,
    start=None,
    end=None,
    flow_unit_label="L/s",
    volume_unit_label="m3",
    flow_to_volume_factor=0.001,
    timestep_seconds=None,
    show_members=True,
    show_group_summary=True,
    summary_quantiles=(0.1, 0.9),
    max_legend_items=16,
    figsize=(16, 14),
    title="Ensemble Outlet Flow Comparison",
    savepath=None,
    show=True,
    return_summary=False,
):
    """Compare outlet flow-component results across ensemble simulations.

    ``results`` can be a simulation manifest CSV path, a manifest DataFrame, a
    list of flow result paths, or a ``{group: [flow_paths...]}`` mapping. For
    multi-configuration comparisons, pass ``group_cols=("base_model",
    "ensemble")`` or any other manifest columns that define the comparison
    groups.
    """
    import warnings
    from matplotlib.lines import Line2D

    if plt is None or np is None or pd is None:
        raise ImportError("matplotlib, numpy, and pandas are required for plotting.")

    flow_columns = {
        "Total": "Flow_model_units",
        "BWF / DWF": "DWF",
        "GWI": "GWI",
        "RDII": "RDII_runoff",
        **(flow_columns or {}),
    }
    hydrograph_panels = [
        ("Total", flow_columns["Total"]),
        ("GWI", flow_columns["GWI"]),
        ("BWF / DWF", flow_columns["BWF / DWF"]),
        ("RDII", flow_columns["RDII"]),
    ]
    summary_components = ["BWF / DWF", "GWI", "RDII"]

    manifest = _coerce_ensemble_manifest(
        results,
        flows_path_col=flows_path_col,
        ensemble_col=ensemble_col,
        realization_col=realization_col,
    )
    if isinstance(group_cols, str):
        group_cols = (group_cols,)
    if group_cols is None:
        group_cols = tuple(col for col in (ensemble_col,) if col in manifest.columns)
    group_cols = tuple(group_cols)

    series_frames = []
    summary_rows = []
    skipped = []

    for member_id, row in manifest.iterrows():
        flow_path = Path(row[flows_path_col])
        if not flow_path.exists():
            skipped.append(str(flow_path))
            continue

        df = _read_flow_components_table(flow_path)
        if start is not None:
            df = df[df["Datetime"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["Datetime"] <= pd.to_datetime(end)]
        if df.empty:
            skipped.append(f"{flow_path} (empty after time filter)")
            continue

        required_columns = {column for _, column in hydrograph_panels}
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            raise ValueError(f"{flow_path} is missing flow columns: {missing}")

        group_label = _ensemble_group_label(row, group_cols)
        simulation_label = _simulation_label(row, group_label, realization_col)
        member_frame = df[["Datetime", *required_columns]].copy()
        member_frame["member_id"] = member_id
        member_frame["group_label"] = group_label
        member_frame["simulation_label"] = simulation_label
        series_frames.append(member_frame)

        for component in summary_components:
            column = flow_columns[component]
            summary_row = {
                "member_id": member_id,
                "group_label": group_label,
                "simulation_label": simulation_label,
                "component": component,
                "volume": _integrate_flow_volume(
                    df,
                    column,
                    flow_to_volume_factor,
                    timestep_seconds=timestep_seconds,
                ),
                "peak_flow": float(df[column].astype(float).max()),
                "flows_path": str(flow_path),
            }
            for col in manifest.columns:
                summary_row[col] = row[col]
            summary_rows.append(summary_row)

    if skipped:
        warnings.warn(
            "Skipped flow results that were missing or empty: "
            + "; ".join(skipped[:5])
            + (f"; ... +{len(skipped) - 5} more" if len(skipped) > 5 else ""),
            stacklevel=2,
        )
    if not series_frames:
        raise ValueError("No usable flow-component results were found.")

    long_flows = pd.concat(series_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    group_order = list(dict.fromkeys(long_flows["group_label"]))
    cmap = plt.get_cmap("tab10" if len(group_order) <= 10 else "tab20")
    color_by_group = {
        group: cmap(i % cmap.N)
        for i, group in enumerate(group_order)
    }

    n_members = long_flows["member_id"].nunique()
    member_alpha = min(0.35, max(0.05, 10.0 / max(n_members, 1)))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.55])
    hydro_grid = grid[:2, 0].subgridspec(2, 2)
    summary_grid = grid[2, 0].subgridspec(2, 3, hspace=0.38, wspace=0.28)
    hydro_axes = [
        fig.add_subplot(hydro_grid[0, 0]),
        fig.add_subplot(hydro_grid[0, 1]),
        fig.add_subplot(hydro_grid[1, 0]),
        fig.add_subplot(hydro_grid[1, 1]),
    ]
    volume_axes = [
        fig.add_subplot(summary_grid[0, i])
        for i in range(len(summary_components))
    ]
    peak_axes = [
        fig.add_subplot(summary_grid[1, i])
        for i in range(len(summary_components))
    ]

    for ax, (panel_label, column) in zip(hydro_axes, hydrograph_panels):
        if show_members:
            for _, member_df in long_flows.groupby("member_id", sort=False):
                group_label = member_df["group_label"].iloc[0]
                ax.plot(
                    member_df["Datetime"],
                    member_df[column],
                    color=color_by_group[group_label],
                    linewidth=0.8,
                    alpha=member_alpha,
                )
        if show_group_summary:
            for group_label, group_df in long_flows.groupby("group_label", sort=False):
                _plot_group_summary_line(
                    ax,
                    group_df,
                    column,
                    color_by_group[group_label],
                    group_label,
                    summary_quantiles,
                )
        ax.set_title(panel_label)
        ax.set_ylabel(f"Flow [{flow_unit_label}]")
        ax.grid(True, alpha=0.25)
    for ax in hydro_axes[2:]:
        ax.set_xlabel("Time")

    for ax, component in zip(volume_axes, summary_components):
        _plot_component_metric_panel(
            ax,
            summary,
            component,
            "volume",
            group_order=group_order,
            color_by_group=color_by_group,
            ylabel=f"Volume [{volume_unit_label}]",
        )
    for ax, component in zip(peak_axes, summary_components):
        _plot_component_metric_panel(
            ax,
            summary,
            component,
            "peak_flow",
            group_order=group_order,
            color_by_group=color_by_group,
            ylabel=f"Peak Flow [{flow_unit_label}]",
        )
    volume_axes[0].text(
        -0.22,
        1.08,
        "Component Volume by Simulation",
        transform=volume_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    peak_axes[0].text(
        -0.22,
        1.08,
        "Component Peak Flow by Simulation",
        transform=peak_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    if len(group_order) <= max_legend_items:
        handles = [
            Line2D([0], [0], color=color_by_group[group], linewidth=2.4, label=group)
            for group in group_order
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(len(handles), 4),
            frameon=False,
        )
    else:
        fig.text(
            0.5,
            0.01,
            f"{len(group_order)} groups plotted; legend omitted.",
            ha="center",
            fontsize=9,
        )

    fig.suptitle(title)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if return_summary:
        return fig, summary
    return fig

def plot_domain_mask(domain_mask, title="Final domain_mask", show_grid=True, savepath=None):
    """
    Plot a binary domain mask (1=inside, 0=outside).

    Parameters
    ----------
    domain_mask : 2D array-like of int/bool
        Mask grid.
    title : str
        Figure title.
    show_grid : bool
        Draw cell grid lines for readability.
    savepath : str or None
        If provided, save figure to this path.
    """
    mask = np.asarray(domain_mask)
    if mask.ndim != 2:
        raise ValueError("domain_mask must be a 2D array.")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mask, origin="upper", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")

    # Colorbar with meaning
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Outside (0)", "Inside (1)"])

    # Optional grid to show cell boundaries
    if show_grid:
        ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
        ax.grid(which="minor", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def _pad_to_box(arr, box_h, box_w, pad_value=0):
    """Center-pad arr into a (box_h, box_w) array."""
    h, w = arr.shape
    out = np.full((box_h, box_w), pad_value, dtype=arr.dtype)
    top = (box_h - h) // 2
    left = (box_w - w) // 2
    out[top:top+h, left:left+w] = arr
    return out

def plot_tetromino_set(tetrominoes, tetromino_colors, ncols=None, savepath=None):
    names = list(tetrominoes.keys())
    shapes = [tetrominoes[k][0] for k in names]

    # Common display box (so I and O are visible)
    max_h = max(s.shape[0] for s in shapes)
    max_w = max(s.shape[1] for s in shapes)
    box_h = max(max_h, 5)
    box_w = max(max_w, 5)

    n = len(names)
    if ncols is None:
        ncols = n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(1.8 * ncols, 1.8 * nrows),
        squeeze=False
    )

    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        shape = _pad_to_box(tetrominoes[name][0], box_h, box_w)
        color = tetromino_colors.get(name, "black")

        # White background, colored blocks only
        cmap = ListedColormap(["white", color])

        ax.imshow(
            shape,
            cmap=cmap,
            interpolation="nearest",
            origin="upper",
            vmin=0,
            vmax=1
        )

        ax.set_title(name, fontsize=11)
        ax.axis("off")

        # Strong grid so yellow O and thin I are visible
        ax.set_xticks(np.arange(-0.5, box_w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, box_h, 1), minor=True)
        ax.grid(which="minor", linewidth=1.1, color="black")

        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.3)
            spine.set_color("black")

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle("SewerTris Block Set", fontsize=14, y=0.98)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_roads(road_lines, road_buffer, crs=None, blocks_path=None, title="Road polygons", savepath=None):
    """
    Plot road polygons (buffer) and optionally centerlines + blocks.
    """
    # Wrap geometries into GeoDataFrames
    gdf_buf = gpd.GeoDataFrame(geometry=[road_buffer], crs=crs)
    gdf_lines = gpd.GeoDataFrame(geometry=[road_lines], crs=crs)

    # Optional: load blocks
    gdf_blocks = None
    if blocks_path is not None:
        gdf_blocks = gpd.read_file(blocks_path)
        if gdf_blocks.crs is None and crs is not None:
            gdf_blocks = gdf_blocks.set_crs(crs)
        elif crs is not None and gdf_blocks.crs != crs:
            gdf_blocks = gdf_blocks.to_crs(crs)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Blocks underneath (optional)
    if gdf_blocks is not None:
        gdf_blocks.plot(ax=ax, alpha=0.15, edgecolor="black", linewidth=0.6)

    # Road polygons (main)
    gdf_buf.plot(ax=ax, alpha=0.5, edgecolor="black", linewidth=0.8)

    # Centerlines on top (optional, nice for clarity)
    gdf_lines.plot(ax=ax, linewidth=1.2)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_blocks_landuse(blocks_gdf, roads_gdf=None, landuse_col=None,
                        title="Blocks colored by land use", savepath=None):
    """
    Plot blocks colored by land use, optionally overlay roads.
    """
    gdf = blocks_gdf.copy()

    # Auto-detect landuse column if not provided
    if landuse_col is None:
        candidates = ["landuse", "LandUse", "LANDUSE", "lu", "LU", "zone", "ZONE", "type", "TYPE"]
        landuse_col = next((c for c in candidates if c in gdf.columns), None)

    if landuse_col is None:
        raise ValueError(
            f"Land use column not found. Available columns:\n{list(gdf.columns)}\n"
            "Pass landuse_col='your_column_name'."
        )

    fig, ax = plt.subplots(figsize=(11, 9))

    # Plot blocks with categorical legend
    gdf.plot(
        ax=ax,
        column=landuse_col,
        categorical=True,
        legend=True,
        linewidth=0.6,
        edgecolor="black"
    )

    # Overlay roads
    if roads_gdf is not None:
        roads_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, alpha=0.9)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_dem_tif(tif_path, title=None, hillshade=False, savepath=None):
    with rasterio.open(tif_path) as src:
        dem = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds

    if nodata is not None:
        dem[dem == nodata] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    # extent in map coordinates
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        # shade works best with NaNs filled; keep NaNs masked for plotting
        dem_fill = np.where(np.isnan(dem), np.nanmedian(dem), dem)
        shaded = ls.shade(dem_fill, cmap=plt.get_cmap("terrain"), vert_exag=1.0, blend_mode="overlay")
        ax.imshow(shaded, extent=extent, origin="upper")
    else:
        im = ax.imshow(dem, extent=extent, origin="upper", cmap="terrain")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Elevation (m)")

    ax.set_title(title or tif_path.split("/")[-1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_final_design_color_by_diameter(
    pipes_path,
    manholes_path,
    blocks_path=None,
    diameter_field=None,
    manhole_color_field=None,
    title="Final sewer design (pipes colored by diameter)",
    linewidth=1.6,
    savepath=None
):
    # --- Load ---
    pipes = gpd.read_file(pipes_path)
    mhs = gpd.read_file(manholes_path)

    if mhs.crs != pipes.crs:
        mhs = mhs.to_crs(pipes.crs)

    blocks = None
    if blocks_path:
        blocks = gpd.read_file(blocks_path)
        if blocks.crs != pipes.crs:
            blocks = blocks.to_crs(pipes.crs)

    # --- Diameter field auto-detect ---
    candidates = [
        "diameter_mm",
        "diameter_m",
        "diam_mm",
        "diameter",
        "D_mm",
        "D",
        "pipe_diam_mm",
    ]
    if diameter_field is None or diameter_field not in pipes.columns:
        diameter_field = next((c for c in candidates if c in pipes.columns), None)
    if diameter_field is None:
        raise ValueError(f"Could not find a diameter field in pipes. Columns:\n{list(pipes.columns)}")

    pipes[diameter_field] = pd.to_numeric(pipes[diameter_field], errors="coerce")
    plot_diameter_field = diameter_field
    if diameter_field.endswith("_m") and pipes[diameter_field].max(skipna=True) <= 10:
        plot_diameter_field = "_diameter_plot_mm"
        pipes[plot_diameter_field] = pipes[diameter_field] * 1000.0

    # --- Manhole color field (optional) ---
    if manhole_color_field is None:
        mh_candidates = ["invert_elev", "invert", "elev", "elevation", "rim_elev", "z"]
        manhole_color_field = next((c for c in mh_candidates if c in mhs.columns), None)

    # Drop pipes without geometry
    pipes = pipes[pipes.geometry.notna()].copy()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 10))

    # Optional blocks backdrop (light outline only)
    if blocks is not None:
        blocks.plot(ax=ax, facecolor="none", edgecolor="0.7", linewidth=0.6, alpha=0.8)

    # Color pipes by diameter (categorical if discrete)
    # If your diameters are standard (200,250,...), categorical legend is best.
    unique_d = np.sort(pipes[plot_diameter_field].dropna().unique())

    if len(unique_d) <= 18:
        # Categorical colors with legend
        pipes["_diam_str"] = pipes[plot_diameter_field].round().astype("Int64").astype(str) + " mm"
        pipes.plot(
            ax=ax,
            column="_diam_str",
            categorical=True,
            legend=True,
            linewidth=linewidth
        )
    else:
        # Continuous colormap + colorbar (if many distinct diameters)
        pipes.plot(
            ax=ax,
            column=plot_diameter_field,
            legend=True,
            linewidth=linewidth
        )

    # Manholes: colored if field exists, otherwise uniform
    if manhole_color_field and manhole_color_field in mhs.columns:
        mhs.plot(ax=ax, column=manhole_color_field, cmap="terrain", markersize=18, edgecolor="k", linewidth=0.3)
        sm = plt.cm.ScalarMappable(cmap="terrain")
        sm.set_array(mhs[manhole_color_field].values)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(f"Manhole {manhole_color_field}")
    else:
        mhs.plot(ax=ax, color="black", markersize=18)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_inflow_from_pipe_length(inp_path, coefficient=0.0001, title=None, savepath=None):
    """
    Plot baseline inflow per node derived from downstream pipe length in [CONDUITS]:
    baseline (L/s) = length(m) * coefficient (L/s/m)

    Uses:
      - [CONDUITS] for FromNode and Length
      - [COORDINATES] for node X,Y
    """
    with open(inp_path, "r") as f:
        lines = f.readlines()

    # -------- Parse CONDUITS: from_node -> length --------
    downstream_lengths = {}
    in_conduits = False
    for line in lines:
        s = line.strip()
        if s.startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits:
            if s.startswith("[") and not s.startswith("[CONDUITS]"):
                break
            if (not s) or s.startswith(";"):
                continue
            parts = s.split()
            # Name  FromNode  ToNode  Length ...
            if len(parts) >= 4:
                from_node = parts[1]
                try:
                    length = float(parts[3])
                except ValueError:
                    continue
                downstream_lengths[from_node] = length

    # -------- Parse COORDINATES: node -> (x,y) --------
    coords = {}
    in_coords = False
    for line in lines:
        s = line.strip()
        if s.startswith("[COORDINATES]"):
            in_coords = True
            continue
        if in_coords:
            if s.startswith("[") and not s.startswith("[COORDINATES]"):
                break
            if (not s) or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                node = parts[0]
                try:
                    x = float(parts[1]); y = float(parts[2])
                except ValueError:
                    continue
                coords[node] = (x, y)

    # -------- Build arrays for plotting --------
    # Nodes that have both a downstream length and coordinates
    nodes = [n for n in downstream_lengths.keys() if n in coords]
    if not nodes:
        raise ValueError("No nodes found with BOTH downstream pipe length (CONDUITS) and coordinates (COORDINATES).")

    xs = np.array([coords[n][0] for n in nodes])
    ys = np.array([coords[n][1] for n in nodes])
    inflows = np.array([downstream_lengths[n] * coefficient for n in nodes])  # L/s

    # For drawing pipes, we need From->To and coordinates
    pipes_xy = []
    in_conduits = False
    for line in lines:
        s = line.strip()
        if s.startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits:
            if s.startswith("[") and not s.startswith("[CONDUITS]"):
                break
            if (not s) or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                u = parts[1]; v = parts[2]
                if u in coords and v in coords:
                    pipes_xy.append((coords[u], coords[v]))

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(11, 9))

    # Pipes (light)
    for (x1, y1), (x2, y2) in pipes_xy:
        ax.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.35)

    # Nodes colored by inflow
    sc = ax.scatter(xs, ys, c=inflows, s=26, edgecolor="k", linewidth=0.25)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Baseline inflow (L/s) = length × coefficient")

    ax.set_title(title or f"Node baseline inflow from downstream pipe length (coef={coefficient:g} L/s/m)")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    # Optional quick stats
    print(f"Nodes plotted: {len(nodes)}")
    print(f"Inflow range: {np.min(inflows):.4f} to {np.max(inflows):.4f} L/s")

def _load_project_for_plotting(project):
    """Return a SewerTrisProject from an object, project folder, or JSON path."""
    if hasattr(project, "project_file") and hasattr(project, "output_dir"):
        return project
    from .project import SewerTrisProject

    return SewerTrisProject.load(project)

def _project_plot_label(project, fallback):
    return (
        project.metadata.get("name")
        or getattr(project, "name", None)
        or Path(project.output_dir).name
        or fallback
    )

def _require_existing_path(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path

def _read_project_flows(project, scenario_name=None):
    paths = []
    if scenario_name:
        paths.append(project.load_run(scenario_name).flows_path)
    paths.append(project.flows_path)

    for path in paths:
        if Path(path).exists():
            import xarray as xr

            df = xr.open_dataset(path).to_dataframe().reset_index()
            if "Datetime" not in df.columns:
                for candidate in ("datetime", "time", "Time"):
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: "Datetime"})
                        break
            return df

    checked = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Flow components not found. Checked: {checked}")

def _flow_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find any of {candidates} in flow data. Columns: {list(df.columns)}"
    )

def _prepare_flow_df(df, start=None, end=None):
    df = df.copy()
    if "Datetime" not in df.columns:
        raise ValueError(f"Flow data must include a Datetime column. Columns: {list(df.columns)}")
    if not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    if start:
        df = df[df["Datetime"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Datetime"] <= pd.to_datetime(end)]
    return df

def _read_inflow_from_swmm(inp_path, coefficient=0.0001):
    with open(inp_path, "r") as f:
        lines = f.readlines()

    downstream_lengths = {}
    in_conduits = False
    for line in lines:
        text = line.strip()
        if text.startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits:
            if text.startswith("[") and not text.startswith("[CONDUITS]"):
                break
            if not text or text.startswith(";"):
                continue
            parts = text.split()
            if len(parts) >= 4:
                try:
                    downstream_lengths[parts[1]] = float(parts[3])
                except ValueError:
                    pass

    coords = {}
    in_coords = False
    for line in lines:
        text = line.strip()
        if text.startswith("[COORDINATES]"):
            in_coords = True
            continue
        if in_coords:
            if text.startswith("[") and not text.startswith("[COORDINATES]"):
                break
            if not text or text.startswith(";"):
                continue
            parts = text.split()
            if len(parts) >= 3:
                try:
                    coords[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass

    pipes_xy = []
    in_conduits = False
    for line in lines:
        text = line.strip()
        if text.startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits:
            if text.startswith("[") and not text.startswith("[CONDUITS]"):
                break
            if not text or text.startswith(";"):
                continue
            parts = text.split()
            if len(parts) >= 3 and parts[1] in coords and parts[2] in coords:
                pipes_xy.append((coords[parts[1]], coords[parts[2]]))

    nodes = [node for node in downstream_lengths if node in coords]
    if not nodes:
        raise ValueError(
            "No nodes found with both downstream pipe length and SWMM coordinates."
        )
    xs = np.array([coords[node][0] for node in nodes])
    ys = np.array([coords[node][1] for node in nodes])
    inflows = np.array([downstream_lengths[node] * coefficient for node in nodes])
    return xs, ys, inflows, pipes_xy

def _project_swmm_input_path(project, scenario_name=None):
    if scenario_name:
        scenario_path = project.load_run(scenario_name).swmm_inp_path
        if scenario_path.exists():
            return scenario_path
    return project.swmm_inp_path

def _auto_landuse_col(gdf, landuse_col=None):
    if landuse_col and landuse_col in gdf.columns:
        return landuse_col
    candidates = [
        "land_use",
        "landuse",
        "LandUse",
        "LANDUSE",
        "lu",
        "LU",
        "zone",
        "ZONE",
        "type",
        "TYPE",
    ]
    return next((col for col in candidates if col in gdf.columns), None)

def _auto_diameter_col(gdf, diameter_field=None):
    if diameter_field and diameter_field in gdf.columns:
        return diameter_field
    candidates = [
        "diameter_mm",
        "diameter_m",
        "diam_mm",
        "diameter",
        "D_mm",
        "D",
        "pipe_diam_mm",
    ]
    col = next((candidate for candidate in candidates if candidate in gdf.columns), None)
    if col is None:
        raise ValueError(f"Could not find a diameter field. Columns: {list(gdf.columns)}")
    return col

def _diameter_values_mm(gdf, diameter_field=None):
    diameter_field = _auto_diameter_col(gdf, diameter_field)
    values = pd.to_numeric(gdf[diameter_field], errors="coerce")
    if diameter_field.endswith("_m") and values.max(skipna=True) <= 10:
        values = values * 1000.0
    return values

def _set_same_extent(axes):
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for ax in axes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
    for ax in axes:
        ax.set_xlim(min(xmins), max(xmaxs))
        ax.set_ylim(min(ymins), max(ymaxs))

def _plot_domain_panel(ax, project, label, show_grid=True):
    _require_existing_path(project.domain_mask_path, f"{label} domain mask")
    mask = np.load(project.domain_mask_path)
    im = ax.imshow(mask, origin="upper", interpolation="nearest")
    if show_grid:
        ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
        ax.grid(which="minor", linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(label)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    return im

def _plot_layout_panel(ax, project, label, cmap="tab20", legend=False):
    path = _require_existing_path(project.layout_blocks_path, f"{label} layout blocks")
    gdf = gpd.read_file(path)
    kwargs = {"ax": ax, "cmap": cmap, "edgecolor": "black", "linewidth": 0.2}
    if "tetro_id" in gdf.columns:
        kwargs.update({"column": "tetro_id", "legend": legend})
    gdf.plot(**kwargs)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def _plot_roads_panel(ax, project, label):
    roads_path = _require_existing_path(project.road_polygons_path, f"{label} road polygons")
    roads = gpd.read_file(roads_path)
    roads.plot(ax=ax, alpha=0.5, edgecolor="black", linewidth=0.8)
    if project.road_centerlines_path.exists():
        lines = gpd.read_file(project.road_centerlines_path)
        if lines.crs != roads.crs:
            lines = lines.to_crs(roads.crs)
        lines.plot(ax=ax, linewidth=1.2)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def _plot_landuse_panel(ax, project, label, landuse_col=None):
    blocks_path = _require_existing_path(project.blocks_path, f"{label} city blocks")
    blocks = gpd.read_file(blocks_path)
    landuse_col = _auto_landuse_col(blocks, landuse_col)
    if landuse_col is None:
        raise ValueError(f"Land use column not found. Columns: {list(blocks.columns)}")
    blocks.plot(
        ax=ax,
        column=landuse_col,
        categorical=True,
        legend=True,
        linewidth=0.6,
        edgecolor="black",
    )
    if project.road_polygons_path.exists():
        roads = gpd.read_file(project.road_polygons_path)
        if roads.crs != blocks.crs:
            roads = roads.to_crs(blocks.crs)
        roads.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def _plot_dem_panel(ax, project, label, hillshade=False, vmin=None, vmax=None):
    path = _require_existing_path(project.dem_path, f"{label} DEM")
    with rasterio.open(path) as src:
        dem = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        dem[dem == nodata] = np.nan
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        dem_fill = np.where(np.isnan(dem), np.nanmedian(dem), dem)
        shaded = ls.shade(dem_fill, cmap=plt.get_cmap("terrain"), vert_exag=1.0, blend_mode="overlay")
        im = ax.imshow(shaded, extent=extent, origin="upper")
    else:
        im = ax.imshow(
            dem,
            extent=extent,
            origin="upper",
            cmap="terrain",
            vmin=vmin,
            vmax=vmax,
        )
    ax.set_title(label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    return im

def _plot_sewer_network_panel(ax, project, label):
    pipes_path = _require_existing_path(project.pipes_path, f"{label} sewer pipes")
    manholes_path = _require_existing_path(project.manholes_path, f"{label} manholes")
    pipes = gpd.read_file(pipes_path)
    manholes = gpd.read_file(manholes_path)
    if manholes.crs != pipes.crs:
        manholes = manholes.to_crs(pipes.crs)
    if project.road_polygons_path.exists():
        roads = gpd.read_file(project.road_polygons_path)
        if roads.crs != pipes.crs:
            roads = roads.to_crs(pipes.crs)
        roads.plot(ax=ax, color="0.9", edgecolor="none", alpha=0.7)

    type_col = "type" if "type" in pipes.columns else None
    if type_col:
        styles = {
            "main": {"color": "red", "linewidth": 2.2, "linestyle": "-"},
            "secondary": {"color": "orange", "linewidth": 1.6, "linestyle": "--"},
            "tertiary": {"color": "green", "linewidth": 1.2, "linestyle": ":"},
        }
        for pipe_type, style in styles.items():
            subset = pipes[pipes[type_col].astype(str).str.lower() == pipe_type]
            if not subset.empty:
                subset.plot(ax=ax, label=pipe_type.title(), **style)
        other = pipes[~pipes[type_col].astype(str).str.lower().isin(styles)]
        if not other.empty:
            other.plot(ax=ax, color="0.35", linewidth=1.0, label="Other")
    else:
        pipes.plot(ax=ax, color="green", linewidth=1.2, label="Pipes")

    color_col = "elevation" if "elevation" in manholes.columns else None
    if color_col:
        manholes.plot(
            ax=ax,
            column=color_col,
            cmap="terrain",
            markersize=18,
            edgecolor="k",
            linewidth=0.3,
        )
    else:
        manholes.plot(ax=ax, color="black", markersize=18)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

def _plot_design_panel(
    ax,
    project,
    label,
    diameter_field=None,
    manhole_color_field=None,
    linewidth=1.6,
    norm=None,
    cmap="viridis",
):
    pipes_path = _require_existing_path(project.pipes_path, f"{label} sewer pipes")
    manholes_path = _require_existing_path(project.manholes_path, f"{label} manholes")
    pipes = gpd.read_file(pipes_path)
    manholes = gpd.read_file(manholes_path)
    if manholes.crs != pipes.crs:
        manholes = manholes.to_crs(pipes.crs)
    if project.blocks_path.exists():
        blocks = gpd.read_file(project.blocks_path)
        if blocks.crs != pipes.crs:
            blocks = blocks.to_crs(pipes.crs)
        blocks.plot(ax=ax, facecolor="none", edgecolor="0.75", linewidth=0.6, alpha=0.8)

    pipes = pipes[pipes.geometry.notna()].copy()
    pipes["_diameter_plot_mm"] = _diameter_values_mm(pipes, diameter_field)
    pipes.plot(
        ax=ax,
        column="_diameter_plot_mm",
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )

    if manhole_color_field is None:
        candidates = ["invert_elev", "invert", "elev", "elevation", "rim_elev", "z"]
        manhole_color_field = next((col for col in candidates if col in manholes.columns), None)
    if manhole_color_field and manhole_color_field in manholes.columns:
        manholes.plot(
            ax=ax,
            column=manhole_color_field,
            cmap="terrain",
            markersize=18,
            edgecolor="k",
            linewidth=0.3,
        )
    else:
        manholes.plot(ax=ax, color="black", markersize=18)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def _plot_flow_panel(ax, df, label, start=None, end=None):
    df = _prepare_flow_df(df, start=start, end=end)
    columns = {
        "Total Flow": _flow_column(df, ["Flow_model_units", "Flow_lps", "flow_lps"]),
        "RDII (Rainfall I&I)": _flow_column(df, ["RDII_runoff", "RDII_lps", "rdii_lps"]),
        "Dry Weather Flow": _flow_column(df, ["DWF", "DWF_lps", "dwf_lps"]),
        "GWI (Groundwater Infiltration)": _flow_column(df, ["GWI", "GWI_lps", "gwi_lps"]),
    }
    styles = {
        "Total Flow": {"linewidth": 2.0, "linestyle": "-"},
        "RDII (Rainfall I&I)": {"linestyle": "--"},
        "Dry Weather Flow": {"linestyle": "-."},
        "GWI (Groundwater Infiltration)": {"linestyle": ":"},
    }
    for flow_label, col in columns.items():
        ax.plot(df["Datetime"], df[col], label=flow_label, **styles[flow_label])
    ax.set_title(label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow [l/s]")
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)

def _plot_inflow_panel(ax, project, label, scenario_name=None, coefficient=0.0001, norm=None):
    inp_path = _require_existing_path(
        _project_swmm_input_path(project, scenario_name=scenario_name),
        f"{label} SWMM input",
    )
    xs, ys, inflows, pipes_xy = _read_inflow_from_swmm(inp_path, coefficient=coefficient)
    for (x1, y1), (x2, y2) in pipes_xy:
        ax.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.35)
    sc = ax.scatter(xs, ys, c=inflows, norm=norm, s=26, edgecolor="k", linewidth=0.25)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return sc

def plot_two_models(
    plot_type,
    project1,
    project2,
    labels=("Original", "Sibling"),
    scenario_name="bwf_gwi_rdii",
    start=None,
    end=None,
    savepath=None,
    figsize=(16, 7),
    show=True,
    **kwargs,
):
    """Compare two completed SewerTris projects with side-by-side panels.

    Parameters
    ----------
    plot_type : str
        One of: ``layout``, ``roads``, ``land_use``, ``dem``, ``sewer_network``,
        ``final_design``, ``flow_components``, ``domain_mask``, ``inflow``, or
        ``all``.
    project1, project2 : SewerTrisProject | str | pathlib.Path
        Loaded projects, project folders, or ``sewertris_project.json`` paths.
    labels : tuple[str, str]
        Panel labels. By default, the left panel is the original model and the
        right panel is the sibling.
    scenario_name : str | None
        Scenario used for flow-component and SWMM-inflow comparisons.
    start, end : str | datetime | None
        Optional time window for ``plot_type="flow_components"``.
    savepath : str | pathlib.Path | None
        Output path. With ``plot_type="all"``, this is treated as a directory.
    figsize : tuple
        Figure size for each comparison figure.
    show : bool
        Display the figure with ``plt.show()``. Set to ``False`` in automated
        tests or scripts that only need the returned figure/save file.
    **kwargs
        Plot-specific options such as ``diameter_field``, ``landuse_col``,
        ``hillshade``, ``coefficient``, or ``show_grid``.
    """
    project1 = _load_project_for_plotting(project1)
    project2 = _load_project_for_plotting(project2)
    left_label = labels[0] if labels else _project_plot_label(project1, "Original")
    right_label = labels[1] if labels and len(labels) > 1 else _project_plot_label(project2, "Sibling")

    aliases = {
        "blocks": "layout",
        "tetris": "layout",
        "road": "roads",
        "landuse": "land_use",
        "land-use": "land_use",
        "land_use_assignment": "land_use",
        "topography": "dem",
        "network": "sewer_network",
        "sewers": "sewer_network",
        "pipes": "final_design",
        "design": "final_design",
        "diameter": "final_design",
        "diameters": "final_design",
        "flow": "flow_components",
        "flows": "flow_components",
        "components": "flow_components",
        "domain": "domain_mask",
        "mask": "domain_mask",
        "swmm_inflow": "inflow",
    }
    plot_key = aliases.get(str(plot_type).lower().replace(" ", "_"), str(plot_type).lower().replace(" ", "_"))

    all_plot_types = [
        "layout",
        "roads",
        "land_use",
        "dem",
        "sewer_network",
        "final_design",
        "flow_components",
    ]
    if plot_key == "all":
        figures = {}
        output_dir = None
        if savepath:
            output_dir = Path(savepath)
            output_dir.mkdir(parents=True, exist_ok=True)
        for key in all_plot_types:
            key_savepath = output_dir / f"{key}.png" if output_dir else None
            figures[key] = plot_two_models(
                key,
                project1,
                project2,
                labels=(left_label, right_label),
                scenario_name=scenario_name,
                start=start,
                end=end,
                savepath=key_savepath,
                figsize=figsize,
                show=show,
                **kwargs,
            )
        return figures

    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    axes = axes[0]

    if plot_key == "domain_mask":
        im1 = _plot_domain_panel(
            axes[0],
            project1,
            left_label,
            show_grid=kwargs.get("show_grid", True),
        )
        im2 = _plot_domain_panel(
            axes[1],
            project2,
            right_label,
            show_grid=kwargs.get("show_grid", True),
        )
        fig.colorbar(im2, ax=axes, fraction=0.046, pad=0.02, label="Mask value")

    elif plot_key == "layout":
        _plot_layout_panel(
            axes[0],
            project1,
            left_label,
            cmap=kwargs.get("cmap", "tab20"),
            legend=kwargs.get("legend", False),
        )
        _plot_layout_panel(
            axes[1],
            project2,
            right_label,
            cmap=kwargs.get("cmap", "tab20"),
            legend=kwargs.get("legend", False),
        )
        _set_same_extent(axes)

    elif plot_key == "roads":
        _plot_roads_panel(axes[0], project1, left_label)
        _plot_roads_panel(axes[1], project2, right_label)
        _set_same_extent(axes)

    elif plot_key == "land_use":
        _plot_landuse_panel(
            axes[0],
            project1,
            left_label,
            landuse_col=kwargs.get("landuse_col"),
        )
        _plot_landuse_panel(
            axes[1],
            project2,
            right_label,
            landuse_col=kwargs.get("landuse_col"),
        )
        _set_same_extent(axes)

    elif plot_key == "dem":
        vmin = vmax = None
        if not kwargs.get("hillshade", False):
            values = []
            for project in (project1, project2):
                path = _require_existing_path(project.dem_path, "DEM")
                with rasterio.open(path) as src:
                    dem = src.read(1).astype(float)
                    if src.nodata is not None:
                        dem[dem == src.nodata] = np.nan
                    values.append(dem)
            vmin = min(np.nanmin(value) for value in values)
            vmax = max(np.nanmax(value) for value in values)
        im1 = _plot_dem_panel(
            axes[0],
            project1,
            left_label,
            hillshade=kwargs.get("hillshade", False),
            vmin=vmin,
            vmax=vmax,
        )
        im2 = _plot_dem_panel(
            axes[1],
            project2,
            right_label,
            hillshade=kwargs.get("hillshade", False),
            vmin=vmin,
            vmax=vmax,
        )
        if not kwargs.get("hillshade", False):
            fig.colorbar(im2, ax=axes, fraction=0.046, pad=0.02, label="Elevation (m)")
        _set_same_extent(axes)

    elif plot_key == "sewer_network":
        _plot_sewer_network_panel(axes[0], project1, left_label)
        _plot_sewer_network_panel(axes[1], project2, right_label)
        _set_same_extent(axes)

    elif plot_key == "final_design":
        diameter_values = []
        for project in (project1, project2):
            pipes = gpd.read_file(_require_existing_path(project.pipes_path, "sewer pipes"))
            diameter_values.append(_diameter_values_mm(pipes, kwargs.get("diameter_field")))
        all_diameters = pd.concat(diameter_values).dropna()
        norm = colors.Normalize(
            vmin=float(all_diameters.min()),
            vmax=float(all_diameters.max()),
        )
        cmap = kwargs.get("cmap", "viridis")
        _plot_design_panel(
            axes[0],
            project1,
            left_label,
            diameter_field=kwargs.get("diameter_field"),
            manhole_color_field=kwargs.get("manhole_color_field"),
            linewidth=kwargs.get("linewidth", 1.6),
            norm=norm,
            cmap=cmap,
        )
        _plot_design_panel(
            axes[1],
            project2,
            right_label,
            diameter_field=kwargs.get("diameter_field"),
            manhole_color_field=kwargs.get("manhole_color_field"),
            linewidth=kwargs.get("linewidth", 1.6),
            norm=norm,
            cmap=cmap,
        )
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02, label="Pipe diameter (mm)")
        _set_same_extent(axes)

    elif plot_key == "flow_components":
        df1 = _read_project_flows(project1, scenario_name=scenario_name)
        df2 = _read_project_flows(project2, scenario_name=scenario_name)
        _plot_flow_panel(axes[0], df1, left_label, start=start, end=end)
        _plot_flow_panel(axes[1], df2, right_label, start=start, end=end)
        ymin, ymax = [], []
        for ax in axes:
            low, high = ax.get_ylim()
            ymin.append(low)
            ymax.append(high)
        for ax in axes:
            ax.set_ylim(min(ymin), max(ymax))
        handles, legend_labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=4)
        fig.subplots_adjust(bottom=0.2)

    elif plot_key == "inflow":
        coefficient = kwargs.get("coefficient", 0.0001)
        inflow_values = []
        for project in (project1, project2):
            inp_path = _require_existing_path(
                _project_swmm_input_path(project, scenario_name=scenario_name),
                "SWMM input",
            )
            inflow_values.append(_read_inflow_from_swmm(inp_path, coefficient=coefficient)[2])
        all_inflows = np.concatenate(inflow_values)
        norm = colors.Normalize(vmin=float(np.min(all_inflows)), vmax=float(np.max(all_inflows)))
        sc1 = _plot_inflow_panel(
            axes[0],
            project1,
            left_label,
            scenario_name=scenario_name,
            coefficient=coefficient,
            norm=norm,
        )
        sc2 = _plot_inflow_panel(
            axes[1],
            project2,
            right_label,
            scenario_name=scenario_name,
            coefficient=coefficient,
            norm=norm,
        )
        fig.colorbar(sc2, ax=axes, fraction=0.046, pad=0.02, label="Baseline inflow (L/s)")
        _set_same_extent(axes)

    else:
        valid = [
            "layout",
            "roads",
            "land_use",
            "dem",
            "sewer_network",
            "final_design",
            "flow_components",
            "domain_mask",
            "inflow",
            "all",
        ]
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use one of: {valid}")

    fig.suptitle(kwargs.get("title", f"{plot_key.replace('_', ' ').title()} Comparison"), y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.08)
    if plot_key == "flow_components":
        fig.subplots_adjust(bottom=0.22)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig

__all__ = [
    "plot_board",
    "plot_filled_board_shapefile",
    "visualize_results",
    "plot_manholes",
    "visualize_sewer_network",
    "plot_sewer_network_all",
    "generate_clustered_rainfall_timeseries",
    "plot_flow_components_v2",
    "plot_ensemble_results",
    "plot_domain_mask",
    "plot_tetromino_set",
    "plot_roads",
    "plot_blocks_landuse",
    "plot_dem_tif",
    "plot_final_design_color_by_diameter",
    "plot_inflow_from_pipe_length",
    "plot_two_models",
]
