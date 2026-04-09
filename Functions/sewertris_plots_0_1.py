from __future__ import annotations
from collections import defaultdict
from scipy.spatial import cKDTree
from math import sqrt

# =======================
# Standard library
# =======================
import os
import random
import math
from typing import Dict, List, Tuple
from types import SimpleNamespace

# =======================
# Scientific stack
# =======================
import numpy as np
import pandas as pd
import networkx as nx
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from skimage.measure import label

# =======================
# Plotting
# =======================
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from matplotlib.colors import ListedColormap

# =======================
# Geospatial stack
# =======================
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.transform import from_origin

import osmnx as ox
from pyproj import CRS, Transformer

from shapely.errors import TopologicalError
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
    box,
)
from shapely.ops import unary_union, linemerge
from shapely.strtree import STRtree

# Shapely version compatibility (make_valid exists in Shapely >= 2.0)
try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None

from shapely.geometry import LineString
from shapely.strtree import STRtree
from collections import defaultdict
import networkx as nx
import random

# Function to download city boundary from OpenStreetMap
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
    if diameter_field is None:
        candidates = ["diameter_mm", "diam_mm", "diameter", "D_mm", "D", "pipe_diam_mm"]
        diameter_field = next((c for c in candidates if c in pipes.columns), None)
    if diameter_field is None:
        raise ValueError(f"Could not find a diameter field in pipes. Columns:\n{list(pipes.columns)}")

    pipes[diameter_field] = pd.to_numeric(pipes[diameter_field], errors="coerce")

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
    unique_d = np.sort(pipes[diameter_field].dropna().unique())

    if len(unique_d) <= 18:
        # Categorical colors with legend
        pipes["_diam_str"] = pipes[diameter_field].astype("Int64").astype(str) + " mm"
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
            column=diameter_field,
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