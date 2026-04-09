from __future__ import annotations
from collections import defaultdict
from scipy.spatial import cKDTree
from math import sqrt

# =======================
# Standard library
# =======================
import os
import random
from math import hypot
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


import sys
from pathlib import Path

SEWERTRIS_DIR = Path("..") / "Functions"   # adjust if needed
sys.path.append(str(SEWERTRIS_DIR))

# =======================
# SewerTris modules (your split files)
# =======================
import sewertris_plots_0_1 as plots
import sewertris_swmm_0_1 as swmm

# =======================
# Optional: reload during development
# =======================
import importlib
importlib.reload(plots)
importlib.reload(swmm)

# Function to download city boundary from OpenStreetMap
def download_city_boundary(city_name: str, save_path: str = "city_boundary.shp"):
    """
    Download the shapefile for the given city or town using OpenStreetMap (OSM).
    """
    # Download city boundary as GeoDataFrame
    gdf = ox.geocode_to_gdf(city_name)

    # Save to shapefile
    gdf.to_file(save_path)
    print(f"Shapefile saved to: {save_path}")

    return save_path


def utm_epsg_from_lon(lon, lat):
    """
    Very simple UTM estimator for WGS84.
    Northern hemisphere → EPSG:326xx
    Southern hemisphere → EPSG:327xx
    """
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"


def can_place(board, piece, pos):
    px, py = pos
    ph, pw = piece.shape
    bh, bw = board.shape
    if px+ph > bh or py+pw > bw:
        return False
    # Check if piece fits in mask and does not overlap
    if np.any((piece == 1) & (board[px:px+ph, py:py+pw] != 0)):
        return False
    return True


def place_piece(board, piece, pos, value):
    px, py = pos
    ph, pw = piece.shape
    board[px:px+ph, py:py+pw][piece==1] = value


def fill_domain_with_tetrominoes_and_blocks(domain_mask, tetrominoes):
    board = np.zeros_like(domain_mask, dtype=int)
    board[domain_mask == 0] = -1  # Mark outside domain

    tetro_keys = list(tetrominoes.keys())
    current_id = 1  # Start unique ID counter
    id_type_map = {}  # Optional: maps current_id to tetromino type (or "block")

    # Initial random fill
    empty_cells = np.argwhere((domain_mask == 1) & (board == 0))
    random.shuffle(empty_cells)

    for cell in empty_cells:
        x, y = cell
        if board[x, y] != 0:
            continue
        random.shuffle(tetro_keys)
        placed = False
        for tkey in tetro_keys:
            rotations = tetrominoes[tkey]
            random.shuffle(rotations)
            for piece in rotations:
                if can_place(board, piece, (x, y)):
                    place_piece(board, piece, (x, y), current_id)
                    id_type_map[current_id] = tkey
                    current_id += 1
                    placed = True
                    break
            if placed:
                break

    # Iterative fill
    changed = True
    while changed:
        changed = False
        empty_cells = np.argwhere((domain_mask == 1) & (board == 0))
        for cell in empty_cells:
            x, y = cell
            if board[x, y] != 0:
                continue
            for tkey in tetro_keys:
                for piece in tetrominoes[tkey]:
                    if can_place(board, piece, (x, y)):
                        place_piece(board, piece, (x, y), current_id)
                        id_type_map[current_id] = tkey
                        current_id += 1
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    # Fill remaining cells
    remaining = np.argwhere((domain_mask == 1) & (board == 0))
    for x, y in remaining:
        board[x, y] = current_id
        id_type_map[current_id] = "block"
        current_id += 1

    return board, id_type_map, current_id


def export_individual_figures_to_shapefile(
    filled_board, cell_size, output_path, id_to_type_map=None, crs="EPSG:3857",
    flip_y=False
):
    """
    Exports each unique tetromino or block as a polygon with a unique ID and optional label.
    
    Parameters:
    - filled_board (2D np.array): Grid with unique IDs for each shape
    - cell_size (float): Size of each square cell
    - output_path (str): File path for the output shapefile
    - id_to_type_map (dict or None): Optional. Maps each shape ID to a tetromino letter or "block"
    - crs (str): Coordinate Reference System (default EPSG:3857)
    - flip_y (bool): If True, invert the Y-axis (North/South flip)
    """
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box
    from shapely.ops import unary_union

    rows, cols = filled_board.shape
    geometries = []
    figure_ids = []
    tetro_id_list = []
    tetro_label_list = []

    unique_ids = np.unique(filled_board)
    unique_ids = unique_ids[unique_ids > 0]  # Skip background and -1

    for shape_id in unique_ids:
        cells = np.argwhere(filled_board == shape_id)
        cell_polys = []

        for i, j in cells:
            x0 = j * cell_size
            if flip_y:
                # No vertical flip: row index maps directly to Y
                y0 = i * cell_size
            else:
                # Original orientation: flip vertically so row 0 is top
                y0 = (rows - 1 - i) * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            cell_polys.append(box(x0, y0, x1, y1))

        merged = unary_union(cell_polys)

        geometries.append(merged)
        figure_ids.append(shape_id)
        tetro_id_list.append(shape_id)
        label = id_to_type_map.get(shape_id, "Unknown") if id_to_type_map else "Unknown"
        tetro_label_list.append(label)

    gdf = gpd.GeoDataFrame({
        "figure_id": figure_ids,
        "tetro_id": tetro_id_list,
        "label": tetro_label_list,
        "geometry": geometries
    }, crs=crs)

    gdf.to_file(output_path)
    print(f"✅ Exported {len(gdf)} figures to {output_path}")


def generate_road_network_from_blocks(blocks_path, road_width=10, simplify_tol=0.0):
    """
    Generates a road network from block polygon boundaries.

    Parameters:
    - blocks_path (str): Path to the block polygon shapefile.
    - road_width (float): Width of the road buffer (in CRS units, e.g., meters).
    - simplify_tol (float): Optional tolerance to simplify geometries before buffering.

    Returns:
    - road_centerlines (MultiLineString): Raw centerlines from polygon borders.
    - road_polygons (Polygon or MultiPolygon): Buffered road surface.
    - crs (CRS): The original CRS of the input shapefile.
    """
    gdf = gpd.read_file(blocks_path)
    # Do NOT change CRS — keep original
    crs = gdf.crs

    # Extract clean polygon boundaries
    boundaries = [make_valid(poly).boundary for poly in gdf.geometry if not poly.is_empty]

    # Merge all boundaries into one multilinestring
    merged_lines = linemerge(unary_union(boundaries))

    # Optionally simplify to reduce complexity
    if simplify_tol > 0:
        merged_lines = merged_lines.simplify(simplify_tol)

    # Buffer centerlines to get road polygons
    road_polygons = merged_lines.buffer(
        road_width / 2.0,
        cap_style=2,
        join_style=2,
        resolution=4
    )

    return merged_lines, road_polygons, crs


def extract_boundary(
    in_path: str,
    out_boundary_lines: str = None,
    out_outer_shell_polygon: str = None,
    keep_holes: bool = True
):
    """
    Extract boundary from a polygon/multipolygon layer.

    Parameters
    ----------
    in_path : str
        Input polygon shapefile/GeoPackage/GeoJSON path.
    out_boundary_lines : str, optional
        Output path for boundary as line(s) (e.g., 'boundary.shp').
        If None, lines aren’t written.
    out_outer_shell_polygon : str, optional
        Output path for outer shell polygon(s) only (holes removed)
        (e.g., 'outer_shell.shp'). If None, not written.
    keep_holes : bool
        If True, the line output includes both outer and inner rings (holes).
        If False, the line output includes only exteriors (outer rings).
    """
    gdf = gpd.read_file(in_path)
    if gdf.empty:
        raise ValueError("Input has no features.")

    crs = gdf.crs

    # Fix invalids (common with slivers/self-intersections)
    gdf = gdf.set_geometry(gdf.geometry.buffer(0))

    # Dissolve all polygons to a single geometry
    # (unary_union returns a shapely geometry directly)
    geom = unary_union([g for g in gdf.geometry if g is not None])

    # Normalize to MultiPolygon for uniform handling
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        # In case inputs weren’t pure polygons
        polys = [Polygon(p) for p in geom if isinstance(p, Polygon)]

    # ----- Boundary as lines -----
    if out_boundary_lines:
        line_parts = []
        for p in polys:
            # exterior ring
            line_parts.append(LineString(p.exterior.coords))
            if keep_holes:
                for interior in p.interiors:
                    line_parts.append(LineString(interior.coords))

        if len(line_parts) == 1:
            boundary_geom = line_parts[0]
        else:
            boundary_geom = MultiLineString(line_parts)

        gdf_lines = gpd.GeoDataFrame({"id": [1]}, geometry=[boundary_geom], crs=crs)
        gdf_lines.to_file(out_boundary_lines)
        print(f"[OK] Boundary lines written to: {out_boundary_lines}")

    # ----- Outer shell polygon(s) only (no holes) -----
    if out_outer_shell_polygon:
        shells = [Polygon(p.exterior) for p in polys]  # drop holes
        gdf_shells = gpd.GeoDataFrame({"id": range(1, len(shells)+1)}, geometry=shells, crs=crs)
        gdf_shells.to_file(out_outer_shell_polygon)
        print(f"[OK] Outer shell polygon(s) written to: {out_outer_shell_polygon}")

    # Return the shapely results in case you want them in-memory
    return {
        "merged_polygon": MultiPolygon(polys) if len(polys) > 1 else polys[0],
        "boundary_lines": boundary_geom if out_boundary_lines else None,
        "outer_shell_polygons": shells if out_outer_shell_polygon else None,
    }


def load_data(boundary_path, roads_path):
    """
    Load and validate input shapefiles
    """
    # Load shapefiles
    boundary = gpd.read_file(boundary_path)
    roads = gpd.read_file(roads_path)
    
    print(f"Original boundary CRS: {boundary.crs}")
    print(f"Original roads CRS: {roads.crs}")
    
    # If CRS is not set, try to set a default UTM zone based on the data
    if boundary.crs is None:
        # Assuming the data is in UTM zone 14N (Stillwater, OK)
        boundary.set_crs(epsg=32614, inplace=True)
        print("Set boundary CRS to EPSG:32614 (UTM Zone 14N)")
    
    if roads.crs is None:
        roads.set_crs(boundary.crs, inplace=True)
        print(f"Set roads CRS to match boundary: {boundary.crs}")
    
    # Ensure both datasets are in the same CRS
    if boundary.crs != roads.crs:
        print(f"Reprojecting roads to match boundary CRS: {boundary.crs}")
        roads = roads.to_crs(boundary.crs)
    
    # Print bounds for verification
    print(f"\nBoundary extent:")
    print(f"X range: {boundary.total_bounds[0]:.2f} to {boundary.total_bounds[2]:.2f}")
    print(f"Y range: {boundary.total_bounds[1]:.2f} to {boundary.total_bounds[3]:.2f}")
    
    return boundary, roads


def find_road_intersections(roads_gdf):
    """
    Find all road intersections to place manholes
    """
    # Convert roads to lines
    if roads_gdf.geometry.type[0] == 'Polygon' or roads_gdf.geometry.type[0] == 'MultiPolygon':
        # Extract all exterior boundaries
        boundaries = []
        for geom in roads_gdf.geometry:
            if geom.geom_type == 'Polygon':
                boundaries.append(geom.exterior)
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    boundaries.append(polygon.exterior)
        
        # Create a new GeoDataFrame with the boundaries
        roads_lines = gpd.GeoDataFrame(geometry=boundaries, crs=roads_gdf.crs)
    else:
        roads_lines = roads_gdf.copy()
    
    # Create network graph
    G = nx.Graph()
    for idx, row in roads_lines.iterrows():
        if row.geometry is not None:
            coords = list(row.geometry.coords)
            for i in range(len(coords)-1):
                G.add_edge(coords[i], coords[i+1])
    
    # Find intersection points
    intersections = []
    for node in G.nodes():
        if G.degree(node) > 2:  # More than 2 connections means intersection
            intersections.append(Point(node))
    
    return gpd.GeoSeries(intersections, crs=roads_gdf.crs)


def determine_outlet_point(boundary_gdf, direction='S'):
    """
    Find the outlet point based on the specified direction
    """
    bounds = boundary_gdf.total_bounds
    centroid = boundary_gdf.geometry.centroid.iloc[0]
    
    if direction == 'N':
        y = bounds[3]  # max y
        x = centroid.x
    elif direction == 'S':
        y = bounds[1]  # min y
        x = centroid.x
    elif direction == 'E':
        x = bounds[2]  # max x
        y = centroid.y
    elif direction == 'W':
        x = bounds[0]  # min x
        y = centroid.y
    else:
        raise ValueError("Direction must be one of: N, S, E, W")
    
    point = Point(x, y)
    # Ensure point is within or on boundary
    if not boundary_gdf.geometry.iloc[0].contains(point):
        point = boundary_gdf.geometry.iloc[0].boundary.interpolate(
            boundary_gdf.geometry.iloc[0].boundary.project(point)
        )
    
    return point


def generate_base_topography(boundary_gdf, roads_gdf, intersections, outlet_point, config):
    """
    Generate initial DEM with guaranteed drainage using geodesic distances and minimum slope enforcement
    """
    from shapely.geometry import Point
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    
    # Print boundary information for debugging
    print(f"Boundary CRS: {boundary_gdf.crs}")
    print(f"Boundary bounds: {boundary_gdf.total_bounds}")
    
    # Get the boundary polygon and add 100m offset
    boundary_poly = boundary_gdf.geometry.iloc[0]
    offset_boundary = boundary_poly.buffer(100.0)  # 100m offset
    bounds = offset_boundary.bounds
    
    # Calculate grid dimensions
    x_min = np.floor(bounds[0] / config.cell_size) * config.cell_size
    y_min = np.floor(bounds[1] / config.cell_size) * config.cell_size
    x_max = np.ceil(bounds[2] / config.cell_size) * config.cell_size
    y_max = np.ceil(bounds[3] / config.cell_size) * config.cell_size
    
    # Calculate number of cells
    width = int(round((x_max - x_min) / config.cell_size))
    height = int(round((y_max - y_min) / config.cell_size))
    
    # Create grid points - note the ordering of y coordinates
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_max, y_min, height)  # Reversed y-coordinates for GeoTIFF compatibility
    xx, yy = np.meshgrid(x, y)
    
    print(f"Grid shape: {xx.shape}")
    print(f"Grid extent:")
    print(f"  X: {x_min:.2f} to {x_max:.2f}")
    print(f"  Y: {y_min:.2f} to {y_max:.2f}")
    
    # 1. Create mask for points within offset boundary
    mask = np.zeros(xx.shape, dtype=bool)
    print("Creating mask...")
    for i in range(height):
        if i % 100 == 0:
            print(f"Processing row {i} of {height}")
        for j in range(width):
            point = Point(xx[i,j], yy[i,j])
            if offset_boundary.contains(point):
                mask[i,j] = True
    
    # Find outlet cell indices
    outlet_x, outlet_y = outlet_point.x, outlet_point.y
    outlet_i = np.abs(yy[:,0] - outlet_y).argmin()
    outlet_j = np.abs(xx[0,:] - outlet_x).argmin()
    print(f"Outlet cell: ({outlet_i}, {outlet_j})")
    
    # 2. Lock the boundary by creating a boundary mask
    boundary_mask = np.zeros_like(mask)
    boundary_mask[0,:] = mask[0,:]
    boundary_mask[-1,:] = mask[-1,:]
    boundary_mask[:,0] = mask[:,0]
    boundary_mask[:,-1] = mask[:,-1]
    boundary_mask = boundary_mask & mask  # Only keep boundary points within the domain
    
    # 3. Compute geodesic distances using Dijkstra's algorithm
    print("Computing geodesic distances...")
    
    # Create graph edges for 8-connected grid
    n = height * width
    row_ind, col_ind, data = [], [], []
    
    for i in range(height):
        for j in range(width):
            if not mask[i,j]:
                continue
                
            current_node = i * width + j
            
            # Check 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                        
                    ni, nj = i + di, j + dj
                    if (0 <= ni < height and 0 <= nj < width and mask[ni,nj]):
                        neighbor_node = ni * width + nj
                        distance = np.sqrt(di**2 + dj**2) * config.cell_size
                        row_ind.append(current_node)
                        col_ind.append(neighbor_node)
                        data.append(distance)
    
    # Create sparse graph
    graph = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    # Run Dijkstra's algorithm from outlet
    outlet_node = outlet_i * width + outlet_j
    distances, _ = dijkstra(graph, indices=[outlet_node], return_predecessors=True)
    dist = distances.reshape(height, width)
    
    # 4. Build monotonic base ramp with minimum slope
    min_slope = 0.002  # Minimum slope (2/1000 or 0.2%)
    ramp = dist * min_slope
    
    # 5. Create initial DEM with large-scale slope and enforce minimum slopes
    # Initial DEM creation similar to before but with guaranteed drainage
    radial_dist = np.sqrt((xx - outlet_x)**2 + (yy - outlet_y)**2)
    normalized_dist = (radial_dist - radial_dist.min()) / (radial_dist.max() - radial_dist.min())
    
    # Initial DEM
    Z0 = np.full_like(xx, np.nan, dtype=float)
    Z0[mask] = (
        config.min_elevation + 
        (config.max_elevation - config.min_elevation) * 
        normalized_dist[mask]
    )
    
    # Enforce minimum slopes using the ramp
    Z = np.maximum(Z0, config.min_elevation + ramp)
    
    # 6. Burn in roads if provided
    if roads_gdf is not None:
        print("Processing roads...")
        road_depth = 0.5  # meters
        sigma = 20.0  # meters
        
        for _, road in roads_gdf.iterrows():
            road_distances = np.zeros_like(xx)
            for i in range(height):
                for j in range(width):
                    if mask[i,j]:
                        point = Point(xx[i,j], yy[i,j])
                        road_distances[i,j] = point.distance(road.geometry)
            
            # Apply road depression
            road_effect = road_depth * np.exp(-road_distances / sigma)
            Z[mask] -= road_effect[mask]
        
        # Re-enforce minimum slopes
        Z = np.maximum(Z, config.min_elevation + ramp)
    
    # 7. Apply light smoothing while preserving drainage
    Z_smooth = gaussian_filter(
        np.where(np.isnan(Z), np.nanmean(Z), Z),
        sigma=0.8
    )
    
    # Blend smoothed and original while maintaining drainage
    Z_final = np.where(
        mask,
        0.8 * Z + 0.2 * Z_smooth,
        np.nan
    )
    
    # Final enforcement of minimum slopes
    Z_final = np.maximum(Z_final, config.min_elevation + ramp)
    
    # Ensure outlet is the lowest point
    local_min = np.nanmin(Z_final[
        max(0, outlet_i-1):min(height, outlet_i+2),
        max(0, outlet_j-1):min(width, outlet_j+2)
    ])
    Z_final[outlet_i, outlet_j] = local_min * 0.98
    
    return Z_final, xx, yy, mask


def optimize_drainage(elevation, xx, yy, mask, roads_gdf, intersections, config):
    """
    Optimize the topography for proper drainage
    """
    # Create depression at road intersections
    for point in intersections:
        # Find nearest grid points
        dist = np.sqrt((xx - point.x)**2 + (yy - point.y)**2)
        local_mask = dist < config.cell_size * 2
        if np.any(local_mask):
            elevation[local_mask] *= 0.95  # Create slight depression
    
    # Smooth the surface while preserving general slope
    elevation_smooth = gaussian_filter(
        elevation,
        sigma=config.smoothing_factor,
        mode='nearest'
    )
    
    # Ensure monotonic decrease towards outlet
    elevation[mask] = np.maximum(
        elevation[mask],
        elevation_smooth[mask]
    )
    
    return elevation


def validate_drainage(elevation, xx, yy, mask, outlet_point):
    """
    Validate that water can drain from all points to the outlet
    """
    # Calculate slopes
    dy, dx = np.gradient(elevation, yy[:,0], xx[0,:])
    slope = np.sqrt(dx**2 + dy**2)
    
    # Flow direction (D8 algorithm simplified)
    flow_dir = np.zeros_like(elevation)
    for i in range(1, elevation.shape[0]-1):
        for j in range(1, elevation.shape[1]-1):
            if mask[i,j]:
                # Check 8 neighbors
                neighborhood = elevation[i-1:i+2, j-1:j+2]
                if np.any(neighborhood < elevation[i,j]):
                    flow_dir[i,j] = 1
    
    return slope, flow_dir


def generate_topography(boundary_path, roads_path, config=None):
    """
    Main function to generate topography
    """
    if config is None:
        config = TopographyConfig()
    
    # Load data
    boundary_gdf, roads_gdf = load_data(boundary_path, roads_path)
    
    # Find road intersections for manhole locations
    intersections = find_road_intersections(roads_gdf)
    
    # Determine outlet point
    outlet_point = determine_outlet_point(boundary_gdf, config.outlet_direction)
    
    # Generate base topography
    elevation, xx, yy, mask = generate_base_topography(
        boundary_gdf, roads_gdf, intersections, outlet_point, config
    )
    
    # Optimize for drainage
    elevation = optimize_drainage(
        elevation, xx, yy, mask, roads_gdf, intersections, config
    )
    
    # Validate drainage
    slope, flow_dir = validate_drainage(elevation, xx, yy, mask, outlet_point)
    
    # Visualize results
    plots.visualize_results(
        elevation, slope, flow_dir, mask,
        boundary_gdf, roads_gdf, intersections, outlet_point, xx, yy
    )
    
    return elevation, xx, yy, mask


def load_blocks_and_roads(blocks_path, roads_path):
    blocks_gdf = gpd.read_file(blocks_path)
    roads_gdf = gpd.read_file(roads_path)
    
    # Ensure same CRS
    if blocks_gdf.crs != roads_gdf.crs:
        roads_gdf = roads_gdf.to_crs(blocks_gdf.crs)
    
    # Merge all road geometries into one
    road_network = roads_gdf.geometry.union_all()
    
    # Convert block polygons to block objects
    blocks = []
    for geom in blocks_gdf.geometry:
        blocks.append(SimpleNamespace(original_poly=geom.buffer(0)))  # Safe make_valid alternative

    return blocks, road_network, blocks_gdf.crs


def cut_blocks(blocks, road_network):
    for block in blocks:
        original = block.original_poly
        if original is None or original.is_empty:
            continue

        try:
            trimmed = original.difference(road_network)
            block.final_poly = trimmed if not trimmed.is_empty else None
        except TopologicalError:
            try:
                block.final_poly = original.buffer(0.1).difference(road_network.buffer(0.1))
            except Exception as e:
                block.final_poly = None
                print(f"[cut_blocks] Failed to cut block: {e}")
    return blocks


def assign_land_use_compact(blocks, land_use_distribution=None, seed=42):
    """
    Assign land use in a more compact / zoned way:
      - COMMERCIAL: near the city center (compact core)
      - INDUSTRIAL: near the outer edge (periphery)
      - RESIDENTIAL: in between core and edge (mainly)
      - PUBLIC, RECREATIONAL: sprinkled on top of RES (as before)

    blocks: list of SimpleNamespace-like objects with .final_poly geometry
    land_use_distribution: optional dict with fractions for each use.
                           Default: same as before.
    """
    random.seed(seed)

    # === 1. Land use config ===
    land_use_types = {
        'RESIDENTIAL': {'color': '#FF9999'},
        'COMMERCIAL':  {'color': '#99CCFF'},
        'INDUSTRIAL':  {'color': '#FFCC99'},
        'PUBLIC':      {'color': '#99FF99'},
        'RECREATIONAL':{'color': '#CC99FF'}
    }

    if land_use_distribution is None:
        land_use_distribution = {
            'RESIDENTIAL': 0.5,
            'COMMERCIAL':  0.2,
            'INDUSTRIAL':  0.15,
            'PUBLIC':      0.1,
            'RECREATIONAL':0.05
        }

    # === 2. Valid blocks & centroids ===
    valid_blocks = []
    for b in blocks:
        geom = getattr(b, "final_poly", None)
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        valid_blocks.append((b, c.x, c.y))

    if not valid_blocks:
        return blocks

    # city center = mean of centroids
    cx = sum(x for _, x, _ in valid_blocks) / len(valid_blocks)
    cy = sum(y for _, _, y in valid_blocks) / len(valid_blocks)

    # compute distance to center
    blocks_with_dist = []
    for b, x, y in valid_blocks:
        d = hypot(x - cx, y - cy)
        blocks_with_dist.append((b, d))

    # sort by distance: center → edge
    blocks_with_dist.sort(key=lambda t: t[1])
    sorted_blocks = [b for b, _ in blocks_with_dist]
    N = len(sorted_blocks)

    # === 3. Target counts per category (fractions) ===
    frac_res = land_use_distribution.get('RESIDENTIAL', 0.5)
    frac_com = land_use_distribution.get('COMMERCIAL',  0.2)
    frac_ind = land_use_distribution.get('INDUSTRIAL',  0.15)
    frac_pub = land_use_distribution.get('PUBLIC',      0.1)
    frac_rec = land_use_distribution.get('RECREATIONAL',0.05)

    # We treat PUBLIC + RECREATIONAL as a subset of RESIDENTIAL band.
    # First decide how many blocks we want for COM + IND:
    n_com = int(N * frac_com)
    n_ind = int(N * frac_ind)

    # sanity
    if n_com + n_ind >= N:
        # degenerate case: fall back to simple distribution
        return _assign_land_use_simple(blocks, land_use_distribution, land_use_types)

    # number of blocks we can label as RES initially
    n_res_band = N - n_com - n_ind

    # PUBLIC + RECREATIONAL will come from inside this RES band
    n_pub = int(N * frac_pub)
    n_rec = int(N * frac_rec)
    n_pub_rec = n_pub + n_rec

    if n_pub_rec > n_res_band:
        # too many public+rec asked, clamp
        n_pub_rec = n_res_band
        # split roughly
        n_pub = n_pub_rec // 2
        n_rec = n_pub_rec - n_pub

    # === 4. First pass: COM core, RES middle, IND edge ===

    # center commercial
    com_indices = list(range(0, n_com))
    # edge industrial
    ind_indices = list(range(N - n_ind, N))
    # middle residential band
    res_indices_band = list(range(n_com, N - n_ind))

    # init all as None
    for b in sorted_blocks:
        b.land_use = None
        b.color = None

    # assign commercial
    for idx in com_indices:
        b = sorted_blocks[idx]
        b.land_use = 'COMMERCIAL'
        b.color = land_use_types['COMMERCIAL']['color']

    # assign industrial
    for idx in ind_indices:
        b = sorted_blocks[idx]
        b.land_use = 'INDUSTRIAL'
        b.color = land_use_types['INDUSTRIAL']['color']

    # assign residential in the middle band (initially all RES)
    for idx in res_indices_band:
        b = sorted_blocks[idx]
        b.land_use = 'RESIDENTIAL'
        b.color = land_use_types['RESIDENTIAL']['color']

    # === 5. Turn some RES blocks into PUBLIC and RECREATIONAL (sprinkled) ===
    res_only_indices = [i for i in res_indices_band if sorted_blocks[i].land_use == 'RESIDENTIAL']

    random.shuffle(res_only_indices)

    # public first
    for idx in res_only_indices[:n_pub]:
        b = sorted_blocks[idx]
        b.land_use = 'PUBLIC'
        b.color = land_use_types['PUBLIC']['color']

    # recreational next
    for idx in res_only_indices[n_pub:n_pub + n_rec]:
        b = sorted_blocks[idx]
        b.land_use = 'RECREATIONAL'
        b.color = land_use_types['RECREATIONAL']['color']

    return blocks


def _assign_land_use_simple(blocks, land_use_distribution, land_use_types):
    """
    Fallback: original non-compact strategy if something goes wrong.
    """
    valid_blocks = [b for b in blocks if getattr(b, "final_poly", None) and not b.final_poly.is_empty]
    sorted_blocks = sorted(valid_blocks, key=lambda b: b.final_poly.area, reverse=True)
    n = len(sorted_blocks)

    land_use_counts = {lu: int(n * pct) for lu, pct in land_use_distribution.items()}
    remaining = n - sum(land_use_counts.values())
    lu_keys = list(land_use_distribution.keys())
    for i in range(remaining):
        land_use_counts[lu_keys[i % len(lu_keys)]] += 1

    for b in sorted_blocks:
        for lu in lu_keys:
            if land_use_counts[lu] > 0:
                b.land_use = lu
                b.color = land_use_types[lu]['color']
                land_use_counts[lu] -= 1
                break
    return blocks


def export_to_shapefile(blocks, crs, output_path):
    gdf = gpd.GeoDataFrame({
        "land_use": [b.land_use for b in blocks if b.final_poly],
        "color": [b.color for b in blocks if b.final_poly],
        "geometry": [b.final_poly for b in blocks if b.final_poly]
    }, crs=crs)
    gdf.to_file(output_path)
    print(f"✅ Exported to {output_path}")
    return gdf


def extract_manholes_from_lines(road_axes_path, dem_path):
    """
    Extract manholes at each vertex (bend) from road centerlines and assign elevation using DEM.

    Parameters:
    - road_axes_path (str): Path to the road axes shapefile (LineString/MultiLineString).
    - dem_path (str): Path to the elevation GeoTIFF.

    Returns:
    - List[dict]: Each dict has 'id', 'location' (Point), 'elevation'.
    """
    # 1. Load shapefile
    gdf = gpd.read_file(road_axes_path)
    crs = gdf.crs

    # 2. Extract vertices from all LineStrings
    points = []
    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            points.extend(list(geom.coords))
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                points.extend(list(line.coords))

    # 3. Convert to unique Point objects
    unique_pts = list({(round(x, 3), round(y, 3)) for x, y in points})  # deduplicated with 3-digit precision
    point_objs = [Point(xy) for xy in unique_pts]

    # 4. Sample elevations from DEM
    with rasterio.open(dem_path) as src:
        if src.crs != crs:
            raise ValueError("CRS mismatch: DEM and shapefile must have the same CRS.")
        coords = [(pt.x, pt.y) for pt in point_objs]
        samples = list(src.sample(coords))
        elevations = [float(v[0]) if v[0] is not None else 0.0 for v in samples]

    # 5. Create manhole dictionaries
    manholes = [{
        'id': f"MH{i+1:03d}",
        'location': pt,
        'elevation': elev,
        'connections': []
    } for i, (pt, elev) in enumerate(zip(point_objs, elevations))]

    print(f"✅ Extracted {len(manholes)} manholes from road centerlines.")
    return manholes


def export_manholes_to_shapefile(manholes, output_path, crs):
    """
    Export manholes to a shapefile.

    Parameters:
    - manholes (list): List of manhole dicts with 'id', 'location', and 'elevation'.
    - output_path (str): Path to the output shapefile (e.g., "manholes.shp").
    - crs (CRS or str): Coordinate reference system to assign (e.g., EPSG code).
    """
    gdf = gpd.GeoDataFrame({
        'id': [mh['id'] for mh in manholes],
        'elevation': [mh['elevation'] for mh in manholes],
        'geometry': [mh['location'] for mh in manholes]
    }, crs=crs)

    gdf.to_file(output_path)
    print(f"✅ Manholes exported to {output_path}")


def generate_main_sewer_path(
    manholes,
    road_buffer,
    block_size=40.0,
    slope_tolerance=-0.01,
    min_pipe_length=5.0,
    prefer_slope=0.6,
):
    """
    Build a graph of admissible manhole-to-manhole connections, then find the
    best path from the highest manhole (head) to the lowest/boundary outlet.
    If the chosen outlet is unreachable, pick the lowest reachable manhole.

    Parameters
    ----------
    manholes : list[dict]
        [{'id': ..., 'location': Point, 'elevation': float}, ...]
    road_buffer : Polygon or MultiPolygon
    block_size : float
        Used for neighbor search radius and 'no crossing' buffer
    slope_tolerance : float
        Minimum allowed slope (e.g. -0.01 = allow up to 1% adverse)
    min_pipe_length : float
        Minimum segment length (m)
    prefer_slope : float
        0..1, weight to favor higher slopes in cost function

    Returns
    -------
    segments : list[(id_from, id_to)]
    path_info : dict
    """
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.strtree import STRtree
    import numpy as np
    import networkx as nx

    # normalize road buffer
    if isinstance(road_buffer, Polygon):
        road_polys = [road_buffer]
    elif isinstance(road_buffer, MultiPolygon):
        road_polys = list(road_buffer.geoms)
    else:
        raise ValueError("road_buffer must be Polygon or MultiPolygon")

    def line_in_road(line):
        for poly in road_polys:
            if poly.covers(line):
                return True
        return False

    # index manholes
    id_to_mh = {mh["id"]: mh for mh in manholes}
    points = [mh["location"] for mh in manholes]
    mh_tree = STRtree(points)
    idx_to_id = {i: manholes[i]["id"] for i in range(len(manholes))}
    all_ids = list(id_to_mh.keys())

    # pick head = highest
    head = max(manholes, key=lambda m: m["elevation"])["id"]
    # pick outlet = lowest
    outlet = min(manholes, key=lambda m: m["elevation"])["id"]

    # build graph
    G = nx.Graph()

    # add all manholes as nodes with elevation
    for mh in manholes:
        G.add_node(mh["id"], elevation=mh["elevation"], geom=mh["location"])

    # build edges (candidate neighbors from STRtree)
    # we can do multi-scale search once for all pairs
    # to keep it simple: for each manhole, look around 4*block_size
    max_radius = block_size * 4.0
    no_cross_radius = block_size * 0.3

    for mh in manholes:
        mh_id = mh["id"]
        pt = mh["location"]
        elev = mh["elevation"]

        nearby_idxs = mh_tree.query(pt.buffer(max_radius))
        for idx in nearby_idxs:
            nbr_id = idx_to_id[idx]
            if nbr_id == mh_id:
                continue

            nbr = id_to_mh[nbr_id]
            nbr_pt = nbr["location"]
            dist = pt.distance(nbr_pt)
            if dist < min_pipe_length:
                continue

            # build line
            line = LineString([pt, nbr_pt])

            # must follow road
            if not line_in_road(line):
                continue

            # avoid crossing through other manholes
            crosses = False
            for other_id in all_ids:
                if other_id in (mh_id, nbr_id):
                    continue
                other_pt = id_to_mh[other_id]["location"]
                if line.distance(other_pt) < no_cross_radius:
                    # check point lies inside segment
                    proj = line.project(other_pt)
                    if 0 < proj < line.length:
                        crosses = True
                        break
            if crosses:
                continue

            # slope from mh -> nbr
            slope = (elev - nbr["elevation"]) / max(dist, 1e-6)
            if slope < slope_tolerance:
                # this edge is too adverse, skip
                continue

            # define cost: we want shorter AND better slope
            # lower cost = better
            # base on distance
            base_cost = dist
            # slope bonus (bigger slope → smaller cost)
            # normalize slope to something reasonable
            slope_bonus = (1.0 - prefer_slope * max(0.0, slope))  # shrink cost if slope>0
            cost = base_cost * slope_bonus

            # add undirected (or directed?) -- sewer is directional, so we can add directed
            G.add_edge(mh_id, nbr_id,
                       distance=dist,
                       slope=slope,
                       cost=cost)

    # now we try to route from head → outlet
    reachable = nx.descendants(nx.DiGraph(G.to_directed()), head)
    reachable.add(head)

    if outlet not in reachable:
        # outlet not reachable → pick the lowest reachable manhole
        # (closest to what we wanted)
        reachable_mhs = [mid for mid in reachable]
        lowest_reachable = min(reachable_mhs, key=lambda mid: id_to_mh[mid]["elevation"])
        print(f"⚠️ Outlet {outlet} unreachable, using lowest reachable manhole {lowest_reachable}")
        target = lowest_reachable
    else:
        target = outlet

    # run shortest path on 'cost'
    try:
        path_ids = nx.shortest_path(G, source=head, target=target, weight="cost")
    except nx.NetworkXNoPath:
        # fallback: just return head
        print("❌ No path found at all.")
        return [], {
            "segments": [],
            "slopes": {},
            "distances": {},
            "cumulative_drop": 0.0,
            "total_length": 0.0,
        }

    # build path_info from path_ids
    segments = []
    slopes = {}
    distances = {}
    cumulative_drop = 0.0
    total_length = 0.0

    for u, v in zip(path_ids[:-1], path_ids[1:]):
        data = G.get_edge_data(u, v)
        seg = (u, v)
        segments.append(seg)
        slopes[seg] = data["slope"]
        distances[seg] = data["distance"]
        elev_drop = id_to_mh[u]["elevation"] - id_to_mh[v]["elevation"]
        cumulative_drop += elev_drop
        total_length += data["distance"]

    avg_slope = (sum(slopes.values()) / len(slopes)) if slopes else 0.0

    print("\nPath Statistics:")
    print(f"Head: {head}  →  Target: {target}")
    print(f"Segments: {len(segments)}")
    print(f"Total length: {total_length:.1f} m")
    print(f"Total drop: {cumulative_drop:.2f} m")
    print(f"Avg slope: {avg_slope:.3%}")

    path_info = {
        "segments": segments,
        "slopes": slopes,
        "distances": distances,
        "cumulative_drop": cumulative_drop,
        "total_length": total_length,
    }
    return segments, path_info


def generate_secondary_pipes(manholes, main_path, road_buffer, block_size=40.0, slope_tolerance=0.0):
    """
    Connect unconnected manholes to the main path with valid secondary pipes,
    avoiding overlap with the main path and illegal crossings.

    Parameters:
    - manholes: list of manhole dicts
    - main_path: list of (from_id, to_id) tuples
    - road_buffer: shapely Polygon representing allowed pipe area
    - block_size: search radius
    - slope_tolerance: minimum allowed slope (e.g., 0 for downhill only)

    Returns:
    - secondary_pipes: list of (from_id, to_id) tuples
    """
    id_to_mh = {mh['id']: mh for mh in manholes}
    all_ids = list(id_to_mh.keys())
    all_pts = [mh['location'] for mh in manholes]
    idx_to_id = {i: all_ids[i] for i in range(len(all_ids))}
    tree = STRtree(all_pts)

    connected_ids = set(u for u, v in main_path) | set(v for u, v in main_path)
    unconnected = set(all_ids) - connected_ids
    secondary_pipes = []

    # Main pipe geometries
    main_lines = [LineString([id_to_mh[u]['location'], id_to_mh[v]['location']]) for u, v in main_path]
    main_tree = STRtree(main_lines)

    def crosses_other_manhole(line, exclude_ids):
        buffer_radius = block_size * 0.3
        candidates = tree.query(line.buffer(buffer_radius))
        for idx in candidates:
            other_id = idx_to_id[idx]
            if other_id in exclude_ids:
                continue
            other_pt = id_to_mh[other_id]['location']
            proj = line.project(other_pt)
            if 0 < proj < line.length:
                return True
        return False

    def overlaps_main_path(line, tolerance=0.01):
        for seg in main_tree.query(line):
            if not isinstance(seg, LineString):
                continue
            if line.equals_exact(seg, tolerance):
                return True
            if line.relate_pattern(seg, "1********"):
                return True
        return False

    while unconnected:
        newly_connected = set()
        for uid in list(unconnected):
            src = id_to_mh[uid]
            pt = src['location']
            elev = src['elevation']
            candidates = tree.query(pt.buffer(block_size * 2))

            best = None
            best_slope = -np.inf

            for idx in candidates:
                tid = idx_to_id[idx]
                if tid not in connected_ids or tid == uid:
                    continue
                target = id_to_mh[tid]
                line = LineString([pt, target['location']])
                if not road_buffer.covers(line):
                    continue
                if crosses_other_manhole(line, exclude_ids={uid, tid}):
                    continue
                if overlaps_main_path(line):
                    continue
                dist = pt.distance(target['location'])
                if dist < 1e-3:
                    continue
                slope = (elev - target['elevation']) / dist
                if slope > best_slope:
                    best = tid
                    best_slope = slope

            if best and best_slope >= slope_tolerance:
                secondary_pipes.append((uid, best))
                newly_connected.add(uid)

        if not newly_connected:
            print("⚠️ Some manholes couldn’t be connected while preserving constraints.")
            break

        connected_ids.update(newly_connected)
        unconnected -= newly_connected

    print(f"✅ Generated {len(secondary_pipes)} secondary pipes (tree structure preserved).")
    return secondary_pipes


def remove_secondary_pipes_overlapping_main(manholes, secondary_pipes, main_pipes, tolerance=0.01):
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    id_map = {mh['id']: mh for mh in manholes}
    main_lines = [LineString([id_map[u]['location'], id_map[v]['location']]) for u, v in main_pipes]
    main_tree = STRtree(main_lines)

    cleaned_secondary = []

    for u, v in secondary_pipes:
        line = LineString([id_map[u]['location'], id_map[v]['location']])
        try:
            p0, p1 = line.boundary.geoms  # ✅ fixed unpacking
        except Exception as e:
            print(f"⚠️ Invalid geometry for line {u}->{v}: {e}")
            continue

        overlap = False

        for seg in main_tree.query(line):
            if not isinstance(seg, LineString):
                continue

            if not line.intersects(seg):
                continue

            intersection = line.intersection(seg)
            if intersection.is_empty:
                continue

            # Allow touching only at endpoints
            if intersection.geom_type == "Point":
                if intersection.equals(p0) or intersection.equals(p1) or \
                   intersection.equals(seg.boundary[0]) or intersection.equals(seg.boundary[1]):
                    continue
                else:
                    overlap = True
                    break

            elif intersection.geom_type == "MultiPoint":
                if all(pt.equals(p0) or pt.equals(p1) or 
                       pt.equals(seg.boundary[0]) or pt.equals(seg.boundary[1]) 
                       for pt in intersection.geoms):
                    continue
                else:
                    overlap = True
                    break

            else:
                # Anything more than touching is considered overlap
                overlap = True
                break

        if not overlap:
            cleaned_secondary.append((u, v))
        else:
            print(f"❌ Removed overlapping secondary pipe: {u} → {v}")

    print(f"✅ Cleaned: {len(secondary_pipes) - len(cleaned_secondary)} secondary pipes removed.")
    return cleaned_secondary


def export_pipes_to_shapefile(pipes, manholes, output_path, crs):
    """
    Export sewer pipes (main, secondary, or tertiary) to a shapefile.

    Parameters:
    - pipes: list of (from_id, to_id) tuples
    - manholes: list of dicts with 'id' and 'location'
    - output_path: output shapefile path
    - crs: CRS to assign (use same CRS as manholes shapefile)
    """
    id_map = {mh['id']: mh for mh in manholes}
    geometries = []
    from_ids = []
    to_ids = []

    for u, v in pipes:
        p1 = id_map[u]['location']
        p2 = id_map[v]['location']
        geometries.append(LineString([p1, p2]))
        from_ids.append(u)
        to_ids.append(v)

    gdf = gpd.GeoDataFrame({
        'from_id': from_ids,
        'to_id': to_ids,
        'geometry': geometries
    }, crs=crs)

    gdf.to_file(output_path)
    print(f"✅ Pipes exported to {output_path}")


def generate_tertiary_pipes(manholes, main_path, secondary_pipes, road_buffer, block_size=60.0):
    id_map = {str(mh['id']): mh for mh in manholes}
    point_map = {str(mh['id']): mh['location'] for mh in manholes}
    point_list = list(point_map.values())
    id_list = list(point_map.keys())
    tree = STRtree(point_list)

    connected = set(str(u) for u, v in main_path + secondary_pipes) | set(str(v) for u, v in main_path + secondary_pipes)
    has_outlet = set(str(u) for u, v in main_path + secondary_pipes)

    raw_tertiary = []
    used_sources = set()
    candidates = [mid for mid in point_map if mid not in has_outlet]

    for radius_factor in [3, 5, 8]:
        radius = block_size * radius_factor
        still_unconnected = []

        for uid in candidates:
            if uid in used_sources:
                continue

            u_pt = point_map[uid]
            u_elev = id_map[uid]['elevation']
            nearby_idxs = tree.query(u_pt.buffer(radius))

            best_slope = -float("inf")
            best_vid = None

            for idx in nearby_idxs:
                v_pt = point_list[idx]
                vid = id_list[idx]

                if vid == uid or vid in used_sources or vid not in connected:
                    continue

                v_elev = id_map[vid]['elevation']
                line = LineString([u_pt, v_pt])
                if not road_buffer.contains(line):
                    continue

                dist = u_pt.distance(v_pt)
                if dist < 1e-3:
                    continue

                slope = (u_elev - v_elev) / (dist + 1e-6)
                if slope > best_slope:
                    best_slope = slope
                    best_vid = vid

            if best_vid:
                line = LineString([point_map[uid], point_map[best_vid]])
                possible_idxs = tree.query(line.buffer(0.01))
                intermediate_ids = []

                for idx in possible_idxs:
                    pt = point_list[idx]
                    mid = id_list[idx]
                    if mid in {uid, best_vid}:
                        continue
                    if line.intersects(pt):
                        intermediate_ids.append(mid)

                if intermediate_ids:
                    all_pts = [uid] + sorted(intermediate_ids, key=lambda m: point_map[uid].distance(point_map[m])) + [best_vid]
                    for u_seg, v_seg in zip(all_pts[:-1], all_pts[1:]):
                        raw_tertiary.append((u_seg, v_seg))
                        connected.add(u_seg)
                        has_outlet.add(u_seg)
                else:
                    raw_tertiary.append((uid, best_vid))
                    connected.add(uid)
                    has_outlet.add(uid)

                used_sources.add(uid)
            else:
                still_unconnected.append(uid)

        candidates = still_unconnected

    # Fallback: connect remaining manholes without slope
    previous_unlinked = set(candidates)

    while True:
        newly_connected = []
        remaining = []

        for uid in previous_unlinked:
            if uid in has_outlet:
                continue

            u_pt = point_map[uid]
            min_dist = float("inf")
            best_vid = None

            for vid in connected:
                if vid == uid:
                    continue
                v_pt = point_map[vid]
                line = LineString([u_pt, v_pt])
                if not road_buffer.contains(line):
                    continue
                dist = u_pt.distance(v_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_vid = vid

            if best_vid:
                line = LineString([point_map[uid], point_map[best_vid]])
                possible_idxs = tree.query(line.buffer(0.01))
                intermediate_ids = []

                for idx in possible_idxs:
                    pt = point_list[idx]
                    mid = id_list[idx]
                    if mid in {uid, best_vid}:
                        continue
                    if line.intersects(pt):
                        intermediate_ids.append(mid)

                if intermediate_ids:
                    all_pts = [uid] + sorted(intermediate_ids, key=lambda m: point_map[uid].distance(point_map[m])) + [best_vid]
                    for u_seg, v_seg in zip(all_pts[:-1], all_pts[1:]):
                        raw_tertiary.append((u_seg, v_seg))
                        connected.add(u_seg)
                        has_outlet.add(u_seg)
                else:
                    raw_tertiary.append((uid, best_vid))
                    connected.add(uid)
                    has_outlet.add(uid)

                newly_connected.append(uid)
            else:
                remaining.append(uid)

        if not newly_connected:
            break
        previous_unlinked = remaining

    # Step 3: DAG pruning (no cycles allowed)
    G = nx.DiGraph()
    G.add_edges_from((str(u), str(v)) for u, v in main_path + secondary_pipes)

    final_tertiary = []
    removed_cycles = 0

    for u, v in raw_tertiary:
        u, v = str(u), str(v)
        if G.has_node(u) and G.has_node(v) and nx.has_path(G, v, u):
            removed_cycles += 1
        else:
            G.add_edge(u, v)
            final_tertiary.append((u, v))

    print(f"✅ Tertiary pipes generated (raw): {len(raw_tertiary)}")
    print(f"🧹 Tertiary pipes removed to prevent cycles: {removed_cycles}")

    # Step 4: Conflict pruning — one outlet per manhole, priority to main/secondary
    outlet_map = defaultdict(list)
    for u, v in main_path:
        outlet_map[str(u)].append((str(u), str(v), 'main'))
    for u, v in secondary_pipes:
        outlet_map[str(u)].append((str(u), str(v), 'secondary'))
    for u, v in final_tertiary:
        outlet_map[str(u)].append((str(u), str(v), 'tertiary'))

    final_pruned_tertiary = set(final_tertiary)
    removed_conflicts = 0

    for uid, pipes in outlet_map.items():
        if len(pipes) <= 1:
            continue
        mains = [p for p in pipes if p[2] == 'main']
        secs = [p for p in pipes if p[2] == 'secondary']
        ter = [p for p in pipes if p[2] == 'tertiary']

        if mains or secs:
            for p in ter:
                if (p[0], p[1]) in final_pruned_tertiary:
                    final_pruned_tertiary.discard((p[0], p[1]))
                    removed_conflicts += 1
        elif len(ter) > 1:
            to_keep = random.choice(ter)
            for p in ter:
                if p != to_keep and (p[0], p[1]) in final_pruned_tertiary:
                    final_pruned_tertiary.discard((p[0], p[1]))
                    removed_conflicts += 1

    print(f"🚫 Tertiary pipes removed due to multiple outlet conflicts: {removed_conflicts}")
    print(f"🌳 Final tree-safe tertiary pipes: {len(final_pruned_tertiary)}")

    return list(final_pruned_tertiary), [mid for mid in point_map if mid not in G.nodes]


def export_tertiary_pipes_to_shapefile(manholes, tertiary_pipes, output_path, crs="EPSG:32614"):
    """
    Export tertiary pipes to a shapefile.

    Parameters:
    - manholes: list of dicts containing 'id' and 'location'
    - tertiary_pipes: list of (from_id, to_id) tuples
    - output_path: full path to the .shp file (e.g., "outputs/tertiary_pipes.shp")
    - crs: coordinate reference system (default: UTM zone 14N)
    """
    if not tertiary_pipes:
        print("⚠️ No tertiary pipes to export.")
        return

    id_map = {mh["id"]: mh for mh in manholes}
    
    records = []
    for u, v in tertiary_pipes:
        line = LineString([id_map[u]["location"], id_map[v]["location"]])
        records.append({
            "from_id": u,
            "to_id": v,
            "geometry": line
        })

    gdf = gpd.GeoDataFrame(records, crs=crs)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path)
    print(f"✅ Tertiary pipes exported to: {output_path}")


def export_pipes_to_shapefile(pipes_main, pipes_sec, pipes_ter, manholes, output_path, crs="EPSG:32618"):
    """
    Export all pipes (main, secondary, tertiary) to a single shapefile with:
    - pipe_id, upstream_m, downstream_m, type, geometry
    - Custom CRS support
    
    Parameters:
    - pipes_main, pipes_sec, pipes_ter: list of (upstream, downstream) tuples
    - manholes: list of dicts with 'id' and 'location' (Shapely Point)
    - output_path: path to save the shapefile
    - crs: coordinate reference system (default EPSG:32618 = UTM Zone 18N)
    """
    # Build manhole ID → Point lookup
    manhole_points = {str(mh['id']): mh['location'] for mh in manholes}

    def build_records(pipe_list, pipe_type, start_idx):
        records = []
        for i, (u, v) in enumerate(pipe_list):
            u, v = str(u), str(v)
            if u in manhole_points and v in manhole_points:
                records.append({
                    'pipe_id': f"P{start_idx + i:04d}",
                    'upstream_m': u,
                    'downstream_m': v,
                    'type': pipe_type,
                    'geometry': LineString([manhole_points[u], manhole_points[v]])
                })
        return records

    rec_main = build_records(pipes_main, 'main', 0)
    rec_sec = build_records(pipes_sec, 'secondary', len(rec_main))
    rec_ter = build_records(pipes_ter, 'tertiary', len(rec_main) + len(rec_sec))

    all_records = rec_main + rec_sec + rec_ter
    gdf = gpd.GeoDataFrame(all_records, crs=crs)
    gdf.to_file(output_path)

    print(f"✅ Exported {len(all_records)} pipes to: {output_path} with CRS: {crs}")


def modify_topography_with_sewers(
    dem_path,
    sewer_path,
    output_path=None,
    buffer_pixels=2,
    densify_step_factor=1.0,
    smoothing_sigma=1.5,
    downhill_bias=0.0,
):
    """
    Carve the DEM along sewer centerlines to create a continuous, gently
    downhill surface toward pipe outlets, and blend smoothly into the
    surrounding terrain.

    Parameters
    ----------
    dem_path : str
        Input DEM GeoTIFF.
    sewer_path : str
        Sewer network (lines) as Shapefile/GeoPackage/etc.
    output_path : str or None
        Where to save the modified DEM. If None, overwrite the input.
    buffer_pixels : int
        Half-width of the trench around pipes in pixels (>=1).
    densify_step_factor : float
        Densify lines every (pixel_size * factor). Use 1.0 for ~1 pixel spacing.
    smoothing_sigma : float
        Gaussian sigma (pixels) to feather trench edges.
    downhill_bias : float
        Optional extra drop per pixel along flow direction (meters/pixel).
        Use small values (e.g., 0.01–0.1) to encourage convergence.

    Returns
    -------
    np.ndarray
        The modified DEM array (same shape as input).
    """
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge
    import rasterio
    from rasterio import features
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    def _to_dem_crs(gdf, target_crs):
        if gdf.crs is None:
            raise ValueError("Sewer file has no CRS; set one before running.")
        return gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf

    def _pixel_sizes(transform):
        # transform.a = pixel width, transform.e = pixel height (usually negative)
        px = abs(transform.a)
        py = abs(transform.e)
        return px, py

    def _densify_line(line: LineString, step: float) -> LineString:
        if line.length <= step:
            return line
        n = max(int(np.ceil(line.length / step)), 2)
        pts = [line.interpolate(d) for d in np.linspace(0, line.length, n)]
        return LineString(pts)

    def _iter_lines(geom):
        if geom is None:
            return
        if isinstance(geom, LineString):
            yield geom
        elif isinstance(geom, MultiLineString):
            for g in geom.geoms:
                yield g

    # 1) Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=False)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        height, width = dem.shape
        px, py = _pixel_sizes(transform)
        pixel_diag = (px + py) * 0.5
        densify_step = max(1e-9, pixel_diag * densify_step_factor)

    # 2) Read and reproject sewer network to DEM CRS
    sewer_gdf = gpd.read_file(sewer_path)
    sewer_gdf = _to_dem_crs(sewer_gdf, crs)

    # Merge small multipart pieces if present (optional, helps continuity)
    try:
        sewer_geom = linemerge(sewer_gdf.geometry.unary_union)
    except Exception:
        sewer_geom = sewer_gdf.geometry.unary_union

    # Normalize to a list of LineStrings
    raw_lines = list(_iter_lines(sewer_geom))
    if not raw_lines:
        raise ValueError("No line geometries found in sewer data.")

    # 3) Densify lines and collect coordinates in pixel space
    densified_lines = [ _densify_line(ln, densify_step) for ln in raw_lines ]
    all_lines = densified_lines

    # For each line, sample DEM along its points and enforce monotonic downstream
    modified = dem.copy()
    if nodata is not None:
        valid_mask = (dem != nodata)
    else:
        # Assume all finite are valid
        valid_mask = np.isfinite(dem)

    def _world_to_rc(x, y, tf):
        # Fast affine inverse
        # r, c = ~tf * (x, y) gives (col, row); we return (row, col)
        col, row = (~tf) * (x, y)
        return int(round(row)), int(round(col))

    # Build a burn mask for trench area (buffered lines)
    buffer_meters = max(1, int(buffer_pixels)) * max(px, py)
    buffered_shapes = [(ln.buffer(buffer_meters), 1) for ln in all_lines]
    trench_mask = features.rasterize(
        buffered_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True
    ).astype(bool)

    # We'll also build an array to hold "target" elevations inside the trench
    target = np.full_like(modified, np.nan, dtype=float)

    for ln in all_lines:
        coords = np.asarray(ln.coords)
        # Convert to (row, col)
        rcs = np.array([_world_to_rc(x, y, transform) for x, y in coords], dtype=int)
        r = np.clip(rcs[:, 0], 0, height - 1)
        c = np.clip(rcs[:, 1], 0, width - 1)

        # Sample DEM along the line; ignore NODATA by interpolating locally
        prof = modified[r, c].astype(float)
        if nodata is not None:
            prof[np.where(prof == nodata)] = np.nan

        # If too many NaNs, skip this segment
        if np.all(~np.isfinite(prof)):
            continue

        # Fill small NaN gaps along profile by linear interpolation
        idx = np.arange(prof.size)
        good = np.isfinite(prof)
        if good.sum() >= 2:
            prof[~good] = np.interp(idx[~good], idx[good], prof[good])
        else:
            # Fallback to nearest finite value
            finite_vals = prof[good]
            if finite_vals.size:
                prof[~good] = finite_vals[-1]

        # Enforce monotonic downstream (choose downstream end as the lower end)
        if prof[0] < prof[-1]:
            prof = prof[::-1]
            r = r[::-1]
            c = c[::-1]

        # Optional downhill bias (drop per pixel)
        if downhill_bias != 0.0:
            prof = prof - downhill_bias * np.arange(prof.size)

        mono = np.minimum.accumulate(prof)

        # Write into the target array
        target[r, c] = mono

    # 4) Spread “centerline” targets across trench using nearest within trench
    #    Compute distance inside trench and feather with Gaussian.
    # If no targets were written, nothing to do
    if np.all(~np.isfinite(target[trench_mask])):
        # Save original DEM if requested, then return
        if output_path is None:
            output_path = dem_path
        with rasterio.open(dem_path) as src:
            profile = src.profile
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(dem, 1)
        return dem

    # Compute distance transform to the nearest target cell (inside trench)
    # First, create a mask of available targets (finite)
    target_mask = np.isfinite(target)
    # For EDT, we want distances to the set where target_mask is True.
    # edt operates on False=features, True=background; invert logic:
    edt_src = ~(target_mask & trench_mask)
    dist = distance_transform_edt(edt_src)  # pixels to nearest target cell

    # Propagate target elevations across trench by “soft” nearest-neighbor
    # Use a Gaussian-weighted copy based on distance (smoother than hard nearest).
    # Convert pixel distance to weight: w = exp(-(d^2)/(2*sigma^2))
    # Sigma in pixels; reuse smoothing_sigma but ensure >0
    sigma_px = max(0.5, float(smoothing_sigma))
    with np.errstate(over='ignore'):
        weights = np.exp(-(dist * dist) / (2.0 * sigma_px * sigma_px))
    # Normalize weights to [0,1] inside trench, 0 outside
    weights[~trench_mask] = 0.0

    # Build a filled “carved” surface: where target is finite, use it; elsewhere,
    # softly pull original DEM toward the nearest target via weights.
    carved = modified.copy().astype(float)
    # Start from original, then blend toward target values where defined
    # For pixels that exactly have a target value, use it
    exact = target_mask
    carved[exact] = target[exact]

    # For trench pixels without exact target, pull toward the nearest target elevation
    # To get that, do a true nearest-propagation: use indices of nearest target
    # Prepare a fast NN via EDT indices: compute argmins by multi-pass
    # (simple but effective: forward/backward fill along rows/cols).
    # We’ll do a 4-pass nearest fill just within trench to get an approximate NN.
    nn = target.copy()
    # Horizontal passes
    for r in range(height):
        last = np.nan
        for c in range(width):
            if trench_mask[r, c] and np.isfinite(nn[r, c]):
                last = nn[r, c]
            elif trench_mask[r, c] and np.isfinite(last):
                nn[r, c] = last
        last = np.nan
        for c in range(width - 1, -1, -1):
            if trench_mask[r, c] and np.isfinite(nn[r, c]):
                last = nn[r, c]
            elif trench_mask[r, c] and np.isfinite(last):
                nn[r, c] = last
    # Vertical passes
    for c in range(width):
        last = np.nan
        for r in range(height):
            if trench_mask[r, c] and np.isfinite(nn[r, c]):
                last = nn[r, c]
            elif trench_mask[r, c] and np.isfinite(last):
                nn[r, c] = last
        last = np.nan
        for r in range(height - 1, -1, -1):
            if trench_mask[r, c] and np.isfinite(nn[r, c]):
                last = nn[r, c]
            elif trench_mask[r, c] and np.isfinite(last):
                nn[r, c] = last

    needs_blend = trench_mask & ~exact & np.isfinite(nn)
    carved[needs_blend] = (1.0 - weights[needs_blend]) * carved[needs_blend] + \
                          weights[needs_blend] * nn[needs_blend]

    # 5) Feather trench edges to avoid sharp steps
    feather = carved.copy()
    feather[~trench_mask] = 0.0
    # Only blur inside trench to keep surroundings intact
    blur = gaussian_filter(feather, sigma=smoothing_sigma)
    # Build a soft mask = blurred(binary trench)
    soft_mask = gaussian_filter(trench_mask.astype(float), sigma=smoothing_sigma)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    blended = modified.copy().astype(float)
    blended = soft_mask * blur + (1.0 - soft_mask) * blended

    # Respect NODATA (keep original value)
    if nodata is not None:
        blended[~valid_mask] = nodata

    # 6) Write out
    if output_path is None:
        output_path = dem_path

    with rasterio.open(dem_path) as src:
        profile = src.profile
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(blended.astype(profile["dtype"]), 1)

    return blended


def enforce_positive_slopes_tiered(
    dem_path: str,
    pipes_path: str,
    manholes_path: str,
    output_path: str,
    *,
    upstream_field: str = "upstream_m",
    downstream_field: str = "downstream",
    manhole_id_field: str = "id",
    type_field: str = "type",                # field that holds 'main'/'secondary'/'tertiary'
    tier_order=("main", "secondary", "tertiary"),
    tier_buffer_m={"main": 4.0, "secondary": 3.0, "tertiary": 2.0},
    Smin: float = 0.002,                     # minimum slope (m/m)
    densify_step_factor: float = 1.0,        # ~1 pixel sampling
    smoothing_sigma_px: float = 1.5,         # feathering smoothness (pixels)
):
    """
    Guarantees every pipe has positive slope (>= Smin), carving the DEM tier-by-tier:
    MAIN first from outlet -> head, then SECONDARY, then TERTIARY.

    The solver computes node target elevations <= current DEM elevations that satisfy
    Zu >= Zv + Smin*L for each directed pipe (u->v). Carving only lowers the DEM.

    Returns
    -------
    modified_dem : np.ndarray
    """
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from shapely.geometry import LineString, MultiLineString, Point
    from shapely.ops import linemerge
    from collections import defaultdict, deque
    from math import sqrt
    from scipy.ndimage import gaussian_filter

    # ------------------ helpers ------------------
    def _iter_lines(geom):
        if isinstance(geom, LineString):
            yield geom
        elif isinstance(geom, MultiLineString):
            for g in geom.geoms:
                yield g

    def _densify(line: LineString, step: float) -> LineString:
        if line.length <= step:
            return line
        n = max(int(np.ceil(line.length / step)), 2)
        pts = [line.interpolate(d) for d in np.linspace(0, line.length, n)]
        return LineString(pts)

    def _world_to_rc(x, y, tf):
        c, r = (~tf) * (x, y)
        return int(round(r)), int(round(c))

    # 0) Load DEM
    with rasterio.open(dem_path) as src:
        Z = src.read(1).astype(np.float64)
        tf = src.transform
        crs = src.crs
        nodata = src.nodata
        H, W = Z.shape
        px = abs(tf.a); py = abs(tf.e)
        pix_diag = 0.5 * (px + py)
        densify_step = max(1e-9, pix_diag * densify_step_factor)

    valid = (Z != nodata) if nodata is not None else np.isfinite(Z)

    # 1) Load pipes & manholes
    pipes = gpd.read_file(pipes_path)
    mans  = gpd.read_file(manholes_path)
    if pipes.crs != crs: pipes = pipes.to_crs(crs)
    if mans.crs  != crs: mans  = mans.to_crs(crs)

    if upstream_field not in pipes.columns or downstream_field not in pipes.columns:
        raise ValueError(f"Pipe attributes must include '{upstream_field}' and '{downstream_field}'.")
    if manhole_id_field not in mans.columns:
        raise ValueError(f"Manholes must have '{manhole_id_field}'.")

    # manhole id -> point & DEM elevation upper bound
    id_to_point = {}
    id_ub = {}
    for _, r in mans.iterrows():
        mid = str(r[manhole_id_field])
        geom = r.geometry
        if not isinstance(geom, Point): 
            continue
        id_to_point[mid] = geom
        rr, cc = _world_to_rc(geom.x, geom.y, tf)
        if 0 <= rr < H and 0 <= cc < W and valid[rr, cc]:
            id_ub[mid] = float(Z[rr, cc])
        else:
            # If outside DEM/invalid, set a very high UB so constraints will lower downstream instead
            id_ub[mid] = np.inf

    # 2) Collect edges by tier + length
    edges_all = []  # (tier, u, v, line, L)
    for _, r in pipes.iterrows():
        u = str(r[upstream_field]); v = str(r[downstream_field])
        if u not in id_to_point or v not in id_to_point:
            continue
        geom = r.geometry
        if geom is None: 
            continue
        lines = list(_iter_lines(geom))
        if not lines: 
            continue
        try:
            line = linemerge(lines) if len(lines) > 1 else lines[0]
        except Exception:
            line = lines[0]
        if not isinstance(line, LineString): 
            continue
        L = float(line.length)
        tier_value = str(r.get(type_field, "") or "").strip().lower()
        edges_all.append((tier_value, u, v, line, L))

    if not edges_all:
        raise ValueError("No valid pipes after filtering by manholes/geometry.")

    # 3) Build directed graph (all tiers) for ordering & outlet selection
    G_out = defaultdict(list)
    G_in  = defaultdict(list)
    nodes = set()
    for tier, u, v, _, L in edges_all:
        G_out[u].append((v, L))
        G_in[v].append((u, L))
        nodes.add(u); nodes.add(v)

    # Pick outlets as nodes with no outgoing edges; if none, pick nodes with minimal out-degree
    outlets = [n for n in nodes if len(G_out[n]) == 0]
    if not outlets:
        mindeg = min(len(G_out[n]) for n in nodes)
        outlets = [n for n in nodes if len(G_out[n]) == mindeg]

    # Choose the **lowest-elevation** outlet (in DEM) as the network outlet
    def _node_dem(n):
        z = id_ub.get(n, np.inf)
        return z if np.isfinite(z) else np.inf
    outlet = min(outlets, key=_node_dem)

    # 4) Feasible node targets under upper bounds (difference constraints)
    # We want Z[u] >= Z[v] + Smin*L  (edge u->v), with Z[n] <= UB[n].
    # We satisfy by propagating caps downstream: Zv <= Zu - Smin*L.
    # Initialize Zt = UB (never above DEM). Then relax from heads → outlet.
    Zt = {n: id_ub.get(n, np.inf) for n in nodes}

    # Topological order (if cyclic, do Kahn but keep all nodes; cycles will be handled by iterative passes)
    indeg = {n: 0 for n in nodes}
    for u in nodes:
        for v, _ in G_out[u]:
            indeg[v] += 1
    q = deque([n for n in nodes if indeg[n] == 0])
    topo = []
    indeg2 = indeg.copy()
    while q:
        n = q.popleft()
        topo.append(n)
        for v, _ in G_out[n]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    has_cycle = (len(topo) != len(nodes))
    if has_cycle:
        # Make a pass order anyway: heads first, then the rest arbitrarily
        rem = [n for n in nodes if n not in topo]
        topo = topo + rem

    # Relax: for each edge u->v, cap v <= u - w. Multiple passes for safety (handles cycles).
    for _ in range(10_000):
        changed = 0
        for u in topo:
            zu = Zt[u]
            if not np.isfinite(zu): 
                continue
            for v, L in G_out[u]:
                cap_v = zu - Smin * L
                if cap_v < Zt[v]:
                    Zt[v] = cap_v
                    changed += 1
        if changed == 0:
            break

    # Ensure not below -inf; Zt may become -huge, that’s fine for carving—DEM will cap it.

    # 5) Compute graph distance (meters) from outlet to every node for outlet→head ordering
    # Dijkstra on reversed edges (we want distances *upstream* from outlet along directed graph)
    import heapq
    dist = {n: np.inf for n in nodes}
    dist[outlet] = 0.0
    pq = [(0.0, outlet)]
    G_in_list = G_in  # v -> [(u, L)]
    while pq:
        d, v = heapq.heappop(pq)
        if d != dist[v]:
            continue
        for u, L in G_in_list[v]:
            nd = d + L
            if nd < dist[u]:
                dist[u] = nd
                heapq.heappush(pq, (nd, u))

    # 6) Carving utilities
    def _paint_centerline_profile(carved, line, zu, zv):
        # densify, compute cumulative length ratio, assign linear profile
        ln = _densify(line, densify_step)
        coords = np.asarray(ln.coords)
        # cumulative s
        ls = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            ls.append(ls[-1] + sqrt(dx*dx + dy*dy))
        Ltot = max(ls[-1], 1e-9)
        s = np.asarray(ls) / Ltot
        prof = zu + (zv - zu) * s  # strictly monotone

        for (x, y), zval in zip(coords, prof):
            rr, cc = _world_to_rc(x, y, tf)
            if 0 <= rr < H and 0 <= cc < W:
                if np.isnan(carved[rr, cc]):
                    carved[rr, cc] = zval
                else:
                    carved[rr, cc] = min(carved[rr, cc], zval)  # honor lowest where multiple pipes overlap

    def _carve_tier(Z_in, tier_edges, buf_m):
        """
        Carve one tier: build a trench mask, paint centerline targets, feather, and only lower DEM.
        Edges are processed outlet→head (increasing 'dist').
        """
        if not tier_edges:
            return Z_in

        # Order edges by distance from outlet (small→large)
        tier_edges_sorted = sorted(tier_edges, key=lambda rec: min(dist.get(rec[1], np.inf), dist.get(rec[2], np.inf)))

        # Rasterize trench
        trench_shapes = [(line.buffer(buf_m), 1) for (_, u, v, line, L) in tier_edges_sorted]
        trench = features.rasterize(
            trench_shapes,
            out_shape=(H, W),
            transform=tf,
            fill=0,
            all_touched=True,
            dtype="uint8"
        ).astype(bool)
        trench &= valid

        carved_center = np.full((H, W), np.nan, dtype=float)

        # Paint centerline profiles using node targets Zt[u], Zt[v]
        for (_, u, v, line, L) in tier_edges_sorted:
            zu, zv = Zt.get(u, np.inf), Zt.get(v, np.inf)
            if not (np.isfinite(zu) and np.isfinite(zv)):
                continue
            _paint_centerline_profile(carved_center, line, zu, zv)

        if np.all(~np.isfinite(carved_center[trench])):
            return Z_in

        # Nearest fill inside trench (4-pass) then feather
        nn = carved_center.copy()
        for r in range(H):
            last = np.nan
            for c in range(W):
                if trench[r, c] and np.isfinite(nn[r, c]): last = nn[r, c]
                elif trench[r, c] and np.isfinite(last):    nn[r, c] = last
            last = np.nan
            for c in range(W-1, -1, -1):
                if trench[r, c] and np.isfinite(nn[r, c]): last = nn[r, c]
                elif trench[r, c] and np.isfinite(last):    nn[r, c] = last
        for c in range(W):
            last = np.nan
            for r in range(H):
                if trench[r, c] and np.isfinite(nn[r, c]): last = nn[r, c]
                elif trench[r, c] and np.isfinite(last):    nn[r, c] = last
            last = np.nan
            for r in range(H-1, -1, -1):
                if trench[r, c] and np.isfinite(nn[r, c]): last = nn[r, c]
                elif trench[r, c] and np.isfinite(last):    nn[r, c] = last

        filled = nn.copy(); filled[~trench] = 0.0
        blur = gaussian_filter(filled, sigma=smoothing_sigma_px)
        soft_mask = gaussian_filter(trench.astype(float), sigma=smoothing_sigma_px)
        soft_mask = np.clip(soft_mask, 0.0, 1.0)

        carved_surface = Z_in.copy()
        carved_surface = soft_mask * blur + (1.0 - soft_mask) * carved_surface

        # Only LOWER
        lower_mask = trench & valid & np.isfinite(carved_surface) & (carved_surface < Z_in)
        Z_out = Z_in.copy()
        Z_out[lower_mask] = carved_surface[lower_mask]
        return Z_out

    # 7) Split edges by tier label
    # normalize labels to lower
    tier_map = defaultdict(list)
    for tier, u, v, line, L in edges_all:
        t = (tier or "").lower()
        tier_map[t].append((t, u, v, line, L))

    # 8) Carve in requested tier order, outlet→head within each tier
    Z_work = Z.copy()
    for t in tier_order:
        edges_t = tier_map.get(t, [])
        if not edges_t:
            continue
        buf = float(tier_buffer_m.get(t, 3.0))
        Z_work = _carve_tier(Z_work, edges_t, buf_m=buf)

    # Optionally carve any remaining pipes with unknown/other tier last
    others = [rec for tt, recs in tier_map.items() if tt not in tier_order for rec in recs]
    if others:
        Z_work = _carve_tier(Z_work, others, buf_m=float(min(tier_buffer_m.values()) if tier_buffer_m else 2.0))

    # Respect nodata
    if nodata is not None:
        Z_work[~valid] = nodata

    # 9) Write out
    with rasterio.open(dem_path) as src:
        profile = src.profile
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(Z_work.astype(profile["dtype"]), 1)

    return Z_work


def _iter_lines(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            if isinstance(g, LineString):
                yield g


def _densify(line: LineString, step_m: float) -> LineString:
    if step_m <= 0 or line.length <= step_m:
        return line
    n = max(int(np.ceil(line.length / step_m)), 2)
    pts = [line.interpolate(d) for d in np.linspace(0, line.length, n)]
    return LineString(pts)


def _world_to_rc(tf, x, y):
    c, r = (~tf) * (x, y)  # (col, row)
    return int(round(r)), int(round(c))


def adjust_manhole_elevations_tiered(
    pipes_path: str,
    manholes_path: str,
    *,
    upstream_field: str = "upstream_m",
    downstream_field: str = "downstream",
    manhole_id_field: str = "id",
    manhole_elev_field: str = "elevation",
    type_field: str = "type",                     # 'main'/'secondary'/'tertiary'
    tier_order=("main", "secondary", "tertiary"),
    Smin: float = 0.002,                          # m/m
    target_crs=None
):
    """
    Raise upstream manhole elevations (never lower) so every directed pipe has slope >= Smin.
    Returns: (pipes_gdf, manholes_gdf with 'elev_adj', outlet_id)
    """
    pipes = gpd.read_file(pipes_path)
    mans  = gpd.read_file(manholes_path)

    if target_crs is not None:
        if pipes.crs != target_crs:
            pipes = pipes.to_crs(target_crs)
        if mans.crs != target_crs:
            mans = mans.to_crs(target_crs)

    # Manhole maps
    id_to_point = {}
    id_to_elev  = {}
    for _, r in mans.iterrows():
        mid = str(r[manhole_id_field])
        geom = r.geometry
        if isinstance(geom, Point):
            id_to_point[mid] = geom
            id_to_elev[mid]  = float(r[manhole_elev_field])

    # Collect edges per tier with lengths
    edges_all = []  # (tier, u, v, line, L)
    for _, r in pipes.iterrows():
        u = str(r[upstream_field]); v = str(r[downstream_field])
        if u not in id_to_point or v not in id_to_point:
            continue
        geom = r.geometry
        if geom is None:
            continue
        lines = list(_iter_lines(geom))
        if not lines:
            continue
        try:
            line = linemerge(lines) if len(lines) > 1 else lines[0]
        except Exception:
            line = lines[0]
        if not isinstance(line, LineString):
            continue
        L = float(line.length)
        tier_value = str(r.get(type_field, "") or "").strip().lower()
        edges_all.append((tier_value, u, v, line, L))

    if not edges_all:
        raise ValueError("No valid pipe geometries found.")

    # Build directed graph
    G_out = defaultdict(list)  # u -> [(v, L)]
    G_in  = defaultdict(list)  # v -> [(u, L)]
    nodes = set()
    for t,u,v,line,L in edges_all:
        G_out[u].append((v, L))
        G_in[v].append((u, L))
        nodes.add(u); nodes.add(v)

    # Choose outlet:
    #   Prefer nodes with NO outgoing edges in the MAIN tier; else any node with no outgoing.
    main_outgoing = set(u for (t,u,_,_,_) in edges_all if t == "main")
    main_targets  = set(v for (t,_,v,_,_) in edges_all if t == "main")
    main_nodes    = main_outgoing | main_targets
    main_outlets  = {n for n in main_nodes if all(t != "main" or u != n for (t,u,_,_,_) in edges_all)}
    candidates = main_outlets if main_outlets else {n for n in nodes if len(G_out[n]) == 0}
    if not candidates:
        candidates = set(nodes)
    outlet = min(candidates, key=lambda n: id_to_elev.get(n, np.inf))

    # Start adjusted elevations at existing manhole elevations
    Zt = {n: id_to_elev[n] for n in nodes}

    # Bucket edges by tier
    tier_map = defaultdict(list)
    for rec in edges_all:
        tier_map[rec[0]].append(rec)

    # Enforce edges in a given set, processing approx outlet->upstream order
    def enforce_edges(edges):
        if not edges:
            return
        # Dijkstra upstream distance from outlet (on reversed edges) for ordering
        import heapq
        dist = {n: np.inf for n in nodes}
        dist[outlet] = 0.0
        pq = [(0.0, outlet)]
        while pq:
            d, v = heapq.heappop(pq)
            if d != dist[v]:
                continue
            for u,L in G_in[v]:
                nd = d + L
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(pq, (nd, u))

        # Sort edges near outlet first
        edges_sorted = sorted(edges, key=lambda rec: min(dist.get(rec[1], np.inf), dist.get(rec[2], np.inf)))

        # Raise upstream until all constraints are satisfied
        for _ in range(10000):
            changed = 0
            for _, u, v, _, L in edges_sorted:
                need = Zt[v] + Smin * L
                if Zt[u] < need:
                    Zt[u] = need
                    changed += 1
            if changed == 0:
                break

    # Process tiers in order
    for t in tier_order:
        enforce_edges(tier_map.get(t, []))
    # Any other types last
    remaining = [rec for k, recs in tier_map.items() if k not in tier_order for rec in recs]
    if remaining:
        enforce_edges(remaining)

    # Attach adjusted elevations
    mans_out = mans.copy()
    mans_out["elev_adj"] = mans_out[manhole_id_field].map(lambda mid: Zt.get(str(mid), np.nan))
    return pipes, mans_out, outlet


def verify_all_pipes_positive_by_nodes(
    pipes_gdf, manholes_gdf, upstream_field="upstream_m", downstream_field="downstream",
    manhole_id_field="id", elev_field="elev_adj", Smin=0.002
):
    """Check Zu - Zv >= Smin * L for every pipe, using adjusted node elevations only."""
    mh_elev = {str(r[manhole_id_field]): float(r[elev_field]) for _, r in manholes_gdf.iterrows()}
    bad = []
    for i, r in pipes_gdf.reset_index().iterrows():
        u = str(r[upstream_field]); v = str(r[downstream_field])
        if u not in mh_elev or v not in mh_elev:
            continue
        geom = r.geometry
        if geom is None:
            continue
        try:
            ln = linemerge(list(_iter_lines(geom))) if isinstance(geom, MultiLineString) else geom
        except Exception:
            ln = geom
        if not isinstance(ln, LineString):
            continue
        L = float(ln.length)
        zu, zv = mh_elev[u], mh_elev[v]
        if (zu - zv) < Smin * max(L, 1e-9) - 1e-9:
            bad.append((i, u, v, (zu - zv) / max(L, 1e-9)))
    return bad


def verify_pipes_on_raster(dem_array, transform, pipes_gdf, Smin=0.002):
    """Sample along each pipe (~1 px) and verify each segment slope >= Smin."""
    H, W = dem_array.shape
    tf = transform
    px = abs(tf.a); py = abs(tf.e)
    step = 0.5 * (px + py)

    def rcxy(x, y):
        return _world_to_rc(tf, x, y)

    bad = []
    for i, r in pipes_gdf.reset_index().iterrows():
        geom = r.geometry
        if geom is None:
            continue
        try:
            ln = linemerge(list(_iter_lines(geom))) if isinstance(geom, MultiLineString) else geom
        except Exception:
            ln = geom
        if not isinstance(ln, LineString) or ln.length <= 0:
            continue
        n = max(int(np.ceil(ln.length / step)), 2)
        coords = [ln.interpolate(d) for d in np.linspace(0, ln.length, n)]
        rc = [rcxy(p.x, p.y) for p in coords]
        rr = np.clip([r for r, c in rc], 0, H-1)
        cc = np.clip([c for r, c in rc], 0, W-1)
        z = dem_array[rr, cc].astype(float)
        # Ensure last point is downstream (lower), reverse if needed
        if z[0] < z[-1]:
            z = z[::-1]; rr = rr[::-1]; cc = cc[::-1]
        segL = []
        for j in range(1, len(coords)):
            dx = coords[j].x - coords[j-1].x
            dy = coords[j].y - coords[j-1].y
            segL.append((dx*dx + dy*dy)**0.5)
        segL = np.asarray(segL)
        drops = z[:-1] - z[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.where(segL > 0, drops/segL, np.inf)
        if np.nanmin(slopes) < Smin - 1e-9:
            j = int(np.nanargmin(slopes))
            bad.append((i, float(slopes[j]), (int(rr[j]), int(cc[j]))))
    return bad


def _collect_controls_for_interp(
    dem_template_path,
    manholes_gdf,
    pipes_gdf,
    *,
    manhole_id_field="id",
    elev_field="elev_adj",
    densify_step_m=None,       # default ≈ 2 px if None/0
    along_pipe_weight=1,       # emulate weights via small repeats (1..3)
    max_repeat=3
):
    """Return (Xc, Yc, Zc, crs, transform, profile, valid_mask) for interpolation."""
    with rasterio.open(dem_template_path) as src:
        profile = src.profile
        tf = src.transform
        crs = src.crs
        nodata = src.nodata
        base = src.read(1)
        H, W = base.shape
        px = abs(tf.a); py = abs(tf.e)

    # default: about 2 pixels spacing to keep N modest
    step = (px + py) if not densify_step_m else float(densify_step_m)

    mh = manholes_gdf.dropna(subset=[elev_field]).copy()
    if mh.crs != crs:
        mh = mh.to_crs(crs)
    Xp = mh.geometry.x.to_numpy(dtype=np.float32)
    Yp = mh.geometry.y.to_numpy(dtype=np.float32)
    Zp = mh[elev_field].to_numpy(dtype=np.float32)

    pipes = pipes_gdf
    if pipes.crs != crs:
        pipes = pipes.to_crs(crs)

    mh_elev = {str(r[manhole_id_field]): float(r[elev_field]) for _, r in mh.iterrows()}
    Xs, Ys, Zs = [], [], []
    for _, row in pipes.iterrows():
        u = str(row.get("upstream_m"))
        v = str(row.get("downstream"))
        if u not in mh_elev or v not in mh_elev:
            continue
        zu, zv = mh_elev[u], mh_elev[v]
        geom = row.geometry
        if geom is None:
            continue
        try:
            line = linemerge(list(_iter_lines(geom))) if isinstance(geom, MultiLineString) else geom
        except Exception:
            line = geom
        if not isinstance(line, LineString) or line.length <= 0:
            continue
        n = max(int(np.ceil(line.length / step)), 2)
        coords = [line.interpolate(d) for d in np.linspace(0, line.length, n)]
        s = np.linspace(0.0, 1.0, n, dtype=np.float64)
        prof = zu + (zv - zu) * s
        Xs.extend([p.x for p in coords]); Ys.extend([p.y for p in coords]); Zs.extend(prof.tolist())

    Xs = np.asarray(Xs, dtype=np.float32)
    Ys = np.asarray(Ys, dtype=np.float32)
    Zs = np.asarray(Zs, dtype=np.float32)

    rep = int(min(max(1, round(along_pipe_weight)), max_repeat))
    if Xs.size and rep > 1:
        Xc = np.concatenate([Xp, np.repeat(Xs, rep)]).astype(np.float32, copy=False)
        Yc = np.concatenate([Yp, np.repeat(Ys, rep)]).astype(np.float32, copy=False)
        Zc = np.concatenate([Zp, np.repeat(Zs, rep)]).astype(np.float32, copy=False)
    elif Xs.size:
        Xc = np.concatenate([Xp, Xs]).astype(np.float32, copy=False)
        Yc = np.concatenate([Yp, Ys]).astype(np.float32, copy=False)
        Zc = np.concatenate([Zp, Zs]).astype(np.float32, copy=False)
    else:
        Xc, Yc, Zc = Xp, Yp, Zp

    valid = (base != nodata) if nodata is not None else np.isfinite(base)
    return Xc, Yc, Zc, crs, tf, profile, valid


def interpolate_dem_idw_tiled(
    dem_template_path: str,
    manholes_gdf,
    pipes_gdf,
    *,
    manhole_id_field="id",
    elev_field="elev_adj",
    densify_step_m=None,
    along_pipe_weight=2,
    power=2.0,
    k=12,
    tile=1024,
    output_path=None
):
    """
    Memory-safe IDW interpolation on the SAME grid/extent as dem_template_path.
    - KDTree with k nearest neighbors
    - Processes raster in tiles (tile x tile)
    - Preserves nodata outside original valid area
    """
    Xc, Yc, Zc, crs, tf, profile, valid = _collect_controls_for_interp(
        dem_template_path, manholes_gdf, pipes_gdf,
        manhole_id_field=manhole_id_field,
        elev_field=elev_field,
        densify_step_m=densify_step_m,
        along_pipe_weight=along_pipe_weight,
        max_repeat=3,
    )

    pts = np.column_stack([Xc, Yc])  # (N,2)
    tree = cKDTree(pts)

    H, W = profile["height"], profile["width"]
    out = np.empty((H, W), dtype=np.float32)

    def block_coords(r0, r1, c0, c1):
        rows = np.arange(r0, r1, dtype=np.int32)
        cols = np.arange(c0, c1, dtype=np.int32)
        X = (cols.astype(np.float64)[None, :] * tf.a + tf.c).astype(np.float32)
        Y = (rows.astype(np.float64)[:, None] * tf.e + tf.f).astype(np.float32)
        return X, Y  # (1,w), (h,1)

    for r0 in range(0, H, tile):
        r1 = min(H, r0 + tile)
        for c0 in range(0, W, tile):
            c1 = min(W, c0 + tile)
            h = r1 - r0; w = c1 - c0

            Xv, Yv = block_coords(r0, r1, c0, c1)
            X_flat = np.broadcast_to(Xv, (h, w)).ravel()
            Y_flat = np.broadcast_to(Yv, (h, w)).ravel()
            Q = np.column_stack([X_flat, Y_flat])

            dists, idxs = tree.query(Q, k=min(k, pts.shape[0]), workers=-1)
            if dists.ndim == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]

            with np.errstate(divide='ignore'):
                wts = 1.0 / np.maximum(dists, 1e-6)**power
            wsum = np.sum(wts, axis=1, keepdims=True)
            wts /= np.maximum(wsum, 1e-12)

            Z_neighbors = Zc[idxs]  # (M,k)
            Z_flat = np.sum(wts * Z_neighbors, axis=1).astype(np.float32)
            out[r0:r1, c0:c1] = Z_flat.reshape(h, w)

    nodata = profile.get("nodata", None)
    if nodata is not None:
        out[~valid] = nodata

    if output_path:
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out.astype(profile["dtype"]), 1)

    return out, profile


def centerline_writeback_linear(
    dem_array, transform, pipes_gdf, manholes_gdf,
    *,
    upstream_field="upstream_m", downstream_field="downstream",
    manhole_id_field="id", elev_field="elev_adj",
    step_m=None, lower_only=True
):
    """
    Set DEM pixels ON the pipe centerlines to the linear profile between adjusted manholes.
    Ensures strictly monotone pixels along lines; optionally 'lower_only'.
    """
    Z = dem_array
    tf = transform
    H, W = Z.shape

    # step ~ 1 pixel if not given
    px = abs(tf.a); py = abs(tf.e)
    if not step_m or step_m <= 0:
        step_m = 0.5 * (px + py)

    # Pull adjusted node elevations
    mh = manholes_gdf
    mh_elev = {str(r[manhole_id_field]): float(r[elev_field]) for _, r in mh.iterrows()}

    for _, r in pipes_gdf.iterrows():
        u = str(r.get(upstream_field)); v = str(r.get(downstream_field))
        if u not in mh_elev or v not in mh_elev:
            continue
        zu, zv = mh_elev[u], mh_elev[v]
        geom = r.geometry
        if geom is None:
            continue
        try:
            line = linemerge(list(_iter_lines(geom))) if isinstance(geom, MultiLineString) else geom
        except Exception:
            line = geom
        if not isinstance(line, LineString) or line.length <= 0:
            continue

        ln = _densify(line, step_m)
        coords = np.asarray(ln.coords)
        # cumulative param s in [0,1]
        ls = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            ls.append(ls[-1] + sqrt(dx*dx + dy*dy))
        Ltot = max(ls[-1], 1e-9)
        s = np.asarray(ls) / Ltot
        prof = zu + (zv - zu) * s

        for (x, y), zval in zip(coords, prof):
            rr, cc = _world_to_rc(tf, x, y)
            if 0 <= rr < H and 0 <= cc < W:
                if lower_only:
                    if zval < Z[rr, cc]:
                        Z[rr, cc] = zval
                else:
                    Z[rr, cc] = zval

    return Z


def build_dem_with_guaranteed_positive_slopes_idw(
    dem_path: str,
    pipes_path: str,
    manholes_path: str,
    output_path: str,
    *,
    upstream_field="upstream_m",
    downstream_field="downstream",
    manhole_id_field="id",
    manhole_elev_field="elevation",
    type_field="type",
    tier_order=("main","secondary","tertiary"),
    Smin=0.002,
    densify_step_m=None,       # controls controls density (default ≈ 2 px)
    along_pipe_weight=2,
    idw_power=2.0,
    idw_k=12,
    idw_tile=1024,
    centerline_writeback=True, # final, lower-only linear profiles on pixels
    verify_on_raster=True
):
    """
    Full pipeline:
      1) Adjust manhole elevations tier-by-tier so Zu-Zv >= Smin*L for all pipes (node-level).
      2) Verify node-level slopes (hard requirement). If any fail, raise.
      3) Tiled KDTree-IDW to build DEM on the SAME grid/extent/CRS/nodata as input.
      4) Optional centerline write-back (lower-only) to eliminate tiny interpolation wiggles.
      5) Optional raster verification (report only).
    """
    # Read DEM profile to get CRS and to ensure same area on output
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        profile = src.profile

    # (1) Adjust nodes (raise upstream)
    pipes_gdf, mans_adj, outlet_id = adjust_manhole_elevations_tiered(
        pipes_path, manholes_path,
        upstream_field=upstream_field,
        downstream_field=downstream_field,
        manhole_id_field=manhole_id_field,
        manhole_elev_field=manhole_elev_field,
        type_field=type_field,
        tier_order=tier_order,
        Smin=Smin,
        target_crs=dem_crs
    )

    # (2) Pre-DEM verification (by nodes)
    node_bad = verify_all_pipes_positive_by_nodes(
        pipes_gdf, mans_adj,
        upstream_field=upstream_field,
        downstream_field=downstream_field,
        manhole_id_field=manhole_id_field,
        elev_field="elev_adj",
        Smin=Smin
    )
    if node_bad:
        sample = "\n".join([f"pipe_idx={i}, u={u}, v={v}, node_slope={s:.6f}" for (i,u,v,s) in node_bad[:10]])
        raise RuntimeError(f"Node-level slope enforcement failed for {len(node_bad)} pipes. Example:\n{sample}")

    # (3) Tiled IDW interpolation on SAME grid/extent
    Znew, profile_out = interpolate_dem_idw_tiled(
        dem_template_path=dem_path,
        manholes_gdf=mans_adj,
        pipes_gdf=pipes_gdf,
        manhole_id_field=manhole_id_field,
        elev_field="elev_adj",
        densify_step_m=densify_step_m,
        along_pipe_weight=along_pipe_weight,
        power=idw_power,
        k=idw_k,
        tile=idw_tile,
        output_path=None  # write after optional write-back
    )

    # (4) Optional centerline write-back (lower-only)
    if centerline_writeback:
        with rasterio.open(dem_path) as src:
            tf = src.transform
            nodata = src.nodata
        Zwb = centerline_writeback_linear(
            dem_array=Znew.copy(),
            transform=tf,
            pipes_gdf=pipes_gdf,
            manholes_gdf=mans_adj,
            upstream_field=upstream_field,
            downstream_field=downstream_field,
            manhole_id_field=manhole_id_field,
            elev_field="elev_adj",
            step_m=0.5 * (abs(tf.a) + abs(tf.e)),  # ~1 px
            lower_only=True
        )
        # Preserve nodata footprint from template
        if profile_out.get("nodata", None) is not None:
            with rasterio.open(dem_path) as src:
                base = src.read(1)
                Zwb[base == profile_out["nodata"]] = profile_out["nodata"]
        Znew = Zwb

    # Write output with EXACT same area/CRS/transform/dtype/nodata
    with rasterio.open(output_path, "w", **profile_out) as dst:
        dst.write(Znew.astype(profile_out["dtype"]), 1)

    # (5) Optional raster-level verification
    if verify_on_raster:
        with rasterio.open(output_path) as src:
            Zpost = src.read(1).astype(float)
            tf = src.transform
        rb = verify_pipes_on_raster(Zpost, tf, pipes_gdf, Smin=Smin)
        if rb:
            worst = min(rb, key=lambda t: t[1])
            print(f"⚠️ Raster check: {len(rb)} pipes have a local slope < Smin (e.g., idx={worst[0]}, slope={worst[1]:.6f}).")
        else:
            print("✅ Raster check: all sampled pipe segments meet the minimum slope.")

    return output_path


def _bilinear_sample(dem: np.ndarray, r: float, c: float, nodata=None):
    """
    Bilinear interpolation at fractional row/col. Returns np.nan if any needed neighbors are nodata/outside.
    """
    h, w = dem.shape
    r0 = int(np.floor(r)); c0 = int(np.floor(c))
    r1 = r0 + 1; c1 = c0 + 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return np.nan
    q11 = dem[r0, c0]; q21 = dem[r0, c1]; q12 = dem[r1, c0]; q22 = dem[r1, c1]
    if nodata is not None:
        if (q11 == nodata) or (q21 == nodata) or (q12 == nodata) or (q22 == nodata):
            return np.nan
    fr = r - r0
    fc = c - c0
    # (1-fr)(1-fc) q11 + (1-fr)fc q21 + fr(1-fc) q12 + fr fc q22
    return (1-fr)*(1-fc)*q11 + (1-fr)*fc*q21 + fr*(1-fc)*q12 + fr*fc*q22


def _window_stats(dem: np.ndarray, mask: np.ndarray, nodata):
    """
    Compute min/mean/median on values where mask=True and not nodata. Returns np.nan if nothing valid.
    """
    vals = dem[mask]
    if nodata is not None:
        vals = vals[vals != nodata]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(vals)), float(np.mean(vals)), float(np.median(vals))


def update_manhole_elevations_from_dem(
    dem_path: str,
    manholes_path: str,
    output_path: str | None = None,
    *,
    elevation_field: str = "elevation",
    create_new_field_if_missing: bool = True,
    sampling: str = "bilinear",        # "nearest" or "bilinear"
    search_radius_m: float | None = None,  # e.g., 3.0 -> sample stats in a disk of this radius
    search_stat: str = "min",          # "min" | "mean" | "median" (used only if search_radius_m given)
    overwrite: bool = False            # if True and output_path is None, overwrite input shapefile
):
    """
    Sample the DEM at each manhole point and write the value into `elevation_field`.

    - If `search_radius_m` is given, elevation is computed from pixels within that radius using `search_stat`.
      (This helps avoid edge artifacts; 'min' is typical for drainage features.)
    - If the field doesn't exist and `create_new_field_if_missing` is True, it will be added.

    Returns: path to written manholes file.
    """
    # 1) Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        tf: Affine = src.transform
        crs_dem = src.crs
        nodata = src.nodata
        height, width = dem.shape
        px = abs(tf.a); py = abs(tf.e)

    # 2) Read manholes and project to DEM CRS if needed
    mans = gpd.read_file(manholes_path)
    if mans.crs != crs_dem:
        mans = mans.to_crs(crs_dem)

    # Ensure geometry are Points
    if not all(isinstance(geom, Point) for geom in mans.geometry):
        raise ValueError("Manhole layer must contain Point geometries only.")

    # Prepare output path
    if output_path is None:
        if overwrite:
            # We'll overwrite the original file
            output_path = manholes_path
        else:
            base, ext = os.path.splitext(manholes_path)
            output_path = base + "_elev_updated" + ext

    # 3) Sampling
    # Helper: world -> (row, col)
    inv = ~tf
    def xy_to_rc(x, y):
        c, r = inv * (x, y)  # returns float col,row in pixel space
        return float(r), float(c)

    # If searching in a radius, precompute a disk mask per unique radius in pixels
    do_radius = (search_radius_m is not None) and (search_radius_m > 0)
    if do_radius:
        # approximate isotropic pixel size
        pix = 0.5 * (px + py)
        rad_px = max(1, int(np.ceil(search_radius_m / max(pix, 1e-9))))
        # build a disk mask template big enough to cover the radius
        yy, xx = np.ogrid[-rad_px:rad_px+1, -rad_px:rad_px+1]
        disk = (xx*xx + yy*yy) <= (rad_px*rad_px)
        stat_choice = search_stat.lower().strip()
        if stat_choice not in ("min", "mean", "median"):
            raise ValueError("search_stat must be one of: 'min', 'mean', 'median'")
    else:
        disk = None
        stat_choice = None

    # Decide sampling mode
    sampling = sampling.lower().strip()
    if sampling not in ("nearest", "bilinear"):
        raise ValueError("sampling must be 'nearest' or 'bilinear'")

    # 4) Compute elevations
    out_vals = np.full(len(mans), np.nan, dtype="float64")

    for i, geom in enumerate(mans.geometry):
        x, y = geom.x, geom.y
        r_f, c_f = xy_to_rc(x, y)

        if do_radius:
            # Build a window around the center pixel, clip to raster bounds
            r0 = int(np.floor(r_f)) - disk.shape[0]//2
            c0 = int(np.floor(c_f)) - disk.shape[1]//2
            r_start = max(0, r0); c_start = max(0, c0)
            r_end   = min(height, r0 + disk.shape[0])
            c_end   = min(width,  c0 + disk.shape[1])

            if r_start >= r_end or c_start >= c_end:
                out_vals[i] = np.nan
                continue

            sub = dem[r_start:r_end, c_start:c_end]
            # Align the disk mask for this clipped window
            dr0 = r_start - r0
            dc0 = c_start - c0
            disk_clip = disk[dr0:dr0+sub.shape[0], dc0:dc0+sub.shape[1]]

            vmin, vmean, vmed = _window_stats(sub, disk_clip, nodata)
            if stat_choice == "min":
                out_vals[i] = vmin
            elif stat_choice == "mean":
                out_vals[i] = vmean
            else:
                out_vals[i] = vmed

        else:
            # Single-point sample
            if sampling == "nearest":
                r = int(round(r_f)); c = int(round(c_f))
                if 0 <= r < height and 0 <= c < width:
                    z = dem[r, c]
                    out_vals[i] = np.nan if (nodata is not None and z == nodata) else float(z)
                else:
                    out_vals[i] = np.nan
            else:  # bilinear
                z = _bilinear_sample(dem, r_f, c_f, nodata=nodata)
                out_vals[i] = z

    # 5) Write back into attribute table
    mans_out = mans.copy()

    if elevation_field in mans_out.columns:
        # Overwrite values where we have a sample (leave others untouched)
        mask = np.isfinite(out_vals)
        mans_out.loc[mask, elevation_field] = out_vals[mask]
    else:
        if not create_new_field_if_missing:
            raise ValueError(f"Field '{elevation_field}' not in layer. Set create_new_field_if_missing=True to add it.")
        # Shapefile warning: field names <= 10 chars
        mans_out[elevation_field] = out_vals

    # 6) Save
    driver = "ESRI Shapefile" if output_path.lower().endswith(".shp") else None
    if driver:
        mans_out.to_file(output_path, driver=driver)
    else:
        # Default to GeoPackage if extension is .gpkg, else let GeoPandas infer
        mans_out.to_file(output_path)

    # 7) Quick report
    n_ok = int(np.isfinite(out_vals).sum())
    n_all = len(out_vals)
    print(f"Updated elevations for {n_ok}/{n_all} manholes. Wrote: {output_path}")

    return output_path


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
    pipes = gpd.read_file(pipes_path)

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
    pipes = gpd.read_file(pipes_path)
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


def british_columbia_peaking_factor(q_lps):
    """
    Estimate peak flow and bounded peaking factor using British Columbia method.

    Peaking Factor = 1 + 11 / sqrt(Q_mgd), bounded between 2.5 and 4.0

    Parameters:
        q_lps (float or pd.Series): Base flow in liters per second (L/s)

    Returns:
        peak_flow_lps (float or pd.Series): Peak flow in L/s
        pf (float or pd.Series): Peaking factor
    """
    import numpy as np

    # Convert to MGD
    q_mgd = q_lps * 0.022824

    # Apply BC formula and bound the result
    raw_pf = 1 + 11 / np.sqrt(q_mgd.clip(lower=0.001))
    pf = np.clip(raw_pf, 2.5, 4.0)  # Clamp to [2.5, 4.0]

    peak_flow_lps = q_lps * pf
    return peak_flow_lps, pf


def assign_pipe_slopes(pipes_path, manholes_path, output_path, minimum_slope=0.001):
    """
    Assigns slopes to each pipe based on manhole elevations and enforces a minimum slope.

    Parameters:
        pipes_path (str): Path to pipes shapefile. Must contain 'upstream_m' and 'downstream'.
        manholes_path (str): Path to manholes shapefile. Must contain 'id' and 'elevation'.
        output_path (str): Path to save updated pipes shapefile with slope attributes.
        minimum_slope (float): Minimum allowable slope (default is 0.001).
    """
    import geopandas as gpd

    # Load data
    pipes = gpd.read_file(pipes_path)
    manholes = gpd.read_file(manholes_path)

    # Rename for consistent joining
    manholes = manholes.rename(columns={"id": "manhole_id", "elevation": "elevation"})

    # Merge upstream elevations
    pipes = pipes.merge(
        manholes[["manhole_id", "elevation"]],
        left_on="upstream_m",
        right_on="manhole_id",
        how="left"
    ).rename(columns={"elevation": "elev_up"}).drop(columns="manhole_id")

    # Merge downstream elevations
    pipes = pipes.merge(
        manholes[["manhole_id", "elevation"]],
        left_on="downstream",
        right_on="manhole_id",
        how="left"
    ).rename(columns={"elevation": "elev_down"}).drop(columns="manhole_id")

    # Calculate pipe length in meters
    pipes["length_m"] = pipes.geometry.length

    # Calculate raw slope
    pipes["slope_raw"] = (pipes["elev_up"] - pipes["elev_down"]) / pipes["length_m"]

    # Apply minimum slope rule
    pipes["slope"] = pipes["slope_raw"].apply(
        lambda s: s if s >= minimum_slope else minimum_slope
    )

    # Optionally: flag pipes where slope was corrected
    pipes["slope_flag"] = pipes["slope_raw"] < minimum_slope

    # Save to file
    pipes.to_file(output_path)
    print(f"✅ Pipe slopes assigned and saved to: {output_path}")


def assign_material_diameter_to_pipes(
    pipes_path: str,
    output_path: str = None,
    *,
    pipe_id_col: str = "pipe_id",
    up_mh_col: str = "upstream_m",
    down_mh_col: str = "downstream",
    flow_col: str = "predesign_",     # <── your requested default
    slope_col: str = "slope",
    material_fractions: dict = {"PVC": 0.6, "CONCRETE": 0.3, "HDPE": 0.1},
    n_by_material: dict = {"PVC": 0.011, "CONCRETE": 0.013, "HDPE": 0.012},
    rng_seed: int = 42,
    standard_diameters_mm = (
        200, 250, 300, 350, 400, 450, 500,
        600, 700, 800, 900, 1000, 1100, 1200,
        1300, 1400, 1500, 1600, 1700, 1800,
        1900, 2000
    ),
    minimum_diameter_mm: int = 200,
    slope_floor: float = 1e-5,
    design_factor: float = 1.00,
    fullness_factor: float = 0.85,
    return_gdf: bool = False
):
    """
    Assigns pipe MATERIAL, Manning's n, theoretical diameter (via Manning's eq.),
    rounds up to commercial sizes, and enforces non-decreasing diameter downstream.
    """

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from collections import defaultdict, deque

    # ------------------------- Load data -------------------------
    gdf = gpd.read_file(pipes_path)

    required = {pipe_id_col, up_mh_col, down_mh_col, flow_col, slope_col}
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize material fractions
    total_frac = sum(material_fractions.values())
    if total_frac <= 0:
        raise ValueError("material_fractions must sum to > 0.")
    material_fractions = {k: v / total_frac for k, v in material_fractions.items()}

    # ---------------------- Assign materials ----------------------
    rng = np.random.default_rng(rng_seed)
    materials = list(material_fractions.keys())
    probs = np.array(list(material_fractions.values()))
    gdf["material"] = rng.choice(materials, size=len(gdf), p=probs)
    gdf["n"] = gdf["material"].map(n_by_material)

    # ------------------ Compute theoretical diameter ------------------
    # Manning for full circular pipe:
    # Q = (1/n) * A * R^(2/3) * S^(1/2)
    # A = πD²/4, R = D/4 -> Q = (1/n)*K*S^(1/2)*D^(8/3)
    # where K = (π/4)*4^(-2/3)
    K = (np.pi / 4.0) * (4.0 ** (-2.0 / 3.0))
    Q_lps = gdf[flow_col].astype(float).clip(lower=0).values
    S_arr = gdf[slope_col].astype(float).clip(lower=slope_floor).values
    n_arr = gdf["n"].astype(float).values

    k = float(fullness_factor)
    if not (0 < k <= 1):
        raise ValueError("fullness_factor must be between 0 and 1.")

    # Adjust Q for design and fullness
    Q_m3s = (Q_lps / 1000.0) * design_factor / (k ** (5 / 3))

    # Compute D in meters
    with np.errstate(all="ignore"):
        D_m = ((Q_m3s * n_arr) / (K * np.sqrt(S_arr))) ** (3.0 / 8.0)

    D_m = np.where(~np.isfinite(D_m) | (D_m <= 0), minimum_diameter_mm / 1000.0, D_m)

    # Fixed: use np.maximum instead of .clip(lower=...)
    gdf["calc_diameter_mm"] = np.maximum(D_m * 1000.0, minimum_diameter_mm)

    # --------------------- Round to commercial diameters ---------------------
    std = np.array(sorted(set(int(x) for x in standard_diameters_mm)), dtype=int)
    std = std[std >= minimum_diameter_mm]
    if std.size == 0:
        raise ValueError("No valid commercial diameters >= minimum_diameter_mm")

    def round_up(d):
        for s in std:
            if d <= s:
                return s
        return std[-1]

    gdf["diameter_mm"] = gdf["calc_diameter_mm"].apply(round_up).astype(int)

    # ------------------ Enforce downstream monotonicity ------------------
    pid = gdf[pipe_id_col].astype(object).values
    up_mh = gdf[up_mh_col].astype(object).values
    dn_mh = gdf[down_mh_col].astype(object).values
    diam = dict(zip(pid, gdf["diameter_mm"].values))

    out_pipes = defaultdict(list)
    in_pipes = defaultdict(list)
    indeg = defaultdict(int)
    manholes = set()

    for p, u, d in zip(pid, up_mh, dn_mh):
        out_pipes[u].append(p)
        in_pipes[d].append(p)
        indeg[d] += 1
        manholes.add(u)
        manholes.add(d)

    queue = deque([mh for mh in manholes if indeg[mh] == 0])
    visited = set()

    while queue:
        mh = queue.popleft()
        visited.add(mh)
        max_in_d = max((diam[p] for p in in_pipes.get(mh, [])), default=minimum_diameter_mm)
        for p in out_pipes.get(mh, []):
            diam[p] = max(diam[p], max_in_d)
            dmh = gdf.loc[gdf[pipe_id_col] == p, down_mh_col].iloc[0]
            indeg[dmh] -= 1
            if indeg[dmh] == 0 and dmh not in visited:
                queue.append(dmh)

    gdf["diameter_mm"] = gdf[pipe_id_col].map(diam).astype(int)

    # ------------------------- Output -------------------------
    if output_path:
        gdf.to_file(output_path)
        print(f"✅ Materials and diameters assigned and saved to: {output_path}")

    if return_gdf:
        return gdf


def assign_invert_elevations(
    pipes_path,
    output_path,
    min_cover=1.4,   # meters
    min_slope=0.005, # minimum design slope
    manhole_drop=0.05 # drop across manhole for continuity
):
    """
    Assigns upstream and downstream invert elevations to each pipe based on slope and network topology.

    Parameters:
        pipes_path (str): Path to input pipe shapefile (must include: pipe_id, upstream_m, downstream, slope, length_m, elev_up).
        output_path (str): Path to save shapefile with 'inv_up' and 'inv_down' fields.
        min_cover (float): Minimum cover depth above pipe in meters.
        min_slope (float): Minimum pipe slope to enforce.
        manhole_drop (float): Drop applied at junctions for outlet pipes.
    """
    import geopandas as gpd
    import numpy as np
    from collections import defaultdict, deque

    # Load pipe shapefile
    pipes = gpd.read_file(pipes_path)

    # Initialize new columns
    pipes["inv_up"] = np.nan
    pipes["inv_down"] = np.nan

    # Create dictionaries for topology
    pipe_dict = pipes.set_index("pipe_id").to_dict("index")
    downstream_to_pipes = defaultdict(list)
    for pid, row in pipe_dict.items():
        downstream_to_pipes[row["downstream"]].append(pid)

    # Identify head pipes (no inflows)
    all_upstream = set(pipes["upstream_m"])
    all_downstream = set(pipes["downstream"])
    head_manholes = all_upstream - all_downstream
    head_pipes = pipes[pipes["upstream_m"].isin(head_manholes)]

    # Build dependency graph (topological sort)
    in_degree = {pid: 0 for pid in pipe_dict}
    for pid, row in pipe_dict.items():
        for other_pid, other_row in pipe_dict.items():
            if other_row["downstream"] == row["upstream_m"]:
                in_degree[pid] += 1

    queue = deque([pid for pid, deg in in_degree.items() if deg == 0])
    inv_results = {}

    while queue:
        pid = queue.popleft()
        row = pipe_dict[pid]
        length = row["length_m"]
        slope = max(row["slope"], min_slope)

        # HEAD PIPE
        if row["upstream_m"] in head_manholes:
            elev_up = row["elev_up"]
            inv_up = elev_up - min_cover
            inv_down = inv_up - slope * length
        else:
            upstream_pipes = [
                up_pid for up_pid, up_row in pipe_dict.items()
                if up_row["downstream"] == row["upstream_m"] and up_pid in inv_results
            ]
            if not upstream_pipes:
                continue  # wait until upstream is processed

            min_inv_down = min(inv_results[up_pid][1] for up_pid in upstream_pipes)
            inv_up = min_inv_down - manhole_drop
            inv_down = inv_up - slope * length

        inv_results[pid] = (inv_up, inv_down)

        # Push downstream pipes to queue
        for next_pid, next_row in pipe_dict.items():
            if next_row["upstream_m"] == row["downstream"]:
                in_degree[next_pid] -= 1
                if in_degree[next_pid] == 0:
                    queue.append(next_pid)

    # Assign to DataFrame
    pipes["inv_up"] = pipes["pipe_id"].map(lambda pid: inv_results.get(pid, (np.nan, np.nan))[0])
    pipes["inv_down"] = pipes["pipe_id"].map(lambda pid: inv_results.get(pid, (np.nan, np.nan))[1])

    # Save output
    pipes.to_file(output_path)
    print(f"✅ Invert elevations assigned and saved to: {output_path}")


def preprocess_pipes_and_manholes(pipes_path, manholes_path, output_pipes_path, output_manholes_path):
    import geopandas as gpd
    import numpy as np

    # Load data
    pipes = gpd.read_file(pipes_path)
    manholes = gpd.read_file(manholes_path)

    # --- PIPES preprocessing ---
    pipes_clean = pipes.copy()

    # Fix/ensure required columns
    if "diameter_mm" not in pipes_clean.columns:
        if "diameter_m" in pipes_clean.columns:
            pipes_clean["diameter_mm"] = pipes_clean["diameter_m"] * 1000
        elif "calc_diame" in pipes_clean.columns:
            pipes_clean["diameter_mm"] = pipes_clean["calc_diame"] * 1000
        else:
            raise ValueError("No diameter column found to derive 'diameter_mm'.")

    # Drop entries with missing required values
    required_pipe_fields = [
        "pipe_id", "upstream_m", "downstream", "length_m",
        "inv_up", "inv_down", "n", "diameter_mm", "geometry"
    ]
    for field in required_pipe_fields:
        if field not in pipes_clean.columns:
            raise ValueError(f"Missing required pipe field: {field}")

    pipes_clean = pipes_clean.dropna(subset=required_pipe_fields)
    pipes_clean = pipes_clean.drop_duplicates(subset="pipe_id")

    # --- MANHOLES preprocessing ---
    manholes_clean = manholes.copy()
    manholes_clean = manholes_clean.dropna(subset=["id", "elevation", "geometry"])
    manholes_clean = manholes_clean.drop_duplicates(subset="id")

    # Save cleaned files
    pipes_clean.to_file(output_pipes_path)
    manholes_clean.to_file(output_manholes_path)

    print(f"✅ Pipes saved to: {output_pipes_path}")
    print(f"✅ Manholes saved to: {output_manholes_path}")
    return pipes_clean, manholes_clean


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

    gdf = gpd.read_file(pipes_path)

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
    pipes = gpd.read_file(pipes_path)
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


def add_predesign_flow(pipes_path: str, out_path: str | None = None, overwrite: bool = True):
    """
    Adds predesign flow = peak_flow_ + cum_gwi_ls + cum_rdii_l to pipes shapefile.

    Parameters
    ----------
    pipes_path : str
        Path to the pipes shapefile/GeoPackage.
    out_path : str | None
        Optional path to save updated file. If None, does not save.
    overwrite : bool
        Allow overwrite of existing file.

    Returns
    -------
    GeoDataFrame with new column 'predesign_ls'
    """

    import geopandas as gpd

    gdf = gpd.read_file(pipes_path)

    # Check required fields
    for col in ["peak_flow_", "cum_gwi_ls", "cum_rdii_l"]:
        if col not in gdf.columns:
            raise ValueError(f"Missing column '{col}' in input file.")

    # Calculate predesign flow
    gdf["predesign_ls"] = (
        gdf["peak_flow_"].astype(float)
        + gdf["cum_gwi_ls"].astype(float)
        + gdf["cum_rdii_l"].astype(float)
    )

    # Save if requested
    if out_path:
        import pathlib
        out_path = pathlib.Path(out_path)
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists and overwrite=False.")
        gdf.to_file(out_path)

    return gdf


def assign_all_dwf_patterns(
    inp_path,
    output_path,
    hourly_id="1",
    hourly_values=None,
    daily_id="2",
    daily_values=None,
    monthly_id="3",
    monthly_values=None,
    weekend_id="4",
    weekend_values=None,
):
    """
    Rewrite [DWF] so each entry uses 4 patterns:
        "hourly_id" "daily_id" "monthly_id" "weekend_id"

    Replace any existing [PATTERNS] section with a new one:

    [DWF]
    ;;Node           Constituent      Baseline   Patterns  
    ;;-------------- ---------------- ---------- ----------
    MH001           FLOW             0.047      "1" "2" "3" "4"
    MH002           FLOW             0.052      "1" "2" "3" "4"
    ...

    [PATTERNS]
    ;;Name           Type       Multipliers
    ;;-------------- ---------- -----------
    1                HOURLY     ...
    ;
    2                DAILY      ...
    ;
    3                MONTHLY    ...
    ;
    4                WEEKEND    ...
    ;
    """

    # ----- defaults if not passed -----
    if hourly_values is None:
        hourly_values = [1.0] * 24
    if daily_values is None:
        daily_values = [1.0] * 7
    if monthly_values is None:
        monthly_values = [1.0] * 12
    if weekend_values is None:
        weekend_values = [0.8] * 24

    # sanity checks
    if len(hourly_values) != 24:
        raise ValueError("hourly_values must have 24 values.")
    if len(daily_values) != 7:
        raise ValueError("daily_values must have 7 values (Mon–Sun).")
    if len(monthly_values) != 12:
        raise ValueError("monthly_values must have 12 values (Jan–Dec).")
    if len(weekend_values) != 24:
        raise ValueError("weekend_values must have 24 values.")

    with open(inp_path, "r") as f:
        lines = f.readlines()

    # ---------------------------------------------------------
    # 1) Rewrite [DWF] and strip any existing [PATTERNS]
    # ---------------------------------------------------------
    new_lines = []
    in_dwf = False
    in_patterns = False
    insert_patterns_index = None  # where we'll inject new [PATTERNS]

    for line in lines:
        stripped = line.strip()

        # ---- remove old [PATTERNS] ----
        if stripped.startswith("[PATTERNS]"):
            in_patterns = True
            insert_patterns_index = len(new_lines)
            continue

        if in_patterns:
            if stripped.startswith("[") and not stripped.startswith("[PATTERNS]"):
                in_patterns = False
                new_lines.append(line)
            continue

        # ---- handle [DWF] section ----
        if stripped.startswith("[DWF]"):
            in_dwf = True
            new_lines.append("[DWF]\n")
            new_lines.append(
                ";;Node           Constituent      Baseline   Patterns  \n"
            )
            new_lines.append(
                ";;-------------- ---------------- ---------- ----------\n"
            )
            continue

        if in_dwf:
            # end of DWF section
            if stripped.startswith("[") and not stripped.startswith("[DWF]"):
                in_dwf = False
                new_lines.append(line)
                continue

            # skip old comments/blank lines in DWF (we already wrote header)
            if not stripped or stripped.startswith(";"):
                continue

            # data line: Node Constituent Baseline [maybe patterns...]
            parts = stripped.split()
            if len(parts) >= 3:
                node, constituent, baseline = parts[0], parts[1], parts[2]
                new_line = (
                    f"{node:<16} {constituent:<16} {baseline:<10} "
                    f"\"{hourly_id}\" \"{daily_id}\" \"{monthly_id}\" \"{weekend_id}\"\n"
                )
                new_lines.append(new_line)
            else:
                # weird line: keep it
                new_lines.append(line)
            continue

        # other sections: copy as-is
        new_lines.append(line)

    if insert_patterns_index is None:
        insert_patterns_index = len(new_lines)

    # ---------------------------------------------------------
    # 2) Build new [PATTERNS] section
    # ---------------------------------------------------------
    def format_pattern(name, ptype, values, per_line):
        block = []
        for i in range(0, len(values), per_line):
            chunk = values[i : i + per_line]
            if i == 0:
                block.append(
                    f"{name:<16} {ptype:<10} "
                    + "   ".join(f"{v:.1f}" for v in chunk)
                    + "  \n"
                )
            else:
                block.append(
                    f"{name:<16} {'':<10} "
                    + "   ".join(f"{v:.1f}" for v in chunk)
                    + "  \n"
                )
        return block

    patterns_block = [
        "\n[PATTERNS]\n",
        ";;Name           Type       Multipliers\n",
        ";;-------------- ---------- -----------\n",
    ]

    # HOURLY 1
    patterns_block += format_pattern(hourly_id, "HOURLY", hourly_values, per_line=6)
    patterns_block.append(";\n")

    # DAILY 2
    patterns_block += format_pattern(daily_id, "DAILY", daily_values, per_line=7)
    patterns_block.append(";\n")

    # MONTHLY 3
    patterns_block += format_pattern(monthly_id, "MONTHLY", monthly_values, per_line=6)
    patterns_block.append(";\n")

    # WEEKEND 4
    patterns_block += format_pattern(weekend_id, "WEEKEND", weekend_values, per_line=6)
    patterns_block.append(";\n")

    final_lines = (
        new_lines[:insert_patterns_index]
        + patterns_block
        + new_lines[insert_patterns_index:]
    )

    with open(output_path, "w") as f:
        f.writelines(final_lines)

    print(
        f"✅ Updated [DWF] with patterns \"{hourly_id}\" \"{daily_id}\" \"{monthly_id}\" \"{weekend_id}\""
    )
    print("✅ Rewrote [PATTERNS] with HOURLY, DAILY, MONTHLY, WEEKEND")
    print(f"   Output saved to: {output_path}")


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


def auto_add_pollutants_to_inp_fixed(inp_path, output_path):
    """
    Keeps [DWF] lines (including all patterns) and appends RAIN/DRY [POLLUTANTS]
    and [REPORT] sections at the end.
    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    in_dwf = False
    cleaned_lines = []

    # Step 1: Preserve DWF section (no pattern truncation)
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("[DWF]"):
            in_dwf = True
            cleaned_lines.append(line)
            continue
        elif stripped.startswith("[") and not stripped.upper().startswith("[DWF]"):
            in_dwf = False
            cleaned_lines.append(line)
            continue

        if in_dwf and stripped and not stripped.startswith(";"):
            # Normalize whitespace but keep ALL tokens (node, constituent, baseline, patterns...)
            parts = stripped.split()
            cleaned_lines.append("    ".join(parts) + "\n")
        else:
            cleaned_lines.append(line)

    # Step 2: Append pollutant and report sections
    cleaned_lines.append("""
[POLLUTANTS]
;;Name           Units  Crain      Cgw        Crdii      Kdecay     SnowOnly   Co-Pollutant     Co-Frac    Cdwf       Cinit     
;;-------------- ------ ---------- ---------- ---------- ---------- ---------- ---------------- ---------- ----------
RAIN             MG/L   100        0.0        0.0        0.0        NO         *                0.0        0.0        0.0       
DRY              MG/L   0.0        0.0        0.0        0.0        NO         *                0.0        100        0.0       

[REPORT]
;;Reporting Options
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL
""")

    # Step 3: Write output
    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)

    print(f"✅ Kept [DWF] patterns and appended [POLLUTANTS] and [REPORT] to {output_path}")


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

class TopographyConfig:
    def __init__(self, 
                 min_elevation=0,
                 max_elevation=100,
                 cell_size=10,  # meters
                 outlet_direction='S',
                 smoothing_factor=1.0):
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        self.cell_size = cell_size
        self.outlet_direction = outlet_direction
        self.smoothing_factor = smoothing_factor

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

def meters_to_crs_units(cell_size_m: float, crs) -> float:
    """Convert meters to CRS units if CRS uses feet; otherwise assume meters."""
    crs_obj = CRS.from_user_input(crs)
    unit = (crs_obj.axis_info[0].unit_name or "").lower()
    if "foot" in unit:
        return cell_size_m / 0.304800609601219  # meters -> US survey foot
    return cell_size_m


def build_domain_mask_from_shapefile(
    shapefile_path: str,
    cell_size_m: float,
    all_touched: bool = False,   # kept for compatibility (not used in point-center method)
):
    """
    Build a domain mask anchored to the ORIGINAL shapefile CRS + extent.

    FIXES / IMPROVEMENTS:
    - If CRS is projected and in FEET, you can pass meters_to_crs_units(cell_size_m, crs) instead,
      OR set cell_size_m to already be in CRS units.
    - Uses union_all (no deprecated unary_union)
    - Grid is anchored to minx/miny (origin), so export overlays correctly
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS. Define it before running.")

    crs_out = gdf.crs
    crs_out_obj = CRS.from_user_input(crs_out)
    is_geographic = crs_out_obj.is_geographic

    # Bounds in original CRS
    minx, miny, maxx, maxy = gdf.total_bounds

    # Union geometry for point-in-polygon
    geom_union = gdf.geometry.union_all()

    if is_geographic:
        # --- compute grid dims in meters via UTM ---
        calc_crs = gdf.estimate_utm_crs()
        gdf_calc = gdf.to_crs(calc_crs)
        minx_c, miny_c, maxx_c, maxy_c = gdf_calc.total_bounds
        width_m  = maxx_c - minx_c
        height_m = maxy_c - miny_c

        n_cols = int(np.ceil(width_m / cell_size_m))
        n_rows = int(np.ceil(height_m / cell_size_m))

        # Uniform grid in calc CRS (meters), anchored to calc bounds
        x_edges_c = minx_c + np.arange(n_cols + 1) * cell_size_m
        y_edges_c = miny_c + np.arange(n_rows + 1) * cell_size_m

        # Transform edges back to original CRS (degrees)
        to_out = Transformer.from_crs(calc_crs, crs_out, always_xy=True)
        midy_c = 0.5 * (miny_c + maxy_c)
        midx_c = 0.5 * (minx_c + maxx_c)

        x_edges = np.array([to_out.transform(x, midy_c)[0] for x in x_edges_c])
        y_edges = np.array([to_out.transform(midx_c, y)[1] for y in y_edges_c])

        # Use median step for export (good enough for small domains)
        cell_w = float(np.median(np.diff(x_edges))) if len(x_edges) > 1 else None
        cell_h = float(np.median(np.diff(y_edges))) if len(y_edges) > 1 else None

        # IMPORTANT: ensure monotonic increasing edges (rarely, transforms can reverse order)
        if len(x_edges) > 1 and x_edges[1] < x_edges[0]:
            x_edges = x_edges[::-1]
        if len(y_edges) > 1 and y_edges[1] < y_edges[0]:
            y_edges = y_edges[::-1]

    else:
        # --- projected CRS: build grid directly in CRS units ---
        # NOTE: if CRS is feet, cell_size_m must already be in feet (use meters_to_crs_units)
        width_u  = maxx - minx
        height_u = maxy - miny

        n_cols = int(np.ceil(width_u / cell_size_m))
        n_rows = int(np.ceil(height_u / cell_size_m))

        x_edges = minx + np.arange(n_cols + 1) * cell_size_m
        y_edges = miny + np.arange(n_rows + 1) * cell_size_m

        cell_w = float(cell_size_m)
        cell_h = float(cell_size_m)

    # Build mask using CELL-CENTER point-in-polygon in ORIGINAL CRS
    domain_mask = np.zeros((n_rows, n_cols), dtype=np.uint8)

    # Row i=0 is the BOTTOM (miny side) since y_edges starts at miny.
    for i in range(n_rows):
        cy = 0.5 * (y_edges[i] + y_edges[i + 1])
        for j in range(n_cols):
            cx = 0.5 * (x_edges[j] + x_edges[j + 1])
            if geom_union.contains(gpd.points_from_xy([cx], [cy], crs=crs_out)[0]):
                domain_mask[i, j] = 1

    # Metadata for export
    grid_meta = dict(
        crs_out=str(crs_out),
        origin_x=float(x_edges[0]),
        origin_y=float(y_edges[0]),
        cell_w=float(cell_w) if cell_w is not None else None,
        cell_h=float(cell_h) if cell_h is not None else None,
        rows=int(n_rows),
        cols=int(n_cols),
        flip_y=False,  # matches export below (no y flip)
    )

    return domain_mask, grid_meta


def export_individual_figures_to_shapefile_georeferenced(
    filled_board,
    output_path,
    grid_meta,
    id_to_type_map=None,
):
    """
    Export each shape as polygons in the ORIGINAL shapefile CRS using grid_meta.

    FIXES:
    - Accepts folder OR .shp file path
    - No vertical flip (y0 = oy + i*ch) so output matches input orientation
    - Writes 'tetro_id' (so your plotting function works)
    """
    # Make output_path a valid shapefile path
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "filled_board.shp")
    if not output_path.lower().endswith(".shp"):
        output_path = output_path + ".shp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows, cols = filled_board.shape
    ox = grid_meta["origin_x"]
    oy = grid_meta["origin_y"]
    cw = grid_meta["cell_w"]
    ch = grid_meta["cell_h"]
    crs_out = grid_meta["crs_out"]

    if cw is None or ch is None:
        raise ValueError("grid_meta cell_w/cell_h are None. Check mask/grid construction.")

    geometries = []
    figure_ids = []
    tetro_ids = []
    labels = []

    unique_ids = np.unique(filled_board)
    unique_ids = unique_ids[unique_ids > 0]  # skip -1 and 0

    for shape_id in unique_ids:
        cells = np.argwhere(filled_board == shape_id)
        cell_polys = []

        for i, j in cells:
            x0 = ox + j * cw
            y0 = oy + i * ch          # ✅ NO (rows-1-i) flip
            x1 = x0 + cw
            y1 = y0 + ch
            cell_polys.append(box(x0, y0, x1, y1))

        merged = unary_union(cell_polys)
        geometries.append(merged)
        figure_ids.append(int(shape_id))
        tetro_ids.append(int(shape_id))  # keep same ID (your plot uses this)
        labels.append(id_to_type_map.get(int(shape_id), "Unknown") if id_to_type_map else "Unknown")

    gdf = gpd.GeoDataFrame(
        {"figure_id": figure_ids, "tetro_id": tetro_ids, "label": labels, "geometry": geometries},
        crs=crs_out
    )
    gdf.to_file(output_path)
    print(f"✅ Exported {len(gdf)} figures to {output_path}")
    return gdf

import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, breadth_first_order

def prepare_sewer_nodes(manholes):
    """
    Convert manholes into indexed arrays and lookup dictionaries.

    Parameters
    ----------
    manholes : list[dict]
        Each dict must contain:
        {
            'id': ...,
            'location': shapely Point,
            'elevation': float
        }

    Returns
    -------
    node_data : dict
        {
            'n', 'ids', 'points', 'elevs', 'xs', 'ys',
            'id_to_idx', 'idx_to_id', 'id_to_mh', 'point_tree'
        }
    """
    if not manholes:
        raise ValueError("manholes list is empty.")

    ids = [mh["id"] for mh in manholes]
    points = [mh["location"] for mh in manholes]
    elevs = np.asarray([mh["elevation"] for mh in manholes], dtype=float)
    xs = np.asarray([pt.x for pt in points], dtype=float)
    ys = np.asarray([pt.y for pt in points], dtype=float)

    id_to_idx = {mid: i for i, mid in enumerate(ids)}
    idx_to_id = {i: mid for i, mid in enumerate(ids)}
    id_to_mh = {mh["id"]: mh for mh in manholes}

    point_tree = STRtree(points)

    return {
        "n": len(manholes),
        "ids": ids,
        "points": points,
        "elevs": elevs,
        "xs": xs,
        "ys": ys,
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "id_to_mh": id_to_mh,
        "point_tree": point_tree,
    }

def prepare_road_index(road_buffer):
    """
    Normalize road buffer and create STRtree index.

    Parameters
    ----------
    road_buffer : Polygon or MultiPolygon

    Returns
    -------
    road_data : dict
        {
            'road_polys', 'road_tree'
        }
    """
    if isinstance(road_buffer, Polygon):
        road_polys = [road_buffer]
    elif isinstance(road_buffer, MultiPolygon):
        road_polys = list(road_buffer.geoms)
    else:
        raise ValueError("road_buffer must be Polygon or MultiPolygon")

    road_tree = STRtree(road_polys)

    return {
        "road_polys": road_polys,
        "road_tree": road_tree,
    }

def line_is_covered_by_road(line, road_data):
    """
    Check whether a line is covered by at least one road polygon.
    Uses spatial filtering via STRtree.
    """
    road_polys = road_data["road_polys"]
    road_tree = road_data["road_tree"]

    candidate_idxs = road_tree.query(line)
    for idx in candidate_idxs:
        if road_polys[idx].covers(line):
            return True
    return False


def segment_crosses_other_manholes(line, i, j, node_data, no_cross_radius):
    """
    Check whether the segment i->j passes too close to another manhole.

    Parameters
    ----------
    line : LineString
    i, j : int
        End node indices
    node_data : dict
    no_cross_radius : float

    Returns
    -------
    bool
    """
    points = node_data["points"]
    point_tree = node_data["point_tree"]

    nearby_idxs = point_tree.query(line.buffer(no_cross_radius))

    for k in nearby_idxs:
        if k == i or k == j:
            continue

        pt_k = points[k]

        if line.distance(pt_k) < no_cross_radius:
            proj = line.project(pt_k)
            if 0.0 < proj < line.length:
                return True

    return False

def build_main_candidate_graph(
    manholes,
    road_buffer,
    block_size=40.0,
    slope_tolerance=-0.01,
    min_pipe_length=5.0,
    prefer_slope=0.6,
):
    """
    Build directed sparse candidate graph for the main sewer path.

    Parameters
    ----------
    manholes : list[dict]
    road_buffer : Polygon or MultiPolygon
    block_size : float
    slope_tolerance : float
        Minimum allowed slope
    min_pipe_length : float
    prefer_slope : float
        Cost preference for positive slope

    Returns
    -------
    graph_data : dict
    """
    node_data = prepare_sewer_nodes(manholes)
    road_data = prepare_road_index(road_buffer)

    n = node_data["n"]
    points = node_data["points"]
    elevs = node_data["elevs"]
    xs = node_data["xs"]
    ys = node_data["ys"]
    point_tree = node_data["point_tree"]

    head_idx = int(np.argmax(elevs))    # highest manhole
    outlet_idx = int(np.argmin(elevs))  # lowest manhole

    max_radius = block_size * 4.0
    no_cross_radius = block_size * 0.3

    rows = []
    cols = []
    data = []
    edge_attrs = {}

    for i in range(n):
        pt_i = points[i]
        elev_i = elevs[i]
        x_i = xs[i]
        y_i = ys[i]

        nearby_idxs = point_tree.query(pt_i.buffer(max_radius))

        for j in nearby_idxs:
            if j == i:
                continue

            # Fast distance check with NumPy
            dx = xs[j] - x_i
            dy = ys[j] - y_i
            dist = float(np.hypot(dx, dy))

            if dist < min_pipe_length:
                continue

            elev_j = elevs[j]
            slope = (elev_i - elev_j) / max(dist, 1e-12)

            # Sewer edge only if admissible
            if slope < slope_tolerance:
                continue

            # Only create geometry after cheap filters pass
            line = LineString([pt_i, points[j]])

            if not line_is_covered_by_road(line, road_data):
                continue

            if segment_crosses_other_manholes(line, i, j, node_data, no_cross_radius):
                continue

            # Cost function: short + favorable slope
            slope_bonus = 1.0 - prefer_slope * max(0.0, slope)
            cost = dist * slope_bonus

            rows.append(i)
            cols.append(j)
            data.append(cost)

            edge_attrs[(i, j)] = {
                "distance": dist,
                "slope": slope,
                "cost": cost,
                "line": line,
            }

    if len(data) == 0:
        A = coo_matrix((n, n)).tocsr()
    else:
        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    return {
        "A": A,
        "edge_attrs": edge_attrs,
        "node_data": node_data,
        "road_data": road_data,
        "head_idx": head_idx,
        "outlet_idx": outlet_idx,
        "parameters": {
            "block_size": block_size,
            "slope_tolerance": slope_tolerance,
            "min_pipe_length": min_pipe_length,
            "prefer_slope": prefer_slope,
            "max_radius": max_radius,
            "no_cross_radius": no_cross_radius,
        },
    }

def get_reachable_nodes(A, start_idx):
    """
    Return all nodes reachable from start_idx in a directed sparse graph.
    """
    reachable_order, _ = breadth_first_order(
        A,
        i_start=start_idx,
        directed=True,
        return_predecessors=True
    )
    return set(reachable_order.tolist())

def choose_target_node(graph_data):
    """
    Choose outlet if reachable, otherwise choose lowest reachable node.
    """
    node_data = graph_data["node_data"]
    A = graph_data["A"]
    head_idx = graph_data["head_idx"]
    outlet_idx = graph_data["outlet_idx"]

    elevs = node_data["elevs"]
    idx_to_id = node_data["idx_to_id"]

    reachable = get_reachable_nodes(A, head_idx)
    reachable.add(head_idx)

    if outlet_idx in reachable:
        return outlet_idx, reachable, False

    target_idx = min(reachable, key=lambda idx: elevs[idx])
    print(
        f"⚠️ Outlet {idx_to_id[outlet_idx]} unreachable, using lowest reachable manhole {idx_to_id[target_idx]}"
    )
    return target_idx, reachable, True

def reconstruct_path_from_predecessors(predecessors, start_idx, target_idx):
    """
    Reconstruct shortest path from predecessor array.
    """
    path = []
    cur = target_idx

    while cur != -9999:
        path.append(cur)
        if cur == start_idx:
            break
        cur = predecessors[cur]

    path.reverse()

    if not path or path[0] != start_idx:
        return []

    return path

def extract_main_path(graph_data, target_idx=None):
    """
    Extract best path from head to target using Dijkstra.

    Parameters
    ----------
    graph_data : dict
    target_idx : int or None

    Returns
    -------
    segments : list[(id_from, id_to)]
    path_info : dict
    """
    A = graph_data["A"]
    edge_attrs = graph_data["edge_attrs"]
    node_data = graph_data["node_data"]
    head_idx = graph_data["head_idx"]

    idx_to_id = node_data["idx_to_id"]
    elevs = node_data["elevs"]

    empty_output = {
        "segments": [],
        "slopes": {},
        "distances": {},
        "lines": {},
        "path_node_ids": [],
        "path_node_indices": [],
        "reachable_node_indices": [],
        "reachable_node_ids": [],
        "used_fallback_target": False,
        "target_id": None,
        "target_idx": None,
        "cumulative_drop": 0.0,
        "total_length": 0.0,
        "avg_slope": 0.0,
    }

    if A.nnz == 0:
        print("❌ No admissible edges found.")
        return [], empty_output

    if target_idx is None:
        target_idx, reachable, used_fallback = choose_target_node(graph_data)
    else:
        reachable = get_reachable_nodes(A, head_idx)
        reachable.add(head_idx)
        used_fallback = False

    dist_arr, predecessors = dijkstra(
        A,
        directed=True,
        indices=head_idx,
        return_predecessors=True
    )

    if np.isinf(dist_arr[target_idx]):
        print("❌ No path found at all.")
        return [], empty_output

    path_idxs = reconstruct_path_from_predecessors(predecessors, head_idx, target_idx)

    if not path_idxs:
        print("❌ Path reconstruction failed.")
        return [], empty_output

    segments = []
    slopes = {}
    distances = {}
    lines = {}
    cumulative_drop = 0.0
    total_length = 0.0

    for u_idx, v_idx in zip(path_idxs[:-1], path_idxs[1:]):
        u_id = idx_to_id[u_idx]
        v_id = idx_to_id[v_idx]
        seg = (u_id, v_id)

        attrs = edge_attrs[(u_idx, v_idx)]

        segments.append(seg)
        slopes[seg] = attrs["slope"]
        distances[seg] = attrs["distance"]
        lines[seg] = attrs["line"]

        cumulative_drop += (elevs[u_idx] - elevs[v_idx])
        total_length += attrs["distance"]

    avg_slope = sum(slopes.values()) / len(slopes) if slopes else 0.0

    print("\nPath Statistics:")
    print(f"Head: {idx_to_id[head_idx]}  →  Target: {idx_to_id[target_idx]}")
    print(f"Segments: {len(segments)}")
    print(f"Total length: {total_length:.1f} m")
    print(f"Total drop: {cumulative_drop:.2f} m")
    print(f"Avg slope: {avg_slope:.3%}")

    path_info = {
        "segments": segments,
        "slopes": slopes,
        "distances": distances,
        "lines": lines,
        "path_node_ids": [idx_to_id[i] for i in path_idxs],
        "path_node_indices": path_idxs,
        "reachable_node_indices": sorted(list(reachable)),
        "reachable_node_ids": [idx_to_id[i] for i in sorted(list(reachable))],
        "used_fallback_target": used_fallback,
        "target_id": idx_to_id[target_idx],
        "target_idx": target_idx,
        "cumulative_drop": cumulative_drop,
        "total_length": total_length,
        "avg_slope": avg_slope,
    }

    return segments, path_info

def generate_main_sewer_path_optimized(
    manholes,
    road_buffer,
    block_size=40.0,
    slope_tolerance=-0.01,
    min_pipe_length=5.0,
    prefer_slope=0.6,
    return_graph_data=False,
):
    """
    High-level wrapper:
      1) Build candidate graph
      2) Extract best main path

    Returns
    -------
    segments, path_info
    or
    segments, path_info, graph_data
    """
    graph_data = build_main_candidate_graph(
        manholes=manholes,
        road_buffer=road_buffer,
        block_size=block_size,
        slope_tolerance=slope_tolerance,
        min_pipe_length=min_pipe_length,
        prefer_slope=prefer_slope,
    )

    segments, path_info = extract_main_path(graph_data)

    if return_graph_data:
        return segments, path_info, graph_data
    return segments, path_info

def build_incidence_matrix(graph_data):
    """
    Build directed node-edge incidence matrix B.

    Convention:
        -1 at upstream node
        +1 at downstream node

    Returns
    -------
    B : scipy.sparse.csr_matrix
    edge_list : list[(u_idx, v_idx)]
    """
    A = graph_data["A"]
    n_nodes = A.shape[0]

    coo = A.tocoo()
    edge_list = list(zip(coo.row.tolist(), coo.col.tolist()))
    n_edges = len(edge_list)

    if n_edges == 0:
        return coo_matrix((n_nodes, 0)).tocsr(), []

    rows = []
    cols = []
    vals = []

    for e_idx, (u, v) in enumerate(edge_list):
        rows.extend([u, v])
        cols.extend([e_idx, e_idx])
        vals.extend([-1.0, 1.0])

    B = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_edges)).tocsr()
    return B, edge_list


def build_edge_table(graph_data):
    """
    Convert graph edges into a simple Python list of dictionaries.
    """
    node_data = graph_data["node_data"]
    idx_to_id = node_data["idx_to_id"]
    edge_attrs = graph_data["edge_attrs"]

    edge_table = []
    for (u_idx, v_idx), attrs in edge_attrs.items():
        edge_table.append({
            "u_idx": u_idx,
            "v_idx": v_idx,
            "u_id": idx_to_id[u_idx],
            "v_id": idx_to_id[v_idx],
            "distance": attrs["distance"],
            "slope": attrs["slope"],
            "cost": attrs["cost"],
            "line": attrs["line"],
        })
    return edge_table

def build_secondary_candidate_edges(
    manholes,
    connected_ids,
    road_buffer,
    main_path,
    block_size=40.0,
    slope_tolerance=0.0,
    prefer_slope=0.7,
):
    """
    Build valid candidate secondary edges from unconnected manholes
    to already-connected nodes.

    Parameters
    ----------
    manholes : list[dict]
    connected_ids : set
        Node IDs already connected to the network
    road_buffer : Polygon or MultiPolygon
    main_path : list[(u_id, v_id)]
    block_size : float
    slope_tolerance : float
    prefer_slope : float

    Returns
    -------
    candidate_edges : dict
        candidate_edges[src_id] = list of candidate dicts
    aux_data : dict
        reusable support data
    """
    import numpy as np
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.strtree import STRtree

    # -----------------------------
    # Prepare manhole data
    # -----------------------------
    ids = [mh["id"] for mh in manholes]
    points = [mh["location"] for mh in manholes]
    elevs = np.asarray([mh["elevation"] for mh in manholes], dtype=float)
    xs = np.asarray([pt.x for pt in points], dtype=float)
    ys = np.asarray([pt.y for pt in points], dtype=float)

    id_to_idx = {mid: i for i, mid in enumerate(ids)}
    idx_to_id = {i: mid for i, mid in enumerate(ids)}
    id_to_mh = {mh["id"]: mh for mh in manholes}
    point_tree = STRtree(points)

    # -----------------------------
    # Prepare road data
    # -----------------------------
    if isinstance(road_buffer, Polygon):
        road_polys = [road_buffer]
    elif isinstance(road_buffer, MultiPolygon):
        road_polys = list(road_buffer.geoms)
    else:
        raise ValueError("road_buffer must be Polygon or MultiPolygon")

    road_tree = STRtree(road_polys)

    def line_in_road(line):
        candidate_idxs = road_tree.query(line)
        for idx in candidate_idxs:
            if road_polys[idx].covers(line):
                return True
        return False

    # -----------------------------
    # Prepare main-path lines
    # -----------------------------
    main_lines = []
    for u, v in main_path:
        if u not in id_to_mh or v not in id_to_mh:
            continue
        main_lines.append(LineString([id_to_mh[u]["location"], id_to_mh[v]["location"]]))

    main_tree = STRtree(main_lines) if main_lines else None

    def overlaps_main_path(line, tolerance=0.01):
        if main_tree is None:
            return False

        for seg in main_tree.query(line):
            if not isinstance(seg, LineString):
                continue

            if line.equals_exact(seg, tolerance):
                return True

            # interior overlap
            if line.relate_pattern(seg, "1********"):
                return True

        return False

    def crosses_other_manhole(line, src_idx, tgt_idx, buffer_radius):
        near_idxs = point_tree.query(line.buffer(buffer_radius))
        for k in near_idxs:
            if k == src_idx or k == tgt_idx:
                continue

            pt_k = points[k]
            if line.distance(pt_k) < buffer_radius:
                proj = line.project(pt_k)
                if 0.0 < proj < line.length:
                    return True
        return False

    # -----------------------------
    # Build candidate edges
    # -----------------------------
    connected_ids = set(connected_ids)
    unconnected_ids = [mid for mid in ids if mid not in connected_ids]

    candidate_edges = {}
    search_radius = block_size * 2.0
    no_cross_radius = block_size * 0.3

    for src_id in unconnected_ids:
        i = id_to_idx[src_id]
        pt_i = points[i]
        elev_i = elevs[i]
        x_i = xs[i]
        y_i = ys[i]

        nearby_idxs = point_tree.query(pt_i.buffer(search_radius))
        candidates_for_src = []

        for j in nearby_idxs:
            tgt_id = idx_to_id[j]

            if tgt_id == src_id:
                continue

            if tgt_id not in connected_ids:
                continue

            dx = xs[j] - x_i
            dy = ys[j] - y_i
            dist = float(np.hypot(dx, dy))

            if dist < 1e-6:
                continue

            slope = (elev_i - elevs[j]) / max(dist, 1e-12)
            if slope < slope_tolerance:
                continue

            line = LineString([pt_i, points[j]])

            if not line_in_road(line):
                continue

            if crosses_other_manhole(line, i, j, no_cross_radius):
                continue

            if overlaps_main_path(line):
                continue

            slope_bonus = 1.0 - prefer_slope * max(0.0, slope)
            cost = dist * slope_bonus

            candidates_for_src.append({
                "src_id": src_id,
                "tgt_id": tgt_id,
                "src_idx": i,
                "tgt_idx": j,
                "distance": dist,
                "slope": slope,
                "cost": cost,
                "line": line,
            })

        candidate_edges[src_id] = candidates_for_src

    aux_data = {
        "ids": ids,
        "points": points,
        "elevs": elevs,
        "xs": xs,
        "ys": ys,
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "id_to_mh": id_to_mh,
    }

    return candidate_edges, aux_data

def select_best_secondary_edges(candidate_edges):
    """
    Select the best candidate edge for each source node.

    Parameters
    ----------
    candidate_edges : dict

    Returns
    -------
    selected_edges : list[(src_id, tgt_id)]
    selected_attrs : dict
        selected_attrs[(src_id, tgt_id)] = candidate dict
    """
    selected_edges = []
    selected_attrs = {}

    for src_id, candidates in candidate_edges.items():
        if not candidates:
            continue

        # best = minimum cost
        best = min(candidates, key=lambda c: (c["cost"], -c["slope"], c["distance"]))
        edge = (best["src_id"], best["tgt_id"])

        selected_edges.append(edge)
        selected_attrs[edge] = best

    return selected_edges, selected_attrs

def generate_secondary_pipes_optimized(
    manholes,
    main_path,
    road_buffer,
    block_size=40.0,
    slope_tolerance=0.0,
    prefer_slope=0.7,
    return_attrs=False,
):
    """
    Iteratively connect unconnected manholes to the already-connected network
    using best valid secondary edges.

    Parameters
    ----------
    manholes : list[dict]
    main_path : list[(u_id, v_id)]
    road_buffer : Polygon or MultiPolygon
    block_size : float
    slope_tolerance : float
    prefer_slope : float
    return_attrs : bool

    Returns
    -------
    secondary_pipes : list[(u_id, v_id)]
    secondary_attrs : dict, optional
    """
    connected_ids = set()
    for u, v in main_path:
        connected_ids.add(u)
        connected_ids.add(v)

    all_ids = {mh["id"] for mh in manholes}
    secondary_pipes = []
    secondary_attrs = {}

    while True:
        unconnected_ids = all_ids - connected_ids
        if not unconnected_ids:
            break

        candidate_edges, aux_data = build_secondary_candidate_edges(
            manholes=manholes,
            connected_ids=connected_ids,
            road_buffer=road_buffer,
            main_path=main_path + secondary_pipes,
            block_size=block_size,
            slope_tolerance=slope_tolerance,
            prefer_slope=prefer_slope,
        )

        selected_edges, selected_edge_attrs = select_best_secondary_edges(candidate_edges)

        if not selected_edges:
            print("⚠️ Some manholes could not be connected while preserving constraints.")
            break

        newly_connected = set()

        for u, v in selected_edges:
            if u in connected_ids:
                continue

            secondary_pipes.append((u, v))
            secondary_attrs[(u, v)] = selected_edge_attrs[(u, v)]
            newly_connected.add(u)

        if not newly_connected:
            print("⚠️ No additional secondary connections could be added.")
            break

        connected_ids.update(newly_connected)

    print(f"✅ Generated {len(secondary_pipes)} secondary pipes.")
    if return_attrs:
        return secondary_pipes, secondary_attrs
    return secondary_pipes

def remove_secondary_pipes_overlapping_main_optimized(
    manholes,
    secondary_pipes,
    main_pipes,
):
    """
    Final safety cleanup: remove secondary pipes that overlap main pipes
    beyond endpoint touching.

    Parameters
    ----------
    manholes : list[dict]
    secondary_pipes : list[(u_id, v_id)]
    main_pipes : list[(u_id, v_id)]

    Returns
    -------
    cleaned_secondary : list[(u_id, v_id)]
    """
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    id_map = {mh["id"]: mh for mh in manholes}

    main_lines = []
    for u, v in main_pipes:
        if u in id_map and v in id_map:
            main_lines.append(LineString([id_map[u]["location"], id_map[v]["location"]]))

    main_tree = STRtree(main_lines) if main_lines else None
    cleaned_secondary = []

    for u, v in secondary_pipes:
        line = LineString([id_map[u]["location"], id_map[v]["location"]])

        p0 = line.coords[0]
        p1 = line.coords[-1]

        overlap = False

        if main_tree is not None:
            for seg in main_tree.query(line):
                if not isinstance(seg, LineString):
                    continue

                if not line.intersects(seg):
                    continue

                inter = line.intersection(seg)

                if inter.is_empty:
                    continue

                if inter.geom_type == "Point":
                    continue

                if inter.geom_type == "MultiPoint":
                    continue

                # line overlap or more complex intersection
                overlap = True
                break

        if not overlap:
            cleaned_secondary.append((u, v))
        else:
            print(f"❌ Removed overlapping secondary pipe: {u} → {v}")

    print(f"✅ Cleaned: {len(secondary_pipes) - len(cleaned_secondary)} secondary pipes removed.")
    return cleaned_secondary

def build_current_network_status(
    manholes,
    main_path,
    secondary_pipes,
):
    """
    Build the current sewer network status from main + secondary pipes.

    Parameters
    ----------
    manholes : list[dict]
        Each dict must contain at least:
        {
            "id": ...,
            "location": shapely Point,
            "elevation": float
        }
    main_path : list[(u, v)]
    secondary_pipes : list[(u, v)]

    Returns
    -------
    network_status : dict
        {
            "all_ids": list[str],
            "main_outlet_id": str or None,
            "full_edges": list[(u, v)],
            "outgoing_from": dict,
            "incoming_to": dict,
            "nodes_in_network": set[str],
            "missing_outlet_ids": list[str],
            "duplicate_sources": list[str],
        }
    """
    ids = [str(mh["id"]) for mh in manholes]
    all_ids = set(ids)

    main_edges = [(str(u), str(v)) for u, v in main_path]
    sec_edges = [(str(u), str(v)) for u, v in secondary_pipes]
    full_edges = main_edges + sec_edges

    main_outlet_id = str(main_edges[-1][1]) if len(main_edges) > 0 else None

    outgoing_from = {}
    incoming_to = {}
    duplicate_sources = []
    nodes_in_network = set()

    for u, v in full_edges:
        nodes_in_network.add(u)
        nodes_in_network.add(v)

        if u in outgoing_from and outgoing_from[u] != v:
            duplicate_sources.append(u)
        else:
            outgoing_from[u] = v

        if v not in incoming_to:
            incoming_to[v] = []
        incoming_to[v].append(u)

    missing_outlet_ids = sorted([
        mid for mid in ids
        if mid != main_outlet_id and mid not in outgoing_from
    ])

    return {
        "all_ids": ids,
        "main_outlet_id": main_outlet_id,
        "full_edges": full_edges,
        "outgoing_from": outgoing_from,
        "incoming_to": incoming_to,
        "nodes_in_network": nodes_in_network,
        "missing_outlet_ids": missing_outlet_ids,
        "duplicate_sources": sorted(list(set(duplicate_sources))),
    }

def generate_tertiary_pipes_backtracking_stop_at_each_manhole(
    manholes,
    main_path,
    secondary_pipes,
    road_buffer,
    city_boundary,
    block_size=60.0,
    neighbor_radius_factor=1.5,
    min_pipe_length=1e-3,
    point_on_line_tol=0.01,
    return_attrs=False,
    max_search_depth=300,
    max_outer_iterations=10000,
):
    """
    Generate tertiary pipes using neighbor-to-neighbor backtracking.

    CRITICAL RULE
    -------------
    A pipe must stop at each manhole.
    Therefore, an edge A->B is VALID only if there is no other manhole lying
    on the segment between A and B.

    Strategy
    --------
    1. Find manholes missing an outlet pipe.
    2. Start from the outermost missing manhole (closest to city boundary).
    3. Search neighbor-to-neighbor paths with backtracking.
    4. Commit the first path that reaches the connected system.
    5. Repeat until no more progress is possible.

    Connected system includes:
    - main-path nodes
    - secondary-pipe nodes
    - previously committed tertiary-pipe nodes

    Returns
    -------
    tertiary_pipes : list[(u, v)]
    still_missing_ids : list[str]
    tertiary_attrs : dict, optional
    """

    import os
    import numpy as np
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.prepared import prep

    # ------------------------------------------------------------------
    # 1) PREPARE NODE DATA
    # ------------------------------------------------------------------
    ids = [str(mh["id"]) for mh in manholes]
    id_map = {str(mh["id"]): mh for mh in manholes}

    points = [id_map[mid]["location"] for mid in ids]
    elevs = np.asarray([id_map[mid]["elevation"] for mid in ids], dtype=float)
    xs = np.asarray([pt.x for pt in points], dtype=float)
    ys = np.asarray([pt.y for pt in points], dtype=float)

    id_to_idx = {mid: i for i, mid in enumerate(ids)}
    idx_to_id = {i: mid for i, mid in enumerate(ids)}

    coords = np.column_stack([xs, ys])
    kd = cKDTree(coords)

    # ------------------------------------------------------------------
    # 2) PREPARE ROAD GEOMETRY
    # ------------------------------------------------------------------
    if not isinstance(road_buffer, (Polygon, MultiPolygon)):
        raise ValueError("road_buffer must be Polygon or MultiPolygon")
    road_prepared = prep(road_buffer)

    # ------------------------------------------------------------------
    # 3) PREPARE CITY BOUNDARY GEOMETRY
    # ------------------------------------------------------------------
    if isinstance(city_boundary, str):
        if not os.path.exists(city_boundary):
            raise FileNotFoundError(f"City boundary file not found: {city_boundary}")
        boundary_gdf = gpd.read_file(city_boundary)
        if boundary_gdf.empty:
            raise ValueError("City boundary shapefile is empty.")
        boundary_geom = boundary_gdf.geometry.union_all()

    elif hasattr(city_boundary, "geometry"):
        if len(city_boundary) == 0:
            raise ValueError("city_boundary GeoDataFrame is empty.")
        boundary_geom = city_boundary.geometry.union_all()

    elif hasattr(city_boundary, "union_all"):
        if len(city_boundary) == 0:
            raise ValueError("city_boundary GeoSeries is empty.")
        boundary_geom = city_boundary.union_all()

    elif isinstance(city_boundary, (Polygon, MultiPolygon)):
        boundary_geom = city_boundary

    else:
        raise ValueError(
            "city_boundary must be a shapefile path, GeoDataFrame, GeoSeries, "
            "Polygon, or MultiPolygon."
        )

    # ------------------------------------------------------------------
    # 4) INITIAL NETWORK STATUS
    # ------------------------------------------------------------------
    main_path = [(str(u), str(v)) for u, v in main_path]
    secondary_pipes = [(str(u), str(v)) for u, v in secondary_pipes]
    base_edges = main_path + secondary_pipes

    main_outlet_id = str(main_path[-1][1]) if main_path else None

    outgoing_from = {}
    network_nodes = set()

    for u, v in base_edges:
        network_nodes.add(u)
        network_nodes.add(v)
        outgoing_from[u] = v

    tertiary_pipes = []
    tertiary_attrs = {}

    # ------------------------------------------------------------------
    # 5) NEIGHBOR SEARCH
    # ------------------------------------------------------------------
    neighbor_radius = block_size * neighbor_radius_factor
    neighbors_by_idx = kd.query_ball_point(coords, r=neighbor_radius)

    # ------------------------------------------------------------------
    # 6) HELPERS
    # ------------------------------------------------------------------
    def nodes_missing_outlet():
        return [mid for mid in ids if mid != main_outlet_id and mid not in outgoing_from]

    def boundary_distance(mid):
        return id_map[mid]["location"].distance(boundary_geom.boundary)

    def distance_to_network(mid):
        i = id_to_idx[mid]
        x0, y0 = xs[i], ys[i]
        best = float("inf")
        for nid in network_nodes:
            j = id_to_idx[nid]
            d = np.hypot(xs[j] - x0, ys[j] - y0)
            if d < best:
                best = d
        return best

    def would_create_cycle_with_temp(u, v, temp_outgoing):
        cur = v
        seen = set()

        while True:
            if cur == u:
                return True
            if cur in seen:
                return False
            seen.add(cur)

            if cur in temp_outgoing:
                cur = temp_outgoing[cur]
            elif cur in outgoing_from:
                cur = outgoing_from[cur]
            else:
                break

        return False

    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom <= 1e-20:
            return np.hypot(px - x1, py - y1), 0.0

        t = ((px - x1) * dx + (py - y1) * dy) / denom
        t_clip = max(0.0, min(1.0, t))
        projx = x1 + t_clip * dx
        projy = y1 + t_clip * dy
        dist = np.hypot(px - projx, py - projy)
        return dist, t_clip

    def has_intermediate_manhole(a, b, tol=0.01):
        """
        Return True if there exists any other manhole on the segment a->b.
        This enforces that a pipe must stop at each manhole.
        """
        ia = id_to_idx[a]
        ib = id_to_idx[b]

        x1, y1 = xs[ia], ys[ia]
        x2, y2 = xs[ib], ys[ib]

        minx = min(x1, x2) - tol
        maxx = max(x1, x2) + tol
        miny = min(y1, y2) - tol
        maxy = max(y1, y2) + tol

        # search nearby candidates from both endpoints
        candidate_ks = set(neighbors_by_idx[ia]) | set(neighbors_by_idx[ib])

        for k in candidate_ks:
            nid = idx_to_id[k]
            if nid in (a, b):
                continue

            px, py = xs[k], ys[k]
            if px < minx or px > maxx or py < miny or py > maxy:
                continue

            dist, t = point_to_segment_distance(px, py, x1, y1, x2, y2)
            if dist <= tol and 0.0 < t < 1.0:
                return True

        return False

    def validate_neighbor_edge(a, b, temp_outgoing):
        """
        Validate ONE direct neighbor edge a -> b.
        Reject it if another manhole lies on the segment.
        """
        ia = id_to_idx[a]
        ib = id_to_idx[b]

        dx = xs[ib] - xs[ia]
        dy = ys[ib] - ys[ia]
        dist = float(np.hypot(dx, dy))

        if dist < min_pipe_length:
            return None

        line = LineString([points[ia], points[ib]])

        if not road_prepared.covers(line):
            return None

        if a in outgoing_from or a in temp_outgoing:
            return None

        if would_create_cycle_with_temp(a, b, temp_outgoing):
            return None

        # critical check: do not allow segment to pass over another manhole
        if has_intermediate_manhole(a, b, tol=point_on_line_tol):
            return None

        slope = (elevs[ia] - elevs[ib]) / max(dist, 1e-12)

        return {
            "segment": (a, b),
            "line": line,
            "distance": dist,
            "slope": slope,
        }

    def ordered_neighbors(current_id, visited, temp_outgoing):
        """
        Order valid neighboring manholes by:
        1. already in connected system
        2. closer to connected system
        3. more inward from boundary
        4. lower elevation
        5. shorter distance
        """
        i = id_to_idx[current_id]
        candidates = []

        for j in neighbors_by_idx[i]:
            if j == i:
                continue

            nid = idx_to_id[j]
            if nid in visited:
                continue

            valid = validate_neighbor_edge(current_id, nid, temp_outgoing)
            if valid is None:
                continue

            score = (
                0 if nid in network_nodes else 1,
                distance_to_network(nid),
                -boundary_distance(nid),
                elevs[j],
                valid["distance"],
            )
            candidates.append((score, nid, valid))

        candidates.sort(key=lambda x: x[0])
        return candidates

    def dfs_find_path(
        current_id,
        start_id,
        visited,
        temp_outgoing,
        temp_segments,
        temp_lines,
        temp_slopes,
        depth,
    ):
        """
        Recursive backtracking over neighbor manholes only.
        """
        if depth > max_search_depth:
            return False

        # success if we reached the connected system after at least one edge
        if current_id in network_nodes and current_id != start_id and len(temp_segments) > 0:
            return True

        for _, nid, valid in ordered_neighbors(current_id, visited, temp_outgoing):
            a, b = valid["segment"]

            # choose
            visited.add(nid)
            temp_outgoing[a] = b
            temp_segments.append((a, b))
            temp_lines.append(valid["line"])
            temp_slopes.append(valid["slope"])

            # recurse
            found = dfs_find_path(
                current_id=nid,
                start_id=start_id,
                visited=visited,
                temp_outgoing=temp_outgoing,
                temp_segments=temp_segments,
                temp_lines=temp_lines,
                temp_slopes=temp_slopes,
                depth=depth + 1,
            )
            if found:
                return True

            # backtrack
            visited.remove(nid)
            temp_outgoing.pop(a, None)
            temp_segments.pop()
            temp_lines.pop()
            temp_slopes.pop()

        return False

    def commit_chain(temp_segments, temp_lines, temp_slopes):
        for (a, b), line, slope in zip(temp_segments, temp_lines, temp_slopes):
            seg_dist = float(line.length)

            tertiary_pipes.append((a, b))
            tertiary_attrs[(a, b)] = {
                "distance": seg_dist,
                "slope": slope,
                "cost": seg_dist,
                "line": line,
                "ptype": "tertiary",
            }

            outgoing_from[a] = b
            network_nodes.add(a)
            network_nodes.add(b)

    # ------------------------------------------------------------------
    # 7) OUTER LOOP OVER MISSING MANHOLES
    # ------------------------------------------------------------------
    print(f"📊 Initial missing outlet pipes: {len(nodes_missing_outlet())}")

    blocked_starts = set()
    outer_it = 0

    while outer_it < max_outer_iterations:
        missing = nodes_missing_outlet()
        if not missing:
            break

        available = [mid for mid in missing if mid not in blocked_starts]
        if not available:
            print("⚠️ No more progress possible with current constraints.")
            break

        outer_it += 1

        # Start with the most external missing manhole
        start_id = min(available, key=boundary_distance)

        visited = {start_id}
        temp_outgoing = {}
        temp_segments = []
        temp_lines = []
        temp_slopes = []

        found = dfs_find_path(
            current_id=start_id,
            start_id=start_id,
            visited=visited,
            temp_outgoing=temp_outgoing,
            temp_segments=temp_segments,
            temp_lines=temp_lines,
            temp_slopes=temp_slopes,
            depth=0,
        )

        if found and temp_segments:
            commit_chain(temp_segments, temp_lines, temp_slopes)
            blocked_starts.clear()
            status = "committed"
        else:
            blocked_starts.add(start_id)
            status = "discarded"

        print(
            f"Iteration {outer_it}: start={start_id}, status={status}, "
            f"chain_len={len(temp_segments)}, remaining={len(nodes_missing_outlet())}"
        )

    # ------------------------------------------------------------------
    # 8) FINAL CLEANUP
    # ------------------------------------------------------------------
    final_by_source = {}
    removed_conflicts = 0

    for u, v in tertiary_pipes:
        attrs = tertiary_attrs[(u, v)]

        if u not in final_by_source:
            final_by_source[u] = (u, v)
        else:
            keep = final_by_source[u]
            keep_attrs = tertiary_attrs[keep]

            new_key = (attrs["cost"], -attrs["slope"], attrs["distance"])
            old_key = (keep_attrs["cost"], -keep_attrs["slope"], keep_attrs["distance"])

            if new_key < old_key:
                final_by_source[u] = (u, v)
                removed_conflicts += 1
            else:
                removed_conflicts += 1

    final_tertiary = list(final_by_source.values())
    final_attrs = {e: tertiary_attrs[e] for e in final_tertiary}

    still_missing_ids = nodes_missing_outlet()

    print("\n📊 FINAL RESULTS")
    print(f"✅ Tertiary pipes generated: {len(final_tertiary)}")
    print(f"🚫 Tertiary conflicts removed: {removed_conflicts}")
    print(f"⚠️ Still missing outlet pipe: {len(still_missing_ids)}")
    if still_missing_ids:
        print(f"   First few: {still_missing_ids[:10]}")

    if return_attrs:
        return final_tertiary, still_missing_ids, final_attrs
    return final_tertiary, still_missing_ids

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

def export_pipes_to_shapefile_2(
    pipes_main,
    pipes_sec,
    pipes_ter,
    manholes,
    output_path,
    crs="EPSG:32618",
    main_attrs=None,
    secondary_attrs=None,
    tertiary_attrs=None,
):
    """
    Export all pipes (main, secondary, tertiary) to a single shapefile.

    Fields exported
    ---------------
    pipe_id       : unique pipe ID
    upstream_m    : upstream manhole ID
    downstream_m  : downstream manhole ID
    ptype         : main / secondary / tertiary
    dist_m        : pipe length
    slope         : slope
    cost          : candidate cost if available
    geometry      : LineString

    Parameters
    ----------
    pipes_main : list[(u, v)]
    pipes_sec : list[(u, v)]
    pipes_ter : list[(u, v)]
    manholes : list[dict]
        Each dict must contain:
        {
            "id": ...,
            "location": shapely Point,
            "elevation": float
        }
    output_path : str
        Output shapefile path
    crs : str
        CRS string, e.g. "EPSG:32618"
    main_attrs : dict or None
        Optional dictionary keyed by (u, v), such as:
        {
            (u, v): {"line": ..., "distance": ..., "slope": ..., "cost": ...}
        }
        For main pipes, you can build this from path_info.
    secondary_attrs : dict or None
    tertiary_attrs : dict or None

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    manhole_map = {str(mh["id"]): mh for mh in manholes}

    def get_pipe_record(u, v, pipe_type, pipe_id, attrs_dict=None):
        u = str(u)
        v = str(v)

        if u not in manhole_map or v not in manhole_map:
            return None

        pt_u = manhole_map[u]["location"]
        pt_v = manhole_map[v]["location"]

        geom = None
        dist = None
        slope = None
        cost = None

        if attrs_dict is not None and (u, v) in attrs_dict:
            attrs = attrs_dict[(u, v)]

            if "line" in attrs and attrs["line"] is not None:
                geom = attrs["line"]

            if "distance" in attrs:
                dist = float(attrs["distance"])

            if "slope" in attrs:
                slope = float(attrs["slope"])

            if "cost" in attrs:
                cost = float(attrs["cost"])

        if geom is None:
            geom = LineString([pt_u, pt_v])

        if dist is None:
            dist = float(pt_u.distance(pt_v))

        if slope is None:
            elev_u = float(manhole_map[u]["elevation"])
            elev_v = float(manhole_map[v]["elevation"])
            slope = (elev_u - elev_v) / max(dist, 1e-12)

        if cost is None:
            cost = dist

        return {
            "pipe_id": pipe_id,
            "upstream_m": u,
            "downstream_m": v,
            "type": pipe_type,
            "dist_m": dist,
            "slope": slope,
            "cost": cost,
            "geometry": geom,
        }

    def build_records(pipe_list, pipe_type, start_idx, attrs_dict=None):
        records = []
        for i, (u, v) in enumerate(pipe_list):
            rec = get_pipe_record(
                u=u,
                v=v,
                pipe_type=pipe_type,
                pipe_id=f"P{start_idx + i:05d}",
                attrs_dict=attrs_dict,
            )
            if rec is not None:
                records.append(rec)
        return records

    rec_main = build_records(
        pipe_list=pipes_main,
        pipe_type="main",
        start_idx=0,
        attrs_dict=main_attrs,
    )

    rec_sec = build_records(
        pipe_list=pipes_sec,
        pipe_type="secondary",
        start_idx=len(rec_main),
        attrs_dict=secondary_attrs,
    )

    rec_ter = build_records(
        pipe_list=pipes_ter,
        pipe_type="tertiary",
        start_idx=len(rec_main) + len(rec_sec),
        attrs_dict=tertiary_attrs,
    )

    all_records = rec_main + rec_sec + rec_ter

    if not all_records:
        print("⚠️ No valid pipes found to export.")
        return None

    df = pd.DataFrame(all_records)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path)

    print(f"✅ Exported {len(all_records)} pipes to: {output_path}")
    print(f"   Main: {len(rec_main)} | Secondary: {len(rec_sec)} | Tertiary: {len(rec_ter)}")
    print(f"   CRS: {crs}")

    return gdf


def build_main_attrs_from_path_info(path_info):
    """
    Convert path_info into a main_attrs dictionary compatible with
    export_pipes_to_shapefile().
    """
    main_attrs = {}

    for seg in path_info["segments"]:
        main_attrs[(str(seg[0]), str(seg[1]))] = {
            "line": path_info["lines"].get(seg, None),
            "distance": path_info["distances"].get(seg, None),
            "slope": path_info["slopes"].get(seg, None),
            "cost": path_info["distances"].get(seg, None),  # default cost for main
        }

    return main_attrs

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
    pipes = gpd.read_file(pipes_path)
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