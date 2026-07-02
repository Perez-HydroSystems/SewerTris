"""Topography and DEM generation utilities."""

from __future__ import annotations

from ._deps import *
from . import plots

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
    # Calculate slopes. np.gradient returns rise/run (a dimensionless tangent);
    # convert to degrees so it matches the "Slope (degrees)" plot/labels.
    dy, dx = np.gradient(elevation, yy[:,0], xx[0,:])
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
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
        sewer_geom = linemerge(sewer_gdf.geometry.union_all())
    except Exception:
        sewer_geom = sewer_gdf.geometry.union_all()

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
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
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
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
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
            base, ext = os.path.splitext(os.fspath(manholes_path))
            output_path = base + "_elev_updated" + ext
    output_path = os.fspath(output_path)

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

__all__ = [
    "TopographyConfig",
    "load_data",
    "find_road_intersections",
    "determine_outlet_point",
    "generate_base_topography",
    "optimize_drainage",
    "validate_drainage",
    "generate_topography",
    "modify_topography_with_sewers",
    "enforce_positive_slopes_tiered",
    "adjust_manhole_elevations_tiered",
    "verify_all_pipes_positive_by_nodes",
    "verify_pipes_on_raster",
    "interpolate_dem_idw_tiled",
    "centerline_writeback_linear",
    "build_dem_with_guaranteed_positive_slopes_idw",
    "update_manhole_elevations_from_dem",
]
