"""Road, block, boundary, and land-use utilities."""

from __future__ import annotations

from ._deps import *

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
        save_vector(gdf_lines, out_boundary_lines)
        print(f"[OK] Boundary lines written to: {out_boundary_lines}")

    # ----- Outer shell polygon(s) only (no holes) -----
    if out_outer_shell_polygon:
        shells = [Polygon(p.exterior) for p in polys]  # drop holes
        gdf_shells = gpd.GeoDataFrame({"id": range(1, len(shells)+1)}, geometry=shells, crs=crs)
        save_vector(gdf_shells, out_outer_shell_polygon)
        print(f"[OK] Outer shell polygon(s) written to: {out_outer_shell_polygon}")

    # Return the shapely results in case you want them in-memory
    return {
        "merged_polygon": MultiPolygon(polys) if len(polys) > 1 else polys[0],
        "boundary_lines": boundary_geom if out_boundary_lines else None,
        "outer_shell_polygons": shells if out_outer_shell_polygon else None,
    }

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
    save_vector(gdf, output_path)
    print(f"✅ Exported to {output_path}")
    return gdf

__all__ = [
    "generate_road_network_from_blocks",
    "extract_boundary",
    "load_blocks_and_roads",
    "cut_blocks",
    "assign_land_use_compact",
    "export_to_shapefile",
]
