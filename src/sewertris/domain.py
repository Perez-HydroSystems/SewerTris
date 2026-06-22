"""Domain and coordinate utilities."""

from __future__ import annotations

from ._deps import *

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


__all__ = [
    "download_city_boundary",
    "utm_epsg_from_lon",
    "meters_to_crs_units",
    "build_domain_mask_from_shapefile",
]
