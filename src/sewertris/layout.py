"""Tetris-style urban layout generation."""

from __future__ import annotations

from ._deps import *


def get_tetromino_set(name: str = "full"):
    """Return a named tetromino dictionary and color map."""
    key = str(name).lower().replace("-", "_").replace(" ", "_")

    tetrominoes_full = {
        'I': [np.array([[1,1,1,1]]), np.array([[1],[1],[1],[1]])],
        'O': [np.array([[1,1],[1,1]])],
        'T': [np.array([[1,1,1],[0,1,0]]), np.array([[0,1],[1,1],[0,1]]), np.array([[0,1,0],[1,1,1]]), np.array([[1,0],[1,1],[1,0]])],
        'S': [np.array([[0,1,1],[1,1,0]]), np.array([[1,0],[1,1],[0,1]])],
        'Z': [np.array([[1,1,0],[0,1,1]]), np.array([[0,1],[1,1],[1,0]])],
        'J': [np.array([[1,0,0],[1,1,1]]), np.array([[1,1],[1,0],[1,0]]), np.array([[1,1,1],[0,0,1]]), np.array([[0,1],[0,1],[1,1]])],
        'L': [np.array([[0,0,1],[1,1,1]]), np.array([[1,0],[1,0],[1,1]]), np.array([[1,1,1],[1,0,0]]), np.array([[1,1],[0,1],[0,1]])],
        'BO': [np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])],
    }
    colors_full = {
        'I': 'cyan',
        'O': 'yellow',
        'T': 'purple',
        'S': 'green',
        'Z': 'red',
        'J': 'blue',
        'L': 'orange',
        'BO': 'grey',
    }

    aliases_full = {"full", "all", "example_02", "stillwater_full"}
    aliases_four = {"i_o_t_s_only", "iots", "4tetromino", "4_tetromino", "four", "example_03"}
    if key in aliases_full:
        return tetrominoes_full, colors_full
    if key in aliases_four:
        selected = ("I", "O", "T", "S")
        return (
            {shape: tetrominoes_full[shape] for shape in selected},
            {shape: colors_full[shape] for shape in selected},
        )
    raise ValueError(f"Unknown tetromino set '{name}'.")

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
            rotations = list(tetrominoes[tkey])
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

    save_vector(gdf, output_path)
    print(f"✅ Exported {len(gdf)} figures to {output_path}")

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
    output_path = os.fspath(output_path)
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "filled_board.gpkg")
    if not output_path.lower().endswith((".shp", ".gpkg")):
        output_path = output_path + ".gpkg"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
    save_vector(gdf, output_path)
    print(f"✅ Exported {len(gdf)} figures to {output_path}")
    return gdf

__all__ = [
    "get_tetromino_set",
    "can_place",
    "place_piece",
    "fill_domain_with_tetrominoes_and_blocks",
    "export_individual_figures_to_shapefile",
    "export_individual_figures_to_shapefile_georeferenced",
]
