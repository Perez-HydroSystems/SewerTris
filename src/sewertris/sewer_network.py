"""Sewer node, pipe, and graph generation utilities."""

from __future__ import annotations

from ._deps import *

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
    if isinstance(city_boundary, (str, os.PathLike)):
        city_boundary_path = os.fspath(city_boundary)
        if not os.path.exists(city_boundary_path):
            raise FileNotFoundError(f"City boundary file not found: {city_boundary_path}")
        boundary_gdf = gpd.read_file(city_boundary_path)
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

def _tertiary_v2_slope_cost(
    distance,
    slope,
    adverse_slope_weight=200.0,
    mild_adverse_slope=-0.005,
    moderate_adverse_slope=-0.01,
    severe_adverse_multiplier=8.0,
):
    """Distance-based cost with soft penalties for adverse slopes."""
    adverse = max(0.0, -float(slope))
    multiplier = 1.0 + adverse_slope_weight * adverse

    if slope < moderate_adverse_slope:
        multiplier += severe_adverse_multiplier
    elif slope < mild_adverse_slope:
        multiplier += severe_adverse_multiplier * 0.35

    return float(distance) * multiplier

def generate_tertiary_pipes_shortest_path_v2(
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
    adverse_slope_weight=200.0,
    mild_adverse_slope=-0.005,
    moderate_adverse_slope=-0.01,
    severe_adverse_multiplier=8.0,
    max_outer_iterations=10000,
):
    """
    Generate tertiary pipes with weighted shortest paths instead of DFS.

    The search preserves the tertiary rule that pipes must stop at intermediate
    manholes. Adverse slopes are allowed, but receive increasing cost penalties
    so the algorithm prefers downhill routes while still trying to connect all
    nodes.
    """
    import heapq
    import os
    import numpy as np
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.prepared import prep

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

    if not isinstance(road_buffer, (Polygon, MultiPolygon)):
        raise ValueError("road_buffer must be Polygon or MultiPolygon")
    road_prepared = prep(road_buffer)

    if isinstance(city_boundary, (str, os.PathLike)):
        city_boundary_path = os.fspath(city_boundary)
        if not os.path.exists(city_boundary_path):
            raise FileNotFoundError(f"City boundary file not found: {city_boundary_path}")
        boundary_gdf = gpd.read_file(city_boundary_path)
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

    neighbor_radius = block_size * neighbor_radius_factor
    neighbors_by_idx = kd.query_ball_point(coords, r=neighbor_radius)

    def nodes_missing_outlet():
        return [mid for mid in ids if mid != main_outlet_id and mid not in outgoing_from]

    def boundary_distance(mid):
        return id_map[mid]["location"].distance(boundary_geom.boundary)

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
        ia = id_to_idx[a]
        ib = id_to_idx[b]

        x1, y1 = xs[ia], ys[ia]
        x2, y2 = xs[ib], ys[ib]

        minx = min(x1, x2) - tol
        maxx = max(x1, x2) + tol
        miny = min(y1, y2) - tol
        maxy = max(y1, y2) + tol

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

    def build_local_edges():
        local_edges = {mid: [] for mid in ids}

        for i, a in enumerate(ids):
            if a == main_outlet_id:
                continue

            for j in neighbors_by_idx[i]:
                if j == i:
                    continue

                b = idx_to_id[j]
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist = float(np.hypot(dx, dy))
                if dist < min_pipe_length:
                    continue

                line = LineString([points[i], points[j]])
                if not road_prepared.covers(line):
                    continue

                if has_intermediate_manhole(a, b, tol=point_on_line_tol):
                    continue

                slope = (elevs[i] - elevs[j]) / max(dist, 1e-12)
                cost = _tertiary_v2_slope_cost(
                    distance=dist,
                    slope=slope,
                    adverse_slope_weight=adverse_slope_weight,
                    mild_adverse_slope=mild_adverse_slope,
                    moderate_adverse_slope=moderate_adverse_slope,
                    severe_adverse_multiplier=severe_adverse_multiplier,
                )
                local_edges[a].append({
                    "src_id": a,
                    "tgt_id": b,
                    "distance": dist,
                    "slope": slope,
                    "cost": cost,
                    "line": line,
                })

        return local_edges

    local_edges = build_local_edges()

    def find_best_path_to_network(start_id):
        queue = [(0.0, 0, start_id)]
        distances = {start_id: 0.0}
        predecessors = {}
        counter = 0

        while queue:
            cur_cost, _, cur_id = heapq.heappop(queue)
            if cur_cost > distances.get(cur_id, float("inf")):
                continue

            if cur_id in network_nodes and cur_id != start_id:
                path = []
                node = cur_id
                while node != start_id:
                    prev, attrs = predecessors[node]
                    path.append(attrs)
                    node = prev
                path.reverse()
                return path, cur_cost

            for attrs in local_edges.get(cur_id, []):
                nxt = attrs["tgt_id"]
                if cur_id in outgoing_from:
                    continue
                if nxt == start_id:
                    continue

                new_cost = cur_cost + attrs["cost"]
                if new_cost < distances.get(nxt, float("inf")):
                    distances[nxt] = new_cost
                    predecessors[nxt] = (cur_id, attrs)
                    counter += 1
                    heapq.heappush(queue, (new_cost, counter, nxt))

        return [], float("inf")

    def commit_path(path):
        for attrs in path:
            u = attrs["src_id"]
            v = attrs["tgt_id"]
            edge = (u, v)

            if u in outgoing_from:
                continue

            tertiary_pipes.append(edge)
            tertiary_attrs[edge] = {
                "distance": attrs["distance"],
                "slope": attrs["slope"],
                "cost": attrs["cost"],
                "line": attrs["line"],
                "ptype": "tertiary_v2",
            }
            outgoing_from[u] = v
            network_nodes.add(u)
            network_nodes.add(v)

    print(f"📊 V2 initial missing outlet pipes: {len(nodes_missing_outlet())}")

    blocked_starts = set()
    outer_it = 0

    while outer_it < max_outer_iterations:
        missing = nodes_missing_outlet()
        if not missing:
            break

        available = [mid for mid in missing if mid not in blocked_starts]
        if not available:
            print("⚠️ V2 no more progress possible with current constraints.")
            break

        outer_it += 1
        start_id = min(available, key=boundary_distance)
        path, path_cost = find_best_path_to_network(start_id)

        if path:
            commit_path(path)
            blocked_starts.clear()
            status = "committed"
        else:
            blocked_starts.add(start_id)
            status = "discarded"

        if outer_it <= 25 or outer_it % 25 == 0 or status == "discarded":
            print(
                f"V2 iteration {outer_it}: start={start_id}, status={status}, "
                f"chain_len={len(path)}, cost={path_cost:.2f}, remaining={len(nodes_missing_outlet())}"
            )

    still_missing_ids = nodes_missing_outlet()

    adverse = [
        attrs["slope"]
        for attrs in tertiary_attrs.values()
        if attrs.get("slope", 0.0) < 0.0
    ]

    print("\n📊 V2 FINAL RESULTS")
    print(f"✅ V2 tertiary pipes generated: {len(tertiary_pipes)}")
    print(f"⚠️ V2 still missing outlet pipe: {len(still_missing_ids)}")
    print(f"↘️ V2 adverse tertiary pipes: {len(adverse)}")
    if adverse:
        print(f"   Worst tertiary slope: {min(adverse):.3%}")
    if still_missing_ids:
        print(f"   First few: {still_missing_ids[:10]}")

    if return_attrs:
        return tertiary_pipes, still_missing_ids, tertiary_attrs
    return tertiary_pipes, still_missing_ids

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

__all__ = [
    "extract_manholes_from_lines",
    "export_manholes_to_shapefile",
    "generate_main_sewer_path",
    "generate_secondary_pipes",
    "remove_secondary_pipes_overlapping_main",
    "export_pipes_to_shapefile",
    "generate_tertiary_pipes",
    "export_tertiary_pipes_to_shapefile",
    "prepare_sewer_nodes",
    "prepare_road_index",
    "line_is_covered_by_road",
    "segment_crosses_other_manholes",
    "build_main_candidate_graph",
    "get_reachable_nodes",
    "choose_target_node",
    "reconstruct_path_from_predecessors",
    "extract_main_path",
    "generate_main_sewer_path_optimized",
    "build_incidence_matrix",
    "build_edge_table",
    "build_secondary_candidate_edges",
    "select_best_secondary_edges",
    "generate_secondary_pipes_optimized",
    "remove_secondary_pipes_overlapping_main_optimized",
    "build_current_network_status",
    "generate_tertiary_pipes_backtracking_stop_at_each_manhole",
    "generate_tertiary_pipes_shortest_path_v2",
    "export_pipes_to_shapefile_2",
    "build_main_attrs_from_path_info",
]
