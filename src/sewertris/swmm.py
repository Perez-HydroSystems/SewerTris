"""SWMM model creation, inflow assignment, simulation, and output utilities."""

from __future__ import annotations

from ._deps import *

def export_swmm_inp(pipes_path, manholes_path, output_path, title="Generated Sewer Network", options_dict=None):
    import geopandas as gpd
    import pandas as pd
    import os
    import math
    from shapely.geometry import Point

    def compute_diameter_manning(Q, n=0.013, S=0.01):
        return ((Q * n) / (0.463 * S**0.5))**(3/8)

    def round_commercial_diameter(d_m):
        commercial_sizes_mm = [100, 150, 200, 250, 300, 375, 450, 525, 600, 750, 900, 1050, 1200]
        d_mm = math.ceil(d_m * 1000)
        for size in commercial_sizes_mm:
            if size >= d_mm:
                return size / 1000
        return commercial_sizes_mm[-1] / 1000

    # Load data
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
    manholes = gpd.read_file(manholes_path)

    # Diameter handling
    diameter_col = next((col for col in ["diameter_mm", "diameter_m", "calc_diame", "diameter_1"] if col in pipes.columns), None)
    if not diameter_col:
        raise KeyError("No diameter column found.")
    pipes["diameter_m"] = pipes[diameter_col].astype(float)
    if pipes["diameter_m"].max() > 10:
        pipes["diameter_m"] /= 1000

    for col in ["length_m", "inv_up", "inv_down"]:
        if col not in pipes.columns:
            raise KeyError(f"Missing '{col}' column in pipes.")
        pipes[col] = pipes[col].astype(float)
    pipes["n"] = pipes.get("n", 0.013)

    # Identify outlet
    all_mh_ids = set(manholes["id"])
    upstream_ids = set(pipes["upstream_m"])
    outlet_candidates = list(all_mh_ids - upstream_ids)
    if not outlet_candidates:
        raise ValueError("No outlet manhole found.")
    last_mh_id = outlet_candidates[0]

    flow_col = next((col for col in [
        "flow_lps",
        "peak_flow_lps_bc",
        "peak_flow_",
        "predesign_ls",
        "predesign_",
        "peakflow",
        "q_peak",
    ] if col in pipes.columns), None)
    if not flow_col:
        raise KeyError("Missing peak flow column.")
    peak_flow = pipes[pipes["downstream"] == last_mh_id][flow_col].sum()
    final_d = round_commercial_diameter(compute_diameter_manning(peak_flow))
    last_inv = pipes[pipes["downstream"] == last_mh_id]["inv_down"].min()

    outlet_mh_id = "OUTLET"
    outlet_pipe_id = "P_OUTLET"
    outlet_length = 5.0
    inv_outlet = last_inv - 0.01 * outlet_length

    new_pipe = pd.DataFrame([{
        'pipe_id': outlet_pipe_id,
        'upstream_m': last_mh_id,
        'downstream': outlet_mh_id,
        'length_m': outlet_length,
        'n': 0.013,
        'inv_up': last_inv,
        'inv_down': inv_outlet,
        'diameter_m': final_d
    }])
    pipes = pd.concat([pipes, new_pipe], ignore_index=True)

    last_geom = manholes.set_index("id").loc[last_mh_id].geometry
    outlet_geom = Point(last_geom.x + 5, last_geom.y)
    outlet_mh = gpd.GeoDataFrame([{
        "id": outlet_mh_id,
        "elevation": inv_outlet + 1,
        "geometry": outlet_geom
    }], geometry="geometry", crs=manholes.crs)
    manholes = pd.concat([manholes, outlet_mh], ignore_index=True)

    inv_map = pipes.set_index("upstream_m")["inv_up"].to_dict()
    manholes["invert_elev"] = manholes["id"].map(inv_map)
    manholes["invert_elev"] = manholes["invert_elev"].fillna(manholes["elevation"] - 1.0)
    manholes["max_depth"] = (manholes["elevation"] - manholes["invert_elev"]).clip(lower=1.0)

    own_flow_col = next((col for col in [
        "own_flow_lps",
        "own_flow_l",
        "flow_lps",
        "base_flow_lps",
    ] if col in pipes.columns), None)
    if own_flow_col is None:
        raise KeyError("Missing own-flow column. Tried: own_flow_lps, own_flow_l, flow_lps, base_flow_lps.")
    dwf_data = pipes.groupby("upstream_m")[own_flow_col].sum().reset_index()
    dwf_data.columns = ["node", "flow_lps"]
    dwf_data = dwf_data[dwf_data["node"].isin(manholes["id"])]

    with open(output_path, "w") as f:
        # TITLE
        f.write("[TITLE]\n;;Project Title/Notes\n")
        f.write(f"{title}\n\n")

        # OPTIONS
        f.write("[OPTIONS]\n;;Option             Value\n")
        default_options = {
            "FLOW_UNITS": "LPS",
            "INFILTRATION": "CURVE_NUMBER",
            "FLOW_ROUTING": "DYNWAVE",
            "LINK_OFFSETS": "DEPTH",
            "MIN_SLOPE": "0",
            "ALLOW_PONDING": "NO",
            "SKIP_STEADY_STATE": "NO",
            "START_DATE": "06/19/2025",
            "START_TIME": "00:00:00",
            "REPORT_START_DATE": "06/19/2025",
            "REPORT_START_TIME": "00:00:00",
            "END_DATE": "06/20/2025",
            "END_TIME": "00:00:00",
            "SWEEP_START": "01/01",
            "SWEEP_END": "12/31",
            "DRY_DAYS": "0",
            "REPORT_STEP": "00:15:00",
            "WET_STEP": "00:05:00",
            "DRY_STEP": "01:00:00",
            "ROUTING_STEP": "0:00:20",
            "RULE_STEP": "00:00:00",
            "INERTIAL_DAMPING": "PARTIAL",
            "NORMAL_FLOW_LIMITED": "BOTH",
            "FORCE_MAIN_EQUATION": "D-W",
            "VARIABLE_STEP": "0.75",
            "LENGTHENING_STEP": "0",
            "MIN_SURFAREA": "1.167",
            "MAX_TRIALS": "8",
            "HEAD_TOLERANCE": "0.0015",
            "SYS_FLOW_TOL": "5",
            "LAT_FLOW_TOL": "5",
            "MINIMUM_STEP": "0.5",
            "THREADS": "1"
        }
        merged_options = default_options if options_dict is None else {**default_options, **options_dict}
        for key, value in merged_options.items():
            f.write(f"{key:22} {value}\n")
        f.write("\n")

        # EVAPORATION
        f.write("[EVAPORATION]\n;;Data Source    Parameters\n;;-------------- ----------------\n")
        f.write("CONSTANT         0.0\nDRY_ONLY         NO\n\n")

        # JUNCTIONS
        f.write("[JUNCTIONS]\n;;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded\n")
        for _, row in manholes[manholes["id"] != outlet_mh_id].iterrows():
            f.write(f"{row['id']:17} {row['invert_elev']:10.2f} {row['max_depth']:10.2f} 0          0          0\n")
        f.write("\n")

        # OUTFALLS
        f.write("[OUTFALLS]\n;;Name           Elevation  Type       Stage Data       Gated    Route To\n")
        f.write(f"{outlet_mh_id:17} {inv_outlet:10.2f} FREE                        NO\n\n")

        # CONDUITS
        f.write("[CONDUITS]\n;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow\n")
        for _, row in pipes.iterrows():
            elev_up = manholes.set_index("id").loc[row["upstream_m"], "invert_elev"]
            elev_down = manholes.set_index("id").loc[row["downstream"], "invert_elev"]
            in_offset = row["inv_up"] - elev_up
            out_offset = row["inv_down"] - elev_down
            f.write(f"{row['pipe_id']:17} {row['upstream_m']:17} {row['downstream']:17} "
                    f"{row['length_m']:10.1f} {row['n']:10.3f} {in_offset:10.3f} {out_offset:10.3f} 0          0\n")
        f.write("\n")

        # XSECTIONS
        f.write("[XSECTIONS]\n;;Link           Shape        Geom1      Geom2      Geom3      Geom4      Barrels    Culvert\n")
        for _, row in pipes.iterrows():
            f.write(f"{row['pipe_id']:17} CIRCULAR     {row['diameter_m']:.3f}     0          0          0          1\n")
        f.write("\n")

        # INFLOWS
        f.write("[INFLOWS]\n;;Node           Constituent      Time Series      Type     Mfactor  Sfactor  Baseline Pattern\n")
        for _, row in dwf_data.iterrows():
            f.write(f"{row['node']:17} FLOW             \"\"               FLOW     1.0      1.0      0.0\n")
        f.write("\n")

        # DWF
        f.write("[DWF]\n;;Node           Constituent      Baseline   Patterns\n")
        for _, row in dwf_data.iterrows():
            f.write(f"{row['node']:17} FLOW             {row['flow_lps']:.3f}     1\n")
        f.write("\n")

        # PATTERNS
        f.write("[PATTERNS]\n;;Name           Type       Multipliers\n")
        f.write("1                HOURLY     1.0   1.0   1.0   1.0   1.0   1.0\n")
        f.write("1                           1.0   1.0   1.0   1.0   1.0   1.0\n")
        f.write("1                           1.0   1.0   1.0   1.0   1.0   1.0\n")
        f.write("1                           1.0   1.0   1.0   1.0   1.0   1.0\n\n")

        # REPORT
        f.write("[REPORT]\n;;Reporting Options\nSUBCATCHMENTS ALL\nNODES ALL\nLINKS ALL\n\n")

        # COORDINATES
        f.write("[COORDINATES]\n;;Node           X-Coord            Y-Coord\n")
        for _, row in manholes.iterrows():
            pt = row.geometry
            f.write(f"{row['id']:17} {pt.x:18.3f} {pt.y:18.3f}\n")
        f.write("\n")

        # VERTICES
        f.write("[VERTICES]\n;;Link           X-Coord            Y-Coord\n\n")

        # MAP
        f.write("[MAP]\nDIMENSIONS 0.000 0.000 10000.000 10000.000\nUnits      None\n\n")

        # TAGS
        f.write("[TAGS]\n\n")

    print(f"✅ SWMM .inp file written to: {output_path}")

def assign_dwf_pattern_to_all_nodes(inp_path, output_path, pattern_id, pattern_values):
    """
    Assigns a repeating HOURLY pattern to all nodes in the [DWF] section
    and inserts a properly formatted [PATTERNS] block into a SWMM .inp file.
    
    Parameters:
    - inp_path (str): Input SWMM .inp file path.
    - output_path (str): Modified output path.
    - pattern_id (str): Pattern name/id to assign (e.g., "1").
    - pattern_values (list of float): 24 HOURLY values.
    """
    if len(pattern_values) != 24:
        raise ValueError("Pattern must have 24 hourly multipliers.")

    with open(inp_path, 'r') as f:
        lines = f.readlines()

    final_lines = []
    new_dwf_block = []
    in_dwf = False
    in_pattern = False

    for line in lines:
        stripped = line.strip()

        # Skip the old [PATTERNS] section
        if stripped.startswith("[PATTERNS]"):
            in_pattern = True
            continue
        if in_pattern:
            if stripped.startswith("[") and not stripped.startswith("[PATTERNS]"):
                in_pattern = False
            else:
                continue

        # Collect and replace [DWF] section
        if stripped.startswith("[DWF]"):
            in_dwf = True
            new_dwf_block.append("\n[DWF]\n")
            new_dwf_block.append(";;Node           Constituent      Baseline   Patterns\n")
            continue
        if in_dwf:
            if stripped.startswith("[") and not stripped.startswith("[DWF]"):
                in_dwf = False
            elif not stripped or stripped.startswith(";"):
                continue
            else:
                parts = stripped.split()
                if len(parts) >= 3:
                    node, param, value = parts[:3]
                    new_dwf_block.append(f"{node:<17} {param:<16} {value:<10} {pattern_id}\n")
            continue

        final_lines.append(line)

    # Format the [PATTERNS] block
    pattern_block = [
        "\n[PATTERNS]\n",
        ";;Name           Type       Multipliers\n",
        ";;-------------- ---------- -----------\n",
        ";qwer\n"
    ]
    for i in range(0, 24, 6):
        chunk = pattern_values[i:i+6]
        if i == 0:
            pattern_block.append(f"{pattern_id:<16} {'HOURLY':<10} " + "   ".join(f"{v:.1f}" for v in chunk) + "\n")
        else:
            pattern_block.append(f"{pattern_id:<16} {'':<10} " + "   ".join(f"{v:.1f}" for v in chunk) + "\n")

    # Insert the pattern block before [DWF], or append at end if [DWF] not found
    try:
        index = next(i for i, l in enumerate(final_lines) if l.strip().startswith("[DWF]"))
    except StopIteration:
        index = len(final_lines)

    final_lines = final_lines[:index] + pattern_block + final_lines[index:]
    final_lines += new_dwf_block

    with open(output_path, 'w') as f:
        f.writelines(final_lines)

    print(f"✅ Pattern '{pattern_id}' correctly assigned and formatted. Output saved to: {output_path}")

def assign_inflow_from_pipe_length(inp_path, output_path, coefficient):
    """
    Assigns inflow baseline values to nodes in a SWMM .inp file based on
    the length of their downstream pipe and a given coefficient (e.g., LPS/m).

    Parameters:
    - inp_path (str): Path to the input .inp file.
    - output_path (str): Path to write the updated .inp file.
    - coefficient (float): Flow coefficient per meter (e.g., LPS/m).
    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    # Parse CONDUITS to map: FromNode -> Length
    downstream_lengths = {}
    in_conduits = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits:
            if stripped.startswith("[") and not stripped.startswith("[CONDUITS]"):
                break
            if not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                from_node = parts[1]
                length = float(parts[3])
                downstream_lengths[from_node] = length

    # Create new [INFLOWS] block
    inflow_lines = [
        "\n[INFLOWS]\n",
        ";;Node           Constituent      Time Series      Type     Mfactor  Sfactor  Baseline Pattern\n",
        ";;-------------- ---------------- ---------------- -------- -------- -------- -------- --------\n"
    ]
    for node, length in downstream_lengths.items():
        baseline = length * coefficient
        inflow_lines.append(f"{node:<17} FLOW             \"\"               FLOW     1.0      1.0      {baseline:.3f}\n")

    # Rebuild the file: skip old [INFLOWS] if present
    new_lines = []
    in_inflows = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[INFLOWS]"):
            in_inflows = True
            continue
        if in_inflows:
            if stripped.startswith("[") and not stripped.startswith("[INFLOWS]"):
                in_inflows = False
            else:
                continue
        if not in_inflows:
            new_lines.append(line)

    # Insert new [INFLOWS] section at the end
    new_lines += inflow_lines

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ INFLOWS section created using coefficient {coefficient:.4f}. File saved to: {output_path}")

def add_subcatchment_data_to_inp(
    inp_path,
    output_path,
    subcatchments_path,
    raingage_id,
    raingage_coords,
    timeseries,
    n_imperv,
    n_perv,
    s_imperv,
    s_perv,
    pct_zero,
    route_to,
    pct_routed,
    infiltration_params,
    imperv_pct,
    width,
    slope,
    curblen
):
    import geopandas as gpd
    import os
    import re

    subgdf = gpd.read_file(subcatchments_path)

    # Step 1: Read original .inp file
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    # Step 2: Extract conduit mapping (pipe -> upstream manhole)
    conduit_lines = []
    in_conduits = False
    for line in lines:
        if line.strip().startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits and line.strip().startswith("["):
            in_conduits = False
        if in_conduits and line.strip() and not line.strip().startswith(";"):
            conduit_lines.append(line.strip())

    pipe_to_manhole_map = {}
    for line in conduit_lines:
        parts = line.split()
        if len(parts) >= 3:
            pipe_id, from_node = parts[0], parts[1]
            pipe_to_manhole_map[pipe_id] = from_node

    # Step 3: Create subcatchment-related blocks
    subcatchments, subareas, infiltration, polygons, timeseries_lines, symbol_lines = [], [], [], [], [], []
    for i, row in subgdf.iterrows():
        sid = f"S{i+1}"
        pipe_id = str(row["pipe_id"])
        outlet = pipe_to_manhole_map.get(pipe_id, "MISSING")
        area_ha = row["area_ha"]

        subcatchments.append(
            f"{sid:<16} {raingage_id:<16} {outlet:<16} {area_ha:<8.2f} {imperv_pct:<8} {width:<8} {slope:<8} {curblen:<8} \n"
        )
        subareas.append(
            f"{sid:<16} {n_imperv:<10} {n_perv:<10} {s_imperv:<10} {s_perv:<10} {pct_zero:<10} {route_to:<10} {pct_routed:<10}\n"
        )
        p1, p2, p3, p4, p5 = infiltration_params
        infiltration.append(
            f"{sid:<16} {p1:<10} {p2:<10} {p3:<10} {p4:<10} {p5:<10}\n"
        )

        geom = row.geometry
        if geom.geom_type == "Polygon":
            for x, y in geom.exterior.coords:
                polygons.append(f"{sid:<16} {x:<18.3f} {y:<18.3f}\n")
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                for x, y in poly.exterior.coords:
                    polygons.append(f"{sid:<16} {x:<18.3f} {y:<18.3f}\n")

    for date, time, val in timeseries:
        timeseries_lines.append(f"{raingage_id:<16} {date:<12} {time:<10} {val:<10.2f}\n")

    symbol_lines.append(f"{raingage_id:<16} {raingage_coords[0]:<18.3f} {raingage_coords[1]:<18.3f}\n")

    # Step 4: Write updated .inp with appended sections
    with open(output_path, 'w') as f:
        f.writelines(lines)
        f.write("\n[RAINGAGES]\n")
        f.write(";;Name           Format    Interval SCF      Source    \n")
        f.write(";;-------------- --------- ------ ------ ----------\n")
        f.write(f"{raingage_id:<16} {'INTENSITY':<10} {'0:15':<6} {'1.0':<6} TIMESERIES {raingage_id}\n")

        f.write("\n[SUBCATCHMENTS]\n")
        f.write(";;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack        \n")
        f.writelines(subcatchments)

        f.write("\n[SUBAREAS]\n")
        f.write(";;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted \n")
        f.writelines(subareas)

        f.write("\n[INFILTRATION]\n")
        f.write(";;Subcatchment   Param1     Param2     Param3     Param4     Param5    \n")
        f.writelines(infiltration)

        f.write("\n[TIMESERIES]\n")
        f.write(";;Name           Date       Time       Value     \n")
        f.writelines(timeseries_lines)

        f.write("\n[Polygons]\n")
        f.write(";;Subcatchment   X-Coord            Y-Coord           \n")
        f.writelines(polygons)

        f.write("\n[SYMBOLS]\n")
        f.write(";;Gage           X-Coord            Y-Coord           \n")
        f.writelines(symbol_lines)

    print(f"✅ SWMM .inp file updated with subcatchments and saved to: {output_path}")

def assign_inflow_from_raster(inp_path, output_path, raster_path, samples_per_pipe=5):
    import rasterio
    import numpy as np
    import re
    
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    # Step 1: Extract node coordinates
    node_coords = {}
    in_coords = False
    for line in lines:
        if line.strip().startswith("[COORDINATES]"):
            in_coords = True
            continue
        if in_coords:
            if line.strip().startswith("["):
                break
            if not line.strip() or line.strip().startswith(";"):
                continue
            match = re.match(r"(\S+)\s+([\d\.\-]+)\s+([\d\.\-]+)", line.strip())
            if match:
                node, x, y = match.group(1), float(match.group(2)), float(match.group(3))
                node_coords[node] = (x, y)

    # Step 2: Extract conduit info
    pipes = []
    in_conduits = False
    for line in lines:
        if line.strip().startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits:
            if line.strip().startswith("["):
                break
            if not line.strip() or line.strip().startswith(";"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pipe_id, from_node, to_node, length = parts[0], parts[1], parts[2], float(parts[3])
                pipes.append((pipe_id, from_node, to_node, length))

    # Step 3: Sample raster
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        height, width = raster.shape
        nodata = src.nodata

        def sample_along_pipe(x1, y1, x2, y2, n):
            xs = np.linspace(x1, x2, n)
            ys = np.linspace(y1, y2, n)
            values = []
            for x, y in zip(xs, ys):
                try:
                    row, col = src.index(x, y)
                    if 0 <= row < height and 0 <= col < width:
                        val = raster[row, col]
                        if nodata is not None and val == nodata:
                            continue
                        values.append(val)
                except:
                    continue
            return values

    # Step 4: Build inflow lines
    inflow_lines = [
        "\n[INFLOWS]\n",
        ";;Node           Constituent      Time Series      Type     Mfactor  Sfactor  Baseline Pattern\n",
        ";;-------------- ---------------- ---------------- -------- -------- -------- -------- --------\n"
    ]
    for pipe_id, from_node, to_node, length in pipes:
        if from_node in node_coords and to_node in node_coords:
            x1, y1 = node_coords[from_node]
            x2, y2 = node_coords[to_node]
            values = sample_along_pipe(x1, y1, x2, y2, samples_per_pipe)
            if values:
                mean_coeff = np.mean(values)
                baseline = mean_coeff * length
                inflow_lines.append(
                    f"{from_node:<17} FLOW             \"\"               FLOW     1.0      1.0      {baseline:.3f}\n"
                )

    # Step 5: Remove existing INFLOWS block
    new_lines = []
    in_inflows = False
    for line in lines:
        if line.strip().startswith("[INFLOWS]"):
            in_inflows = True
            continue
        if in_inflows:
            if line.strip().startswith("["):
                in_inflows = False
            else:
                continue
        if not in_inflows:
            new_lines.append(line)

    # Step 6: Write new file
    new_lines += inflow_lines
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ INFLOWS assigned based on raster and pipe length. Output: {output_path}")

def add_subcatchment_data_with_rdii_raster(
    inp_path,
    output_path,
    subcatchments_path,
    rdii_raster_path,
    raingage_id,
    raingage_coords,
    timeseries,
    n_imperv,
    n_perv,
    s_imperv,
    s_perv,
    pct_zero,
    route_to,
    pct_routed,
    infiltration_params,
    width,
    slope,
    curblen,
    rdii_to_imperv_scale=(0.0, 3.0)  # min and max RDII used for rescaling
):
    import geopandas as gpd
    import rasterio
    import rasterio.mask
    import numpy as np
    from rasterstats import zonal_stats

    subgdf = gpd.read_file(subcatchments_path)

    # Step 1: Read original .inp file
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    # Step 2: Extract conduit mapping (pipe -> upstream manhole)
    conduit_lines = []
    in_conduits = False
    for line in lines:
        if line.strip().startswith("[CONDUITS]"):
            in_conduits = True
            continue
        if in_conduits and line.strip().startswith("["):
            in_conduits = False
        if in_conduits and line.strip() and not line.strip().startswith(";"):
            conduit_lines.append(line.strip())

    pipe_to_manhole_map = {}
    for line in conduit_lines:
        parts = line.split()
        if len(parts) >= 3:
            pipe_id, from_node = parts[0], parts[1]
            pipe_to_manhole_map[pipe_id] = from_node

    # Step 3: Compute average RDII density per polygon
    stats = zonal_stats(
        subgdf,
        rdii_raster_path,
        stats=["mean"],
        nodata=-9999
    )
    rdii_values = [s["mean"] if s["mean"] is not None else 0 for s in stats]

    # Step 4: Rescale RDII to impervious percentage (0–100%)
    rdii_min, rdii_max = rdii_to_imperv_scale
    imperv_pcts = []
    for val in rdii_values:
        pct = 100.0 * (val - rdii_min) / (rdii_max - rdii_min)
        pct = max(0.0, min(pct, 100.0))  # clamp between 0–100
        imperv_pcts.append(round(pct, 2))

    # Step 5: Create subcatchment-related blocks
    subcatchments, subareas, infiltration, polygons, timeseries_lines, symbol_lines = [], [], [], [], [], []
    for i, row in subgdf.iterrows():
        sid = f"S{i+1}"
        pipe_id = str(row["pipe_id"])
        outlet = pipe_to_manhole_map.get(pipe_id, "MISSING")
        area_ha = row["area_ha"]
        imperv_pct = imperv_pcts[i]

        subcatchments.append(
            f"{sid:<16} {raingage_id:<16} {outlet:<16} {area_ha:<8.2f} {imperv_pct:<8} {width:<8} {slope:<8} {curblen:<8} \n"
        )
        subareas.append(
            f"{sid:<16} {n_imperv:<10} {n_perv:<10} {s_imperv:<10} {s_perv:<10} {pct_zero:<10} {route_to:<10} {pct_routed:<10}\n"
        )
        p1, p2, p3, p4, p5 = infiltration_params
        infiltration.append(
            f"{sid:<16} {p1:<10} {p2:<10} {p3:<10} {p4:<10} {p5:<10}\n"
        )

        geom = row.geometry
        if geom.geom_type == "Polygon":
            for x, y in geom.exterior.coords:
                polygons.append(f"{sid:<16} {x:<18.3f} {y:<18.3f}\n")
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                for x, y in poly.exterior.coords:
                    polygons.append(f"{sid:<16} {x:<18.3f} {y:<18.3f}\n")

    for date, time, val in timeseries:
        timeseries_lines.append(f"{raingage_id:<16} {date:<12} {time:<10} {val:<10.2f}\n")

    symbol_lines.append(f"{raingage_id:<16} {raingage_coords[0]:<18.3f} {raingage_coords[1]:<18.3f}\n")

    # Step 6: Write updated .inp with appended sections
    with open(output_path, 'w') as f:
        f.writelines(lines)
        f.write("\n[RAINGAGES]\n")
        f.write(";;Name           Format    Interval SCF      Source    \n")
        f.write(";;-------------- --------- ------ ------ ----------\n")
        f.write(f"{raingage_id:<16} {'INTENSITY':<10} {'0:15':<6} {'1.0':<6} TIMESERIES {raingage_id}\n")

        f.write("\n[SUBCATCHMENTS]\n")
        f.write(";;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack        \n")
        f.writelines(subcatchments)

        f.write("\n[SUBAREAS]\n")
        f.write(";;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted \n")
        f.writelines(subareas)

        f.write("\n[INFILTRATION]\n")
        f.write(";;Subcatchment   Param1     Param2     Param3     Param4     Param5    \n")
        f.writelines(infiltration)

        f.write("\n[TIMESERIES]\n")
        f.write(";;Name           Date       Time       Value     \n")
        f.writelines(timeseries_lines)

        f.write("\n[Polygons]\n")
        f.write(";;Subcatchment   X-Coord            Y-Coord           \n")
        f.writelines(polygons)

        f.write("\n[SYMBOLS]\n")
        f.write(";;Gage           X-Coord            Y-Coord           \n")
        f.writelines(symbol_lines)

    print(f"✅ SWMM .inp file updated with RDII-based impervious percentages and saved to: {output_path}")

def get_flow_components_from_node_pyswmm(inp_path, link_id="P_OUTLET", node_id="OUTLET"):
    """
    Tracer-assisted hydrograph separation at a location in the network.

    Logic:
      - DWF is estimated using TR_DWF
      - GWI is estimated using TR_GWI
      - RDII_runoff is computed as:

            RDII_runoff = Total flow - DWF - GWI

    This avoids relying on TR_RUNOFF being exactly 100%.

    NOTE:
    Flow is in SWMM project units.
    Pollutant concentrations are assumed to use 100 mg/L source tagging.
    """

    from pyswmm import Simulation, Links, Nodes
    import pandas as pd

    inp_path = os.fspath(inp_path)

    times, flows = [], []
    tr_runoff, tr_dwf, tr_gwi = [], [], []

    with Simulation(inp_path) as sim:

        link = Links(sim)[link_id]
        node = Nodes(sim)[node_id]

        for _ in sim:

            times.append(sim.current_time)

            q = float(link.flow)
            flows.append(q)

            tr_runoff.append(float(node.pollut_quality.get("TR_RUNOFF", 0.0)))
            tr_dwf.append(float(node.pollut_quality.get("TR_DWF", 0.0)))
            tr_gwi.append(float(node.pollut_quality.get("TR_GWI", 0.0)))

    df = pd.DataFrame({
        "Datetime": times,
        "Flow_model_units": flows,
        "TR_RUNOFF": tr_runoff,
        "TR_DWF": tr_dwf,
        "TR_GWI": tr_gwi,
    })

    # Clip concentrations to reduce numerical noise
    df["TR_RUNOFF"] = df["TR_RUNOFF"].clip(0.0, 100.0)
    df["TR_DWF"]    = df["TR_DWF"].clip(0.0, 100.0)
    df["TR_GWI"]    = df["TR_GWI"].clip(0.0, 100.0)

    # Components estimated from reliable tracers
    df["DWF"] = df["Flow_model_units"] * df["TR_DWF"] / 100.0
    df["GWI"] = df["Flow_model_units"] * df["TR_GWI"] / 100.0

    # RDII as remaining wet-weather flow
    df["RDII_runoff"] = (
        df["Flow_model_units"]
        - df["DWF"]
        - df["GWI"]
    )

    # Avoid tiny negative values from numerical noise
    df["RDII_runoff"] = df["RDII_runoff"].clip(lower=0.0)

    # Diagnostic only: RDII estimated directly from TR_RUNOFF
    df["RDII_from_TR_RUNOFF"] = (
        df["Flow_model_units"] * df["TR_RUNOFF"] / 100.0
    )

    # Diagnostic closure
    df["Residual"] = (
        df["Flow_model_units"]
        - (
            df["RDII_runoff"]
            + df["DWF"]
            + df["GWI"]
        )
    )

    return df[
        [
            "Datetime",
            "Flow_model_units",

            # Raw tracer concentrations
            "TR_RUNOFF",
            "TR_DWF",
            "TR_GWI",

            # Main separated flows
            "RDII_runoff",
            "DWF",
            "GWI",

            # Diagnostic comparison
            "RDII_from_TR_RUNOFF",

            # Closure check
            "Residual",
        ]
    ]

def run_swmm_and_plot(inp_path, monitored_nodes=None, monitored_links=None):
    """
    Run SWMM model and extract time-series for specified nodes and links.
    
    Parameters:
        inp_path (str): Path to .inp file
        monitored_nodes (list): List of manhole IDs (nodes) to monitor
        monitored_links (list): List of pipe IDs (links) to monitor
    """
    import matplotlib.pyplot as plt
    from pyswmm import Simulation, Nodes, Links
    import pandas as pd

    inp_path = os.fspath(inp_path)

    if monitored_nodes is None:
        monitored_nodes = []
    if monitored_links is None:
        monitored_links = []

    # Containers for results
    time_index = []
    node_depths = {node: [] for node in monitored_nodes}
    link_flows = {link: [] for link in monitored_links}

    # Run Simulation
    with Simulation(inp_path) as sim:
        nodes = Nodes(sim)
        links = Links(sim)

        for step in sim:
            time_index.append(sim.current_time)

            for node_id in monitored_nodes:
                node_depths[node_id].append(nodes[node_id].depth)

            for link_id in monitored_links:
                link_flows[link_id].append(links[link_id].flow)

    # Convert to DataFrames
    df_depths = pd.DataFrame(node_depths, index=pd.to_datetime(time_index))
    df_flows = pd.DataFrame(link_flows, index=pd.to_datetime(time_index))

    # Plotting
    if not df_depths.empty:
        df_depths.plot(figsize=(12, 5), title="Node Depths Over Time")
        plt.xlabel("Time")
        plt.ylabel("Depth (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if not df_flows.empty:
        df_flows.plot(figsize=(12, 5), title="Link Flows Over Time")
        plt.xlabel("Time")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_depths, df_flows

def auto_add_pollutants_to_inp_fixed(inp_path, output_path):
    """
    Adds 3 tracer pollutants and ensures baseline FLOW inflows are tagged with TR_GWI
    via a pollutant inflow using the correct SWMM keyword: CONCEN (not CONC).

    Experiment meaning:
      - TR_RUNOFF: tags subcatchment runoff (your "RDII") via Crain=100
      - TR_DWF:    tags DWF via Cdwf=100
      - TR_GWI:    tags baseline node FLOW inflows (your "GWI") by adding:
                   Node TR_GWI "" CONCEN 1.0 1.0 100 [Pattern optional]

    Protections:
      - Preserves [DWF] lines (keeps all tokens/patterns).
      - Removes existing [POLLUTANTS] and [REPORT] to avoid duplicates.
      - Inserts [POLLUTANTS] BEFORE [INFLOWS] so pollutants exist when referenced.
      - Inserts [REPORT] before [END].
      - Adds TR_GWI only if missing for that node.
    """
    import re

    with open(inp_path, "r") as f:
        lines = f.readlines()

    def is_header(s: str) -> bool:
        s = s.strip()
        return s.startswith("[") and s.endswith("]")

    def drop_sections(lines_in, drop_upper):
        out = []
        in_drop = False
        for line in lines_in:
            s = line.strip()
            if is_header(s):
                in_drop = (s.upper() in drop_upper)
                if in_drop:
                    continue
                out.append(line)
                continue
            if in_drop:
                continue
            out.append(line)
        return out

    # Remove existing [POLLUTANTS] and [REPORT] to avoid duplicates
    lines = drop_sections(lines, {"[POLLUTANTS]", "[REPORT]"})

    # Preserve [DWF] (keep all tokens/patterns)
    cleaned = []
    in_dwf = False
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("[DWF]"):
            in_dwf = True
            cleaned.append(line)
            continue
        elif stripped.startswith("[") and not upper.startswith("[DWF]"):
            in_dwf = False
            cleaned.append(line)
            continue

        if in_dwf and stripped and not stripped.startswith(";"):
            if ";" in line:
                body, comment = line.split(";", 1)
                parts = body.strip().split()
                cleaned.append("    ".join(parts) + "    ;" + comment)
            else:
                parts = stripped.split()
                cleaned.append("    ".join(parts) + "\n")
        else:
            cleaned.append(line)

    # Pollutants block (must appear BEFORE INFLOWS if INFLOWS references TR_GWI)
    pollutants_block = """[POLLUTANTS]
;;Name        Units  Crain  Cgw   Crdii  Kdecay  SnowOnly  Co-Pollutant  Co-Frac  Cdwf  Cinit
;;----------  -----  -----  ----  -----  ------  --------  ------------  -------  ----  -----
TR_RUNOFF     MG/L   100    0.0   0.0    0.0     NO        *             0.0      0.0   0.0
TR_DWF        MG/L   0.0    0.0   0.0    0.0     NO        *             0.0      100   0.0
TR_GWI        MG/L   0.0    0.0   0.0    0.0     NO        *             0.0      0.0   0.0
"""

    report_block = """[REPORT]
;;Reporting Options
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL
"""

    def parse_inflows_row(line: str):
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("["):
            return None

        body = line.split(";", 1)[0].strip()
        parts = re.split(r"\s+", body)

        # Accept both:
        # 7 tokens: node constituent timeseries type mfactor sfactor baseline
        # 8 tokens: ... + pattern
        if len(parts) < 7:
            return None

        node, constituent, typ_series, inflow_type, mf, sf, baseline = parts[:7]
        pattern = parts[7] if len(parts) >= 8 else ""
        return dict(
            node=node,
            constituent=constituent,
            timeseries=typ_series,
            inflow_type=inflow_type,
            mf=mf,
            sf=sf,
            baseline=baseline,
            pattern=pattern,
        )

    def format_inflows_row(node, constituent, timeseries, inflow_type, mf, sf, baseline, pattern=""):
        if pattern:
            return f"{node:<18}{constituent:<16}{timeseries:<16}{inflow_type:<8}{mf:<8}{sf:<8}{baseline:<8}{pattern}\n"
        return f"{node:<18}{constituent:<16}{timeseries:<16}{inflow_type:<8}{mf:<8}{sf:<8}{baseline}\n"

    out = []
    in_inflows = False
    inflows_block = []
    pollutants_inserted = False
    report_inserted = False

    def flush_inflows(block_lines):
        tagged_nodes = set()

        parsed = []
        for ln in block_lines:
            row = parse_inflows_row(ln)
            parsed.append((ln, row))
            if row is None:
                continue
            if row["constituent"].upper() == "TR_GWI" and row["inflow_type"].upper() in {"CONCEN", "MASS"}:
                tagged_nodes.add(row["node"])

        rebuilt = []
        for raw, row in parsed:
            rebuilt.append(raw)
            if row is None:
                continue

            # For each baseline FLOW inflow, add TR_GWI as CONCEN if missing
            if row["constituent"].upper() == "FLOW" and row["inflow_type"].upper() == "FLOW":
                node = row["node"]
                if node not in tagged_nodes:
                    rebuilt.append(format_inflows_row(
                        node=node,
                        constituent="TR_GWI",
                        timeseries=row["timeseries"],     # keep "" like your FLOW line
                        inflow_type="CONCEN",              # ✅ correct SWMM keyword
                        mf=row["mf"],
                        sf=row["sf"],
                        baseline="100",
                        pattern=row["pattern"]
                    ))
                    tagged_nodes.add(node)

        return rebuilt

    for line in cleaned:
        s = line.strip()

        # Insert pollutants BEFORE first INFLOWS header
        if (not pollutants_inserted) and s.upper().startswith("[INFLOWS]"):
            out.append("\n" + pollutants_block + "\n")
            pollutants_inserted = True

        if s.upper().startswith("[INFLOWS]"):
            in_inflows = True
            inflows_block = [line]
            continue

        if in_inflows:
            if is_header(s) and not s.upper().startswith("[INFLOWS]"):
                out.extend(flush_inflows(inflows_block))
                in_inflows = False
                out.append(line)
            else:
                inflows_block.append(line)
        else:
            # Insert report before [END]
            if (not report_inserted) and s.upper().startswith("[END]"):
                out.append("\n" + report_block + "\n")
                report_inserted = True
            out.append(line)

    if in_inflows:
        out.extend(flush_inflows(inflows_block))

    # Fallbacks
    if not pollutants_inserted:
        out.append("\n" + pollutants_block + "\n")
    if not report_inserted:
        out.append("\n" + report_block + "\n")

    with open(output_path, "w") as f:
        f.writelines(out)

    print(f"✅ Tagged inp written (uses CONCEN) -> {output_path}")

__all__ = [
    "export_swmm_inp",
    "assign_dwf_pattern_to_all_nodes",
    "assign_inflow_from_pipe_length",
    "add_subcatchment_data_to_inp",
    "assign_inflow_from_raster",
    "add_subcatchment_data_with_rdii_raster",
    "get_flow_components_from_node_pyswmm",
    "run_swmm_and_plot",
    "auto_add_pollutants_to_inp_fixed",
]
