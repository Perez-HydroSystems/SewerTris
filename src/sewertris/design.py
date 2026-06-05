"""Pipe predesign, slope, material, diameter, and invert utilities."""

from __future__ import annotations

from ._deps import *

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
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
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
    gdf = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))

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
    gdf["diameter_m"] = gdf["diameter_mm"].astype(float) / 1000.0

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
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))

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
    pipes = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))
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

def add_predesign_flow(pipes_path: str, out_path: str | None = None, overwrite: bool = True):
    """
    Adds predesign flow = peak flow + cumulative GWI + cumulative RDII.

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

    gdf = ensure_pipe_topology_aliases(gpd.read_file(pipes_path))

    def _first_existing(candidates, label):
        col = next((name for name in candidates if name in gdf.columns), None)
        if col is None:
            raise ValueError(
                f"Missing {label} column. Tried: {', '.join(candidates)}"
            )
        return col

    peak_col = _first_existing(
        [
            "peak_flow_lps_bc",
            "peak_flow_",
            "peak_flow",
            "peakflow",
            "q_peak",
            "cumulative_flow_lps",
            "cumulative",
            "cumulative_",
            "cumulativ",
        ],
        "peak flow",
    )
    gwi_col = _first_existing(
        ["cum_gwi_ls", "cum_gwi_l", "gwi_cumulative"],
        "cumulative GWI",
    )
    rdii_col = _first_existing(
        ["cum_rdii_ls", "cum_rdii_l", "rdii_cumulative"],
        "cumulative RDII",
    )

    # Calculate predesign flow
    gdf["predesign_ls"] = (
        gdf[peak_col].astype(float)
        + gdf[gwi_col].astype(float)
        + gdf[rdii_col].astype(float)
    )
    gdf["predesign_"] = gdf["predesign_ls"]

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

__all__ = [
    "british_columbia_peaking_factor",
    "assign_pipe_slopes",
    "assign_material_diameter_to_pipes",
    "assign_invert_elevations",
    "preprocess_pipes_and_manholes",
    "add_predesign_flow",
    "assign_all_dwf_patterns",
]
