import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import sewertris as st


def main(inp_path):
    inp_path = Path(inp_path)

    if inp_path.suffix.lower() != ".inp":
        raise ValueError(f"Input is not an INP file: {inp_path}")

    df = st.get_flow_components_from_node_pyswmm(
        inp_path=str(inp_path),
        link_id="P_OUTLET"
    )

    ds = df.set_index("Datetime").to_xarray()

    output_nc = inp_path.with_name(f"{inp_path.stem}_flows.nc")
    ds.to_netcdf(output_nc)

    print(f"Saved: {output_nc}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing INP path argument.")

    main(sys.argv[1])
