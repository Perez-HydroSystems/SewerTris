"""Tests for New_Development/sewer_overflows.quantify_sewer_overflows.

The module lives outside ``src/`` (it is not part of the installed ``sewertris``
package yet), so it is loaded here by file path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "New_Development" / "sewer_overflows.py"

_spec = importlib.util.spec_from_file_location("sewer_overflows", _MODULE_PATH)
sewer_overflows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sewer_overflows)

quantify_sewer_overflows = sewer_overflows.quantify_sewer_overflows
_compute_overflow_metrics = sewer_overflows._compute_overflow_metrics
read_node_coordinates = sewer_overflows.read_node_coordinates
read_conduit_links = sewer_overflows.read_conduit_links
plot_manhole_overflows = sewer_overflows.plot_manhole_overflows

_TIMESTEP_COLS = [
    "n_overflowing_manholes",
    "total_overflow_rate_lps",
    "mean_overflow_rate_lps",
]

_EXAMPLE_OUT = (
    _REPO_ROOT
    / "Examples"
    / "output_example_1_project"
    / "scenarios"
    / "bwf_gwi_rdii"
    / "sewer_model.out"
)
_EXAMPLE_INP = _EXAMPLE_OUT.with_suffix(".inp")


def _synthetic_rates():
    """Wide flooding-rate matrix (LPS) with known overflows at 900 s spacing."""
    idx = pd.date_range("2025-01-01", periods=5, freq="15min")
    rates = pd.DataFrame(
        {"A": [0, 10, 20, 0, 0], "B": [0, 0, 0, 5, 0], "C": [0, 0, 0, 0, 0]},
        index=idx,
        dtype=float,
    )
    rates.index.name = "Datetime"
    return rates


def test_metrics_synthetic_values():
    """Core metrics on synthetic data with a guaranteed non-zero overflow."""
    rates = _synthetic_rates()
    timestep_df, node_summary_df, system_summary, wide = _compute_overflow_metrics(
        rates, threshold_lps=0.0, return_node_timeseries=True
    )

    # Per-step counts / totals (items 1, 3, 4).
    assert list(timestep_df.columns) == _TIMESTEP_COLS
    assert timestep_df.index.name == "Datetime"
    assert timestep_df["n_overflowing_manholes"].tolist() == [0, 1, 1, 1, 0]
    assert timestep_df["total_overflow_rate_lps"].tolist() == [0, 10, 20, 5, 0]
    assert timestep_df.iloc[2]["mean_overflow_rate_lps"] == 20.0

    # Per-node volume (item 5): trapezoid of A over elapsed seconds -> m3.
    # 0.5*900*10 + 0.5*900*30 = 4500 + 13500 = 18000 L over the two intervals...
    # trapz([0,10,20,0,0], [0,900,1800,2700,3600]) = 27000 L = 27 m3.
    assert node_summary_df.loc["A", "overflow_volume_m3"] == pytest.approx(27.0)
    assert node_summary_df.loc["A", "peak_overflow_rate_lps"] == 20.0
    assert node_summary_df.loc["A", "overflow_duration_min"] == pytest.approx(30.0)

    # Only overflowing manholes are listed, worst first; C is dropped.
    assert list(node_summary_df.index) == ["A", "B"]

    # System summary (items 6, 7).
    assert system_summary["total_overflow_volume_m3"] == pytest.approx(
        node_summary_df["overflow_volume_m3"].sum()
    )
    assert system_summary["n_overflowing_manholes"] == 2
    assert system_summary["mean_overflow_volume_per_manhole_m3"] == pytest.approx(
        system_summary["total_overflow_volume_m3"] / 2
    )
    assert system_summary["peak_n_overflowing_manholes"] == 1
    assert system_summary["n_nodes"] == 3

    # Optional per-node x time matrix (item 2).
    assert wide.shape == (5, 3)


def test_threshold_filters_counting():
    """threshold_lps only counts manholes whose rate exceeds it."""
    rates = _synthetic_rates()
    _, node_summary_df, system_summary = _compute_overflow_metrics(
        rates, threshold_lps=6.0
    )
    # B peaks at 5 LPS < 6, so only A qualifies.
    assert list(node_summary_df.index) == ["A"]
    assert system_summary["n_overflowing_manholes"] == 1


def test_empty_when_no_overflow():
    """A model that never floods yields no overflowing manholes but valid frames."""
    idx = pd.date_range("2025-01-01", periods=3, freq="15min")
    rates = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=idx)
    rates.index.name = "Datetime"
    timestep_df, node_summary_df, system_summary = _compute_overflow_metrics(rates)

    assert (timestep_df["n_overflowing_manholes"] == 0).all()
    assert node_summary_df.empty
    assert system_summary["total_overflow_volume_m3"] == 0.0
    assert system_summary["mean_overflow_volume_per_manhole_m3"] == 0.0


@pytest.mark.skipif(not _EXAMPLE_OUT.exists(), reason="example .out not present")
def test_read_out_structure():
    """Reading a real .out results file returns the documented structure."""
    pytest.importorskip("pyswmm")
    timestep_df, node_summary_df, system_summary = quantify_sewer_overflows(_EXAMPLE_OUT)

    assert list(timestep_df.columns) == _TIMESTEP_COLS
    assert timestep_df.index.name == "Datetime"
    assert len(timestep_df) > 0
    assert (timestep_df["n_overflowing_manholes"] >= 0).all()
    assert (timestep_df["total_overflow_rate_lps"] >= -1e-9).all()
    assert (timestep_df["n_overflowing_manholes"] <= system_summary["n_nodes"]).all()

    # System total must equal the sum of the per-manhole volumes it reports.
    assert system_summary["total_overflow_volume_m3"] == pytest.approx(
        node_summary_df["overflow_volume_m3"].sum()
    )


@pytest.mark.skipif(not _EXAMPLE_INP.exists(), reason="example .inp not present")
def test_read_node_coordinates_and_links():
    """Coordinates and pipe connectivity parse out of the .inp cleanly."""
    coords = read_node_coordinates(_EXAMPLE_INP)
    assert list(coords.columns) == ["x", "y"]
    assert coords.index.name == "node_id"
    assert len(coords) > 0
    assert coords[["x", "y"]].notna().all().all()

    links = read_conduit_links(_EXAMPLE_INP)
    assert links and all(len(pair) == 2 for pair in links)
    # Endpoints reference known nodes.
    a, b = links[0]
    assert a in coords.index and b in coords.index


def test_plot_manhole_overflows_smoke():
    """The spatial plot builds without a display and carries a colorbar."""
    import matplotlib
    matplotlib.use("Agg")

    coords = pd.DataFrame(
        {"x": [0.0, 100.0, 200.0], "y": [0.0, 50.0, 0.0]},
        index=pd.Index(["A", "B", "C"], name="node_id"),
    )
    _, node_summary_df, _, _ = _compute_overflow_metrics(
        _synthetic_rates(), return_node_timeseries=True
    )
    fig, ax = plot_manhole_overflows(
        coords, node_summary_df, links=[("A", "B"), ("B", "C")]
    )
    assert ax.collections  # at least one scatter drawn
    assert fig.axes[-1].get_ylabel().startswith("Total overflow volume")


@pytest.mark.slow
def test_inp_and_out_agree_on_volume(swmm_ran):
    """Live .inp re-run and the finalized .out agree on total overflow volume."""
    scenario, _, _ = swmm_ran
    inp_path = scenario.swmm_inp_path
    out_path = inp_path.with_suffix(".out")

    _, _, sys_inp = quantify_sewer_overflows(
        inp_path, source="inp", step_advance_seconds=900
    )
    _, _, sys_out = quantify_sewer_overflows(out_path, source="out")

    assert sys_inp["total_overflow_volume_m3"] == pytest.approx(
        sys_out["total_overflow_volume_m3"], abs=1e-6
    )
