"""Shared imports for SewerTris implementation modules.

This module centralizes optional scientific/geospatial imports so package modules
can be imported even before the full SewerTris conda environment is active. The
actual functions still require their runtime dependencies when called.
"""

from __future__ import annotations

from collections import defaultdict
from math import hypot, sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple
import math
import os
import random
import sys
import warnings

_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".sewertris_cache"
_MPL_CACHE_DIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_DIR = _CACHE_ROOT / "xdg"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency availability varies by env
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    from scipy.interpolate import griddata
except Exception:  # pragma: no cover
    griddata = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:
    from scipy.sparse import coo_matrix, csr_matrix
except Exception:  # pragma: no cover
    coo_matrix = None
    csr_matrix = None

try:
    from scipy.sparse.csgraph import breadth_first_order, dijkstra
except Exception:  # pragma: no cover
    breadth_first_order = None
    dijkstra = None

try:
    from skimage.measure import label
except Exception:  # pragma: no cover
    label = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.colors import LightSource, ListedColormap
except Exception:  # pragma: no cover
    plt = None
    cm = None
    colors = None
    LightSource = None
    ListedColormap = None

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

try:
    import rasterio
    from rasterio import features
    from rasterio.features import rasterize
    from rasterio.mask import mask
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover
    rasterio = None
    features = None
    rasterize = None
    mask = None
    from_origin = None

try:
    import osmnx as ox
except Exception:  # pragma: no cover
    ox = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None

try:
    from shapely.errors import TopologicalError
except Exception:  # pragma: no cover
    TopologicalError = Exception

try:
    from shapely.geometry import (
        Point,
        LineString,
        MultiLineString,
        Polygon,
        MultiPolygon,
        box,
    )
except Exception:  # pragma: no cover
    Point = None
    LineString = None
    MultiLineString = None
    Polygon = None
    MultiPolygon = None
    box = None

try:
    from shapely.ops import linemerge, unary_union
except Exception:  # pragma: no cover
    linemerge = None
    unary_union = None

try:
    from shapely.strtree import STRtree
except Exception:  # pragma: no cover
    STRtree = None

try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover
    make_valid = None


def save_vector(gdf, path, **kwargs):
    """Write a GeoDataFrame to *path*, silencing harmless ESRI Shapefile warnings.

    Shapefiles cap field names at 10 characters, so columns such as
    ``downstream_m`` are silently truncated (to ``downstream``) on write.
    SewerTris expects this and restores the full names on read via
    :func:`ensure_pipe_topology_aliases`, so the resulting geopandas truncation
    warning and the pyogrio ``Normalized/laundered field name`` warning are pure
    noise. This wrapper filters those two specific messages and leaves every
    other warning untouched.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Column names longer than 10 characters",
        )
        warnings.filterwarnings(
            "ignore",
            message="Normalized/laundered field name",
        )
        gdf.to_file(path, **kwargs)


def ensure_pipe_topology_aliases(gdf):
    """Ensure both full and shapefile-truncated downstream field names exist."""
    if gdf is None or not hasattr(gdf, "columns"):
        return gdf
    if "downstream" not in gdf.columns and "downstream_m" in gdf.columns:
        gdf = gdf.copy()
        gdf["downstream"] = gdf["downstream_m"]
    if "downstream_m" not in gdf.columns and "downstream" in gdf.columns:
        gdf = gdf.copy()
        gdf["downstream_m"] = gdf["downstream"]
    return gdf
