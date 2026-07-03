"""Public API for SewerTris."""

from __future__ import annotations

from . import plots, swmm
from .config import *
from .design import *
from .domain import *
from .io import *
from .layout import *
from .plots import *
from .hydrology import *
from .project import SewerTrisProject, SewerTrisScenario
from .ensemble import (
    run_project_sibling,
    run_project_sibling_from_file,
    run_project_simulation,
)
from .roads import *
from .sewer_network import *
from .swmm import *
from .topography import *
from ._deps import save_vector


__version__ = "0.1.0"

__all__ = [
    "SewerTrisProject",
    "SewerTrisScenario",
    "run_project_sibling",
    "run_project_sibling_from_file",
    "run_project_simulation",
    "__version__",
    "plots",
    "swmm",
    "save_vector",
]

for _module in (
    plots,
    swmm,
):
    __all__.extend(getattr(_module, "__all__", []))

for _name in list(globals()):
    if not _name.startswith("_") and _name not in __all__:
        __all__.append(_name)
