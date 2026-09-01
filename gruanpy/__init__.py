# Public API for the gruanpy package
# --- GDP ---
from .gdp.download import search_gdp, download_gdp, exec_cds_request
from .gdp.read import read_cdm, read_gdp, read_netcdf, read_pkl
from .gdp.time_utils import *
# --- Physics ---
from .physics.formulas import *
from .physics.constants import *
from .physics.pblh import *

# --- SSM ---
from .ssm.statsmodels.local_linear_level import LocalLinearLevel
from .ssm.statsmodels.local_linear_trend import LocalLinearTrend
from .ssm.statsmodels.preprocessing import data_prep

# Public API
__all__ = [
    # GDP
    "search_gdp",
    "download_gdp",
    "exec_cds_request",
    "read_cdm",
    "read_gdp",
    "read_netcdf",
    "read_pkl",

    # SSM
    "LocalLinearLevel",
    "LocalLinearTrend",
    "data_prep",
]

# Add physics exports explicitly (functions + constants + classes)
__all__ += [
    name for name, obj in globals().items()
    if getattr(obj, "__module__", "").startswith("gruanpy.physics")
]

__all__ += [
    name for name, obj in globals().items()
    if getattr(obj, "__module__", "").startswith("gruanpy.time_utils")
]