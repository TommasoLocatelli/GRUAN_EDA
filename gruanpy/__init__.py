# Public API for the gruanpy package

# --- GDP ---
from .gdp.download import search_gdp, download_gdp, exec_cds_request
from .gdp.read import read_cdm, read_gdp
from .gdp.validate import validation_pipeline

# --- Physics ---
from .physics.formulas import *
from .physics.constants import *
from .physics.pblh import *

# --- SSM (statsmodels implementations) ---
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
    "validation_pipeline",

    # SSM
    "LocalLinearLevel",
    "LocalLinearTrend",
    "data_prep",
]

# Add physics exports automatically (formulas/constants/pblh)
for name in list(globals()):
    if not name.startswith("_"):
        __all__.append(name)
