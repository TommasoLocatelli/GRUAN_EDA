# ssm_preprocess.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try to import gruanpy; raise a clear error if unavailable
try:
    import gruanpy as gp
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError("gruanpy is required by ssm_preprocess.py. Install it or provide a mock for tests.") from exc


def _safe_column(data, name: str) -> np.ndarray:
    """Return column values as 1D float ndarray; raise KeyError if missing."""
    if name not in data.columns:
        raise KeyError(f"Column '{name}' not found in input data")
    return np.asarray(data[name].values, dtype=float)


def _uncertainty_to_variance(uncertainty: np.ndarray) -> np.ndarray:
    """
    Convert reported uncertainty to variance.
    The original code used (unc * 0.5)**2; keep that behavior but document it.
    """
    return (np.asarray(uncertainty, dtype=float) * 0.5) ** 2


def load_gdp(path: str | Path):
    """Load a GRUAN GDP file using gruanpy.read and return the object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GDP file not found: {path}")
    return gp.read(str(path))


def preprocess_profile(path: str | Path, upper_bound: float = 3000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a GRUAN GDP file and return:
        obs      : ndarray (n, 8)
        meas_var : ndarray (n, 8)

    Observations order: [z, Lz, T, p, RH, r, u, v]
    Measurement variances follow the same order.

    Notes:
    - This function intentionally does not normalize data or build model components.
    - It warns (but does not raise) if NaNs are present so callers can decide how to handle them.
    """
    gdp = load_gdp(path)

    # determine PBLH upper bound using gruanpy helper
    upper_bound_val = gp._find_upper_bound(gdp.data[['alt']], upper_bound=upper_bound, return_value=True)

    # restrict to first part of profile
    data = gdp.data[gdp.data['alt'] <= upper_bound_val]

    # extract observations (1D arrays)
    z   = _safe_column(data, 'alt')
    Lz  = _safe_column(data, 'vspeed')
    T   = _safe_column(data, 'temp')
    p   = _safe_column(data, 'press')
    RH  = _safe_column(data, 'rh')
    r   = _safe_column(data, 'wvmr_mass')
    u   = _safe_column(data, 'wzon')
    v   = _safe_column(data, 'wmeri')

    # extract uncertainties and convert to variances
    z_var  = _uncertainty_to_variance(_safe_column(data, 'alt_gph_uc'))
    Lz_var = _uncertainty_to_variance(_safe_column(data, 'vspeed_uc'))
    T_var  = _uncertainty_to_variance(_safe_column(data, 'temp_uc'))
    p_var  = _uncertainty_to_variance(_safe_column(data, 'press_uc'))
    RH_var = _uncertainty_to_variance(_safe_column(data, 'rh_uc'))
    r_var  = _uncertainty_to_variance(_safe_column(data, 'wvmr_mass_uc'))
    u_var  = _uncertainty_to_variance(_safe_column(data, 'wzon_uc'))
    v_var  = _uncertainty_to_variance(_safe_column(data, 'wmeri_uc'))

    # stack into arrays (n, 8)
    obs = np.column_stack([z, Lz, T, p, RH, r, u, v])
    meas_var = np.column_stack([z_var, Lz_var, T_var, p_var, RH_var, r_var, u_var, v_var])

    # NaN check: warn but return arrays so caller can handle missing data
    if np.isnan(obs).any() or np.isnan(meas_var).any():
        logger.warning("obs or meas_var contains NaN values; caller should handle missing data")

    return obs, meas_var
