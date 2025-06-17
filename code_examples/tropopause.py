import numpy as np

"""
Tropopause Calculation Exercise

https://www.ncl.ucar.edu/Document/Functions/Built-in/trop_wmo.shtml

"""

def trop_wmo(p, t, punit, opt):
    """
    Calculate the WMO tropopause pressure level.

    Parameters
    ----------
    p : array_like
        Pressure levels, monotonically increasing (top-to-bottom).
    t : array_like
        Temperatures in Kelvin, same shape as p.
    punit : int
        0 for hPa, 1 for Pa.
    opt : bool or dict
        If True and 'lapsec' in opt, use opt['lapsec'] as lapse rate threshold (K/km).
        Otherwise, default is 2.0 K/km.

    Returns
    -------
    tropopause_p : float or np.ndarray
        Pressure at the tropopause level.
    """
    p = np.asarray(p)
    t = np.asarray(t)

    # Convert pressure to hPa if needed
    if punit == 1:
        p = p / 100.0

    # Lapse rate threshold
    lapsec = 2.0
    if isinstance(opt, dict) and opt.get('lapsec') is not None:
        lapsec = opt['lapsec']

    # Ensure rightmost dimension is level
    if p.shape != t.shape:
        raise ValueError("p and t must have the same shape")

    # Compute log-pressure height (approximate)
    Rd = 287.05  # J/(kg·K)
    g0 = 9.80665  # m/s^2

    # Calculate geopotential height using hypsometric equation
    # z = -Rd * T_mean / g0 * ln(p2/p1)
    def compute_height(p_levels, t_levels):
        z = np.zeros_like(p_levels)
        for k in range(1, len(p_levels)):
            t_mean = 0.5 * (t_levels[k-1] + t_levels[k])
            z[k] = z[k-1] - (Rd * t_mean / g0) * np.log(p_levels[k] / p_levels[k-1])
        return z

    # Support for multi-dimensional arrays
    def find_tropopause_1d(p1d, t1d):
        z = compute_height(p1d, t1d)
        # Calculate lapse rate (K/km) between levels
        dz = np.diff(z) / 1000.0  # km
        dt = np.diff(t1d)
        lapse_rate = -dt / dz  # K/km

        # WMO tropopause: lowest level where lapse rate <= lapsec K/km over 2 km above
        for k in range(len(lapse_rate)):
            z0 = z[k]
            z2km = z0 + 2.0  # km
            # Find indices within 2 km above
            idx_above = np.where((z > z0) & (z <= z2km))[0]
            if len(idx_above) == 0:
                continue
            # Mean lapse rate over this layer
            dz_sum = z[idx_above[-1]] - z0
            dt_sum = t1d[idx_above[-1]] - t1d[k]
            mean_lapse = -dt_sum / dz_sum if dz_sum != 0 else np.inf
            if lapse_rate[k] <= lapsec and mean_lapse <= lapsec:
                return p1d[k]
        return np.nan

    # Handle multi-dimensional arrays
    if p.ndim == 1:
        return find_tropopause_1d(p, t)
    else:
        # Apply along all axes except last
        shape = p.shape[:-1]
        trop_p = np.full(shape, np.nan, dtype=p.dtype)
        it = np.nditer(np.zeros(shape), flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            trop_p[idx] = find_tropopause_1d(p[idx], t[idx])
            it.iternext()
        return trop_p
    
# Example usage

# Example data (from NCL example)
p = np.array([
    1008.,1000.,950.,900.,850.,800.,750.,700.,650.,600.,
    550.,500.,450.,400.,350.,300.,250.,200.,
    175.,150.,125.,100.,80.,70.,60.,50.,
    40.,30.,25.,20.
])

t = np.array([
    29.3,28.1,23.5,20.9,18.4,15.9,13.1,10.1,6.7,3.1,
    -0.5,-4.5,-9.0,-14.8,-21.5,-29.7,-40.0,-52.4,
    -59.2,-66.5,-74.1,-78.5,-76.0,-71.6,-66.7,-61.3,
    -56.3,-51.7,-50.7,-47.5
])

# Convert temperature to Kelvin
t = t + 273.15

# Reverse order to top-to-bottom
p = p[::-1]
t = t[::-1]

ptrop = trop_wmo(p, t, 0, False)
print("Tropopause pressure (hPa):", ptrop)