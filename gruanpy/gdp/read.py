import pandas as pd
import xarray as xr
import os
from gruanpy.physics.pblh import apply_upper_bound
from gruanpy.physics.formulas import virtual_potential_temperature, virtual_potential_temperature_uncertainty
from gruanpy.gdp.data_models import GDP, NETCDF
import gruanpy
import pickle

def read_gdp(file_path, only_global_attrs=False, upper_bound = False, columns = False):
    """
    Read a GRUAN GDP NetCDF file and return a GDP object.
    """
    content = xr.open_dataset(file_path)

    # Global attributes
    global_attrs = pd.DataFrame(
        content.attrs.items(),
        columns=["Attribute", "Value"]
    )

    if not only_global_attrs:
        # Data variables
        data = content.to_dataframe().sort_values(by="alt")
        data = data.reset_index()

        if upper_bound == True:
            data = apply_upper_bound(data)
        elif type(upper_bound) in [int, float]:
            data = apply_upper_bound(data, upper_bound=upper_bound)

        if columns:
            # Virtual potential temperature
            if ("theta_v" in columns and "theta_v" not in data.columns) or ("theta_v_uc" in columns and "theta_v_uc" not in data.columns):
                T = data["temp"]
                p = data["press"]
                r = data["wvmr_mass"] / 1e6
                data["theta_v"] = virtual_potential_temperature(T, p, r)
                T_unc = data["temp_uc"]
                p_unc = data["press_uc"]
                r_unc = data["wvmr_mass_uc"]
                r_unc = data["wvmr_mass_uc"] / 1e6
                data["theta_v_uc"] = virtual_potential_temperature_uncertainty(
                    T, p, r, T_unc, p_unc, r_unc
                )

            valid_cols = [c for c in columns if c in data.columns]
            missing_cols = set(columns) - set(valid_cols)

            if missing_cols:
                print(f"Warning: these columns do not exist: {missing_cols}")

            data = data[valid_cols]


        variables_attrs = pd.DataFrame([
            {**var.attrs, "variable": var_name}
            for var_name, var in content.data_vars.items()
        ])
    else:
        data = None
        variables_attrs = None

    return GDP(global_attrs, data, variables_attrs)


def read_cdm(file_path):
    """
    Read a CDM-format GDP file (.nc or .csv).
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".nc", ".netcdf"]:
        return read_gdp(file_path)

    elif ext == ".csv":
        data = pd.read_csv(file_path)
        return GDP(None, data, None)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def read_netcdf(file_path):
    """
    Read a generic NetCDF file into a NETCDF object.
    """
    content = xr.open_dataset(file_path)

    global_attrs = pd.DataFrame(
        content.attrs.items(),
        columns=["Attribute", "Value"]
    )

    data = content.to_dataframe().reset_index()

    variables_attrs = pd.DataFrame([
        {**var.attrs, "variable": var_name}
        for var_name, var in content.data_vars.items()
    ])

    return NETCDF(global_attrs, data, variables_attrs)

def read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)