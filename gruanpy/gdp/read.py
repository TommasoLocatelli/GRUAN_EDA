import pandas as pd
import xarray as xr
import os
from gruanpy.gdp.data_models import GDP, NETCDF

def read_gdp(file_path, only_global_attrs=False):
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
