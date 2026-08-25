import xarray as xr
import pandas as pd

content = xr.open_dataset(r'data\cloudnet-collection-af9aa392ac834f67\20260816_cabauw_wls200s-wind_dca88604.nc')

# Global attributes
global_attrs = pd.DataFrame(content.attrs.items(),
                            columns=["Attribute", "Value"])

# Data variables
data = content.to_dataframe()
data = data.reset_index()
variables_attrs = pd.DataFrame([
    {**var.attrs, "variable": var_name}
    for var_name, var in content.data_vars.items()
])

print(data.head())