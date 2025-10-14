import pandas as pd
import xarray as xr
from gruanpy.data_models.gdp import GDP
import os

class ReadingManager:
    def __init__(self):
        pass

    def read(self, file_path):
        content=xr.open_dataset(file_path)
        global_attrs=pd.DataFrame(content.attrs.items(), columns=['Attribute', 'Value'])
        data = content.to_dataframe().sort_values(by='alt') # Ensure data is sorted by altitude
        data = data.reset_index()  # Reset index to have a clean DataFrame
        variables_attrs = pd.DataFrame([
            {**var.attrs, 'variable': var_name} 
            for var_name, var in content.data_vars.items()
        ])

        gdp=GDP(global_attrs, data, variables_attrs)
        return gdp

    def read_cdm(self, file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in ['.nc', '.netcdf']:
            return self.read(file_path)
        elif ext == '.csv':
            data = pd.read_csv(file_path)
            gdp = GDP(None, data, None)
            return gdp
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
