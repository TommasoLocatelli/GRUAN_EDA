import pandas as pd
import xarray as xr
from helpers.format_manager.gdp import Gdp

class FormatManager:
    def __init__(self):
        self.data=None
        self.global_attrs=None
        self.variables=None
        self.variables_attrs=None
        self.gdp=None

    def read_nc_file(self, path):
        #Open .nc file
        self.data=xr.open_dataset(path)
        #Extract global attributes
        self.global_attrs=pd.DataFrame(self.data.attrs.items(), columns=['Attribute', 'Value'])
        #Extract variables data
        self.variables=self.data.to_dataframe()
        #Extract variables attributes
        variables_attrs = []
        for var_name, var in self.data.data_vars.items():
            attrs = var.attrs
            attrs['variable'] = var_name
            variables_attrs.append(attrs)
        self.variables_attrs=pd.DataFrame(variables_attrs)
        self.gdp=(Gdp(self.global_attrs, self.variables, self.variables_attrs))

    def return_dataframes(self):
        if self.data:
            return self.global_attrs, self.variables, self.variables_attrs
        
    def return_gdp(self):
        if self.gdp:
            return self.gdp
        
    # add some method to save read data with a more usable format (pickle, csv, etc)