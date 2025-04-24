import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import GRUANpy as Gp

gp=Gp()
file_path=r'gdp\NYA-RS-01_2_RS-11G-GDP_001_20200319T000000_1-002-001.nc'
gdp=gp.read(file_path)
print(gdp.global_attrs)
print(gdp.data.head())
print(gdp.variables_attrs.head())