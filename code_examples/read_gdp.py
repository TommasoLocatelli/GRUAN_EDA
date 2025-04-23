import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy.helpers.read.reading_manager import ReadingManager as RM

rm=RM()
file_path=r'gdp\NYA-RS-01_2_RS-11G-GDP_001_20200319T000000_1-002-001.nc'
gdp=rm.read(file_path)
print(gdp.global_attrs)
print(gdp.data.head())
print(gdp.variables_attrs.head())