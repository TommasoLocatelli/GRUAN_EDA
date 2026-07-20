"""
Example code to read GDP data.
"""
import sys
import os
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp

file_paths=[r'gdp\exmpls\BAR-RS-01_2_RS41-GDP_001_20180227T193000_1-000-001.nc',
    r'gdp\exmpls\BAR-RS-01_2_RS92-GDP_002_20090101T060000_1-000-001.nc',
    r'gdp\exmpls\LIN-RS-01_2_IMS-100-GDP_002_20220107T093400_1-002-001.nc',
    r'gdp\exmpls\LIN-RS-01_2_RS-11G-GDP_001_20141008T000000_1-004-001.nc']
file_path=file_paths[3]
gdp=gp.read(file_path)
print(gdp.global_attrs)
print(gdp.data)
print(gdp.variables_attrs)