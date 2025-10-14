import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp


folder = r'gdp\products_RS41-GDP-1_POT_2025'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]:
    print(f'Processing file: {file_path}')
# Look at thermodynamic and dynamic variable profiles

# Look at PBLH methods variable of interest profiles

# Look at PBLH results

# how does the result change if gradient is modeled?

# monte carlo simulation of PBLH methods proof of concept