import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import GRUANpy as Gp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from gruanpy.data_models.ggd import Ggd

gp=Gp()

gdp_folder=r'gdp\products_RS41-GDP-1_LIN_2017'
gdp_files = [os.path.join(gdp_folder, f) for f in os.listdir(gdp_folder) if f.endswith('.nc')]
gdps=[]
for file in tqdm(gdp_files[:10] , desc="Reading GDPs"):
    gdps.append(gp.read(file))

ggds=[]
for gdp in tqdm(gdps, desc="Spatial Gridding"):
    bin_column = 'press'
    target_columns = ['temp', 'rh']
    bin_size = (gdp.data[bin_column].max()-gdp.data[bin_column].min()) / 100
    ggd = gp.spatial_gridding(gdp, bin_column, target_columns, bin_size)
    ggds.append(ggd)

target_columns = ['temp', 'rh']
bin_size = 7

tggd=gp.temporal_gridding(ggds, target_columns, bin_size)
print(tggd.metadata.head())
print(tggd.data.head())
#print(tggd['time'].min(), tggd['time'].max())

