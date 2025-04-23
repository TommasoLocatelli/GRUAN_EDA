import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy.helpers.read.reading_manager import ReadingManager as RM
from gruanpy.helpers.grid.gridding_manager import GriddingManager as GM
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from gruanpy.data_models.ggd import Ggd

rm=RM()
gm=GM()

gdp_folder=r'C:\Users\tomma\Documents\SDC\Repos\GRUAN_EDA\gdp\products_RS41-GDP-1_LIN_2017'
gdp_files = [os.path.join(gdp_folder, f) for f in os.listdir(gdp_folder) if f.endswith('.nc')]
gdps=[]
for file in tqdm(gdp_files[:10] , desc="Reading GDPs"):
    gdps.append(rm.read(file))

ggds=[]
for gdp in tqdm(gdps, desc="Spatial Gridding"):
    bin_column = 'press'
    target_columns = ['temp', 'rh']
    bin_size = (gdp.data[bin_column].max()-gdp.data[bin_column].min()) / 100
    ggd = gm.spatial_gridding(gdp, bin_column, target_columns, bin_size)
    ggds.append(ggd)

def temporal_gridding(ggds, target_columns, bin_size):
    # aggregate data from all ggds

    data=pd.DataFrame()
    for ggd in ggds:
        start_time_str = ggd.metadata[ggd.metadata['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
        start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        ggd_data = ggd.data.copy()
        ggd_data['time'] = start_time
        print(start_time)
        data = pd.concat([data, ggd_data], ignore_index=True)

    bin='time_bin'
    data[bin] = (data['time'].dt.day // bin_size) * bin_size + bin_size / 2
    first_data = data['time'].min()
    binned_data = data.groupby(bin)[target_columns].mean().reset_index() # 3.12
    binned_data['time'] = first_data + pd.to_timedelta(binned_data[bin], unit='D')

    for col in target_columns:
        print(data.columns)
        binned_data[col + '_uc'] = data.groupby(bin)[col + '_uc'].apply(
                    lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) #3.13
        binned_data[col + '_var'] = data.groupby(bin)[col].apply(
                    lambda x: ((x-x.mean())**2).sum()/(len(x)*(len(x)-1))**0.5 ).reset_index(drop=True) #3.14
        binned_data[col + '_uc_sc']=data.groupby(bin)[col + '_uc_scor'].apply(
                    lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True).reset_index(drop=True) #3.15
        binned_data[col + '_unc']= (binned_data[col+'_uc']**2 + binned_data[col + '_var']**2 + binned_data[col + '_uc_sc']**2)**0.5 #3.16
        binned_data[col + '_cor']=data.groupby(bin)[col + '_uc_tcor'].mean().reset_index(drop=True) #3.17
        binned_data[col+'_u']=(binned_data[col+'_unc']**2 + binned_data[col+'_cor']**2)**0.5 #3.18
    
    return binned_data

target_columns = ['temp', 'rh']
bin_size = 7

tggd=temporal_gridding(ggds, target_columns, bin_size)
print(tggd.head())
print(tggd.columns)
#print(tggd['time'].min(), tggd['time'].max())

