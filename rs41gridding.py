# imports
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime

# helpers
from helpers.download_manager.download_manager import DownloadManager
DM=DownloadManager()
from helpers.format_manager.format_manager import FormatManager
FM=FormatManager()
from helpers.grid_manager.grid_manager import GridManager
GM=GridManager()

# COSTANTS

# download documentation
files_name=[
    'GRUAN-TD-8_RS41_v1.0.0_20230628_final.pdf',
    'GRUAN-TN-13_RS41-GDP_UserGuide_v1.0_2022-11-21.pdf'
]
base_url="https://www.gruan.org/gruan/editor/documents/gruan/"
for file_name in files_name:
    pdf_document = "doc\\" + file_name
    if not os.path.exists(pdf_document):
        DM.download_pdf_from_url(pdf_url=base_url + file_name, downloaded_pdf_name=file_name)
        DM.move_file(source_path='download\\' + file_name, destination_path=pdf_document)

# download gdp
# 2017 zip at https://www.gruan.org/data/file-archive/rs41-gdp1-at-lc

# read data
gdp_folder = 'data\\gdp\\products_RS41-GDP-1_LIN_2017'
gdp_files = [os.path.join(gdp_folder, f) for f in os.listdir(gdp_folder) if f.endswith('.nc')]
gdps=[]
for file in tqdm(gdp_files[:5] , desc="Reading GDPs"):
    FM.read_nc_file(file)
    gdp=FM.return_gdp()
    gdps.append(gdp)

# Variable Spatial Gridding
binsize=1000 #expressed in meters
for gdp in gdps:
    #break
    bins=range(0, int(gdp.data['alt'].max()), binsize)
    bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
    gridata = pd.DataFrame(bin_alt, columns=['alt'])
    variables = ['temp']#'press', 'temp', 'rh', 'wdir', 'wspeed']
    for var in variables:
        gridata = GM.rs41_spatial_gridding(gdp.data, binsize, [var])
        var_uc = var+'_uc'
        var_u = var+'_u'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot original and gridded data
        ax1.scatter(gdp.data[var], gdp.data['alt'], label='Original Data', alpha=0.5)
        ax1.scatter(gridata[var], gridata['alt'], label='Gridded Data', color='red', alpha=0.5)
        ax1.fill_betweenx(gdp.data['alt'], gdp.data[var] - gdp.data[var_uc], gdp.data[var] + gdp.data[var_uc], color='blue', alpha=0.2, label='Original Uncertainty')
        ax1.fill_betweenx(gridata['alt'], gridata[var] - gridata[var_u], gridata[var] + gridata[var_u], color='red', alpha=0.2, label='Gridded Uncertainty')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Altitude (m)')
        ax1.legend()
        ax1.set_title('Temperature Spatial Gridding')

        # Plot uncertainty distribution over altitude
        ax2.plot(gdp.data['temp_uc'], gdp.data['alt'], label='Combined')
        ax2.plot(gdp.data['temp_uc_ucor'], gdp.data['alt'], label='Uncorrelated')
        ax2.plot(gdp.data['temp_uc_scor'], gdp.data['alt'], label='Spatially Correlated')
        ax2.plot(gdp.data['temp_uc_tcor'], gdp.data['alt'], label='Temporally Correlated')
        ax2.set_xlabel('Temperature Uncertainty (K)')
        ax2.set_ylabel('Altitude (m)')
        ax2.legend()
        ax2.set_title('Original Temperature Uncertainty Components')

        plt.tight_layout()
        plt.show()
    
# Variable Temporal Gridding
if False:
    time_binsize=5
    spatial_binsize=10000
    vars=['temp']
    temp_grid = GM.rs41_temporal_gridding(gdps, time_binsize, spatial_binsize, vars)
    #print(temp_grid.head(), temp_grid.columns)    
    # Plot temperature trend over time for each altitude
    unique_alts = temp_grid['alt'].unique()
    for alt in unique_alts:
        alt_data = temp_grid[temp_grid['alt'] == alt]
        plt.plot(alt_data['time'], alt_data['temp'], label=f'Altitude {alt} m')
        plt.fill_between(alt_data['time'], alt_data['temp'] - alt_data['temp_u'], alt_data['temp'] + alt_data['temp_u'], alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.title('Temperature Trend over Time for Each Altitude')

    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f'temperature_trend_over_time_{timestamp}.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()