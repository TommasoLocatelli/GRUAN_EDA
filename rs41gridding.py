# imports
import pandas as pd
import matplotlib.pyplot as plt
import os

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
for file in gdp_files[:10]:
    FM.read_nc_file(file)
    glb_attrs, data, vars_attrs=FM.return_dataframes()
    gdp=[glb_attrs, data, vars_attrs]
    gdps.append(gdp)

# Variable Spatial Gridding
binsize=1000 #expressed in meters
for gdp in gdps:
    bins=range(0, int(gdp[1]['alt'].max()), binsize)
    bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
    gridata = pd.DataFrame(bin_alt, columns=['alt'])
    variables = ['temp']#'press', 'temp', 'rh', 'wdir', 'wspeed']
    for var in variables:
        gridata = GM.rs41_spatial_griding(gdp[1], binsize, [var])
        var_uc = var+'_uc'
        var_u = var+'_u'
        # plot
        plt.scatter(gdp[1][var], gdp[1]['alt'], label='Original Data', alpha=0.5)
        plt.scatter(gridata[var], gridata['alt'], label='Gridded Data', color='red', alpha=0.5)
        plt.fill_betweenx(gdp[1]['alt'], gdp[1][var] - gdp[1][var_uc], gdp[1][var] + gdp[1][var_uc], color='blue', alpha=0.2, label='Original Uncertainty')    
        plt.fill_betweenx(gridata['alt'], gridata[var] - gridata[var_u], gridata[var] + gridata[var_u], color='red', alpha=0.2, label='Gridded Uncertainty')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Altitude (m)')
        plt.legend()
        plt.title('Temperature Spatial Gridding')
        plt.show()
    
# Variable Temporal Gridding