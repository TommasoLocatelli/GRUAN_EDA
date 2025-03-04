# imports
import pandas as pd
import matplotlib.pyplot as plt
import os

# helpers
from helpers.download_manager.download_manager import DownloadManager
DM=DownloadManager()
from helpers.format_manager.format_manager import FormatManager
FM=FormatManager()

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
for file in gdp_files[:2]:
    FM.read_nc_file(file)
    glb_attrs, data, vars_attrs=FM.return_dataframes()
    gdp=[glb_attrs, data, vars_attrs]
    gdps.append(gdp)
    print(data)
    break

# Variable Spatial Gridding
binsize=10000 #expressed in meters
for gdp in gdps:
    bins=range(0, int(gdp[1]['alt'].max()), binsize)
    bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
    gridata = pd.DataFrame(bin_alt, columns=['alt'])
    variables = ['temp']#'press', 'temp', 'rh', 'wdir', 'wspeed']
    for var in variables:
        bin_avg = gdp[1].groupby(pd.cut(gdp[1]['alt'], bins), observed=True)[var].mean().reset_index(drop=True) # 3.5
        uc_ucor = var + '_uc_ucor'
        bin_ucor = gdp[1].groupby(pd.cut(gdp[1]['alt'], bins), observed=True)[uc_ucor].apply(
            lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) #3.6
        bin_uvar = gdp[1].groupby(pd.cut(gdp[1]['alt'], bins), observed=True)[var].apply(
            lambda x: ((x.var())/len(x))**0.5 ).reset_index(drop=True) #3.7
        bin_uc = (bin_ucor**2+bin_uvar**2)**0.5 #3.8
        uc_scor = var+'_uc_scor'
        bin_sc = gdp[1].groupby(pd.cut(gdp[1]['alt'], bins), observed=True)[uc_scor].mean().reset_index(drop=True) #3.9
        uc_tcor = var+'_uc_tcor'
        bin_tc = gdp[1].groupby(pd.cut(gdp[1]['alt'], bins), observed=True)[uc_tcor].mean().reset_index(drop=True) #3.8
        bin_u = (bin_uc**2+bin_sc**2+bin_tc**2)**0.5
        gridata[var] = bin_avg
        var_u = var+'_u'
        gridata[var_u] = bin_u
        var_uc = var+'_uc'

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