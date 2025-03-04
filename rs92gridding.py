"""
Look at GRUAN-TN-13 section 3.4 Usage of files but applied to RS92-GDP
"""

# imports
from noaa_ftp import NOAA
import pandas as pd
import matplotlib.pyplot as plt

# helpers
from helpers.download_manager.download_manager import DownloadManager
DM=DownloadManager()
from helpers.format_manager.format_manager import FormatManager
FM=FormatManager()

# COSTANTS
FTP="ftp.ncdc.noaa.gov"

#"""
# download documentation
file_name="GRUAN-TD-4_RS92-GDP_BriefDescription_v2_1_1.pdf"
pdf_document = "doc\\"+file_name
DM.download_pdf_from_url(pdf_url="https://www.gruan.org/gruan/editor/documents/gruan/GRUAN-TD-4_RS92-GDP_v2.1.1_2021-07-14.pdf",
            downloaded_pdf_name=file_name)
DM.move_file(source_path='download\\'+file_name, destination_path=pdf_document)

file_name="GRUAN-TN-13_RS41-GDP_UserGuide_v1_0.pdf"
pdf_document = "doc\\"+file_name
DM.download_pdf_from_url(pdf_url="https://www.gruan.org/gruan/editor/documents/gruan/GRUAN-TN-13_RS41-GDP_UserGuide_v1.0_2022-11-21.pdf",
            downloaded_pdf_name=file_name)
DM.move_file(source_path='download\\'+file_name, destination_path=pdf_document)
"""

"""
# download gdp
dir_path=r'pub/data/gruan/processing/level2/RS92-GDP/version-002/LIN/2021'
file_name=r'LIN-RS-01_2_RS92-GDP_002_20210125T132400_1-000-001.nc'
noaa = NOAA(FTP, dir_path).download(file_name)
DM.move_file(file_name, 'data\\gdp\\'+file_name)

dir_path=r'pub/data/gruan/processing/level2/RS92-GDP/version-002/LIN/2021'
file_name=r'LIN-RS-01_2_RS92-GDP_002_20210302T181500_1-000-001.nc'
noaa = NOAA(FTP, dir_path).download(file_name)
DM.move_file(file_name, 'data\\gdp\\'+file_name)

dir_path=r'pub/data/gruan/processing/level2/RS92-GDP/version-002/LIN/2011'
file_name=r'LIN-RS-01_2_RS92-GDP_002_20110101T120000_1-000-001.nc'
noaa = NOAA(FTP, dir_path).download(file_name)
DM.move_file(file_name, 'data\\gdp\\'+file_name)

dir_path=r'pub/data/gruan/processing/level2/RS92-GDP/version-002/LIN/2011'
file_name=r'LIN-RS-01_2_RS92-GDP_002_20110301T060000_1-000-001.nc'
noaa = NOAA(FTP, dir_path).download(file_name)
DM.move_file(file_name, 'data\\gdp\\'+file_name)


#"""

# read data
ncs=[
    r'data\\gdp\\LIN-RS-01_2_RS92-GDP_002_20210125T132400_1-000-001.nc',
    r'data\\gdp\\LIN-RS-01_2_RS92-GDP_002_20210302T181500_1-000-001.nc',
    r'data\gdp\LIN-RS-01_2_RS92-GDP_002_20110101T120000_1-000-001.nc',
    r'data\gdp\LIN-RS-01_2_RS92-GDP_002_20110301T060000_1-000-001.nc'
]
gdps=[]
for nc in ncs:
    FM.read_nc_file(nc)
    glb_attrs, data, vars_attrs=FM.return_dataframes()
    gdp=[glb_attrs, data, vars_attrs]
    gdps.append(gdp)

# Temperature Spatial Gridding
binsize=10000 ##binsize=input('Type the binsize in meters: ')
for gdp in gdps:
    data=gdp[1]
    bins=range(0,#int(data['alt'].min()), 
               int(data['alt'].max()), 
               binsize)

    # QUESTION: how to compute the uncertainty of the temperature?
    # OBS: eq. 3.11 depends on 3.9 and 3.10, for which uscor and tscor are needed--they are not in the data for rs92
    # ASSUMPTION: I will use eq. 3.8 after having applied eq. 3.6 on the total uncertainty

    bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
    avg_temp = data.groupby(pd.cut(data['alt'], bins), observed=True)['temp'].mean().reset_index(drop=True) # (3.5)

    # (3.6) applied to the total uncertainty
    avg_u_temp_squared = data.groupby(pd.cut(data['alt'], bins), observed=True)['u_temp'].apply(
        lambda x: (x**2).sum()**0.5 / len(x)).reset_index(drop=True)

    sd_temp = data.groupby(pd.cut(data['alt'], bins), observed=True)['temp'].std().reset_index(drop=True) # (3.7)
    gridata = pd.DataFrame({'alt': bin_alt, 'temp': avg_temp, 'ucor_temp': avg_u_temp_squared, 'var_temp': sd_temp})
    gridata['uc_temp'] = (gridata['ucor_temp']**2 + gridata['var_temp']**2)**0.5 # (3.8)
    gdp.append(gridata)

    # plot
    plt.scatter(data['temp'], data['alt'], label='Original Data', alpha=0.5)
    plt.scatter(gridata['temp'], gridata['alt'], label='Gridded Data', color='red', alpha=0.5)
    plt.fill_betweenx(data['alt'], data['temp'] - data['u_temp'], data['temp'] + data['u_temp'], color='blue', alpha=0.2, label='Original Uncertainty')    
    plt.fill_betweenx(gridata['alt'], gridata['temp'] - gridata['uc_temp'], gridata['temp'] + gridata['uc_temp'], color='red', alpha=0.2, label='Gridded Uncertainty')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.legend()
    plt.title('Temperature Spatial Gridding')
    plt.show()

# Temporal gridding by year
grids=[gdp[3] for gdp in gdps]
dates = [gdp[1].index[0] for gdp in gdps]
years = list(set([date.year for date in dates]))
temporal_gridding = pd.DataFrame()
for year in years:
    yearly_grids = [grid for grid, date in zip(grids, dates) if date.year == year]
    if yearly_grids:
        combined_data = pd.concat(yearly_grids)
        avg_temp_by_alt = combined_data.groupby('alt')['temp'].mean().reset_index() # (3.12)
        avg_temp_by_alt['uc_temp'] = combined_data.groupby('alt')['uc_temp'].apply(
            lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) # (3.13)
        avg_temp_by_alt['var_temp'] = combined_data.groupby('alt')['temp'].std().reset_index(drop=True) # (3.14)
        avg_temp_by_alt['year'] = year
        avg_temp_by_alt['u_temp'] = (avg_temp_by_alt['uc_temp']**2 + avg_temp_by_alt['var_temp']**2)**0.5 # (3.16) spatially correlated uncertainty are missing
        temporal_gridding = pd.concat([temporal_gridding, avg_temp_by_alt], ignore_index=True)

unique_alts = temporal_gridding['alt'].unique()
plt.figure(figsize=(10, 6))
for alt in unique_alts:
    alt_data = temporal_gridding[temporal_gridding['alt'] == alt]
    plt.plot(alt_data['year'], alt_data['temp'], label=f'Altitude {alt} m')
    plt.fill_between(alt_data['year'], alt_data['temp'] - alt_data['u_temp'], alt_data['temp'] + alt_data['u_temp'], alpha=0.2)
plt.xlabel('Year')
plt.ylabel('Temperature (K)')
plt.title('Temperature Series for Each Altitude')
plt.legend()
plt.show()