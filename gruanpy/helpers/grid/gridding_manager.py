import pandas as pd
from gruanpy.data_models.ggd import Ggd
from datetime import datetime

class GriddingManager:
    def __init__(self):
        pass

    def _mandatory_levels(self):
        # Mandatory levels in mb (Millibars) = hPa (Hectopascals) https://glossary.ametsoc.org/wiki/Mandatory_level
        lvls=[
            1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 
            100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
        ]  
        return lvls
    
    def spatial_gridding(self, gdp, bin_column, target_columns, bin_size=100, mandatory_levels_flag=True): #TN-13
        # bin_size is ignore if mandatory_levels_flag is True
        assert bin_column in ['alt', 'press'] 

        # spatial gridding
        data=gdp.data
        # 
        bin = bin_column+'_bin'
        if mandatory_levels_flag:
            bin_lvl = 'mand_lvl'
            data[bin_lvl] = data['press'].apply(
                lambda x: min(self._mandatory_levels(), key=lambda lvl: abs(lvl - x)) #python crazy syntax to select the nearest lvl
            )
            lvls_mean = data.groupby(bin_lvl)[bin_column].mean().to_dict()
            data[bin] = data[bin_lvl].apply(lambda x: lvls_mean[x])
            binned_data = data.groupby(bin_lvl)[target_columns].mean().reset_index() # 3.5
            binned_data[bin_column] = binned_data[bin_lvl].apply(lambda x: lvls_mean[x])
        else:
            data[bin] = (data[bin_column] // bin_size) * bin_size + bin_size / 2
            binned_data = data.groupby(bin)[target_columns].mean().reset_index() # 3.5
        
        for col in target_columns:
            binned_data[col + '_uc_ucor_avg'] = data.groupby(bin)[col + '_uc_ucor'].apply(
                        lambda x: (((x**2).sum())**0.5)/len(x)
                        ).reset_index(drop=True) #3.6
            binned_data[col + '_var'] = data.groupby(bin)[col].apply(
                        lambda x: (((x-x.mean())**2).sum()/(len(x)*(len(x)-1)))**0.5
                        ).reset_index(drop=True) #3.7
            binned_data[col + '_uc_ucor'] = (
                binned_data[col+'_uc_ucor_avg']**2 + binned_data[col + '_var']**2)**0.5 #3.8
            if col+'_uc_scor' in data.columns:
                binned_data[col + '_uc_scor'] = data.groupby(bin)[col + '_uc_scor'].mean().reset_index(drop=True) #3.9
            else:
                binned_data[col + '_uc_scor']=0
            if col+'_uc_tcor' in data.columns:
                binned_data[col + '_uc_tcor'] = data.groupby(bin)[col + '_uc_tcor'].mean().reset_index(drop=True) #3.10
            else:
                binned_data[col + '_uc_tcor']=0
            binned_data[col+'_uc']=(binned_data[col+'_uc_ucor']**2 + binned_data[col+'_uc_scor']**2 + binned_data[col+'_uc_tcor']**2)**0.5 #3.11

        # add metadata
        metadata = gdp.global_attrs[gdp.global_attrs['Attribute'].str.contains('Product|Measurement', case=False)]
        metadata = metadata._append({'Attribute': 'g.Gridding.Type', 'Value': 'Spatial Gridding'}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.BinColumn', 'Value': bin_column}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.BinSize', 'Value': str(bin_size)}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.TargetColumns', 'Value': ', '.join(target_columns)}, ignore_index=True)

        ggd=Ggd(metadata, binned_data)
        return ggd
    
    def temporal_gridding(self, ggds, target_columns, bin_size, lvl_column='mand_lvl'):
        # merge data from all ggds in a single table
        data=pd.DataFrame()
        for ggd in ggds:
            start_time_str = ggd.metadata[ggd.metadata['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
            start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            ggd_data = ggd.data.copy()
            ggd_data['time'] = start_time
            data = pd.concat([data, ggd_data], ignore_index=True)

        # temporal gridding
        bin='time_bin'
        data[bin] = (data['time'].dt.day // bin_size) * bin_size + bin_size / 2
        first_data = data['time'].min()
        
        binned_data = data.groupby([bin, lvl_column])[target_columns].mean().reset_index() # 3.12

        binned_data['time'] = first_data + pd.to_timedelta(binned_data[bin], unit='D')
        for col in target_columns:
            binned_data[col + '_uc_ucor_avg'] = data.groupby([bin,lvl_column])[col + '_uc_ucor'].apply(
                        lambda x: (((x**2).sum())**0.5)/len(x)
                        ).reset_index(drop=True) #3.13
            binned_data[col + '_var'] = data.groupby([bin,lvl_column])[col].apply(
                        lambda x: ((((x-x.mean())**2).sum())/(len(x)*max((len(x)-1),1)))**0.5
                        ).reset_index(drop=True) #3.14
            binned_data[col + '_uc_sc']=data.groupby([bin,lvl_column])[col + '_uc_scor'].apply(
                        lambda x: (((x**2).sum())**0.5)/len(x)
                        ).reset_index(drop=True) #3.15
            binned_data[col + '_uc_ucor']=(
                binned_data[col+'_uc_ucor_avg']**2 + binned_data[col + '_var']**2 + binned_data[col + '_uc_sc']**2)**0.5 #3.16
            binned_data[col + '_cor']=data.groupby([bin,lvl_column])[col + '_uc_tcor'].mean().reset_index(drop=True) #3.17
            binned_data[col+'_uc']=(
                binned_data[col+'_uc_ucor']**2 + binned_data[col+'_cor']**2)**0.5 #3.18
        
        # add metadata
        metadata = pd.DataFrame()
        for ggd in ggds:
            ggd_metadata = ggd.metadata[ggd.metadata['Attribute'].str.contains('Product|Measurement', case=False)]
            metadata = pd.concat([metadata, ggd_metadata], ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.Type', 'Value': 'Temporal Gridding'}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.BinSize', 'Value': str(bin_size)}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.TargetColumns', 'Value': ', '.join(target_columns)}, ignore_index=True)
        
        ggd=Ggd(metadata, binned_data)
        return ggd