import pandas as pd
from gruanpy.data_models.ggd import Ggd

class GriddingManager:
    def __init__(self):
        pass

    def spatial_gridding(self, gdp, bin_column, target_columns, bin_size): #TN-13
        data=gdp.data
        bin = bin_column+'_bin'
        data[bin] = (data[bin_column] // bin_size) * bin_size + bin_size / 2
        binned_data = data.groupby(bin)[target_columns].mean().reset_index() # 3.5
        for col in target_columns:
            binned_data[col + '_uc_ucor'] = data.groupby(bin)[col + '_uc_ucor'].apply(
                        lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) #3.6
            binned_data[col + '_var'] = data.groupby(bin)[col].apply(
                        lambda x: ((x-x.mean())**2).sum()/(len(x)*(len(x)-1))**0.5 ).reset_index(drop=True) #3.7
            binned_data[col + '_uc'] = (binned_data[col+'_uc_ucor']**2 + binned_data[col + '_var']**2)**0.5 #3.8
            if col+'_uc_scor' in data.columns:
                binned_data[col + '_uc_scor'] = data.groupby(bin)[col + '_uc_scor'].mean().reset_index(drop=True) #3.9
            else:
                binned_data[col + '_uc_scor']=0
            if col+'_uc_tcor' in data.columns:
                binned_data[col + '_uc_tcor'] = data.groupby(bin)[col + '_uc_tcor'].mean().reset_index(drop=True) #3.10
            else:
                binned_data[col + '_uc_tcor']=0
            binned_data[col+'_u']=(binned_data[col+'_uc']**2 + binned_data[col+'_uc_scor']**2 + binned_data[col+'_uc_tcor']**2)**0.5 #3.11

        metadata = gdp.global_attrs[gdp.global_attrs['Attribute'].str.contains('Product|Measurement', case=False)]
        
        metadata = metadata._append({'Attribute': 'g.Gridding.Type', 'Value': 'Spatial Gridding'}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.BinColumn', 'Value': bin_column}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.BinSize', 'Value': str(bin_size)}, ignore_index=True)
        metadata = metadata._append({'Attribute': 'g.Gridding.TargetColumns', 'Value': ', '.join(target_columns)}, ignore_index=True)

        ggd=Ggd(metadata, binned_data)
        return ggd