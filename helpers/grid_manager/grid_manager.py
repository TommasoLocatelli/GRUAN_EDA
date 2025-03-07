import pandas as pd
from helpers.format_manager.gdp import Gdp
from tqdm import tqdm

class GridManager:

    def __init__(self):
        pass

    def rs41_spatial_gridding(self, data, binsize, vars):
        """
        "Spatial gridding (or binning ) always refers to a particular
        measured profile (sounding). Gridding is usually done by applying averaging windows running
        along the profiles or time series, where the use of different kernel shapes enables appropriate
        types of weighting" (TN13, p. 3.4)
        """
        assert isinstance(data, pd.DataFrame), 'data should be a pandas DataFrame'
        assert 'alt' in data.columns, 'data should have a column named alt'
        assert isinstance(binsize, int), 'binsize should be an integer'
        assert isinstance(vars, list) and all(var in data.columns for var in vars), 'vars should be a list of column names in data'

        bins=range(0, int(data['alt'].max()//binsize+2)*binsize, binsize) # make sure to exceed the max value by more than binsize
        bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
        gridata = pd.DataFrame(bin_alt, columns=['alt'])
        for var in vars:
            gridata[var] = data.groupby(pd.cut(data['alt'], bins), observed=True)[var].mean().reset_index(drop=True) # 3.5
            uc_ucor = var + '_uc_ucor' # original uncorrelated uncertainty
            u_ucor = var + '_u_ucor'
            gridata[u_ucor] = data.groupby(pd.cut(data['alt'], bins), observed=True)[uc_ucor].apply(
                lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) #3.6
            uc_var = var+'_var'
            gridata[uc_var] = data.groupby(pd.cut(data['alt'], bins), observed=True)[var].apply(
                lambda x: ((x-x.mean())**2).sum()/(len(x)*(len(x)-1))**0.5 ).reset_index(drop=True) #3.7
            u_uc = var+'_u_uc'
            gridata[u_uc] = (gridata[u_ucor]**2+gridata[uc_var]**2)**0.5 #3.8
            uc_scor = var+'_uc_scor' # original spattialy correlated uncertainty
            u_sc = var+'_u_sc'
            gridata[u_sc]  = data.groupby(pd.cut(data['alt'], bins), observed=True)[uc_scor].mean().reset_index(drop=True) #3.9
            uc_tcor = var+'_uc_tcor' # original temporally correlated uncertainty
            u_tc = var+'_u_tc'
            gridata[u_tc] = data.groupby(pd.cut(data['alt'], bins), observed=True)[uc_tcor].mean().reset_index(drop=True) #3.10
            u = var+'_u'
            gridata[u] = (gridata[u_uc]**2+gridata[u_sc]**2+gridata[u_tc]**2)**0.5 #3.11
        return gridata
    
    def rs41_temporal_gridding(self, gdps, time_binsize, spatial_binsize, vars):
        """
        "If a temporal gridding is to be carried out (e.g. a monthly mean), it should follow the spatial
        gridding as second step. That is, a temporal gridding should be applied to vertical profiles which
        were first converted to the same altitude grid." (TN13, p. 3.4.1.2)
        """
        assert isinstance(gdps, list) and all(isinstance(gdp, Gdp) for gdp in gdps), 'gdps should be a list of Gdp instances'
        assert isinstance(time_binsize, int), 'time_binsize should be an integer' # in days
        assert isinstance(spatial_binsize, int), 'spatial_binsize should be an integer' # in meters 
        assert isinstance(vars, list) and all(var in gdps[0].data.columns for var in vars), 'vars should be a list of column names in data'

        # Spatial Gridding
        for gdp in gdps:
            gridata=self.rs41_spatial_gridding(gdp.data, spatial_binsize, vars)
            gdp.set_gridata(gridata)

        # Temporal Gridding
        times=[gdp.time for gdp in gdps]
        print(times)
        min_time = min(times).replace(hour=0, minute=0, second=0, microsecond=0)
        #print(times)
        days_range=range(0, int((max(times)-min_time).days+time_binsize+1), time_binsize)
        bins=[min_time+pd.Timedelta(days=i) for i in days_range]
        #print(bins)
        gdps = sorted([gdp for gdp in gdps], key=lambda x: x.time)
        temporal_grid = pd.DataFrame()
        i=0 # index to keep track of the grids
        for bin in tqdm(bins, desc="Temporal Gridding Progress"):
            bin_grids = []
            for gdp in gdps[i:]:
                if gdp.time < bin:
                    bin_grids.append(gdp.gridata)
                else:
                    i = gdps.index(gdp)
                    break
                if bin_grids:
                    bins_data = pd.concat(bin_grids)
                    partial_temporal_grid = pd.DataFrame()
                    partial_temporal_grid['alt'] = bins_data['alt'].unique()
                    partial_temporal_grid['time'] = bin
                    for var in vars:
                        partial_temporal_grid[var] = bins_data.groupby('alt')[var].mean().reset_index(drop=True) #3.12
                        u_uc = var + '_u_uc'
                        partial_temporal_grid[u_uc] = bins_data.groupby('alt')[u_uc].apply(
                        lambda x: (x**2).sum()**0.5 / len(x)).reset_index(drop=True) #3.13
                        #uc_var = var + '_var'
                        #partial_temporal_grid[uc_var] = bins_data.groupby('alt')
                        #u_sc = var + '_u_sc'
                        #partial_temporal_grid[u_sc] = bins_data.groupby('alt')[var + '_u_sc'].mean().reset_index(drop=True)
                        #u_tc = var + '_u_tc'
                        #partial_temporal_grid[u_tc] = bins_data.groupby('alt')[var + '_u_tc'].mean().reset_index(drop=True)
                        #u = var + '_u'
                        #partial_temporal_grid[u] = (partial_temporal_grid[u_uc]**2 + partial_temporal_grid[u_sc]**2 + partial_temporal_grid[u_tc]**2)**0.5
                    partial_temporal_grid['time'] = bin
                    temporal_grid = pd.concat([temporal_grid, partial_temporal_grid], ignore_index=True)
        return temporal_grid