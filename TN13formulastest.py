import pandas as pd
import numpy as np

df = pd.DataFrame({
    'alt': range(0, 10, 1),
    'var': np.random.randn(10),
    'var_uc_ucor': np.random.randn(10),
    'var_uc_scor': np.random.randn(10),
    'var_uc_tcor': np.random.randn(10)
})

binsize=5
bins=range(0, int(df['alt'].max()), binsize)
bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
var='var'
gridata = pd.DataFrame(bin_alt, columns=['alt'])
bin_avg = df.groupby(pd.cut(df['alt'], bins), observed=True)[var].mean().reset_index(drop=True) # 3.5
uc_ucor = var + '_uc_ucor'
bin_ucor = df.groupby(pd.cut(df['alt'], bins), observed=True)[uc_ucor].apply(
    lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) #3.6
bin_uvar = df.groupby(pd.cut(df['alt'], bins), observed=True)[var].apply(
    lambda x: ((x.var())/len(x))**0.5 ).reset_index(drop=True) #3.7
bin_uc = (bin_ucor**2+bin_uvar**2)**0.5 #3.8
uc_scor = var+'_uc_scor'
bin_sc = df.groupby(pd.cut(df['alt'], bins), observed=True)[uc_scor].mean().reset_index(drop=True) #3.9
uc_tcor = var+'_uc_tcor'
bin_tc = df.groupby(pd.cut(df['alt'], bins), observed=True)[uc_tcor].mean().reset_index(drop=True) #3.8
bin_u = (bin_uc**2+bin_sc**2+bin_tc**2)**0.5
gridata[var] = bin_avg
var_u = var+'_u'
gridata[var_u] = bin_u
var_uc = var+'_uc'

print(df)
print(bins)
print(gridata)
