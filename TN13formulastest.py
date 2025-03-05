# fare i conti della serva

import pandas as pd
import numpy as np

alt_values = [float(i)+0.5 for i in range(0, 10, 1)]
var_values = np.random.randn(10)
var_uc_ucor_values = np.random.randn(10)
var_uc_scor_values = np.random.randn(10)
var_uc_tcor_values = np.random.randn(10)

df = pd.DataFrame({
    'alt': alt_values,
    'var': var_values,
    'var_uc_ucor': var_uc_ucor_values,
    'var_uc_scor': var_uc_scor_values,
    'var_uc_tcor': var_uc_tcor_values
})
print(df)

binsize=5
bins=range(0, int(df['alt'].max()//binsize+2)*binsize, binsize) # make sure to exceed the max value by more than binsize
bin_alt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)] # take the middle of the bin
var='var'
gridata = pd.DataFrame(bin_alt, columns=['alt'])
#gridata['count'] = df.groupby(pd.cut(df['alt'], bins), observed=True)[var].count().reset_index(drop=True)
#gridata['avg'] = df.groupby(pd.cut(df['alt'], bins), observed=True)['alt'].mean().reset_index(drop=True)
gridata[var] = df.groupby(pd.cut(df['alt'], bins), observed=True)[var].mean().reset_index(drop=True) # 3.5
uc_ucor = var + '_uc_ucor' # original uncorrelated uncertainty
u_ucor = var + '_u_ucor'
gridata[u_ucor] = df.groupby(pd.cut(df['alt'], bins), observed=True)[uc_ucor].apply(
    lambda x: (x**2).sum()**0.5/len(x)).reset_index(drop=True) #3.6
uc_var = var+'_var'
gridata[uc_var] = df.groupby(pd.cut(df['alt'], bins), observed=True)[var].apply(
    lambda x: ((x.var())/len(x))**0.5 ).reset_index(drop=True) #3.7
u_uc = var+'_u_uc'
gridata[u_uc] = (gridata[u_ucor]**2+gridata[uc_var]**2)**0.5 #3.8
uc_scor = var+'_uc_scor' # original spattialy correlated uncertainty
u_sc = var+'_u_sc'
gridata[u_sc]  = df.groupby(pd.cut(df['alt'], bins), observed=True)[uc_scor].mean().reset_index(drop=True) #3.9
uc_tcor = var+'_uc_tcor' # original temporally correlated uncertainty
u_tc = var+'_u_tc'
gridata[u_tc] = df.groupby(pd.cut(df['alt'], bins), observed=True)[uc_tcor].mean().reset_index(drop=True) #3.10
u = var+'_u'
gridata[u] = (gridata[u_uc]**2+gridata[u_sc]**2+gridata[u_tc]**2)**0.5 #3.11

print(gridata)

# test
print('-----------------var-----------------')
print(gridata[var][0], gridata[var][1])
print(var_values[:binsize].mean(), var_values[binsize:].mean())
print(gridata[var][0]==var_values[:binsize].mean(), gridata[var][1]==var_values[binsize:].mean())

print('-----------------var_u_cor-----------------')
print(gridata[u_ucor][0], gridata[u_ucor][1])
test=[(sum([u**2 for u in var_uc_ucor_values[:binsize]])**0.5)/len(var_uc_ucor_values[:binsize]), (sum([u**2 for u in var_uc_ucor_values[binsize:]])**0.5)/len(var_uc_ucor_values[binsize:])]
print(test[0], test[1])
print(gridata[u_ucor][0]==test[0], gridata[u_ucor][1]==test[1])