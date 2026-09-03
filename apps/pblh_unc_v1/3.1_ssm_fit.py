import pickle
import gruanpy as gp
import tqdm
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

gdps = hko
gdps = {**hko, **lau, **lin}
print(len(gdps))

variables = ['alt', 'theta', 'rh', 'wmeri', 'wzon']

results=dict()
for pid, gdp in tqdm.tqdm(gdps.items()):
    data=gdp.data
    pid_results=dict()
    for var in variables:
        endog = data[var].values.astype(float)
        endog_uc = data[var+'_uc'].values.astype(float)
        measurement_sigma2 = (endog_uc * 0.5)**2

        lll_mle = UnivariateLLL(endog=endog)
        lll_mle_results = lll_mle.fit(method='powell',
                maxiter=200,
                full_output=1,
                disp=5)
        
        lll_gruan=UnivariateLLL(endog=endog, measurement_sigma2=measurement_sigma2)
        lll_gruan_results = lll_mle.fit(method='powell',
                maxiter=200,
                full_output=1,
                disp=5)
        
        llt_mle=UnivariateLLT(endog=endog)
        llt_mle_results = lll_mle.fit(method='powell',
                maxiter=200,
                full_output=1,
                disp=5)
        
        llt_gruan=UnivariateLLT(endog=endog, measurement_sigma2=measurement_sigma2)
        llt_gruan_results = lll_mle.fit(method='powell',
                maxiter=200,
                full_output=1,
                disp=5)

        var_result={var:{
            'lll_mle': (lll_mle, lll_mle_results),
            'lll_gruan': (lll_gruan, lll_gruan_results),
            'llt_mle': (llt_mle, llt_mle_results),
            'llt_gruan': (llt_gruan, llt_gruan_results)
        }}
        pid_results[var]=var_result

    pid_results[var]=var_result
    #results = model.fit_constrained({'sigma2.measurement': 0})
    break