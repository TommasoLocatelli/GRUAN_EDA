import pickle
import gruanpy as gp
import tqdm

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
#lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
#lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

hko = gp.read_pkl(hko_path)
#lau = gp.read_pkl(lau_path)
#lin = gp.read_pkl(lin_path)

gdps = hko
#gdps = {**hko, **lau, **lin}
print(len(gdps))

for pid, gdp in tqdm.tqdm(gdps.items()):
    from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT
    data=gdp.data
    z = data['theta'].values.astype(float)
    z_unc  = data['theta_uc'].values
    z_var  = (z_unc * 0.5)**2
    model = UnivariateLLT(endog=z, measurement_sigma2=z_var)
    #results = model.fit(method='powell',
    #            maxiter=200,
    #            full_output=1,
    #            disp=5)
    results = model.fit_constrained({'sigma2.level': 0})
    print(results.summary())
    break