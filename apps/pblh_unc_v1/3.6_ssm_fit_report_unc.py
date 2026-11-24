import gruanpy as gp
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT, DeterministicLevelLLT, UnivariateLLT_AR1
from ssm_fit_utils import report_smoothing_goodness
from tqdm import tqdm

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_unc.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_unc.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_unc.pkl"

dataset = gp.read_pkl(hko_path)

opt='powell'
vars=['wzon', 'wmeri', 'alt', 'rh']#, 'theta_v']

vars_ucor = {'alt': 'alt_gph_uc_ucor',
            'wzon': 'wzon_uc_ucor',
            'rh': 'rh_uc_ucor',
            'wmeri': 'wmeri_uc_ucor'}

models = [UnivariateLLL, DeterministicLevelLLT]#, UnivariateLLT

for pid, gdp in list(dataset.items())[:1]:
    data=gdp.data
    for var in tqdm(vars):
        for model in models:
            prior_meas_unc_flag = True
            gruan_unc_coef = 1
            prior_meas_unc = 'gruan'
            if gruan_unc_coef !=1:
                prior_meas_unc+=str(gruan_unc_coef)
            try:
                uncs=data[vars_ucor[var]]
                uncs = uncs.fillna(uncs.median())
                measurement_sigma2=(uncs*gruan_unc_coef*0.5)**2  
                ssm=model(data[var], measurement_sigma2)
                model_name = ssm.__class__.__name__
                results=ssm.fit(maxiter=200, method=opt, disp=True)
                print(
                    f"Creating report? converged={results.mle_retvals['converged']}"
                )
                if results.mle_retvals['converged']:
                    print("REPORT CREATED")
                    report_smoothing_goodness(
                        results=results, 
                        observations=data[var],
                        ssm_class=model, 
                        measurement_unc=uncs, 
                        path=rf'apps\pblh_unc_v1\reports\3.6_ssm_fit_unc\{var}\3.6_ssm_fit_{pid}_{var}_{model_name}_{vars_ucor[var]}.pdf',
                        plot=False,
                        zoom=(60,120),
                        var=var)
            except Exception as e:
                print(e)
            try:
                uncs=data[var+'_uc']
                uncs = uncs.fillna(uncs.median())
                measurement_sigma2=(uncs*gruan_unc_coef*0.5)**2  
                ssm=model(data[var], measurement_sigma2)
                model_name = ssm.__class__.__name__
                results=ssm.fit(maxiter=200, method=opt, disp=True)
                print(
                    f"Creating report? converged={results.mle_retvals['converged']}"
                )
                if results.mle_retvals['converged']:
                    print("REPORT CREATED")
                    report_smoothing_goodness(
                        results=results, 
                        observations=data[var],
                        ssm_class=model, 
                        measurement_unc=uncs, 
                        path=rf'apps\pblh_unc_v1\reports\3.6_ssm_fit_unc\{var}\3.6_ssm_fit_{pid}_{var}_{model_name}_{var}_uc.pdf',
                        plot=False,
                        zoom=(60,120),
                        var=var)
            except Exception as e:
                print(e)
            try:
                prior_meas_unc_flag=False
                model_name = ssm.__class__.__name__
                ssm=model(data[var])
                prior_meas_unc = 'mle'
                results=ssm.fit(maxiter=500, method=opt, disp=True)
                print(
                    f"Creating report? converged={results.mle_retvals['converged']}"
                )
                if results.mle_retvals['converged']:
                    print("REPORT CREATED")
                    report_smoothing_goodness(
                        results=results, 
                        observations=data[var], 
                        measurement_unc=data[var+'_uc'], 
                        path=rf'apps\pblh_unc_v1\reports\3.6_ssm_fit_unc\{var}\3.6_ssm_fit_{pid}_{var}_{model_name}_{prior_meas_unc}.pdf',
                        plot=False,
                        zoom=(60,120),
                        var=var)
            except Exception as e:
                    print(e)
