import gruanpy as gp
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT, DeterministicLevelLLT, UnivariateLLT_AR1
from ssm_fit_utils import report_smoothing_goodness

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)

for pid, gdp in dataset.items():
    data=gdp.data
    opts=['powell']#, 'lbfgs']
    vars=['theta_v']#'alt', ,  'wzon']#,'rh', 'wmeri']
    models = [UnivariateLLT]#, UnivariateLLT]#, DeterministicLevelLLT]
    #, UnivariateLLT_AR1]
    #from statsmodels.tsa.arima.model import ARIMA
    #model = ARIMA(data[var], trend="t", order=(1, 1, 1))

    for var in vars:
        for opt in opts:
            for model in models:
                prior_meas_unc_flag = True
                for gruan_unc_coef in [1,]:# 0.1, 10]:
                    prior_meas_unc = 'gruan'
                    if gruan_unc_coef !=1:
                        prior_meas_unc+=str(gruan_unc_coef)
                    measurement_sigma2=(data[var+'_uc']*gruan_unc_coef*0.5)**2  
                    ssm=model(data[var], measurement_sigma2)
                    results=ssm.fit(maxiter=200, method=opt, disp=True)
                    model_name = ssm.__class__.__name__
                    report_smoothing_goodness(
                        results=results, 
                        observations=data[var],
                        ssm_class=model, 
                        measurement_unc=data[var+'_uc']*gruan_unc_coef, 
                        path=rf'apps\pblh_unc_v1\reports\3.1_ssm_fit\{var}\3.1_ssm_fit_{pid}_{var}_{model_name}_{opt}_{prior_meas_unc}.pdf',
                        plot=False,
                        zoom=(80,100))

                prior_meas_unc_flag=False
                ssm=model(data[var])
                prior_meas_unc = 'mle'
                results=ssm.fit(maxiter=500, method=opt, disp=True)
                report_smoothing_goodness(
                    results=results, 
                    observations=data[var], 
                    measurement_unc=data[var+'_uc'], 
                    path=rf'apps\pblh_unc_v1\reports\3.1_ssm_fit\{var}\3.1_ssm_fit_{pid}_{var}_{model_name}_{opt}_{prior_meas_unc}.pdf',
                    plot=False,
                    zoom=(80,100))
        break
    break