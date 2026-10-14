import gruanpy as gp
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT, DeterministicLevelLLT, UnivariateLLT_AR1
from ssm_fit_utils import report_smoothing_goodness

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)

for pid, gdp in dataset.items():
    data=gdp.data
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    optimizers=['newton', 'lbfgs', 'powell']
    vars=['alt', 'theta_v', 'rh', 'wzon', 'wmeri']
    models = [UnivariateLLL, UnivariateLLT, DeterministicLevelLLT, UnivariateLLT_AR1]
    unc=[var+'_uc' for var in vars]

    opt='powell'
    var='alt'
    measurement_sigma2=(data[var+'_uc']*0.5)**2
    measurement_sigma2*=1
    model=UnivariateLLT
    prior_meas_unc_flag = False
    if prior_meas_unc_flag:
        ssm=model(data[var], measurement_sigma2)
        prior_meas_unc = 'gruan'
    else:
        ssm=model(data[var])
        prior_meas_unc = 'mle'
    model_name = ssm.__class__.__name__
    results=ssm.fit(maxiter=200, method=opt, disp=True)

    #from statsmodels.tsa.arima.model import ARIMA
    #model = ARIMA(data[var], trend="t", order=(1, 1, 1))
    #results = model.fit()

    #print(results.summary())

    report_smoothing_goodness(
        results=results, 
        observations=data[var], 
        meas_unc=data[var+'_uc'], 
        path=rf'apps\pblh_unc_v1\reports\3.1_ssm_fit\{var}\3.1_ssm_fit_{pid}_{var}_{model_name}_{prior_meas_unc}.pdf',
        plot=False,
        zoom=(80,100))

    break