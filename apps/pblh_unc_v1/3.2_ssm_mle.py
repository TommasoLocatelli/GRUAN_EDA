import gruanpy as gp
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT, DeterministicLevelLLT
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

paths = [hko_path, lau_path, lin_path]
for path in paths:
    dataset = gp.read_pkl(path)

    variables_labels=['alt', 'theta_v', 'rh', 'wzon', 'wmeri']
    uncertainties_labels=[var+'_uc' for var in variables_labels]
    labels=variables_labels+uncertainties_labels
    models_classes=[ UnivariateLLL, UnivariateLLT, DeterministicLevelLLT]
    optimizers=['powell', 'lbfgs', 'nm']

    mle = {}

    for pid, gdp in tqdm(list(dataset.items())[:2]):

        data = gdp.data[labels]

        mle[pid] = {}

        for var in variables_labels:

            unc_label=var+'_uc'

            mle[pid][var] = {}

            for model_cls in models_classes:

                model_name = model_cls.__name__
                mle[pid][var][model_name] = {}

                for opt in optimizers:

                    mle[pid][var][model_name][opt] = {}

                    # with uncertainties
                    try:
                        uncs=data[unc_label]
                        uncs = uncs.fillna(uncs.median())
                        ssm = model_cls(
                            data[var],
                            uncs
                        )

                        results = ssm.fit(
                            maxiter=100,
                            method=opt,
                            disp=False
                        )

                        mle[pid][var][model_name][opt]["gruan"] = {
                            "mle_retvals": results.mle_retvals,
                            "mle_settings": results.mle_settings,
                            "summary": results.summary()
                        }

                    except Exception as e:
                        mle[pid][var][model_name][opt]["gruan"] = {
                            "error": str(e)
                        }

                    # without uncertainties
                    try:
                        ssm = model_cls(
                            data[var]
                        )

                        results = ssm.fit(
                            maxiter=100,
                            method=opt,
                            disp=False
                        )

                        mle[pid][var][model_name][opt]["mle"] = {
                            "mle_retvals": results.mle_retvals,
                            "mle_settings": results.mle_settings,
                            "summary": results.summary()
                        }

                    except Exception as e:
                        mle[pid][var][model_name][opt]["mle"] = {
                            "error": str(e)
                        }

    output_path = f"apps\\pblh_unc_v1\\pkls\\mle_{path[26:-17]}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(mle, f, protocol=pickle.HIGHEST_PROTOCOL)
