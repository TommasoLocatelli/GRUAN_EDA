import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

hko_path = r"apps\pblh_unc_v1\pkls\mle_2024__HKO-RS-01_2024.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

path = hko_path
mle = gp.read_pkl(hko_path)

print(type(mle), len(mle))
raise

for pid, pid_dict in mle.items():
    for model_name, model_dict in pid_dict.items():
        for opt, opt_dict in model_dict.items():
            for fit_type in ["gruan", "mle"]:

                fit = opt_dict.get(fit_type)

                if fit is None:
                    continue

                if "error" in fit:
                    print(
                        pid,
                        model_name,
                        opt,
                        fit_type,
                        "ERROR:",
                        fit["error"]
                    )
                    continue

                converged = fit["mle_retvals"].get(
                    "converged",
                    False
                )

                print(
                    pid,
                    model_name,
                    opt,
                    fit_type,
                    converged
                )