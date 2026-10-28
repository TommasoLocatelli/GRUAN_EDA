import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

mle_summary=gp.read_pkl(r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl")
mask = (
    ~mle_summary["converged"]
    & mle_summary["error"].fillna("").str.contains(
        "measurement_sigma2 must contain finite, nonneg",
        case=False,
        regex=False,
    )
)

bad_pids = list(set(mle_summary.loc[
    mask,
    "pid"
].tolist()))

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

paths = [hko_path,lau_path, lin_path]
variables_labels=['alt', 'theta_v', 'rh', 'wzon', 'wmeri']
uncertainties_labels=[var+'_uc' for var in variables_labels]
labels=variables_labels+uncertainties_labels
models_classes=[MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel]
optimizers=['powell', 'lbfgs']

for path in paths:
    dataset = gp.read_pkl(path)
    subdataset = {pid: dataset[pid] for pid in bad_pids if pid in dataset}
    print(len(subdataset))

    mle = {}

    for pid, gdp in tqdm(subdataset.items()):

        data = gdp.data[labels]

        # Check for NaN
        if data.isna().any().any():
            bad_cols = data.columns[data.isna().any()].tolist()
            print(
                f"PID={pid}: NaN values found in columns {bad_cols}"
            )

        # Check for +/- inf
        if np.isinf(data.to_numpy()).any():
            inf_cols = data.columns[np.isinf(data).any()].tolist()
            print(
                f"PID={pid}: Inf values found in columns {inf_cols}"
            )

        # Check for negative values in uncertainties
        neg_mask = (data[uncertainties_labels] < 0)
        if neg_mask.any().any():
            bad_cols = neg_mask.columns[neg_mask.any()].tolist()
            print(
                f"PID={pid}: Negative uncertainty values found in columns {bad_cols}"
            )