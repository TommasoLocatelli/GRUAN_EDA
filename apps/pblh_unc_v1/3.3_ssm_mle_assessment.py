import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path

def get_value(table, row_label):
    row_label = row_label.strip()

    for row in table:
        if len(row) >= 2 and row[0].strip() == row_label:
            return row[1]

        if len(row) >= 4 and row[2].strip() == row_label:
            return row[3]

    return None


if False: # read mle pkls and produce a summary pickle
    hko_path = r"apps\pblh_unc_v1\pkls\mle_2024__HKO-RS-01_2024.pkl"
    lau_path = r"apps\pblh_unc_v1\pkls\mle_2024__LAU-RS-02_2024.pkl"
    lin_path = r"apps\pblh_unc_v1\pkls\mle_2024__LIN-RS-01_2024.pkl"

    paths = [
        hko_path,
        lau_path,
        lin_path,
    ]

    rows = []

    for path in paths:

        # Extract site from filename
        site = Path(path).stem.split("__")[1].split("-")[0]
        # e.g. HKO, LAU, LIN

        mle = gp.read_pkl(path)
        print(f"Loading site: {site}")
        for pid, pid_dict in tqdm(mle.items()):
            for model_name, model_dict in pid_dict.items():
                for optimizer, opt_dict in model_dict.items():

                    for fit_type in ["gruan", "mle"]:

                        fit = opt_dict.get(fit_type)

                        if fit is None:
                            continue

                        if "error" in fit:
                            rows.append({
                                "site": site,
                                "pid": pid,
                                "model": model_name,
                                "optimizer": optimizer,
                                "fit_type": fit_type,
                                "converged": False,
                                "error": fit["error"],
                            })
                            continue

                        else:
                            converged = fit.get("mle_retvals", {}).get("converged", False)

                            summary = fit.get("summary")

                            if summary is None:
                                continue

                            t1 = summary.tables[0].data
                            t2 = summary.tables[1].data
                            t3 = summary.tables[2].data

                            row = {
                                "site": site,
                                "pid": pid,
                                "model": model_name,
                                "optimizer": optimizer,
                                "fit_type": fit_type,
                                "converged": converged,

                                "nobs": get_value(t1, "No. Observations:"),
                                "loglik": get_value(t1, "Log Likelihood"),
                                "aic": get_value(t1, "AIC"),
                                "bic": get_value(t1, "BIC"),

                                "prob_q": get_value(t3, "Prob(Q):"),
                                "prob_h": get_value(t3, "Prob(H) (two-sided):"),
                                "prob_jb": get_value(t3, "Prob(JB):"),
                            }
                            row["error"] = None

                            for param_row in t2[1:]:  # skip header
                                param_name = param_row[0].strip()
                                coef = pd.to_numeric(param_row[1], errors="coerce")

                                row[param_name] = coef

                            rows.append(row)

    df = pd.DataFrame(rows)

    numeric_cols = [
        "nobs",
        "loglik",
        "aic",
        "bic",
        "prob_q",
        "prob_h",
        "prob_jb",
    ]

    coef_cols = [
        c for c in df.columns
        if c.startswith("sigma2.")
    ]

    df[numeric_cols + coef_cols] = (
        df[numeric_cols + coef_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    out_path = r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl"

    df.to_pickle(out_path)

    print(f"Saved dataframe to: {out_path}")

if True:
    mle_summary=gp.read_pkl(r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl")
    print(mle_summary.head())
    print(mle_summary.shape)