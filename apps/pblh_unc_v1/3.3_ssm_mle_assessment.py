import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def get_value(table, row_label):
    row_label = row_label.strip()

    for row in table:
        if len(row) >= 2 and row[0].strip() == row_label:
            return row[1]

        if len(row) >= 4 and row[2].strip() == row_label:
            return row[3]

    return None

if True: # Read mle pkls and produce a summary pickle
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

            for var, var_dict in pid_dict.items():

                for model_name, model_dict in var_dict.items():

                    for optimizer, opt_dict in model_dict.items():

                        for fit_type in ["gruan", "mle"]:

                            fit = opt_dict.get(fit_type)

                            if fit is None:
                                continue

                            if "error" in fit:
                                rows.append({
                                    "site": site,
                                    "pid": pid,
                                    "variable": var,
                                    "model": model_name,
                                    "optimizer": optimizer,
                                    "fit_type": fit_type,
                                    "converged": False,
                                    "error": fit["error"],
                                })
                                continue

                            else:
                                converged = fit.get("mle_retvals", {}).get("converged", False)
                                warnflag = fit.get("mle_retvals", {}).get("warnflag", 0)
                                iterations = fit.get("mle_retvals", {}).get("iterations", 1)
                                                                                        
                                summary = fit.get("summary")

                                if summary is None:
                                    continue

                                res_metrics = fit.get("res_metrics", {})

                                t1 = summary.tables[0].data
                                t2 = summary.tables[1].data
                                t3 = summary.tables[2].data

                                row = {
                                    "site": site,
                                    "pid": pid,
                                    "variable": var, 
                                    "model": model_name,
                                    "optimizer": optimizer,
                                    "fit_type": fit_type,
                                    "converged": converged,
                                    "warnflag": warnflag,
                                    "iterations": iterations,

                                    "nobs": get_value(t1, "No. Observations:"),
                                    "loglik": get_value(t1, "Log Likelihood"),
                                    "aic": get_value(t1, "AIC"),
                                    "bic": get_value(t1, "BIC"),

                                    "prob_q": get_value(t3, "Prob(Q):"),
                                    "prob_h": get_value(t3, "Prob(H) (two-sided):"),
                                    "prob_jb": get_value(t3, "Prob(JB):"),

                                    "rmse": res_metrics.get("rmse"),
                                    "mae": res_metrics.get("mae"),
                                    "bias": res_metrics.get("bias"),
                                    "ss_res": res_metrics.get("ss_res"),
                                    "ss_tot": res_metrics.get("ss_tot"),
                                    "r2": res_metrics.get("r2"),
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

        "rmse",
        "mae",
        "bias",
        "ss_res",
        "ss_tot",
        "r2",
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

elif False: # Wich is the best optimizer? Powell
    # Link at the warnflag documentation
    # https://www.statsmodels.org/devel/dev/generated/statsmodels.base.model.LikelihoodModelResults.html#:~:text=the%20standard%20errors.-,Notes,-The%20covariance%20of
    mle_summary=gp.read_pkl(r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl")

    for var, df_var in mle_summary.groupby("variable"):

        print(f"\n{'='*80}")
        print(f"Variable: {var}")
        print(f"{'='*80}")

        conv_summary = (
            df_var
            .groupby(["optimizer",])
            .agg(
                n=("converged", "size"),
                n_converged=("converged", "sum"),
                prop_converged=("converged", "mean"),
            )
            .reset_index()
            .sort_values("prop_converged", ascending=False)
        )

        print(conv_summary)

        conv_summary = (
            df_var
            .groupby(["model", "fit_type", "optimizer"])
            .agg(
                n=("converged", "size"),
                n_converged=("converged", "sum"),
                prop_converged=("converged", "mean"),
            )
            .reset_index()
            .sort_values("prop_converged", ascending=False)
        )

        print(conv_summary)

        failed = mle_summary.loc[~mle_summary["converged"]]

        print(len(failed))
        print(
            failed[
                [
                    "site",
                    "pid",
                    "variable",
                    "model",
                    "optimizer",
                    "fit_type",
                    "warnflag",
                    "iterations",
                    "error",
                ]
            ]
        )

elif False: # Which is the best model? REMEMBER TO GET RID OF USELESS OPTIMIZER

    mle_summary = gp.read_pkl(
        r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl"
    )

    print(mle_summary.columns)

    mle_summary = mle_summary[
        (mle_summary["optimizer"] == "powell")
        & (mle_summary["converged"])
    ]

    print(mle_summary.shape)

    for var, df_var in mle_summary.groupby("variable"):

        print("\n" + "=" * 80)
        print(f"Variable: {var}")
        print("=" * 80)

        # AIC winner per profile and fit_type
        best_aic = df_var.loc[
            df_var.groupby(
                ["pid", "fit_type"]
            )["aic"].idxmin()
        ]

        aic_counts = (
            best_aic
            .groupby(["fit_type", "model"])
            .size()
            .unstack(fill_value=0)
        )

        print("\nAIC winners")
        print(aic_counts)

        # BIC winner per profile and fit_type
        best_bic = df_var.loc[
            df_var.groupby(
                ["pid", "fit_type"]
            )["bic"].idxmin()
        ]

        bic_counts = (
            best_bic
            .groupby(["fit_type", "model"])
            .size()
            .unstack(fill_value=0)
        )

        print("\nBIC winners")
        print(bic_counts)

elif False: # it is better with or without gruan unc?
    mle_summary = gp.read_pkl(
        r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl"
    )

    mle_summary = mle_summary[
        (mle_summary["optimizer"] == "powell")
        & (mle_summary["converged"])
    ]

    for var, df_var in mle_summary.groupby("variable"):

        print("\n" + "=" * 80)
        print(f"Variable: {var}")
        print("=" * 80)

        # AIC winner per profile and fit_type
        best_aic = df_var.loc[
            df_var.groupby(
                ["pid", "model"]
            )["aic"].idxmin()
        ]

        aic_counts = (
            best_aic
            .groupby(["fit_type", "model"])
            .size()
            .unstack(fill_value=0)
        )

        print("\nAIC winners")
        print(aic_counts)

        # BIC winner per profile and fit_type
        best_bic = df_var.loc[
            df_var.groupby(
                ["pid", "model"]
            )["bic"].idxmin()
        ]

        bic_counts = (
            best_bic
            .groupby(["fit_type", "model"])
            .size()
            .unstack(fill_value=0)
        )

        print("\nBIC winners")
        print(bic_counts)

elif False: # compare GRUAN and MLE meas unc
    # ============================================================================
    # MLE measurement uncertainty
    # ============================================================================

    mle_summary = gp.read_pkl(
        r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl"
    )

    mle_summary = mle_summary[
        (mle_summary["optimizer"] == "powell")
        & (mle_summary["converged"])
        & (mle_summary["fit_type"] == "mle")
    ]

    print(mle_summary.columns)

    mle_summary["sigma.measurement"] = np.sqrt(
        mle_summary["sigma2.measurement"]
    )

    mle_stats = (
        mle_summary
        .groupby("variable")["sigma.measurement"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            q05=lambda x: x.quantile(0.05),
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            q95=lambda x: x.quantile(0.95),
            n="count",
        )
    )

    print("\nMLE sigma statistics")
    print(mle_stats)


    # ============================================================================
    # GRUAN uncertainty
    # ============================================================================

    hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
    lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
    lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

    paths = [lin_path, hko_path, lau_path]

    variables = ['alt', 'theta_v', 'rh', 'wzon', 'wmeri']
    uncertainties = [f"{v}_uc" for v in variables]

    # store all values
    gruan_values = {v: [] for v in variables}

    for path in paths:

        dataset = gp.read_pkl(path)

        for pid, gdp in dataset.items():

            sigma = gdp.data[uncertainties] * 0.5

            for var in variables:
                gruan_values[var].extend(
                    sigma[f"{var}_uc"].dropna().values
                )

    gruan_stats = pd.DataFrame({
        var: {
            "mean": np.mean(vals),
            "median": np.median(vals),
            "std": np.std(vals),
            "q05": np.quantile(vals, 0.05),
            "q25": np.quantile(vals, 0.25),
            "q75": np.quantile(vals, 0.75),
            "q95": np.quantile(vals, 0.95),
            "n": len(vals),
        }
        for var, vals in gruan_values.items()
    }).T

    print("\nGRUAN sigma statistics")
    print(gruan_stats)


    # ============================================================================
    # Ratio comparison
    # ============================================================================

    comparison = pd.DataFrame(index=variables)

    comparison["MLE_mean"] = mle_stats["mean"]
    comparison["MLE_median"] = mle_stats["median"]

    comparison["GRUAN_mean"] = gruan_stats["mean"]
    comparison["GRUAN_median"] = gruan_stats["median"]

    comparison["ratio_mean"] = (
        comparison["GRUAN_mean"]
        / comparison["MLE_mean"]
    )

    comparison["ratio_median"] = (
        comparison["GRUAN_median"]
        / comparison["MLE_median"]
    )

    print("\nComparison")
    print(comparison)


    # ============================================================================
    # Boxplot comparison
    # ============================================================================

    fig, axes = plt.subplots(
        nrows=len(variables),
        figsize=(8, 3 * len(variables))
    )

    for ax, var in zip(axes, variables):

        mle_vals = mle_summary.loc[
            mle_summary["variable"] == var,
            "sigma.measurement"
        ].values

        gruan_vals = np.asarray(gruan_values[var])

        ax.boxplot(
            [mle_vals, gruan_vals],
            labels=["MLE", "GRUAN"],
            showfliers=False,
        )

        ax.set_title(var)
        ax.set_ylabel("sigma")

    plt.tight_layout()
    plt.show()

elif True: # unc impac on residuals
    
    mle_summary = gp.read_pkl(
        r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl"
    )

    mle_summary = mle_summary[
        (mle_summary["optimizer"] == "powell")
        & (mle_summary["converged"])
        & (mle_summary["fit_type"] == "mle")
    ]

    print(mle_summary.columns)
    print(mle_summary[["rmse","r2"]].head())

elif False: # SSM assumption are meeted? To be UPDATED
    mle_summary=gp.read_pkl(r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl")
    print(mle_summary.head())

    mle_summary = mle_summary[mle_summary["optimizer"] == "powell"]
    mle_summary = mle_summary[
        mle_summary["model"].isin(
            ["MultivariateLLL", "MultivariateLLT", "MultivariateLLTDetLevel"]
    )]

    print(mle_summary.shape)
    p_value_threshold = 0
    assumption_summary = (
        mle_summary
        .groupby(["model", "fit_type"])
        .agg(
            n=("pid", "size"),
            prob_h_pass=("prob_h", lambda x: (x > p_value_threshold).sum()),
            prob_jb_pass=("prob_jb", lambda x: (x > p_value_threshold).sum()),
            prob_q_pass=("prob_q", lambda x: (x > p_value_threshold).sum()),
        )
    )

    print(assumption_summary)

elif False: # Look at estimated parameters. To be UPDATED
    mle_summary = gp.read_pkl(r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl")

    mle_summary = mle_summary[mle_summary["optimizer"] == "powell"]
    mle_summary = mle_summary[
        mle_summary["model"].isin(
            ["MultivariateLLT", "MultivariateLLTDetLevel"]
        )
    ]

    # Select columns containing "sigma"
    sigma_cols = [c for c in mle_summary.columns if "sigma" in c.lower()]

    # Average sigma parameters by model and fit_type
    sigma_means = (
        mle_summary
        .groupby(["model", "fit_type"])[sigma_cols]
        .transform(lambda x: x**0.5)
        .mean()
    )

    for (model, fit_type), group in mle_summary.groupby(["model", "fit_type"]):
        print(f"\nModel={model}, fit_type={fit_type}")
        print(group[sigma_cols].mean().to_string().replace("sigma2.", ""))