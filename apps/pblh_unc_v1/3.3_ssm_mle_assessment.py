import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

TEXT_SIZE=22
plt.rcParams.update({
    "axes.titlesize": TEXT_SIZE,       # Subplot titles
    "axes.labelsize": TEXT_SIZE,       # Axis labels
    "xtick.labelsize": TEXT_SIZE,      # Tick labels
    "ytick.labelsize": TEXT_SIZE,
    "legend.fontsize": TEXT_SIZE,      # Legend text
    "figure.titlesize": TEXT_SIZE,     # Suptitle
})

def get_value(table, row_label):
    row_label = row_label.strip()

    for row in table:
        if len(row) >= 2 and row[0].strip() == row_label:
            return row[1]

        if len(row) >= 4 and row[2].strip() == row_label:
            return row[3]

    return None

if False: # Read mle pkls and produce a summary pickle
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

elif False: # which optimizer found the most likelihood parameters
    mle_summary=gp.read_pkl(r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl")
    print(mle_summary.columns)

    converged = mle_summary.loc[mle_summary["converged"]]

    idx = (
        converged
        .groupby(["pid", "variable", "model", "fit_type"])["loglik"]
        .idxmax()
    )

    best_fits = converged.loc[idx]

    optimizer_order = ["powell", "lbfgs", "nm"]

    table = pd.crosstab(
        best_fits["variable"],
        best_fits["optimizer"],
        normalize="index",
    ) * 100

    table = table.reindex(columns=optimizer_order, fill_value=0)

    print(table.round(1))

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

elif False: # compare GRUAN and estimated sigma
    # step 1, save sigma values
    mle_summary = gp.read_pkl(
        r"apps\pblh_unc_v1\pkls\mle_summary_2024.pkl"
    )

    mle_summary = mle_summary[
        (mle_summary["optimizer"] == "powell")
        & (mle_summary["converged"])
    ]

    mle_summary["sigma.measurement"] = np.sqrt(
        mle_summary["sigma2.measurement"]
    )

    mle_summary["sigma.level"] = np.sqrt(
        mle_summary["sigma2.level"]
    )

    mle_summary["sigma.trend"] = np.sqrt(
        mle_summary["sigma2.trend"]
    )

    hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
    lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
    lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

    paths = [lin_path, hko_path, lau_path]

    variables = ['alt', 'theta_v', 'rh', 'wzon', 'wmeri']

    for path in  paths:
        dataset = gp.read_pkl(path)
        for pid, gdp in tqdm(dataset.items()):
            for var in variables:
                sigma=(gdp.data[var+'_uc']*0.5).mean()
                mle_summary.loc[
                    (mle_summary["pid"] == pid)
                    & (mle_summary["variable"] == var)
                    & (mle_summary["fit_type"] == "gruan"),
                    "sigma.measurement"
                ] = sigma

    out_path = r"apps\pblh_unc_v1\pkls\mle_sigma_2024.pkl"
    mle_summary.to_pickle(out_path)
    print(f"Saved dataframe to: {out_path}")

elif True:  # compare GRUAN and estimated sigma
    # step 2, scatter plots
    mle_summary = gp.read_pkl(
        r"apps\pblh_unc_v1\pkls\mle_sigma_2024.pkl"
    )

    variables = ['alt', 'theta_v', 'rh', 'wzon', 'wmeri']

    var_to_labels = {'alt': 'Altitude ($m$)', 
        'theta_v': "Virtual Potential Temperature ($K$)",
        'rh': "Relative Humidity ($\%$)", 
        'wzon': "Zonal Wind Velocity ($m\,s^{-1}$)",
        'wmeri': "Meridional Wind Velocity ($m\,s^{-1}$)"}
    
    models_to_labels = {"UnivariateLLL": "LLL",
                        "UnivariateLLT": "LLT",
                        "DeterministicLevelLLT": "DetLLT"}
    
    fit_type_to_labels = {"mle":"MLE", "gruan":"GRUAN"}

    sigma_labels = {
    "sigma.measurement": r"$\sigma$ Measurement",
    "sigma.level": r"$\sigma$ Level",
    "sigma.trend": r"$\sigma$ Trend",
    }

    pairs = [
        ("sigma.measurement", "sigma.level"),
        ("sigma.measurement", "sigma.trend"),
        ("sigma.level", "sigma.trend"),
    ]
    mle_summary["model"] = mle_summary["model"].map(models_to_labels)
    mle_summary["fit_type"] = mle_summary["fit_type"].map(fit_type_to_labels)

    # ------------------------------------------------------------------
    # Create a standalone legend figure once
    # ------------------------------------------------------------------
    fig_tmp, ax_tmp = plt.subplots()

    sns.scatterplot(
        data=mle_summary,
        x="sigma.measurement",
        y="sigma.level",
        hue="model",
        style="fit_type",
        s=100,
        ax=ax_tmp,
    )

    handles, labels = ax_tmp.get_legend_handles_labels()
    plt.close(fig_tmp)

    handles, labels = ax_tmp.get_legend_handles_labels()

    # Rename section headers
    labels = [
        "Model" if label == "model"
        else "Fit Type" if label == "fit_type"
        else label
        for label in labels
    ]

    fig_leg, ax_leg = plt.subplots(figsize=(10, 2))

    ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=True,
    )

    ax_leg.axis("off")
    plt.show()

    # ------------------------------------------------------------------
    # Scatter plots
    # ------------------------------------------------------------------
    for var in variables:

        df = mle_summary[mle_summary["variable"] == var].copy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (xcol, ycol) in zip(axes, pairs):

            sns.scatterplot(
                data=df,
                x=xcol,
                y=ycol,
                hue="model",
                style="fit_type",
                s=100,
                legend=False,
                ax=ax,
            )
            #ax.set_xscale("log")
            #ax.set_yscale("log")
            #ax.set_title(f"{xcol} vs {ycol}")

            ax.set_xlabel(sigma_labels[xcol])
            ax.set_ylabel(sigma_labels[ycol])

        fig.suptitle(var_to_labels[var])
        plt.tight_layout()
        fig.subplots_adjust(
            left=0.07,
            bottom=0.116,
            right=0.987,
            top=0.922,
            wspace=0.307,
            hspace=0.200,
        )
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