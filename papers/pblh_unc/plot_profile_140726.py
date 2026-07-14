import matplotlib.pyplot as plt
from methodology import *
from code_examples.visual_config.color_map import map_labels_to_colors
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd

sim_proxy = Line2D([0], [0], color="gray", alpha=0.8, linewidth=1, label="Sim. profiles")

def plot_ssm_diagnostics_with_hist(
    pid, where, when, tod,
    # observed
    alt_o, thv_o, rh_o, u_o, v_o,
    thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
    # smoothed
    alt_s, thv_s, rh_s, u_s, v_s,
    thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
    # simulations
    simulations,
    # PBLH info (pm, thv, rh, ri) including "samples"
    pblh_info,
    # color map
    map_labels_to_colors,
    # plotting parameters
    FONTE_SIZE=12, alpha=0.75, alpha_unc=0.30, alpha_sim=0.02,
    scatter_obs=False
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle(f"Profile {pid} - {where} - {when} - {tod}", fontsize=14)

    # =========================================================
    # 1. θv PANEL (top-left)
    # =========================================================
    ax = axes[0, 0]

    if scatter_obs:
        ax.scatter(thv_o, alt_o, label="Obs. θv",
                   color=map_labels_to_colors['virtual_theta'], alpha=alpha)
    else:
        ax.plot(thv_o, alt_o, label="Obs. θv",
                color=map_labels_to_colors['virtual_theta'], alpha=alpha)

    ax.plot(thv_s, alt_s, label="Smoothed θv",
            color=map_labels_to_colors['virtual_temp'], linewidth=2, alpha=alpha)

    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["thv"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    ax.fill_betweenx(alt_o, thv_o - thv_o_unc, thv_o + thv_o_unc,
                     color=map_labels_to_colors['virtual_theta_uc'], alpha=alpha_unc)
    ax.fill_betweenx(alt_s, thv_s - thv_s_unc, thv_s + thv_s_unc,
                     color=map_labels_to_colors['virtual_temp_uc'], alpha=alpha_unc)

    # pm method
    if "pm" in pblh_info:
        z = pblh_info["pm"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_pm'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH pm std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_pm'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH pm median = {z['median']:.0f} m")
        ax.fill_between([thv_o.min(), thv_o.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_pm_uc'], alpha=alpha_unc,
                        label=f"PBLH pm 95% = [{z['low']:.0f}, {z['high']:.0f}] m")

    # thv gradient method
    if "thv" in pblh_info:
        z = pblh_info["thv"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_theta'],
                   linestyle='-.', linewidth=1.4,
                   label=f"PBLH θv-grad std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_theta'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH θv-grad median = {z['median']:.0f} m")
        ax.fill_between([thv_o.min(), thv_o.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_theta_uc'], alpha=alpha_unc,
                        label=f"PBLH θv-grad 95% = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel(r"$\theta_v$ [K]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Virtual Potential Temperature", fontsize=FONTE_SIZE)
    ax.grid(True)
    ax.legend(loc="upper left")

    # =========================================================
    # 2. RH PANEL (top-right)
    # =========================================================
    ax = axes[0, 1]

    if scatter_obs:
        ax.scatter(rh_o, alt_o, label="Obs. RH",
                   color=map_labels_to_colors['rh'], alpha=alpha)
    else:
        ax.plot(rh_o, alt_o, label="Obs. RH",
                color=map_labels_to_colors['rh'], alpha=alpha)

    ax.plot(rh_s, alt_s, label="Smoothed RH",
            color=map_labels_to_colors['press'], linewidth=2, alpha=alpha)

    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["rh"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    ax.fill_betweenx(alt_o, rh_o - rh_o_unc, rh_o + rh_o_unc,
                     color=map_labels_to_colors['rh_uc'], alpha=alpha_unc)
    ax.fill_betweenx(alt_s, rh_s - rh_s_unc, rh_s + rh_s_unc,
                     color=map_labels_to_colors['press_uc'], alpha=alpha_unc)

    if "rh" in pblh_info:
        z = pblh_info["rh"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_rh'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH RH std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_rh'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH RH median = {z['median']:.0f} m")
        ax.fill_between([rh_o.min(), rh_o.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_rh_uc'], alpha=alpha_unc,
                        label=f"PBLH RH 95% = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel("RH [%]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Relative Humidity", fontsize=FONTE_SIZE)
    ax.grid(True)
    ax.legend(loc="upper left")

    # =========================================================
    # 3. WIND PANEL (bottom-left)
    # =========================================================
    ax = axes[1, 0]

    if scatter_obs:
        ax.scatter(u_o, alt_o, label="Obs. u",
                   color=map_labels_to_colors['q'], alpha=alpha)
        ax.scatter(v_o, alt_o, label="Obs. v",
                   color=map_labels_to_colors['uspeed'], alpha=alpha)
    else:
        ax.plot(u_o, alt_o, label="Obs. u",
                color=map_labels_to_colors['q'], alpha=alpha)
        ax.plot(v_o, alt_o, label="Obs. v",
                color=map_labels_to_colors['uspeed'], alpha=alpha)

    ax.plot(u_s, alt_s, label="Smoothed u",
            color=map_labels_to_colors['wspeed'], linewidth=2, alpha=alpha)
    ax.plot(v_s, alt_s, label="Smoothed v",
            color=map_labels_to_colors['es'], linewidth=2, alpha=alpha)

    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["u"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)
        ax.plot(sim_vars["v"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    ax.fill_betweenx(alt_o, u_o - u_o_unc, u_o + u_o_unc,
                     color=map_labels_to_colors['q_uc'], alpha=alpha_unc)
    ax.fill_betweenx(alt_o, v_o - v_o_unc, v_o + v_o_unc,
                     color=map_labels_to_colors['uspeed_uc'], alpha=alpha_unc)
    ax.fill_betweenx(alt_s, u_s - u_s_unc, u_s + u_s_unc,
                     color=map_labels_to_colors['wspeed_uc'], alpha=alpha_unc)
    ax.fill_betweenx(alt_s, v_s - v_s_unc, v_s + v_s_unc,
                     color=map_labels_to_colors['es_uc'], alpha=alpha_unc)

    if "ri" in pblh_info:
        z = pblh_info["ri"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_Ri'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH Ri std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_Ri'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH Ri median = {z['median']:.0f} m")
        ax.fill_between([min(u_o.min(), v_o.min()), max(u_o.max(), v_o.max())],
                        z["low"], z["high"],
                        color=map_labels_to_colors['pblh_Ri_uc'], alpha=alpha_unc,
                        label=f"PBLH Ri 95% = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel("Wind [m/s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Zonal and Meridional Wind", fontsize=FONTE_SIZE)
    ax.grid(True)
    ax.legend(loc="upper left")

    # =========================================================
    # 4. HISTOGRAM PANEL (bottom-right)
    # =========================================================
    ax = axes[1, 1]
    ax.set_title("Monte‑Carlo PBLH Distributions", fontsize=FONTE_SIZE)

    colors = {
        "pm":  map_labels_to_colors['pblh_pm'],
        "thv": map_labels_to_colors['pblh_theta'],
        "rh":  map_labels_to_colors['pblh_rh'],
        "ri":  map_labels_to_colors['pblh_Ri']
    }

    # ---- COMMON BINS ----
    all_samples = np.concatenate([
        pblh_info[m]["samples"]
        for m in ["pm", "thv", "rh", "ri"]
        if "samples" in pblh_info[m]
    ])
    common_bins = np.linspace(all_samples.min(), all_samples.max(), 41)

    # ---- HISTOGRAMS WITH NUMBERS IN LEGEND ----
    for method in ["pm", "thv", "rh", "ri"]:
        if method in pblh_info and "samples" in pblh_info[method]:
            z = pblh_info[method]
            samples = z["samples"]

            ax.hist(samples, bins=common_bins, alpha=0.6, density=True,
                    color=colors[method],
                    label=f"{method.upper()} median={z['median']:.0f} m, 95%=[{z['low']:.0f},{z['high']:.0f}]")

    ax.set_xlabel("PBLH [m]")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

