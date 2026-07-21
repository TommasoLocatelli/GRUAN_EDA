import matplotlib.pyplot as plt
from methodology import *
from applications.visual_config.color_map import map_labels_to_colors
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd

sim_proxy = Line2D([0], [0], color="gray", alpha=0.8, linewidth=1, label="Sim. profiles")

TEXT_SIZE=15
#c="""
plt.rcParams.update({
    #"font.size": 5,            # Base font size
    "axes.titlesize": TEXT_SIZE,       # Subplot titles
    "axes.labelsize": TEXT_SIZE,       # Axis labels
    "xtick.labelsize": TEXT_SIZE,      # Tick labels
    "ytick.labelsize": TEXT_SIZE,
    "legend.fontsize": 15,      # Legend text
    "figure.titlesize": TEXT_SIZE,     # Suptitle
})

def plot_ssm_diagnostics_short(
    pid, where, when, tod,
    # observed
    alt_o, thv_o, rh_o, u_o, v_o,
    thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
    # smoothed
    alt_s, thv_s, rh_s, u_s, v_s,
    thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
    # simulations
    simulations,
    # PBLH info (pm, thv, rh, ri)
    pblh_info,
    # color map
    map_labels_to_colors,
    # plotting parameters
    FONTE_SIZE=12, alpha=0.75, alpha_unc=0.30, alpha_sim=0.05,
    scatter_obs = False
):

    # Allow disabling PBLH plotting
    if not pblh_info:
        pblh_info = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f"Profile Id: {pid}, {where}, {when}, {tod.capitalize()}", fontsize=14)

    # =========================================================
    # 1. θv PANEL — pm + thv gradient
    # =========================================================
    ax = axes[0]
    ax.legend(handles=[sim_proxy] + ax.get_legend_handles_labels()[0])

    if scatter_obs:
        ax.scatter(thv_o, alt_o, label="Obs. θv",
            color=map_labels_to_colors['theta'], alpha=alpha)
    else:
        ax.plot(thv_o, alt_o, label="Obs. θv",
            color=map_labels_to_colors['theta'], alpha=alpha)
    
    ax.plot(thv_s, alt_s, label="Smoothed θv",
            color=map_labels_to_colors['temp'], linewidth=2, alpha=alpha)

    # Simulated profiles (single legend entry)
    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["thv"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)


    # Uncertainty bands
    ax.fill_betweenx(alt_o, thv_o - thv_o_unc, thv_o + thv_o_unc,
                     color=map_labels_to_colors['theta_uc'], alpha=alpha_unc,
                     label="Obs θv Unc.")

    ax.fill_betweenx(alt_s, thv_s - thv_s_unc, thv_s + thv_s_unc,
                     color=map_labels_to_colors['temp_uc'], alpha=alpha_unc,
                     label="Smoothed θv Unc.")

    # --- pm method ---
    if pblh_info and "pm" in pblh_info:
        z = pblh_info["pm"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_pm'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH pm std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_pm'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH pm median = {z['median']:.0f} m")
        ax.fill_between([thv_o.min(), thv_o.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_pm_uc'], alpha=alpha_unc,
                        label=f"pm unc. = [{z['low']:.0f}, {z['high']:.0f}] m")

    # --- thv gradient method ---
    if pblh_info and "thv" in pblh_info:
        z = pblh_info["thv"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_theta'],
                   linestyle='-.', linewidth=1.4,
                   label=f"PBLH θv std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_theta'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH θv median = {z['median']:.0f} m")
        ax.fill_between([thv_o.min(), thv_o.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_theta_uc'], alpha=alpha_unc,
                        label=f"θv Unc. = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel(r"$\theta_v$ [K]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Virtual Potential Temperature", fontsize=FONTE_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[sim_proxy] + handles, loc="upper left")
    ax.grid(True)

    # =========================================================
    # 2. RH PANEL — rh method
    # =========================================================
    ax = axes[1]
    
    ax.legend(handles=[sim_proxy] + ax.get_legend_handles_labels()[0])


    if scatter_obs:
        ax.scatter(rh_o, alt_o, label="Obs. RH",
            color=map_labels_to_colors['rh'], alpha=alpha)
    else:
        ax.plot(rh_o, alt_o, label="Obs. RH",
            color=map_labels_to_colors['rh'], alpha=alpha)
    
    ax.plot(rh_s, alt_s, label="Smoothed RH",
            color=map_labels_to_colors['press'], linewidth=2, alpha=alpha)

    # Simulated profiles
    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["rh"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    # Uncertainty bands
    ax.fill_betweenx(alt_o, rh_o - rh_o_unc, rh_o + rh_o_unc,
                     color=map_labels_to_colors['rh_uc'], alpha=alpha_unc,
                     label="Obs RH Unc.")

    ax.fill_betweenx(alt_s, rh_s - rh_s_unc, rh_s + rh_s_unc,
                     color=map_labels_to_colors['press_uc'], alpha=alpha_unc,
                     label="Smoothed RH Unc.")

    # --- RH method ---
    if pblh_info and "rh" in pblh_info:
        z = pblh_info["rh"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_rh'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH RH std = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_rh'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH RH median = {z['median']:.0f} m")
        ax.fill_between([rh_o.min(), rh_o.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_rh_uc'], alpha=alpha_unc,
                        label=f"RH unc. = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel("RH [%]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Relative Humidity", fontsize=FONTE_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[sim_proxy] + handles, loc="upper left")
    ax.grid(True)

    # =========================================================
    # 3. WIND PANEL — Ri method
    # =========================================================
    ax = axes[2]
        
    ax.legend(handles=[sim_proxy] + ax.get_legend_handles_labels()[0])

    if scatter_obs:
        #ax.scatter(u_o, alt_o, label="Obs. u",
        #    color=map_labels_to_colors['q_uc'], alpha=alpha)
        ax.scatter(v_o, alt_o, label="Obs. v",
            color=map_labels_to_colors['uspeed'], alpha=alpha)
    else:
        #ax.plot(u_o, alt_o, label="Obs. u",
        #    color=map_labels_to_colors['q_uc'], alpha=alpha)
        ax.plot(v_o, alt_o, label="Obs. v",
            color=map_labels_to_colors['uspeed'], alpha=alpha)
    
    #ax.plot(u_s, alt_s, label="Smoothed u",
    #        color=map_labels_to_colors['wspeed'], linewidth=2, alpha=alpha)
    ax.plot(v_s, alt_s, label="Smoothed v",
            color=map_labels_to_colors['es'], linewidth=2, alpha=alpha)

    # Unc. profiles
    first_sim = True
    for sim in simulations:
        sim_vars = smooth_variables(sim)
        #ax.plot(sim_vars["u"], sim_vars["alt"],
        #        color="gray", alpha=alpha_sim, linewidth=1,
        #        label="Sim. profiles" if first_sim else None)
        ax.plot(sim_vars["v"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)
        first_sim = False

    # Uncertainty bands
    #ax.fill_betweenx(alt_o, u_o - u_o_unc, u_o + u_o_unc,
    #                 color=map_labels_to_colors['uspeed'], alpha=alpha_unc,
    #                 label="Obs u Unc.")

    ax.fill_betweenx(alt_o, v_o - v_o_unc, v_o + v_o_unc,
                     color=map_labels_to_colors['uspeed_uc'], alpha=alpha_unc,
                     label="Obs v Unc.")

    #ax.fill_betweenx(alt_s, u_s - u_s_unc, u_s + u_s_unc,
    #                 color=map_labels_to_colors['wspeed_uc'], alpha=alpha_unc,
    #                 label="Smoothed u Unc.")

    ax.fill_betweenx(alt_s, v_s - v_s_unc, v_s + v_s_unc,
                     color=map_labels_to_colors['es_uc'], alpha=alpha_unc,
                     label="Smoothed v Unc.")

    # --- Ri method ---
    if pblh_info and "ri" in pblh_info:
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
                        label=f"Rm unc = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Meridional Wind Speed", fontsize=FONTE_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[sim_proxy] + handles, loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_ssm_diagnostics_with_violin(
    pid, where, when, tod, launch_time_local,
    # observed
    alt_o, thv_o, rh_o, u_o, v_o,
    thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
    # smoothed
    alt_s, thv_s, rh_s, u_s, v_s,
    thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
    # simulations (ignored)
    simulations,
    # PBLH info (pm, thv, rh, ri) including "samples"
    pblh_info,
    # color map
    map_labels_to_colors,
    # plotting parameters
    FONTE_SIZE=12, alpha=0.75, alpha_unc=0.30,
    scatter_obs=False
):

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    lt_str = launch_time_local.strftime("%Y-%m-%d %H:%M:%S %Z")

    print(
        f"Profile Id: {pid}\n"
        f"Site: {where}\n"
        f"Local Launch Time: {lt_str}\n"
        )

    # =========================================================
    # 1. θv PANEL — pm + thv gradient
    # =========================================================
    ax = axes[0]

    if scatter_obs:
        ax.scatter(thv_o, alt_o, #label="Obs. θv",
            color=map_labels_to_colors['theta'], alpha=alpha)
    else:
        ax.plot(thv_o, alt_o, #label="Obs. θv",
            color=map_labels_to_colors['theta'], alpha=alpha)

    ax.plot(thv_s, alt_s, #label="Smoothed θv",
            color=map_labels_to_colors['temp'], linewidth=2, alpha=alpha)

    # Uncertainty bands
    ax.fill_betweenx(alt_o, thv_o - thv_o_unc, thv_o + thv_o_unc,
                     color=map_labels_to_colors['theta_uc'], 
                     #label="Obs θv Unc.",
                     alpha=alpha_unc)

    ax.fill_betweenx(alt_s, thv_s - thv_s_unc, thv_s + thv_s_unc,
                     color=map_labels_to_colors['temp_uc'], 
                     #label="Smoothed θv Unc.",
                     alpha=alpha_unc)

    # --- pm method ---
    if pblh_info and "pm" in pblh_info:
        z = pblh_info["pm"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_pm'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH PM = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_pm'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH PM MC = {z['median']:.0f} m")
        ax.fill_between([thv_o.min()-thv_o_unc.max(), thv_o.max()+thv_o_unc.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_pm_uc'], alpha=alpha_unc,
                        label=f"PM MC Unc. = [{z['low']:.0f}, {z['high']:.0f}] m")

    # --- thv gradient method ---
    if pblh_info and "thv" in pblh_info:
        z = pblh_info["thv"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_theta'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH θv = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_theta'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH θv MC = {z['median']:.0f} m")
        ax.fill_between([thv_o.min()-thv_o_unc.max(), thv_o.max()+thv_o_unc.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_theta_uc'], alpha=alpha_unc,
                        label=f"θv MC Unc.= [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel(r"$\theta_v$ [K]")
    ax.set_ylabel("Altitude [m]")
    #ax.set_title("θv Profile, PM & θv PBLH", fontsize=FONTE_SIZE)
    ax.legend(loc="upper left")
    ax.grid(True)

    ymin = -50
    ymax = max(alt_o.max()+50, alt_s.max()+50)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=1)

    # =========================================================
    # 2. RH PANEL — rh method
    # =========================================================
    ax = axes[1]

    if scatter_obs:
        ax.scatter(rh_o, alt_o, #label="Obs. RH",
            color=map_labels_to_colors['rh'], alpha=alpha)
    else:
        ax.plot(rh_o, alt_o, #label="Obs. RH",
            color=map_labels_to_colors['rh'], alpha=alpha)

    ax.plot(rh_s, alt_s, #label="Smoothed RH",
            color=map_labels_to_colors['press'], linewidth=2, alpha=alpha)

    # Uncertainty bands
    ax.fill_betweenx(alt_o, rh_o - rh_o_unc, rh_o + rh_o_unc,
                     color=map_labels_to_colors['rh_uc'],
                     #label="Obs RH Unc.",
                     alpha=alpha_unc)

    ax.fill_betweenx(alt_s, rh_s - rh_s_unc, rh_s + rh_s_unc,
                     color=map_labels_to_colors['press_uc'],
                     #label="Smoothed RH Unc.",
                     alpha=alpha_unc)

    # --- RH method ---
    if pblh_info and "rh" in pblh_info:
        z = pblh_info["rh"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_rh'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH RH = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_rh'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH RH MC = {z['median']:.0f} m")
        ax.fill_between([rh_o.min()-rh_o_unc.max(), rh_o.max()+rh_o_unc.max()], z["low"], z["high"],
                        color=map_labels_to_colors['pblh_rh_uc'], alpha=alpha_unc,
                        label=f"RH MC Unc. = [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel("RH [%]")
    #ax.set_title("Relative Humidity Profile, RH PBLH", fontsize=FONTE_SIZE)
    ax.legend(loc="upper left")
    ax.grid(True)

    ymin = -50
    ymax = max(alt_o.max()+50, alt_s.max()+50)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='lower center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=1)
    ax.set_yticklabels([])

    # =========================================================
    # 3. WIND PANEL — Ri method
    # =========================================================
    ax = axes[2]

    if scatter_obs:
        ax.scatter(v_o, alt_o, #label="Obs. v",
            color=map_labels_to_colors['uspeed'], alpha=alpha)
    else:
        ax.plot(v_o, alt_o, #label="Obs. v",
            color=map_labels_to_colors['uspeed'], alpha=alpha)

    ax.plot(v_s, alt_s, #label="Smoothed v",
            color=map_labels_to_colors['es'], linewidth=2, alpha=alpha)

    # Uncertainty bands
    ax.fill_betweenx(alt_o, v_o - v_o_unc, v_o + v_o_unc,
                     color=map_labels_to_colors['uspeed_uc'], 
                     #label="Obs v Unc.",
                     alpha=alpha_unc)

    ax.fill_betweenx(alt_s, v_s - v_s_unc, v_s + v_s_unc,
                     color=map_labels_to_colors['es_uc'],
                     #label="Smoothed v Unc.",
                     alpha=alpha_unc)

    # --- Ri method ---
    if pblh_info and "ri" in pblh_info:
        z = pblh_info["ri"]
        ax.axhline(z["value"], color=map_labels_to_colors['pblh_Ri'],
                   linestyle='--', linewidth=1.4,
                   label=f"PBLH RM = {z['value']:.0f} m")
        ax.axhline(z["median"], color=map_labels_to_colors['pblh_Ri'],
                   linestyle='-', linewidth=1.6,
                   label=f"PBLH RM MC = {z['median']:.0f} m")
        ax.fill_between([v_o.min()-v_o_unc.max(), v_o.max()+v_o_unc.max()],
                        z["low"], z["high"],
                        color=map_labels_to_colors['pblh_Ri_uc'], alpha=alpha_unc,
                        label=f"RM MC Unc.= [{z['low']:.0f}, {z['high']:.0f}] m")

    ax.set_xlabel("Wind speed [m/s]")
    #ax.set_title("Meridional Wind Speed, RM PBLH", fontsize=FONTE_SIZE)
    ax.legend(loc="upper left")
    ax.grid(True)

    ymin = -50
    ymax = max(alt_o.max()+50, alt_s.max()+50)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='lower center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=1)
    ax.set_yticklabels([])

    # =========================================================
    # 4. VIOLIN PANEL — Monte Carlo PBLH distributions (seaborn)
    # =========================================================
    ax = axes[3]
    #ax.set_title("Monte‑Carlo PBLH Distributions", fontsize=FONTE_SIZE)

    # Prepare dataframe for seaborn
    df_violin = []
    for method in ["pm", "thv", "rh", "ri"]:
        if method in pblh_info and "samples" in pblh_info[method]:
            samples = pblh_info[method]["samples"]
            df_violin.append(pd.DataFrame({
                "PBLH": samples,
                "Method": method.upper()
            }))

    df_violin = pd.concat(df_violin, ignore_index=True)

    sns.violinplot(
        data=df_violin,
        x="Method",
        y="PBLH",
        ax=ax,
        palette={
            "PM":  map_labels_to_colors['pblh_pm'],
            "THV": map_labels_to_colors['pblh_theta'],
            "RH":  map_labels_to_colors['pblh_rh'],
            "RI":  map_labels_to_colors['pblh_Ri']
        },
        cut=0,
        inner="quartile",
        linewidth=1.2
    )

    ax.grid(True)

    # >>> Match y-axis limits with other panels <<<
    ymin = -50
    ymax = max(alt_o.max()+50, alt_s.max()+50)
    ax.set_ylim(ymin, ymax)

    axes[3].set_position([
    axes[3].get_position().x0 + 0.03,
    axes[3].get_position().y0,
    axes[3].get_position().width * 0.85,
    axes[3].get_position().height])
    ax.set_ylabel('')
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(
    top=0.65,
    bottom=0.09,
    left=0.07,
    right=0.99,
    hspace=0.20,
    wspace=0.25   # leggermente più largo del tuo 0.223
    )
    plt.show()

