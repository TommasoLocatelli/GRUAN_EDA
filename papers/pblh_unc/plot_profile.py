import matplotlib.pyplot as plt
from methodology import *
from code_examples.visual_config.color_map import map_labels_to_colors

def plot_ssm_diagnostics_short(
    pid,
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
    FONTE_SIZE=12, alpha=0.75, alpha_unc=0.30, alpha_sim=0.02,
    scatter_obs = False
):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f"SSM Diagnostics — Profile {pid}", fontsize=12)

    # =========================================================
    # 1. θv PANEL — pm + thv gradient
    # =========================================================
    ax = axes[0]

    if scatter_obs:
        ax.scatter(thv_o, alt_o, label="Observed θv",
            color=map_labels_to_colors['virtual_theta'], alpha=alpha)
    else:
        ax.plot(thv_o, alt_o, label="Observed θv",
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

    # --- pm method ---
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

    # --- thv gradient method ---
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
    ax.legend(loc="upper left")
    ax.grid(True)

    # =========================================================
    # 2. RH PANEL — rh method
    # =========================================================
    ax = axes[1]

    if scatter_obs:
        ax.scatter(rh_o, alt_o, label="Observed RH",
            color=map_labels_to_colors['rh'], alpha=alpha)
    else:
        ax.plot(rh_o, alt_o, label="Observed RH",
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

    # --- RH method ---
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
    ax.legend(loc="upper left")
    ax.grid(True)

    # =========================================================
    # 3. WIND PANEL — Ri method
    # =========================================================
    ax = axes[2]

    if scatter_obs:
        ax.scatter(u_o, alt_o, label="Observed u",
            color=map_labels_to_colors['q'], alpha=alpha)
        ax.scatter(v_o, alt_o, label="Observed v",
            color=map_labels_to_colors['uspeed'], alpha=alpha)
    else:
        ax.plot(u_o, alt_o, label="Observed u",
            color=map_labels_to_colors['q'], alpha=alpha)
        ax.plot(v_o, alt_o, label="Observed v",
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

    # --- Ri method ---
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
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_ssm_diagnostics_full(
    pid,
    alt_o, thv_o, rh_o, u_o, v_o,
    thv_g_o, rh_g_o, ri_o,
    alt_o_unc, thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
    thv_g_o_unc, rh_g_o_unc, ri_o_unc,
    alt_s, thv_s, rh_s, u_s, v_s,
    thv_g_s, rh_g_s, ri_s,
    alt_s_unc, thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
    thv_g_s_unc, rh_g_s_unc, ri_s_unc,
    simulations,
    pblh_info,
    map_labels_to_colors,
    *,
    FONTE_SIZE=8, alpha=0.75, alpha_unc=0.30, alpha_sim=0.02
):

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()
    plt.suptitle(f"SSM Diagnostics — Profile {pid}", fontsize=12)

    # =========================================================
    # 1. θv PANEL — pm + thv gradient
    # =========================================================
    ax = axes[0]

    ax.plot(thv_o, alt_o, label="Observed θv",
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

    # --- pm method ---
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

    # --- thv gradient method ---
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
    ax.legend(loc="upper left")
    ax.grid(True)

    # =========================================================
    # 2. RH PANEL — rh method
    # =========================================================
    ax = axes[1]

    ax.plot(rh_o, alt_o, label="Observed RH",
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

    # --- RH method ---
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
    ax.legend(loc="upper left")
    ax.grid(True)

    # =========================================================
    # 3. WIND PANEL — Ri method
    # =========================================================
    ax = axes[2]

    ax.plot(u_o, alt_o, label="Observed u",
            color=map_labels_to_colors['q'], alpha=alpha)
    ax.plot(v_o, alt_o, label="Observed v",
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

    # --- Ri method ---
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
    ax.legend(loc="upper left")
    ax.grid(True)

    # =========================================================
    # 4. θv gradient vs altitude
    # =========================================================
    ax = axes[3]
    ax.plot(thv_g_o, alt_o, label="Observed θv gradient",
            color=map_labels_to_colors['virtual_theta'], alpha=alpha)
    ax.plot(thv_g_s, alt_s, label="Smoothed θv gradient",
            color=map_labels_to_colors['virtual_temp'], linewidth=2, alpha=alpha)

    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["thv_grad"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    ax.fill_betweenx(alt_o, thv_g_o - thv_g_o_unc, thv_g_o + thv_g_o_unc,
                     color=map_labels_to_colors['virtual_theta_uc'], alpha=alpha_unc)

    ax.fill_betweenx(alt_s, thv_g_s - thv_g_s_unc, thv_g_s + thv_g_s_unc,
                     color=map_labels_to_colors['virtual_temp_uc'], alpha=alpha_unc)

    ax.set_xlabel(r"$d\theta_v/dz$ [K/m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Virtual Potential Temperature Gradient", fontsize=FONTE_SIZE)
    ax.legend()
    ax.grid(True)

    # =========================================================
    # 5. RH gradient vs altitude
    # =========================================================
    ax = axes[4]
    ax.plot(rh_g_o, alt_o, label="Observed RH gradient",
            color=map_labels_to_colors['rh'], alpha=alpha)
    ax.plot(rh_g_s, alt_s, label="Smoothed RH gradient",
            color=map_labels_to_colors['press'], linewidth=2, alpha=alpha)

    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["rh_grad"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    ax.fill_betweenx(alt_o, rh_g_o - rh_g_o_unc, rh_g_o + rh_g_o_unc,
                     color=map_labels_to_colors['rh_uc'], alpha=alpha_unc)

    ax.fill_betweenx(alt_s, rh_g_s - rh_g_s_unc, rh_g_s + rh_g_s_unc,
                     color=map_labels_to_colors['press_uc'], alpha=alpha_unc)

    ax.set_xlabel("RH Gradient [%/m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Relative Humidity Gradient", fontsize=FONTE_SIZE)
    ax.legend()
    ax.grid(True)

    # =========================================================
    # 6. Richardson number vs altitude
    # =========================================================
    ax = axes[5]
    ax.plot(ri_o, alt_o, label="Observed Ri",
            color=map_labels_to_colors['rh'], alpha=alpha)
    ax.plot(ri_s, alt_s, label="Smoothed Ri",
            color=map_labels_to_colors['Ri_b'], linewidth=2, alpha=alpha)

    for sim in simulations:
        sim_vars = smooth_variables(sim)
        ax.plot(sim_vars["rich"], sim_vars["alt"],
                color="gray", alpha=alpha_sim, linewidth=1)

    ax.fill_betweenx(alt_o, ri_o - ri_o_unc, ri_o + ri_o_unc,
                     color=map_labels_to_colors['rh_uc'], alpha=alpha_unc)

    ax.fill_betweenx(alt_s, ri_s - ri_s_unc, ri_s + ri_s_unc,
                     color=map_labels_to_colors['Ri_b_uc'], alpha=alpha_unc)

    ax.axvline(0.25, color='black', linestyle='--', alpha=alpha,
               label="Ri = 0.25")

    ax.set_xlabel("Richardson Number")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Bulk Richardson Number", fontsize=FONTE_SIZE)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
