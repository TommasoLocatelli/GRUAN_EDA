# plot GDP uncertainties
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gruanpy as gp
from code_examples.visual_config.color_map import map_labels_to_colors
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

#c="""
plt.rcParams.update({
    #"font.size": 5,            # Base font size
    "axes.titlesize": 14,       # Subplot titles
    "axes.labelsize": 14,       # Axis labels
    "xtick.labelsize": 14,      # Tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,      # Legend text
    "figure.titlesize": 14,     # Suptitle
})
#plt.rcParams["figure.dpi"] = 300
#plt.rcParams["savefig.dpi"] = 300
#"""

folder = r'gdp\icm16'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]

for file_path in file_paths[:5]:
    gdp = gp.read(file_path)
    gdp.data = gdp.data[gdp.data['alt'] <= 2000]
    gdp.data = gdp.data[gdp.data['alt'] > 1800] 
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    when = when[:10] + ' ' + when[11:16]
    gdp.data = gdp.data[gdp.data.index % 3==0]

    plt.figure(figsize=(12, 4))
    plt.suptitle(f'RS41-GDP.1: {where}, {when}', fontsize=20)
    DOTS_SIZE = 20

    # ---------------------------------------------------------
    # 1) TEMPERATURE
    # ---------------------------------------------------------
    plt.subplot(1, 3, 1)
    for t, a, t_uc, a_uc in zip(gdp.data['temp'], gdp.data['alt'],
                                gdp.data['temp_uc'], gdp.data['alt_uc']):
        ellip = Ellipse((t, a), t_uc, a_uc,
                        color=map_labels_to_colors['temp_uc'], alpha=0.3)
        plt.gca().add_patch(ellip)

    proxy_temp_uc = mpatches.Ellipse((0,0), 1, 1,
                                    color=map_labels_to_colors['temp_uc'], alpha=0.3)
    proxy_temp = plt.Line2D([0], [0], marker='o', linestyle='',
                            color=map_labels_to_colors['temp'], markersize=6)

    plt.scatter(gdp.data['temp'], gdp.data['alt'],
                color=map_labels_to_colors['temp'], s=DOTS_SIZE)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.grid(True)
    plt.legend([proxy_temp, proxy_temp_uc],
            ["Temperature", "Temperature uncertainty"],loc='upper right')

    # ---------------------------------------------------------
    # 2) RELATIVE HUMIDITY
    # ---------------------------------------------------------
    plt.subplot(1, 3, 2)
    for rh, a, rh_uc, a_uc in zip(gdp.data['rh'], gdp.data['alt'],
                                gdp.data['rh_uc'], gdp.data['alt_uc']):
        ellip = Ellipse((rh, a), rh_uc, a_uc,
                        color=map_labels_to_colors['rh_uc'], alpha=0.3)
        plt.gca().add_patch(ellip)

    proxy_rh_uc = mpatches.Ellipse((0,0), 1, 1,
                                color=map_labels_to_colors['rh_uc'], alpha=0.3)
    proxy_rh = plt.Line2D([0], [0], marker='o', linestyle='',
                        color=map_labels_to_colors['rh'], markersize=6)

    plt.scatter(gdp.data['rh'], gdp.data['alt'],
                color=map_labels_to_colors['rh'], s=DOTS_SIZE)
    plt.xlabel('Relative Humidity (%)')
    plt.gca().set_ylabel('')
    plt.grid(True)
    plt.legend([proxy_rh, proxy_rh_uc],
            ["RH", "RH uncertainty"],loc='upper right')

    # ---------------------------------------------------------
    # 3) WIND SPEED
    # ---------------------------------------------------------
    plt.subplot(1, 3, 3)
    for ws, a, ws_uc, a_uc in zip(gdp.data['wspeed'], gdp.data['alt'],
                                gdp.data['wspeed_uc'], gdp.data['alt_uc']):
        ellip = Ellipse((ws, a), ws_uc, a_uc,
                        color=map_labels_to_colors['wspeed_uc'], alpha=0.3)
        plt.gca().add_patch(ellip)

    proxy_ws_uc = mpatches.Ellipse((0,0), 1, 1,
                                color=map_labels_to_colors['wspeed_uc'], alpha=0.3)
    proxy_ws = plt.Line2D([0], [0], marker='o', linestyle='',
                        color=map_labels_to_colors['wspeed'], markersize=6)

    plt.scatter(gdp.data['wspeed'], gdp.data['alt'],
                color=map_labels_to_colors['wspeed'], s=DOTS_SIZE)
    plt.xlabel('Wind Speed (m/s)')
    plt.gca().set_ylabel('')
    plt.grid(True)
    plt.legend([proxy_ws, proxy_ws_uc],
            ["Wind speed", "Wind speed uncertainty"],loc='upper right')

    plt.tight_layout()
    plt.show()
