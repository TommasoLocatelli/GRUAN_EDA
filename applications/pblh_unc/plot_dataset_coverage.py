import pickle
import pytz
from datetime import datetime
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gruanpy as gp
import matplotlib.pyplot as plt
import numpy as np

TEXT_SIZE=22
#c="""
plt.rcParams.update({
    #"font.size": 5,            # Base font size
    "axes.titlesize": TEXT_SIZE,       # Subplot titles
    "axes.labelsize": TEXT_SIZE,       # Axis labels
    "xtick.labelsize": TEXT_SIZE,      # Tick labels
    "ytick.labelsize": TEXT_SIZE,
    "legend.fontsize": TEXT_SIZE,      # Legend text
    "figure.titlesize": TEXT_SIZE,     # Suptitle
})

# -----------------------------
# 1. Load GDP dictionaries
# -----------------------------

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

hko = load_pkl(r"applications/pblh_unc/pkls/gdp_2024_HKO-RS-01_2024.pkl")
lau = load_pkl(r"applications/pblh_unc/pkls/gdp_2024_LAU-RS-02_2024.pkl")
lin = load_pkl(r"applications/pblh_unc/pkls/gdp_2024_LIN-RS-01_2024.pkl")

print("Loaded:")
print("HKO:", len(hko))
print("LAU:", len(lau))
print("LIN:", len(lin))

# -----------------------------
# 2. Timezone mapping
# -----------------------------

SITE_TIMEZONES = {
    "LIN": "Europe/Berlin",
    "HKO": "Asia/Hong_Kong",
    "LAU": "Pacific/Auckland",
    "POT": "Europe/Rome",
}

# -----------------------------
# 3. Extract UTC timestamp
# -----------------------------

def get_utc_time(gdp):
    return gdp.global_attrs.loc[
        gdp.global_attrs['Attribute'] == 'g.Measurement.StandardTime',
        'Value'
    ].values[0]

# -----------------------------
# 4. Extract GRUAN TimeOfDay
# -----------------------------

def get_time_of_day(gdp):
    return gdp.global_attrs.loc[
        gdp.global_attrs["Attribute"] == "g.Measurement.TimeOfDay",
        "Value"
    ].values[0]

# -----------------------------
# 5. Convert UTC → local time
# -----------------------------

def to_local_time(utc_timestamp, site_key):
    tz = pytz.timezone(SITE_TIMEZONES[site_key])
    utc_dt = datetime.fromisoformat(utc_timestamp.replace("Z", "+00:00"))
    return utc_dt.astimezone(tz)

# -----------------------------
# 6. Season classification
# -----------------------------

def classify_season(dt, site_key):
    m = dt.month

    if site_key == "LAU":  # southern hemisphere
        if m in (12, 1, 2): return "summer"
        if m in (3, 4, 5): return "autumn"
        if m in (6, 7, 8): return "winter"
        return "spring"

    # northern hemisphere
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5): return "spring"
    if m in (6, 7, 8): return "summer"
    return "autumn"

# -----------------------------
# 7. Analyze a site
# -----------------------------

def analyze_site(dataset, site_key):
    day_night = Counter()
    seasons = Counter()
    hours = Counter()   # <-- NEW: count launches per local hour

    for pid, gdp in dataset.items():
        utc = get_utc_time(gdp)
        local = to_local_time(utc, site_key)

        dn = get_time_of_day(gdp)
        ss = classify_season(local, site_key)
        hh = local.hour  # <-- local hour (0–23)

        day_night[dn] += 1
        seasons[ss] += 1
        hours[hh] += 1

    return day_night, seasons, hours

# -----------------------------
# 8. Run analysis
# -----------------------------

hko_dn, hko_season, hko_hours = analyze_site(hko, "HKO")
lau_dn, lau_season, lau_hours = analyze_site(lau, "LAU")
lin_dn, lin_season, lin_hours = analyze_site(lin, "LIN")

# -----------------------------
# 9. Print results
# -----------------------------

print("\n=== HKO (Hong Kong) ===")
print("Day/Night:", hko_dn)
print("Seasons:", hko_season)
print("Hours:", hko_hours)

print("\n=== LAU (Lauder) ===")
print("Day/Night:", lau_dn)
print("Seasons:", lau_season)
print("Hours:", lau_hours)

print("\n=== LIN (Lindenberg) ===")
print("Day/Night:", lin_dn)
print("Seasons:", lin_season)
print("Hours:", lin_hours)


# ---------------------------------------------------------
# Prepare stacked arrays
# ---------------------------------------------------------

def prepare_stacked(counter_dicts, categories):
    stacked = []
    for counter in counter_dicts:
        stacked.append([counter.get(cat, 0) for cat in categories])
    return np.array(stacked)

dn_categories = ["daytime", "nighttime", "twilight"]
season_categories = ["winter", "spring", "summer", "autumn"]

dn_stacked = prepare_stacked([lin_dn, hko_dn, lau_dn], dn_categories)
season_stacked = prepare_stacked([lin_season, hko_season, lau_season], season_categories)

sites = ["LIN", "HKO", "LAU"]

# ---------------------------------------------------------
# Plot: two subplots
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ---------------------------------------------------------
# Subplot 1: Day/Night/Twilight
# ---------------------------------------------------------

ax = axes[0]
bottom = np.zeros(len(sites))

colors_dn = ["#1f77b4", "#ff7f0e", "#9467bd"]  # day, night, twilight

for i, cat in enumerate(dn_categories):
    ax.bar(sites, dn_stacked[:, i], bottom=bottom, label=cat, color=colors_dn[i])
    bottom += dn_stacked[:, i]

#ax.set_title("Day/Night/Twilight Distribution per Site")
ax.set_ylabel("Number of launches")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2)

# ---------------------------------------------------------
# Subplot 2: Seasons
# ---------------------------------------------------------

ax = axes[1]
bottom = np.zeros(len(sites))

colors_season = ["#2ca02c", "#d62728", "#8c564b", "#17becf"]

for i, cat in enumerate(season_categories):
    ax.bar(sites, season_stacked[:, i], bottom=bottom, label=cat, color=colors_season[i])
    bottom += season_stacked[:, i]

#ax.set_title("Seasonal Distribution per Site")
ax.set_ylabel("Number of launches")
ax.set_ylabel('')        # remove axis label
#ax.set_yticklabels([])        # remove tick marks + numbers
# or ax.set_yticklabels([]) if you want ticks but no numbers
ax.grid(axis="y", alpha=0.3)
ax.legend(loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2)

plt.tight_layout()
plt.subplots_adjust(
    top=0.775,
    bottom=0.10,
    left=0.10,
    right=0.95,
    hspace=0.20,
    wspace=0.25   # leggermente più largo del tuo 0.223
    )
plt.show()
