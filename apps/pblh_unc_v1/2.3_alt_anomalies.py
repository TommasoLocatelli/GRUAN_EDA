from tqdm import tqdm
import gruanpy as gp

hko_path = r"apps\\pblh_unc_v1\\pkls\\gdp_2024__HKO-RS-01_2024.pkl"
lau_path = r"apps\\pblh_unc_v1\\pkls\\gdp_2024__LAU-RS-02_2024.pkl"
lin_path = r"apps\\pblh_unc_v1\\pkls\\gdp_2024__LIN-RS-01_2024.pkl"

hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

# merge all pid→gdp dictionaries
merged = {**hko, **lau, **lin}

alt_uc_results = []
alt_low_results = []

for pid, gdp in tqdm(merged.items(), desc="Checking anomalies"):
    alt_uc_series = gdp.data["alt_uc"]
    alt_series = gdp.data["alt"]

    # extract global attributes
    site = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Site.Name"]["Value"].values[0]
    time = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Measurement.StartTime"]["Value"].values[0]
    dataprod = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Product.FullKey"]["Value"].values[0]
    site_altitude_raw = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Site.Altitude"]["Value"].values[0]
    site_altitude = float(site_altitude_raw[:-2])  # strip " m"

    # fix known Lauder bug: use measurement system altitude instead
    if gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Site.Key"]["Value"].values[0] == "LAU":
        site_altitude = 3.0

    # --- alt_uc anomalies ---
    alt_uc_idx = alt_uc_series[alt_uc_series > 250].index

    for idx in alt_uc_idx:
        anomaly_time = gdp.data["time"].iloc[idx]
        anomaly_value = alt_uc_series.iloc[idx]
        altitude = alt_series.iloc[idx]

        alt_uc_results.append([
            pid, site, time, dataprod,
            idx, anomaly_time, anomaly_value, altitude
        ])

    # --- altitude too low anomalies ---
    alt_low_idx = alt_series[alt_series < site_altitude].index

    for idx in alt_low_idx:
        anomaly_time = gdp.data["time"].iloc[idx]
        altitude = alt_series.iloc[idx]
        diff = site_altitude - altitude  # severity

        alt_low_results.append([
            pid, site, time, dataprod,
            idx, anomaly_time, altitude, site_altitude, diff
        ])

# sort both anomaly lists
alt_uc_results.sort(key=lambda row: row[6], reverse=True)   # sort by alt_uc
alt_low_results.sort(key=lambda row: row[8], reverse=True)  # sort by severity (diff)

# save alt_uc anomalies
with open("alt_uc_anomalies.csv", "w") as f:
    f.write("pid,site,start_time,data_product,index,anomaly_time,alt_uc,altitude\n")
    for row in alt_uc_results:
        f.write(",".join(map(str, row)) + "\n")

# save low-altitude anomalies
with open("alt_low_anomalies.csv", "w") as f:
    f.write("pid,site,start_time,data_product,index,anomaly_time,altitude,site_altitude,diff\n")
    for row in alt_low_results:
        f.write(",".join(map(str, row)) + "\n")

print("Saved", len(alt_uc_results), "alt_uc anomalies")
print("Saved", len(alt_low_results), "low-altitude anomalies")
