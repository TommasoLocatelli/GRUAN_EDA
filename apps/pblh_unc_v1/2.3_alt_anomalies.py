from tqdm import tqdm
import gruanpy as gp
import numpy as np
import pandas as pd

hko_path = r"apps\\pblh_unc_v1\\pkls\\gdp_2024__HKO-RS-01_2024.pkl"
lau_path = r"apps\\pblh_unc_v1\\pkls\\gdp_2024__LAU-RS-02_2024.pkl"
lin_path = r"apps\\pblh_unc_v1\\pkls\\gdp_2024__LIN-RS-01_2024.pkl"

hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

# merge all pid→gdp dictionaries
merged = {**hko, **lau, **lin}

print(f'Numner of gdps {len(merged)}')

anomalous_rows=[]

for pid, gdp in tqdm(merged.items(), desc="Checking anomalies"):
    data = gdp.data[['time', 'alt', 'alt_uc']].copy()

    # --- Extract metadata ---
    site = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Site.Name"]["Value"].values[0]
    start_time = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Measurement.StartTime"]["Value"].values[0]
    dataprod = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Product.FullKey"]["Value"].values[0]

    site_altitude_raw = gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Site.Altitude"]["Value"].values[0]
    site_altitude = float(site_altitude_raw[:-2])  # strip " m"

    # Special case for LAU
    if gdp.global_attrs[gdp.global_attrs["Attribute"] == "g.Site.Key"]["Value"].values[0] == "LAU":
        site_altitude = 3.0

    # --- Add constant columns ---
    data["pid"] = pid
    data["site"] = site
    data["start_time"] = start_time
    data["dataprod"] = dataprod
    data["site_altitude"] = site_altitude

    data["A.1"] = (data["alt"] < site_altitude).astype(int)
    data["A.2"] = ((data["alt"] + data["alt_uc"]) < site_altitude).astype(int)

    data["B"] = (data["alt_uc"] > 250).astype(int)

    data = data[(data["A.1"] == 1) | (data["A.2"] == 1) | (data["B"] == 1)]

    # Append filtered DataFrame
    anomalous_rows.append(data)

# --- Combine all anomalies into one DataFrame ---
anomalous_rows = pd.concat(anomalous_rows, ignore_index=True)

anomalous_rows.to_csv("altitude_anomalies_2024.csv", index=False)
print("Saved altitude_anomalies_2024.csv")

print("\nNumber of anomalies per type:")
print("A.1:", anomalous_rows["A.1"].sum())
print("A.2:", anomalous_rows["A.2"].sum())
print("B:", anomalous_rows["B"].sum())

print("\nAnomaly combinations:")
print(anomalous_rows[["A.1","A.2","B"]].value_counts().reset_index())

cols_B = ["site", "alt", "alt_uc", "site_altitude", "B"]
cols_A2 = ["site", "alt", "alt_uc", "site_altitude", "A.2", "margin"]

print("\nTop 5 most extreme B anomalies (largest alt_uc):")
print(
    anomalous_rows[anomalous_rows["B"] == 1]
    .sort_values("alt_uc", ascending=False)
    .head(5)[cols_B]
)

print("\nTop 5 most extreme A.2 anomalies (largest negative margin):")
print(
    anomalous_rows[anomalous_rows["A.2"] == 1]
    .assign(margin = anomalous_rows["alt"] + anomalous_rows["alt_uc"] - anomalous_rows["site_altitude"])
    .sort_values("margin")
    .head(5)[cols_A2]
)


