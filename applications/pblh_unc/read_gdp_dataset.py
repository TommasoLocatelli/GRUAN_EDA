import sys
import os
import time
import random
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gruanpy as gp
import tqdm

start_time = time.time()   # <-- start timer

random.seed(42)

# select list of unzipped annual gdp folder
folders = [
    r'gdp\products_RS41-GDP-1_LIN-RS-01_2024'
]

for folder in folders:
    dataset = {}
    if not os.path.isdir(folder):
        print(f"Warning: folder not found -> {folder}")
        continue

    nc_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nc")
    ]

    print(f"{folder}: {len(nc_files)} .nc files")

    if len(nc_files) == 0:
        print(f"No .nc files found in {folder}")
        continue

    if len(nc_files) > 800:
        nc_files = [
        f for f in nc_files
        if ("T000000" in f or "T120000" in f)
        ]
        print(f"Keeping only T00 and T12 files: {len(nc_files)} found")

    for nc in tqdm.tqdm(nc_files[:]):
        try:
            g = gp.read(nc)
            pid = g.global_attrs[g.global_attrs['Attribute']=='g.Product.Id']['Value'].values[0]
            upper_bound=gp._find_upper_bound(g.data[['alt']], upper_bound=6000, return_value=True)
            g.data[g.data['alt'] <= upper_bound]

            if pid in dataset:
                print(f"Duplicate Product.Id detected: {pid} (skipping {nc})")
                continue

            dataset[pid] = g

        except Exception as e:
            print(f"Error reading {nc}: {e}")

    print(f"\nTotal unique profiles loaded: {len(dataset)}")

    output_path = f"papers\\pblh_unc\\gdp_2024_{folder[24:]}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved to: {output_path}")

# <-- end timer
end_time = time.time()
elapsed = end_time - start_time
print(f"Total execution time: {elapsed:.2f} seconds")
