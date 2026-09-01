import pandas as pd
import matplotlib.pyplot as plt

def count_profiles_with_missing(md_list):
        return sum([not df.empty for df in md_list])

def missing_count_per_profile(md_list):
        counts = []
        for df in md_list:
            if df.empty:
                counts.append(0)
            else:
                counts.append(df["missing_count"].sum())
        return counts

def gap_sizes(md_list):
        sizes = []
        for df in md_list:
            if df.empty:
                continue
            for gaps in df["gaps"]:
                for g in gaps:
                    sizes.append(g["length"])
        return sizes

def gap_positions(md_list):
        positions = []
        for df in md_list:
            if df.empty:
                continue
            for idx_list in df["indices"]:
                positions.extend(idx_list)
        return positions

def aggregate_missing_by_variable(md_list):
    """
    md_list: list of DataFrames (one per profile)
    Returns: dict {variable: {profiles, missing_total, gap_sizes, positions}}
    """
    agg = {}

    for df in md_list:
        if df.empty:
            continue

        for _, row in df.iterrows():
            var = row["variable"]
            if var not in agg:
                agg[var] = {
                    "profiles": 0,
                    "missing_total": 0,
                    "gap_sizes": [],
                    "positions": []
                }

            agg[var]["profiles"] += 1
            agg[var]["missing_total"] += row["missing_count"]

            # gap sizes
            for g in row["gaps"]:
                agg[var]["gap_sizes"].append(g["length"])

            # positions
            agg[var]["positions"].extend(row["indices"])

    return agg

def missing_summary_table(agg_dict):
    rows = []
    for var, info in agg_dict.items():
        rows.append({
            "variable": var,
            "profiles_with_missing": info["profiles"],
            "missing_total": info["missing_total"],
            "mean_gap_size": np.mean(info["gap_sizes"]) if info["gap_sizes"] else 0,
            "max_gap_size": max(info["gap_sizes"]) if info["gap_sizes"] else 0,
            "mean_position": np.mean(info["positions"]) if info["positions"] else 0
        })
    return pd.DataFrame(rows)
def plot_missing_pie(agg_dict, site_name):
    variables = list(agg_dict.keys())
    totals = [agg_dict[v]["missing_total"] for v in variables]

    plt.figure(figsize=(10, 10))
    plt.pie(
        totals,
        labels=variables,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(f"{site_name} – Portion of Missing Data per Variable")
    plt.axis("equal")  # keeps the pie circular
    plt.show()

def combine_sites(*site_dicts):
    combined = {}

    for site in site_dicts:
        for var, info in site.items():
            if var not in combined:
                combined[var] = 0
            combined[var] += info["missing_total"]

    return combined

def plot_combined_missing_pie(combined_dict):
    variables = list(combined_dict.keys())
    totals = list(combined_dict.values())

    plt.figure(figsize=(10, 10))
    plt.pie(
        totals,
        labels=variables,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Portion of Missing Data per Variable – All Sites Combined")
    plt.axis("equal")
    plt.show()