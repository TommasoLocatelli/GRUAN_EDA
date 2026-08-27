import pandas as pd
import numpy as np

def missing_data(data, columns=None):
    """
    Check missing values in selected columns.
    Returns:
        DataFrame with columns:
            variable, missing_count, indices
    """
    if data is None:
        return pd.DataFrame()

    df = data

    # Select columns
    if columns is None:
        df_selected = df
    else:
        valid_cols = [c for c in columns if c in df.columns]
        df_selected = df[valid_cols]

        missing_cols = set(columns) - set(valid_cols)
        if missing_cols:
            print(f"Warning: these columns do not exist: {missing_cols}")

    results = []

    for col in df_selected.columns:
        mask = df_selected[col].isna()
        if mask.any():
            idx = df.index[mask].tolist()
            results.append({
                "variable": col,
                "missing_count": len(idx),
                "indices": idx
            })

    if len(results) == 0:
        return pd.DataFrame()

    return pd.DataFrame(results)

def physics_constraint(data, columns=None):
    """
    Check physical constraints.
    Returns:
        DataFrame with columns:
            variable, violation_count, indices
    """

    if data is None:
        return pd.DataFrame()

    df = data

    # Select columns
    if columns is None:
        df_selected = df
    else:
        valid_cols = [c for c in columns if c in df.columns]
        df_selected = df[valid_cols]

        missing_cols = set(columns) - set(valid_cols)
        if missing_cols:
            print(f"Warning: these columns do not exist: {missing_cols}")

    # Physical constraints
    constraints = {
        "alt":      (0, 40000),
        "temp":     (150, 350),
        "rh":       (0, 150),
        "wmeri":    (-150, 150),
        "wzon":     (-150, 150),
    }

    # Uncertainty constraints
    uc_constraints = {
        "alt_uc":   (0, 20000),
        "temp_uc":  (0, 200),
        "rh_uc":    (0, 75),
        "wmeri_uc": (0, 150),
        "wzon_uc":  (0, 150),
    }

    summary = []

    for col in df_selected.columns:
        series = df_selected[col]

        # Uncertainty variables
        if col.endswith("_uc"):
            if col not in uc_constraints:
                continue

            low, high = uc_constraints[col]
            mask = (series < low) | (series > high)

            if mask.any():
                idx = df.index[mask].tolist()
                summary.append({
                    "variable": col,
                    "violation_count": len(idx),
                    "indices": idx
                })

            continue

        # Main variables
        if col in constraints:
            low, high = constraints[col]
            mask = (series < low) | (series > high)

            if mask.any():
                idx = df.index[mask].tolist()
                summary.append({
                    "variable": col,
                    "violation_count": len(idx),
                    "indices": idx
                })

        # Variables without constraints → skip

    if len(summary) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(summary)

def detect_outliers(data, columns=None, method="iqr", z_thresh=3.5, iqr_factor=5):
    """
    Detect outliers.
    Returns:
        DataFrame with columns:
            variable, outlier_count, indices
    """

    if data is None:
        return pd.DataFrame()

    df = data

    # Select columns
    if columns is None:
        df_selected = df.select_dtypes(include=[np.number])
    else:
        valid_cols = [c for c in columns if c in df.columns]
        df_selected = df[valid_cols]

        missing_cols = set(columns) - set(valid_cols)
        if missing_cols:
            print(f"Warning: these columns do not exist: {missing_cols}")

    summary = []

    for col in df_selected.columns:
        series = df_selected[col].dropna()

        if len(series) == 0:
            continue

        # Z-score method
        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            z_scores = (series - mean) / std
            mask = np.abs(z_scores) > z_thresh

        # IQR method
        elif method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            low = q1 - iqr_factor * iqr
            high = q3 + iqr_factor * iqr
            mask = (series < low) | (series > high)

        else:
            raise ValueError("method must be 'zscore' or 'iqr'")

        bad_idx = series[mask].index.tolist()

        if len(bad_idx) > 0:
            summary.append({
                "variable": col,
                "outlier_count": len(bad_idx),
                "indices": bad_idx
            })

    if len(summary) == 0:
        return pd.DataFrame()

    return pd.DataFrame(summary)

def altitude_drops(data):
    """
    Detect non‑increasing time values in GDP data.
    Since data is sorted by altitude, altitude drops are redundant.
    
    Returns:
        DataFrame with columns:
            check_type, count, indices
    """

    if data is None:
        return pd.DataFrame()

    df = data

    # Required columns
    required = ["time", "alt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return pd.DataFrame()

    summary = []

    # -------------------------
    # TIME NOT INCREASING
    # -------------------------
    time_diff = df["time"].diff()
    mask_time = time_diff <= pd.Timedelta(0)

    if mask_time.any():
        idx = df.index[mask_time].tolist()
        summary.append({
            "check_type": "time_not_increasing",
            "count": len(idx),
            "indices": idx
        })
    
    # -------------------------
    # Final output
    # -------------------------
    if len(summary) == 0:
        return pd.DataFrame()

    return pd.DataFrame(summary)
