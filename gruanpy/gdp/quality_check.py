import pandas as pd
import numpy as np

def missing_data(data, columns=None):
    """
    Check missing values in selected columns.
    Returns:
        dict: {column: DataFrame of rows with missing values}
        DataFrame: combined problematic rows
    """
    if data is None:
        print("No data table found in GDP object.")
        return {}, pd.DataFrame()

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

    missing_rows = {}

    for col in df_selected.columns:
        mask = df_selected[col].isna()
        if mask.any():
            missing_rows[col] = df.loc[mask]

    if len(missing_rows) == 0:
        print("No missing values found.")
        return {}, pd.DataFrame()

    print("\nMissing values detected:")
    for col, rows in missing_rows.items():
        print(f"{col}: {len(rows)} missing")

    combined = pd.concat(missing_rows.values()).drop_duplicates()
    return missing_rows, combined

def physics_constraint(data, columns=None):
    """
    Check physical constraints and return problematic rows.
    Variables without defined constraints are ignored (no QC needed).
    """

    if data is None:
        print("No data table found in GDP object.")
        return {}, pd.DataFrame()

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
        "temp":     (150, 350),     # Kelvin
        "rh":       (0, 150),       # supersaturation allowed
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

    violations = {}

    for col in df_selected.columns:
        series = df_selected[col]

        # Uncertainty variables
        if col.endswith("_uc"):
            if col not in uc_constraints:
                # No constraint defined → skip
                continue

            low, high = uc_constraints[col]
            mask = (series < low) | (series > high)
            if mask.any():
                violations[col] = df.loc[mask]
            continue

        # Main variables
        if col in constraints:
            low, high = constraints[col]
            mask = (series < low) | (series > high)
            if mask.any():
                violations[col] = df.loc[mask]
        else:
            # No constraint defined → skip
            continue

    # Combine violations
    if len(violations) == 0:
        print("All selected columns satisfy physical constraints.")
        return {}, pd.DataFrame()

    combined = pd.concat(violations.values()).drop_duplicates()

    return violations, combined

def detect_outliers(data, columns=None, method="iqr", z_thresh=3.5, iqr_factor=1.5):
    """
    Detect outliers and return problematic rows.
    """
    if data is None:
        print("No data table found in GDP object.")
        return {}, pd.DataFrame()

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

    outlier_rows = {}

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

        bad_idx = series[mask].index
        if len(bad_idx) > 0:
            outlier_rows[col] = df.loc[bad_idx]

    if len(outlier_rows) == 0:
        print("No outliers detected.")
        return {}, pd.DataFrame()

    print("\nOutliers detected:")
    for col, rows in outlier_rows.items():
        print(f"{col}: {len(rows)} outliers")

    combined = pd.concat(outlier_rows.values()).drop_duplicates()
    return outlier_rows, combined

def altitude_drops(data):
    """
    Detect altitude drops in GDP data.

    Returns
    -------
    dict
        {
            "time_not_increasing": DataFrame of problematic rows,
            "altitude_drops": DataFrame of problematic rows
        }
    DataFrame
        Combined DataFrame of all problematic rows
    """

    if data is None:
        print("No data table found in GDP object.")
        return {}, pd.DataFrame()

    df = data

    # Check required columns
    required = ["time", "alt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return {}, pd.DataFrame()

    problems = {}

    # -------------------------
    # 1. TIME NOT INCREASING
    # -------------------------
    time_diff = df["time"].diff()
    mask_time = time_diff <= pd.Timedelta(0)  # non-increasing or equal

    if mask_time.any():
        problems["time_not_increasing"] = df.loc[mask_time]
        print(f"Time not strictly increasing: {mask_time.sum()} rows")

    # -------------------------
    # 2. ALTITUDE DROPS
    # -------------------------
    alt_diff = df["alt"].diff()
    mask_alt = alt_diff < 0  # altitude decreases

    if mask_alt.any():
        problems["altitude_drops"] = df.loc[mask_alt]
        print(f"Altitude drops detected: {mask_alt.sum()} rows")

    # -------------------------
    # Combine all problematic rows
    # -------------------------
    if len(problems) == 0:
        print("No altitude drops or time-order issues detected.")
        return {}, pd.DataFrame()

    combined = pd.concat(problems.values()).drop_duplicates()

    return problems, combined
