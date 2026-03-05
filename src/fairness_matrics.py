from src.db_config import fetch_latest_records
import pandas as pd
import numpy as np

def compute_group_positive_rates(df: pd.DataFrame,
                                 group_col: str = "race",
                                 positive_col: str = "prediction") -> dict:
    """
    Compute the average positive prediction rate for each group.
    
    positive_rate[g] = (# rows with prediction == 1 and group == g) / (# rows with group == g)
    
    Returns:
        dict: {group_value: positive_rate (float between 0 and 1)}
              Groups with no samples are skipped.
    """
    
    sub = df[[group_col, positive_col]].dropna(subset=[group_col, positive_col])

    
    sub[positive_col] = pd.to_numeric(sub[positive_col], errors="coerce")
    sub = sub.dropna(subset=[positive_col])

    
    rates = (
        sub.groupby(group_col)[positive_col]
        .mean()  
        .to_dict()
    )

    return rates


def compute_demographic_parity_difference(df: pd.DataFrame,
                                          group_col: str = "race",
                                          positive_col: str = "prediction") -> float:
    """
    Demographic Parity Difference (DPD):
    
        DPD = max(positive_rate_g) - min(positive_rate_g) across all groups.
    
    Higher values indicate more disparity between groups.
    """
    rates = compute_group_positive_rates(df, group_col, positive_col)
    if not rates:
        return np.nan  

    values = list(rates.values())
    return float(max(values) - min(values))


def compute_disparate_impact(df: pd.DataFrame,
                             group_col: str = "race",
                             positive_col: str = "prediction") -> float:
    """
    Disparate Impact (DI):
    
        DI = min(positive_rate_g) / max(positive_rate_g)
    
    This is a group-agnostic version (no fixed 'protected' group).
    DI is between 0 and 1 (if max_rate > 0).
    
    - DI close to 1  → groups are treated similarly.
    - DI near 0      → large disparity.
    """
    rates = compute_group_positive_rates(df, group_col, positive_col)
    if not rates:
        return np.nan

    values = list(rates.values())
    max_rate = max(values)
    min_rate = min(values)

    if max_rate == 0:
        
        return np.nan

    return float(min_rate / max_rate)


def compute_fairness_metrics(df: pd.DataFrame,
                             group_col: str = "race",
                             positive_col: str = "prediction") -> dict:
    """
    Wrapper that computes:
      - group-wise positive rates
      - demographic parity difference
      - disparate impact
    
    Returns:
        {
          "group_positive_rates": {group: rate, ...},
          "demographic_parity_difference": float,
          "disparate_impact": float
        }
    """
    group_rates = compute_group_positive_rates(df, group_col, positive_col)
    dpd = compute_demographic_parity_difference(df, group_col, positive_col)
    di = compute_disparate_impact(df, group_col, positive_col)

    return {
        "group_positive_rates": group_rates,
        "demographic_parity_difference": dpd,
        "disparate_impact": di,
    }





# looping
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Fairness metric functions

def compute_group_positive_rates(df: pd.DataFrame,
                                 group_col: str = "race",
                                 positive_col: str = "prediction") -> dict:
    """
    Compute the average positive prediction rate for each group.

    positive_rate[g] = (# rows with prediction == 1 and group == g)
                       / (# rows with group == g)

    Returns:
        dict: {group_value: positive_rate (float between 0 and 1)}
    """
    sub = df[[group_col, positive_col]].dropna(subset=[group_col, positive_col])

    sub[positive_col] = pd.to_numeric(sub[positive_col], errors="coerce")
    sub = sub.dropna(subset=[positive_col])

    rates = (
        sub.groupby(group_col)[positive_col]
        .mean()  
        .to_dict()
    )

    return rates


def compute_demographic_parity_difference(df: pd.DataFrame,
                                          group_col: str = "race",
                                          positive_col: str = "prediction") -> float:
    """
    Demographic Parity Difference (DPD):

        DPD = max(positive_rate_g) - min(positive_rate_g) across all groups.
    """
    rates = compute_group_positive_rates(df, group_col, positive_col)
    if not rates:
        return np.nan

    values = list(rates.values())
    return float(max(values) - min(values))


def compute_disparate_impact(df: pd.DataFrame,
                             group_col: str = "race",
                             positive_col: str = "prediction") -> float:
    """
    Disparate Impact (DI):

        DI = min(positive_rate_g) / max(positive_rate_g)

    (group-agnostic; we don’t fix a “protected group” here)
    """
    rates = compute_group_positive_rates(df, group_col, positive_col)
    if not rates:
        return np.nan

    values = list(rates.values())
    max_rate = max(values)
    min_rate = min(values)

    if max_rate == 0:
        
        return np.nan

    return float(min_rate / max_rate)


def compute_fairness_metrics(df: pd.DataFrame,
                             group_col: str = "race",
                             positive_col: str = "prediction") -> dict:
    """
    Wrapper to compute:
      - group-wise positive rates
      - demographic parity difference
      - disparate impact
    """
    group_rates = compute_group_positive_rates(df, group_col, positive_col)
    dpd = compute_demographic_parity_difference(df, group_col, positive_col)
    di = compute_disparate_impact(df, group_col, positive_col)

    return {
        "group_positive_rates": group_rates,
        "demographic_parity_difference": dpd,
        "disparate_impact": di,
    }



# Print fairness matrix printer

def print_fairness_matrix(df: pd.DataFrame,
                          group_col: str = "race",
                          positive_col: str = "prediction"):
    """
    Prints a clear fairness matrix:
        - Positive rate for each group
        - Demographic Parity Difference
        - Disparate Impact
    """
    metrics = compute_fairness_metrics(df, group_col, positive_col)

    group_rates = metrics["group_positive_rates"]
    dpd = metrics["demographic_parity_difference"]
    di = metrics["disparate_impact"]

    print("\n======================= FAIRNESS MATRIX =======================")
    print(f"Group Attribute : {group_col}")
    print("----------------------------------------------------------------")
    print(f"{'Group':25s} | {'Positive Rate':>15s}")
    print("-" * 55)

    for group, rate in group_rates.items():
        print(f"{group:25s} | {rate:15.3f}")

    print("-" * 55)
    print(f"{'Demographic Parity Difference':25s} : {dpd:.4f}")
    if pd.isna(di):
        print(f"{'Disparate Impact':25s} : NA")
    else:
        print(f"{'Disparate Impact':25s} : {di:.4f}")

    print("================================================================\n")


# Sliding-window live monitor (integrated with fairness matrix)

import time
from datetime import datetime
import pandas as pd

def monitor_fairness_sliding_window(
    window_size: int = 100,
    interval_seconds: int = 10,
    group_col: str = "race",
    positive_col: str = "prediction"
):
    """
    Periodically:
      - fetch the latest `window_size` records from predictions_log
      - convert them to a DataFrame
      - compute & print the fairness matrix
    """
    while True:
        
        records = fetch_latest_records(window_size)   

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{now_str}] Fetched records: {0 if records is None else len(records)}")

        if not records:
            print("No data available for fairness computation yet.")
        else:
            
            df = pd.DataFrame(records)

            
            if "timestamp" in df.columns:
                
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                df = df.sort_values("timestamp", ascending=False).head(window_size)

            print(f"Using {len(df)} records (sliding window, max {window_size}).")
            
            print_fairness_matrix(df, group_col=group_col, positive_col=positive_col)

        time.sleep(interval_seconds)


# Entry point

if __name__ == "__main__":
    monitor_fairness_sliding_window(
        window_size=100,
        interval_seconds=10,
        group_col="race",
        positive_col="prediction"
    )
