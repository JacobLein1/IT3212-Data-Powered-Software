import numpy as np
import pandas as pd
from scipy import stats


def transform_data(data_in):
    """Log + z-score standardize the value columns of the cleaned wide data.
    """
    
    value_cols = ["Area harvested", "Yield", "Production"]
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    log_z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]

    data = data_in.copy()
    for col, log_col, log_z_col in zip(value_cols, log_cols, log_z_cols):
        data[log_col] = np.log1p(data[col])
        data[log_z_col] = (data[log_col] - data[log_col].mean()) / data[log_col].std()

    print("Transform statistics (log1p -> z-score):")
    for col, log_z_col in zip(value_cols, log_z_cols):
        print(
            f"{col}: "
            f"skew {stats.skew(data[col]):.0f} -> {stats.skew(data[log_z_col]):.2f}, "
            f"kurtosis {stats.kurtosis(data[col]):.0f} -> {stats.kurtosis(data[log_z_col]):.2f}"
        )

    return data
