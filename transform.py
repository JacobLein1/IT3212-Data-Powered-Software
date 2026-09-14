import numpy as np
import pandas as pd
from scipy import stats


def transform_data(data_in, z_params=None):
    """Log + z-score standardize the value columns of the cleaned wide data.

    When `z_params` is None the z-score mean/std are fitted on `data_in`;
    otherwise the given statistics are applied as-is (use this to transform
    held-out data with the training statistics). Returns the transformed
    frame together with the fitted `z_params`.
    """
    
    value_cols = ["Area harvested", "Yield", "Production"]
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    log_z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]

    data = data_in.copy()
    for col, log_col in zip(value_cols, log_cols):
        data[log_col] = np.log1p(data[col])

    if z_params is None:
        z_params = {log_col: (data[log_col].mean(), data[log_col].std()) for log_col in log_cols}
    for log_col, log_z_col in zip(log_cols, log_z_cols):
        mean, std = z_params[log_col]
        data[log_z_col] = (data[log_col] - mean) / std

    print("Transform statistics (log1p -> z-score):")
    for col, log_z_col in zip(value_cols, log_z_cols):
        print(
            f"{col}: "
            f"skew {stats.skew(data[col]):.0f} -> {stats.skew(data[log_z_col]):.2f}, "
            f"kurtosis {stats.kurtosis(data[col]):.0f} -> {stats.kurtosis(data[log_z_col]):.2f}"
        )

    return data, z_params


def one_hot_encode(data_in, columns=None):
    """One-hot encode the categorical columns as int8 dummy columns.

    The original categorical columns are replaced by their dummies, so the
    returned frame is a numeric feature matrix; the input frame is not
    modified.
    """
    if columns is None:
        columns = ["Area", "Item"]

    data = data_in.copy()
    num_cols_before = data.shape[1]
    category_counts = {col: data[col].nunique() for col in columns}
    data = pd.get_dummies(data, columns=columns, dtype=np.int8)

    print("Encoding statistics (one-hot):")
    for col in columns:
        print(f"{col}: {category_counts[col]} dummy columns")
    print(f"Columns: {num_cols_before} -> {data.shape[1]}")

    return data
