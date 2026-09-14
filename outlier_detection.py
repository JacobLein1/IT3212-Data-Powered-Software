import pandas as pd


VALUE_COLS = ['Area harvested', 'Yield', 'Production']
GROUP_COLS = ['Area', 'Item']


def moving_average_outlier_mask(data_in, columns=VALUE_COLS, window=5, threshold=0.5):
    """
    Flag outliers by comparing each value to the simple moving average of its
    own (Area, Item) time series.

    For every group the values are sorted by Year and a centered moving
    average over `window` years is computed. A value is an outlier if its
    relative deviation from the moving average exceeds `threshold`:

        |value - moving_average| / moving_average > threshold

    A relative threshold is used because the value columns span several
    orders of magnitude across items and areas.

    Returns a boolean DataFrame (same index as `data_in`) with one column per
    value column, True where the value is an outlier.
    """
    data = data_in.sort_values(GROUP_COLS + ['Year'])
    grouped = data.groupby(GROUP_COLS)

    mask = pd.DataFrame(False, index=data.index, columns=columns)
    for col in columns:
        moving_average = grouped[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=2).mean()
        )
        # avoid division by zero; a zero moving average gives no reference to compare against
        moving_average = moving_average.replace(0, pd.NA)
        deviation = (data[col] - moving_average).abs() / moving_average
        mask[col] = deviation.gt(threshold).fillna(False).astype(bool)

    return mask.reindex(data_in.index)


def remove_outliers(data_in, columns=VALUE_COLS, window=5, threshold=0.5):
    """
    Remove rows where any value column is a moving-average outlier
    (see `moving_average_outlier_mask`) and print statistics about it.
    """
    mask = moving_average_outlier_mask(data_in, columns, window, threshold)
    row_mask = mask.any(axis=1)

    total_rows = len(data_in)
    print(f"Moving average outlier detection (window={window}, threshold={threshold}):")
    for col in columns:
        n = mask[col].sum()
        print(f"{col}: {n} outliers ({round(n / total_rows * 100, 2)}%)")
    print(f"Rows removed: {row_mask.sum()} ({round(row_mask.sum() / total_rows * 100, 2)}%)")

    return data_in.loc[~row_mask].reset_index(drop=True)
