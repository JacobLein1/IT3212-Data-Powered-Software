import pandas as pd


VALUE_COLS = ['Area harvested', 'Yield', 'Production']
GROUP_COLS = ['Area', 'Item']
WINDOW = 5
THRESHOLD = 0.5



def moving_average_outlier_mask(data_in, columns=VALUE_COLS, window=WINDOW, threshold=THRESHOLD):
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



def remove_outliers(data_in, columns=VALUE_COLS, window=WINDOW, threshold=THRESHOLD):
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



def plot_outlier_examples(data_in, columns=VALUE_COLS, window=WINDOW, threshold=THRESHOLD, n_examples=3, examples=None):
    """
    Plot example (Area, Item) time series with the moving average and the
    points flagged as outliers highlighted.

    `examples` is an optional list of (Area, Item) tuples. If not given, the
    `n_examples` groups with the most flagged outliers are used.
    """
    import matplotlib.pyplot as plt

    mask = moving_average_outlier_mask(data_in, columns, window, threshold)

    if examples is None:
        counts = mask.any(axis=1).groupby([data_in['Area'], data_in['Item']]).sum()
        examples = counts.sort_values(ascending=False).head(n_examples).index.tolist()

    for area, item in examples:
        group_idx = data_in.index[(data_in['Area'] == area) & (data_in['Item'] == item)]
        subset = data_in.loc[group_idx].sort_values('Year')
        subset_mask = mask.loc[subset.index]

        fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
        for ax, col in zip(axes, columns):
            moving_average = subset[col].rolling(window, center=True, min_periods=2).mean()
            ax.plot(subset['Year'], subset[col], marker='o', markersize=3, label=col)
            ax.plot(subset['Year'], moving_average, linestyle='--', label=f"{window}-year moving average")
            ax.fill_between(
                subset['Year'],
                moving_average * (1 - threshold),
                moving_average * (1 + threshold),
                alpha=0.15,
                label=f"±{threshold:.0%} band",
            )
            dropped = subset[subset_mask[col]]
            ax.scatter(dropped['Year'], dropped[col], color='red', zorder=3, label="dropped")
            ax.set_title(col)
            ax.set_xlabel("Year")
            ax.legend(fontsize=7)

        fig.suptitle(f"{item} — {area}")
        plt.tight_layout()
        plt.show()
