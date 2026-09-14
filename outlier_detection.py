import os
import warnings

import numpy as np
import pandas as pd


VALUE_COLS = ['Area harvested', 'Yield', 'Production']
GROUP_COLS = ['Area', 'Item']

# consistency constant so that MAD estimates the standard deviation for normal data
MAD_TO_STD = 1.4826


# ============================================================
# Moving average
# ============================================================
def moving_average_outlier_mask(data_in, columns=VALUE_COLS, window=5, threshold=0.5):
    """
    Flag outliers by comparing each value to the simple moving average of its
    own (Area, Item) time series.

    For every group the values are sorted by Year and a centered moving
    average over `window` years is computed. A value is an outlier if its
    relative deviation from the moving average exceeds `threshold`:

        |value - moving_average| / moving_average > threshold

    Note: the threshold is the same for every series, so series with a large
    natural year-to-year variation (e.g. rain-fed crops) get most of their
    peaks flagged. See `arima_outlier_mask` for a per-series adaptive method.

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


# ============================================================
# ARIMA
# ============================================================
def arima_fit_series(series, order=(1, 1, 0)):
    """
    Fit an ARIMA model to one time series (indexed by Year) and return
    (fitted values, log residuals, residual scale). The first two are Series
    indexed by Year, the scale is a scalar.

    The series is reindexed to consecutive years (gaps become NaN, which the
    state-space ARIMA handles) and modelled on a log1p scale so that
    residuals are relative deviations (a residual of 0.1 is roughly 10% off
    the model prediction). The residual scale is a robust standard deviation
    (median/MAD instead of mean/std) so the outliers we are looking for do
    not inflate the scale they are measured against.
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = series.sort_index()
    series = series.reindex(range(series.index.min(), series.index.max() + 1))
    y = np.log1p(series.values.astype(float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ARIMA(y, order=order, trend='c' if order[1] == 0 else 'n').fit()

    fitted = pd.Series(np.expm1(result.fittedvalues), index=series.index)
    resid = pd.Series(result.resid, index=series.index)

    # the first d fitted values/residuals are meaningless (no differenced value to predict from)
    fitted.iloc[:order[1]] = np.nan
    resid.iloc[:order[1]] = np.nan
    resid[series.isna()] = np.nan

    resid = resid - resid.median()
    scale = resid.abs().median() * MAD_TO_STD
    return fitted, resid, scale


def _arima_group_residuals(group, columns, order, min_points):
    """Log residuals and residual scale for every value column of one (Area, Item) group."""
    out = pd.DataFrame(np.nan, index=group.index, columns=[f"{col}_resid" for col in columns] + [f"{col}_scale" for col in columns])
    if len(group) < min_points:
        return out

    series = group.set_index('Year')
    for col in columns:
        try:
            _, resid, scale = arima_fit_series(series[col], order)
        except Exception:
            continue  # leave the column unflagged if the model fails to fit
        out[f"{col}_resid"] = resid.reindex(series.index).values
        out[f"{col}_scale"] = scale
    return out


def arima_residuals(data_in, columns=VALUE_COLS, order=(1, 1, 0), min_points=10, cache="arima_residuals.csv", n_jobs=-1):
    """
    Log residual and per-series residual scale of every value w.r.t. an
    ARIMA model of its own (Area, Item) time series (see `arima_fit_series`).
    Groups with fewer than `min_points` rows are not scored (NaN).

    Fitting ~36k models takes a few minutes, so the result is cached to
    `cache` and reused as long as the index of `data_in` is unchanged.
    """
    expected_columns = [f"{col}_resid" for col in columns] + [f"{col}_scale" for col in columns]
    if cache and os.path.exists(cache):
        cached = pd.read_csv(cache, index_col=0)
        if cached.index.equals(data_in.index) and list(cached.columns) == expected_columns:
            return cached
        print(f"Cached ARIMA residuals in {cache} do not match the data, refitting.")

    from joblib import Parallel, delayed

    groups = [group for _, group in data_in.groupby(GROUP_COLS)]
    print(f"Fitting ARIMA{order} to {len(groups)} series x {len(columns)} columns...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_arima_group_residuals)(group, columns, order, min_points) for group in groups
    )
    residuals = pd.concat(results).reindex(data_in.index)

    if cache:
        residuals.to_csv(cache)
    return residuals


def arima_outlier_mask(data_in, columns=VALUE_COLS, order=(1, 1, 0), threshold=3.5, min_scale=0.1, min_points=10, cache="arima_residuals.csv", n_jobs=-1):
    """
    Flag values whose ARIMA log residual exceeds `threshold` robust standard
    deviations of their own series:

        |residual| / max(scale, min_scale) > threshold

    `min_scale` is a floor on the per-series scale (on the log scale, so 0.1
    is roughly 10%): many series in the data are carried forward unchanged
    for years, which makes their MAD tiny and would flag every real change.
    The floor says we never assume a series to be more predictable than
    +/- min_scale per year.

    Returns a boolean DataFrame like `moving_average_outlier_mask`.
    """
    residuals = arima_residuals(data_in, columns, order, min_points, cache, n_jobs)
    mask = pd.DataFrame(False, index=data_in.index, columns=columns)
    for col in columns:
        scale = residuals[f"{col}_scale"].clip(lower=min_scale)
        z = residuals[f"{col}_resid"].abs() / scale
        mask[col] = z.gt(threshold).fillna(False).astype(bool)
    return mask


# ============================================================
# Removal
# ============================================================
def outlier_mask(data_in, method="arima", columns=VALUE_COLS, **kwargs):
    if method == "arima":
        return arima_outlier_mask(data_in, columns, **kwargs)
    if method == "moving_average":
        return moving_average_outlier_mask(data_in, columns, **kwargs)
    raise ValueError(f"Unknown outlier detection method: {method}")


def remove_outliers(data_in, method="arima", columns=VALUE_COLS, **kwargs):
    """
    Remove rows where any value column is an outlier according to `method`
    ("arima" or "moving_average", extra kwargs are passed on to the mask
    function) and print statistics about it.
    """
    mask = outlier_mask(data_in, method, columns, **kwargs)
    row_mask = mask.any(axis=1)

    total_rows = len(data_in)
    print(f"Outlier detection ({method}, {kwargs or 'default parameters'}):")
    for col in columns:
        n = mask[col].sum()
        print(f"{col}: {n} outliers ({round(n / total_rows * 100, 2)}%)")
    print(f"Rows removed: {row_mask.sum()} ({round(row_mask.sum() / total_rows * 100, 2)}%)")

    return data_in.loc[~row_mask].reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================
def plot_outlier_examples(data_in, method="arima", columns=VALUE_COLS, n_examples=3, examples=None, save_dir=None, **kwargs):
    """
    Plot example (Area, Item) time series with the model reference (moving
    average or ARIMA fitted values), the acceptance band, and the points
    flagged as outliers highlighted.

    `examples` is an optional list of (Area, Item) tuples. If not given, the
    `n_examples` groups with the most flagged outliers are used.

    If `save_dir` is given, each figure is saved there as
    "{area}_{item}_{method}.png" instead of being shown.
    """
    import matplotlib.pyplot as plt

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    mask = outlier_mask(data_in, method, columns, **kwargs)

    if examples is None:
        counts = mask.any(axis=1).groupby([data_in['Area'], data_in['Item']]).sum()
        examples = counts.sort_values(ascending=False).head(n_examples).index.tolist()

    for area, item in examples:
        group_idx = data_in.index[(data_in['Area'] == area) & (data_in['Item'] == item)]
        subset = data_in.loc[group_idx].sort_values('Year')
        subset_mask = mask.loc[subset.index]

        fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
        for ax, col in zip(axes, columns):
            years = subset['Year']
            if method == "moving_average":
                window = kwargs.get("window", 5)
                threshold = kwargs.get("threshold", 0.5)
                reference = subset[col].rolling(window, center=True, min_periods=2).mean().values
                lower, upper = reference * (1 - threshold), reference * (1 + threshold)
                reference_label = f"{window}-year moving average"
                band_label = f"±{threshold:.0%} band"
            else:
                order = kwargs.get("order", (1, 1, 0))
                threshold = kwargs.get("threshold", 3.5)
                min_scale = kwargs.get("min_scale", 0.1)
                fitted, _, scale = arima_fit_series(subset.set_index('Year')[col], order)
                reference = fitted.reindex(years).values
                half_width = threshold * max(scale, min_scale)
                lower = np.expm1(np.log1p(reference) - half_width)
                upper = np.expm1(np.log1p(reference) + half_width)
                reference_label = f"ARIMA{order} fitted"
                band_label = f"±{threshold} robust std band"

            ax.plot(years, subset[col], marker='o', markersize=3, label=col)
            ax.plot(years, reference, linestyle='--', label=reference_label)
            ax.fill_between(years, lower, upper, alpha=0.15, label=band_label)
            dropped = subset[subset_mask[col]]
            ax.scatter(dropped['Year'], dropped[col], color='red', zorder=3, label="dropped")
            ax.set_title(col)
            ax.set_xlabel("Year")
            if method == "arima":
                ax.set_yscale('log')  # the model and its band are on the log scale
            ax.legend(fontsize=7)

        fig.suptitle(f"{item} — {area} ({method})")
        plt.tight_layout()
        if save_dir:
            safe_area = area.replace("/", "-")
            safe_item = item.replace("/", "-")
            fig.savefig(os.path.join(save_dir, f"{safe_area}_{safe_item}_{method}.png"), dpi=150)
            plt.close(fig)
        else:
            plt.show()
