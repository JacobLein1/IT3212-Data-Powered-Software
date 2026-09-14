import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_item_time_series(df, item, country):
    elements = ["Area harvested", "Yield", "Production"]

    subset = df[(df['Item'] == item) & (df['Area'] == country)].sort_values('Year')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, element in zip(axes, elements):
        ax.plot(subset['Year'], subset[element], marker='o', markersize=2)
        ax.set_title(element)
        ax.set_xlabel("Year")

    axes[0].set_ylabel("Value")
    fig.suptitle(item)
    plt.tight_layout()
    plt.show()


def plot_transform_steps_histograms(data, save_path=None):
    """Histograms of each transformation step: raw -> log1p -> log1p+z.

    Raw panels use a log-scaled y axis, otherwise the pathological skew
    hides everything but a single spike at zero. The log1p and log1p+z
    panels have identical shape on purpose: z-scoring changes only the
    axis (mean 0, std 1), never the shape.
    """
    value_cols = ["Area harvested", "Yield", "Production"]
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    log_z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for i, (col, log_col, z_col) in enumerate(zip(value_cols, log_cols, log_z_cols)):
        stages = [
            (data[col].dropna(), "raw", col, "tab:blue"),
            (data[log_col].dropna(), "log1p", log_col, "tab:orange"),
            (data[z_col].dropna(), "log1p + z", z_col, "tab:green"),
        ]
        for j, (values, stage, label, color) in enumerate(stages):
            ax = axes[i, j]
            if stage == "raw":
                ax.hist(values, bins=80, color=color, edgecolor="none")
                ax.set_yscale("log")
                ax.set_ylabel("count (log scale)")
            else:
                ax.hist(values, bins=80, density=True, color=color, edgecolor="none")
                ax.set_ylabel("density")
                if stage == "log1p + z":
                    xs = np.linspace(values.min(), values.max(), 200)
                    ax.plot(xs, stats.norm.pdf(xs), "k--", linewidth=1, label="normal pdf")
                    ax.legend()
            ax.set_xlabel(label)
            ax.set_title(
                f"{col} — {stage}\n(skew {stats.skew(values):.2f}, kurt {stats.kurtosis(values):.2f})"
            )

    fig.suptitle("Transformation steps: raw -> log1p -> log1p + z-score")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_transform_steps_qq(data, save_path=None):
    """QQ plots vs the normal distribution at each transformation step."""
    value_cols = ["Area harvested", "Yield", "Production"]
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    log_z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(data), size=min(len(data), 20000), replace=False)
    sample = data.iloc[idx]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, (col, log_col, z_col) in enumerate(zip(value_cols, log_cols, log_z_cols)):
        stats.probplot(sample[col].dropna(), dist="norm", plot=axes[i, 0])
        axes[i, 0].set_title(f"{col} — raw")
        stats.probplot(sample[log_col].dropna(), dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f"{col} — log1p")
        stats.probplot(sample[z_col].dropna(), dist="norm", plot=axes[i, 2])
        axes[i, 2].set_title(f"{col} — log1p + z")

    fig.suptitle("QQ plots vs normal: raw -> log1p -> log1p + z-score")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_transform_steps_boxplots(data, save_path=None):
    """Boxplots at each transformation step: raw -> log1p -> log1p+z.

    A random sample of 10000 rows is used so the thousands of outlier
    flier points stay readable.
    """
    value_cols = ["Area harvested", "Yield", "Production"]
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    log_z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]
    sample = data.sample(10000, random_state=42)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for i, (col, log_col, z_col) in enumerate(zip(value_cols, log_cols, log_z_cols)):
        axes[i, 0].boxplot(sample[col].dropna(), tick_labels=["raw"])
        axes[i, 0].set_yscale("log")
        axes[i, 0].set_title(f"{col} — raw (log-scaled y)")

        axes[i, 1].boxplot(sample[log_col].dropna(), tick_labels=["log1p"])
        axes[i, 1].set_title(f"{col} — log1p")

        axes[i, 2].boxplot(sample[z_col].dropna(), tick_labels=["log1p + z"])
        axes[i, 2].set_title(f"{col} — log1p + z")

    fig.suptitle("Boxplots: raw -> log1p -> log1p + z-score")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_item_transform_steps(data, item, areas=None, save_path=None):
    """One item, several areas: each element as raw / log1p / log1p+z (columns).

    Shows why log helps: raw panels are dominated by the largest producer,
    log panels reveal the dynamics of all of them.
    """
    subset = data[data["Item"] == item]
    elements = ["Area harvested", "Yield", "Production"]
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]
    subset = subset.dropna(subset=elements)

    if areas is None:
        areas = ["China", "India", "United States", "Norway", "Nigeria"]
    areas = [a for a in areas if a in set(subset["Area"].unique())]
    if not areas:
        areas = subset.groupby("Area")["Production"].mean().nlargest(5).index.tolist()

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex="col")
    for i, (element, log_col, z_col) in enumerate(zip(elements, log_cols, z_cols)):
        element_data = subset[subset["Area"].isin(areas)]
        for area, group in element_data.groupby("Area"):
            group = group.sort_values("Year")
            axes[i, 0].plot(group["Year"], group[element], label=area)
            axes[i, 1].plot(group["Year"], group[log_col], label=area)
            axes[i, 2].plot(group["Year"], group[z_col], label=area)

        axes[i, 0].set_ylabel(element)
        axes[i, 0].set_title(f"{element} — raw")
        axes[i, 1].set_title(f"{element} — log1p")
        axes[i, 2].set_title(f"{element} — log1p + z")

    axes[0, 2].legend(fontsize=8)
    fig.suptitle(f"{item}: raw -> log1p -> log1p + z-score")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_all_elements_in_area(df, area):
    """For one area: one subplot per element, with one line per item (crop) over time."""
    subset = df[df['Area'] == area]
    elements = subset['Element'].unique()

    fig, axes = plt.subplots(1, len(elements), figsize=(6 * len(elements), 4))
    if len(elements) == 1:
        axes = [axes]

    for ax, element in zip(axes, elements):
        element_data = subset[subset['Element'] == element]
        for item, group in element_data.groupby('Item'):
            group = group.sort_values('Year')
            ax.plot(group['Year'], group['Value'], marker='o', markersize=2, label=item)
        ax.set_title(element)
        ax.set_xlabel("Year")

    axes[0].set_ylabel("Value")
    axes[-1].legend(fontsize=6, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(area)
    plt.tight_layout()
    plt.show()


def plot_item_all_areas(df, item):
    """For one item: one subplot per element, with one line per area over time."""
    subset = df[df['Item'] == item]
    elements = subset['Element'].unique()

    fig, axes = plt.subplots(1, len(elements), figsize=(6 * len(elements), 4))
    if len(elements) == 1:
        axes = [axes]

    for ax, element in zip(axes, elements):
        element_data = subset[subset['Element'] == element]
        for area, group in element_data.groupby('Area'):
            group = group.sort_values('Year')
            ax.plot(group['Year'], group['Value'], marker='o', markersize=2, label=area)
        ax.set_title(element)
        ax.set_xlabel("Year")

    axes[0].set_ylabel("Value")
    axes[-1].legend(fontsize=6, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(item)
    plt.tight_layout()
    plt.show()

def find_and_plot_partially_missing_series(df, value_cols=None, start=0, end=None):
    """Find Area/Item series with at least one Year where exactly two of the
    element values are missing simultaneously, plot each one (elements in
    subplots, side by side), and highlight those years in red.

    `start`/`end` select which slice of the found series to plot (like
    Python slicing), so you can page through the results instead of always
    plotting from the beginning."""
    if value_cols is None:
        value_cols = ["Area harvested", "Yield", "Production"]

    two_missing_mask = df[value_cols].isna().sum(axis=1) == 2
    affected = df.loc[two_missing_mask, ["Area", "Item"]].drop_duplicates()

    print(f"Found {len(affected)} Area/Item series with at least one year missing two of three values")

    affected = affected.iloc[start:end]

    for _, row in affected.iterrows():
        area, item = row["Area"], row["Item"]
        subset = df[(df["Area"] == area) & (df["Item"] == item)].sort_values("Year")
        missing_years = subset.loc[subset[value_cols].isna().sum(axis=1) == 2, "Year"]

        fig, axes = plt.subplots(1, len(value_cols), figsize=(5 * len(value_cols), 4))
        if len(value_cols) == 1:
            axes = [axes]

        for ax, element in zip(axes, value_cols):
            ax.plot(subset["Year"], subset[element], marker="o", markersize=2)
            for year in missing_years:
                ax.axvspan(year - 0.5, year + 0.5, color="red", alpha=0.15)
            ax.set_title(element)
            ax.set_xlabel("Year")

        axes[0].set_ylabel("Value")
        fig.suptitle(f"{item} — {area} (years missing two of three values highlighted)")
        plt.tight_layout()
        plt.show()

    return affected

def plot_raw_distributions(data, save_path=None):
    """Histograms of the raw value columns, before any transformation.

    The y axis is log-scaled so the long right tail is visible at all;
    on a linear axis everything but a single spike at zero disappears.
    """
    value_cols = ["Area harvested", "Yield", "Production"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, value_cols):
        values = data[col].dropna()
        ax.hist(values, bins=80, edgecolor="none")
        ax.set_yscale("log")
        ax.set_title(
            f"{col}"
        )
        ax.set_xlabel(col)
        ax.set_ylabel("count (log scale)")
        stats_text = (
            f"mean = {values.mean():.3g}\n"
            f"std = {values.std():.3g}\n"
            f"median = {values.median():.3g}\n"
            f"min = {values.min():g}, max = {values.max():.3g}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    fig.suptitle("Value distributions")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_log_distributions(data, save_path=None):
    """Histograms of the log1p columns, one panel per value column."""
    log_cols = ["area_harvested_log", "yield_log", "production_log"]
    labels = ["Area harvested", "Yield", "Production"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, label in zip(axes, log_cols, labels):
        values = data[col].dropna()
        ax.hist(values, bins=80, density=True, edgecolor="none", color="tab:orange")
        xs = np.linspace(values.min(), values.max(), 200)
        ax.plot(
            xs,
            stats.norm.pdf(xs, values.mean(), values.std()),
            "k--",
            linewidth=1,
            label="fitted normal",
        )
        ax.set_title(f"{label}")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        stats_text = (
            f"mean = {values.mean():.3g}\n"
            f"std = {values.std():.3g}\n"
            f"median = {values.median():.3g}\n"
            f"min = {values.min():.3g}, max = {values.max():.3g}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    fig.suptitle("Value distributions post log-transform")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_log_z_distributions(data, save_path=None):
    """Histograms of the log1p+z columns, one panel per value column."""
    log_z_cols = ["area_harvested_log_z", "yield_log_z", "production_log_z"]
    labels = ["Area harvested", "Yield", "Production"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, label in zip(axes, log_z_cols, labels):
        values = data[col].dropna()
        ax.hist(values, bins=80, density=True, edgecolor="none", color="tab:green")
        xs = np.linspace(values.min(), values.max(), 200)
        ax.plot(
            xs,
            stats.norm.pdf(xs, values.mean(), values.std()),
            "k--",
            linewidth=1,
            label="fitted normal",
        )
        ax.set_title(f"{label}")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        stats_text = (
            f"mean = {values.mean():.3g}\n"
            f"std = {values.std():.3g}\n"
            f"median = {values.median():.3g}\n"
            f"min = {values.min():.3g}, max = {values.max():.3g}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    fig.suptitle("Value distributions post z-score transform")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
