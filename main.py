import os

import pandas as pd
import numpy as np
from clean import clean_data
from outlier_detection import remove_outliers, plot_outlier_examples
from transform import transform_data, one_hot_encode
from plot import plot_item_time_series, plot_all_elements_in_area, plot_item_all_areas
from plot import (
    plot_raw_distributions,
    plot_log_distributions,
    plot_log_z_distributions,
    plot_transform_steps_histograms,
    plot_transform_steps_qq,
    plot_transform_steps_boxplots,
    plot_item_transform_steps,
)
from sklearn.decomposition import PCA

crop1 = pd.read_csv("food.bank/crop1.csv")

print(crop1.head())
print(crop1.info())
print(crop1.describe(include="object"))

crop1_wide = (
    crop1.pivot(
        index=['Area', 'Item', 'Year'],
        columns='Element',
        values='Value',
    )
    .reset_index()
    .rename_axis(columns=None)
)

print(crop1_wide)
num_missing_vals = crop1_wide.isna().sum().sum()
print(f"Number of missing values: {num_missing_vals}")
num_value_rows = len(crop1_wide) * 3 #3 is number of value columns we have in the dataset
num_rows = len(crop1_wide)
print(f"Number of rows: {num_rows}")
print("Missing values per collumn")
print(crop1_wide.isna().sum())
print(f"Percentage of values that are missing: {round((num_missing_vals/num_value_rows)*100, 2)}%")
num_missing_vals_rows = crop1_wide.isna().any(axis=1).sum()
print(f"Rows with at least one missing value: {num_missing_vals_rows}")
print(f"Percentage of rows with at least one missing value: {round((num_missing_vals_rows/num_rows)*100, 2)}%")
print(crop1_wide)


#plot_item_time_series(crop1, "Almonds, with shell", "Afghanistan")

#plot_all_elements_in_area(crop1, "Afghanistan")

#plot_item_all_areas(crop1, "Almonds, with shell")


def gini_impurity(df, column):
    """Gini impurity of the value distribution in `column`."""
    probs = df[column].value_counts(normalize=True)
    gini = 1 - np.sum(probs ** 2)
    return gini


"""print(f"Gini impurity of Item: {gini_impurity(crop1, 'Item'):.4f}")
print(f"Gini impurity of Area: {gini_impurity(crop1, 'Area'):.4f}")
print(f"Gini impurity of Unit: {gini_impurity(crop1, 'Unit'):.4f}")
print(f"Gini impurity of Element: {gini_impurity(crop1, 'Element'):.4f}")
print(f"Gini impurity of Item (wide): {gini_impurity(crop1_wide, 'Item'):.4f}")
print(f"Gini impurity of Area (wide): {gini_impurity(crop1_wide, 'Area'):.4f}")
"""

def variance(df, columns):
    """Variance of one or more numeric columns."""
    return df[columns].var()

print(variance(crop1, ['Value', 'Year']))


# DATA CLEANING!
data_cleaned = clean_data(crop1_wide.copy())

# OUTLIER DETECTION!

# ARIMA residuals per Area/Item series; the fits are cached in arima_residuals.csv after the first run
crop1_wide_filtered = remove_outliers(data_cleaned, method="arima", threshold=3.5, min_scale=0.1)
#crop1_wide_filtered = remove_outliers(data_cleaned, method="moving_average", window=5, threshold=0.5)

# plot_outlier_examples(data_cleaned, method="arima", threshold=3.5, min_scale=0.1)
# plot_outlier_examples(data_cleaned, method="moving_average", window=5, threshold=0.5)
plot_outlier_examples(data_cleaned, method="arima", threshold=3.5, min_scale=0.1, examples=[("Botswana", "Maize"), ("Eastern Europe", "Mushrooms and truffles"), ("Afghanistan", "Sugar beet")], save_dir="plots/outlier_examples_arima")
plot_outlier_examples(data_cleaned, method="moving_average",  window=5, threshold=0.5, examples=[("Botswana", "Maize"), ("Eastern Europe", "Mushrooms and truffles"), ("Afghanistan", "Sugar beet")], save_dir="plots/outlier_examples_moving_average")
# plot_outlier_examples(data_cleaned, method="arima", examples=[("Botswana", "Maize"), ("Eastern Europe", "Mushrooms and truffles")])
# plot_outlier_examples(data_cleaned, method="moving_average", examples=[("Botswana", "Maize"), ("Eastern Europe", "Mushrooms and truffles")])

# TRAIN-TEST SPLIT
split_year = 2008 
min_train_years = 10

grouped_years = crop1_wide_filtered.groupby(["Area", "Item"])["Year"]
spans_both = (grouped_years.transform("min") <= split_year) & (grouped_years.transform("max") > split_year)
train_year_count = (crop1_wide_filtered["Year"] <= split_year).groupby(
    [crop1_wide_filtered["Area"], crop1_wide_filtered["Item"]]
).transform("sum")
keep = spans_both & (train_year_count >= min_train_years)
crop1_panel = crop1_wide_filtered.loc[keep].reset_index(drop=True)
n_series_before = grouped_years.ngroups
n_series_after = crop1_panel.groupby(["Area", "Item"]).ngroups
print(f"Panel filter (both periods, >= {min_train_years} train years): kept {n_series_after}/{n_series_before} series")
print(f"Rows: {len(crop1_wide_filtered)} -> {len(crop1_panel)}")

train_raw = crop1_panel.loc[crop1_panel["Year"] <= split_year].reset_index(drop=True)
test_raw = crop1_panel.loc[crop1_panel["Year"] > split_year
].reset_index(drop=True)

# DATA TRANSFORMATION!
train_transformed, z_params = transform_data(train_raw)
test_transformed, _ = transform_data(test_raw, z_params=z_params)

# ONE-HOT ENCODING!
train_encoded = one_hot_encode(train_transformed)
test_encoded = one_hot_encode(test_transformed).reindex(columns=train_encoded.columns, fill_value=0)

target_cols = ["Area harvested", "Yield", "Production"]
derived_cols = [c for c in train_encoded.columns if c.endswith(("_log", "_log_z"))]
X_train = train_encoded.drop(columns=target_cols + derived_cols)
y_train = train_encoded[target_cols]
X_test = test_encoded.drop(columns=target_cols + derived_cols)
y_test = test_encoded[target_cols]

print(f"Train: {X_train.shape[0]} rows (years <= {split_year})")
print(f"Test:  {X_test.shape[0]} rows (years > {split_year})")
print(X_train)


plot_outlier_examples(
    data_cleaned,
    method="arima",
    examples=[("Botswana", "Maize"), ("Afghanistan", "Sugar beet")],
    save_dir="plots/outlier_examples_arima",
)
plot_outlier_examples(
    data_cleaned,
    method="moving_average",
    examples=[("Botswana", "Maize"), ("Afghanistan", "Sugar beet")],
    save_dir="plots/outlier_examples_moving_average",
)
# plot_outlier_examples(data_cleaned, method="arima", threshold=3.5, min_scale=0.1)
# plot_outlier_examples(data_cleaned, method="moving_average", window=5, threshold=0.5)
# plot_outlier_examples(data_cleaned, method="arima", examples=[("Botswana", "Maize"), ("Eastern Europe", "Mushrooms and truffles")])
# plot_outlier_examples(data_cleaned, method="moving_average", examples=[("Botswana", "Maize"), ("Eastern Europe", "Mushrooms and truffles")])


# PLOTS (from train_transformed, fitted on training data only)

os.makedirs("plots", exist_ok=True)
plot_raw_distributions(train_transformed, save_path="plots/raw_distributions.png")
plot_log_distributions(train_transformed, save_path="plots/log_distributions.png")
plot_log_z_distributions(train_transformed, save_path="plots/log_z_distributions.png")
plot_transform_steps_histograms(train_transformed, save_path="plots/transform_steps_histograms.png")
plot_transform_steps_qq(train_transformed, save_path="plots/transform_steps_qq.png")
plot_transform_steps_boxplots(train_transformed, save_path="plots/transform_steps_boxplots.png")
plot_item_transform_steps(train_transformed, item="Wheat", save_path="plots/wheat_transform_steps.png")

from sklearn.decomposition import PCA

pca_cols = [
    "area_harvested_log_z",
    "yield_log_z",
    "production_log_z"
]

# Fit PCA only on training data
pca = PCA(n_components=3)

train_pca = pca.fit_transform(train_transformed[pca_cols])
test_pca = pca.transform(test_transformed[pca_cols])

print("Explained variance:")
print(pca.explained_variance_ratio_)

print("Cumulative explained variance:")
print(pca.explained_variance_ratio_.cumsum())

print("PCA components:")
print(pca.components_)

# Create PCA columns
pca_cols = ["area_harvested_log_z",
            "yield_log_z",
            "production_log_z"
]
# Fit PCA only on the training data

pca = PCA(n_components=3)

train_pca = pca.fit_transform(train_transformed[pca_cols])
test_pca = pca.transform(test_transformed[pca_cols])

print("Explained variance from PCA:")
print(pca.explained_variance_ratio_)

print("Cumulative explained variance from PCA:")
print(pca.explained_variance_ratio_.cumsum())

print("PCA components:")
print(pca.components_)

pca = PCA(n_components=2)

train_pca = pca.fit_transform(train_transformed[pca_cols])

test_pca = pca.transform(test_transformed[pca_cols])

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print("Explain variance ratio sum:")
print(pca.explained_variance_ratio.sum())