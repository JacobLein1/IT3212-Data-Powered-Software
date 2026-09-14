import pandas as pd
import numpy as np
from clean import clean_data
from outlier_detection import remove_outliers, plot_outlier_examples
from plot import plot_item_time_series, plot_all_elements_in_area, plot_item_all_areas
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


value_cols = ["Area harvested", "Production", "Yield"]


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

"""

Calculate average variance, and variance per item/year group, for Yield. Also remove outliers based on a moving average per Area/Item time series, and print the number of rows removed.

"""

# OUTLIER DETECTION!

crop1_wide_filtered = remove_outliers(data_cleaned)

#plot_outlier_examples(data_cleaned)
plot_outlier_examples(data_cleaned, examples=[("Eastern Europe", "Mushrooms and truffles")])

# ============================================================
# 1. Variasjonskoeffisient (CV) per Item/Year — bedre enn ren varians
#    fordi den er sammenlignbar på tvers av varer med ulikt Yield-nivå
# ============================================================
def yield_cv_within_items(df):
    """
    Standardavvik, gjennomsnitt og variasjonskoeffisient (CV = std/mean)
    for Yield innad i hver Item/Year-gruppe.
    Lav CV -> Yield er stabilt på tvers av land -> godt å estimere med.
    Høy CV -> stor spredning mellom land -> upålitelig å estimere med.
    """
    stats_df = df.groupby(['Item', 'Year'])['Yield'].agg(['mean', 'std', 'count'])
    stats_df['cv'] = stats_df['std'] / stats_df['mean']

    # Krever minst 2 land for at std/cv skal gi mening
    stats_df = stats_df[stats_df['count'] >= 2]

    print("10 mest USTABILE Item/Year-grupper (høyest CV):")
    print(stats_df.sort_values('cv', ascending=False).head(10))

    print("\n10 mest STABILE Item/Year-grupper (lavest CV):")
    print(stats_df.sort_values('cv', ascending=True).head(10))

    print(f"\nGjennomsnittlig CV på tvers av alle Item/Year-grupper: {stats_df['cv'].mean():.3f}")
    print(f"Median CV på tvers av alle Item/Year-grupper: {stats_df['cv'].median():.3f}")

    return stats_df


#cv_stats = yield_cv_within_items(crop1_wide_filtered)


# ============================================================
# 2. Aggregert CV per Item (på tvers av alle år) — for å se hvilke
#    varetyper generelt er "trygge" å bruke Yield-estimering på
# ============================================================
def cv_summary_per_item(cv_stats_df):
    """Gjennomsnittlig CV per Item, aggregert over alle år."""
    summary = (
        cv_stats_df
        .groupby('Item')['cv']
        .mean()
        .sort_values(ascending=False)
    )
    print("Items med høyest gjennomsnittlig CV (minst pålitelige å estimere fra):")
    print(summary.head(10))
    print("\nItems med lavest gjennomsnittlig CV (mest pålitelige å estimere fra):")
    print(summary.tail(10))
    return summary


#item_cv_summary = cv_summary_per_item(cv_stats)


# ============================================================
# 3. Direkte test av estimeringsfeil: bruk median Yield per Item/Year
#    til å estimere Production fra Area harvested, og mål feilen
#    mot faktisk Production
# ============================================================
def evaluate_yield_estimation(df):
    """
    Tester hvor god metoden 'Production = Area harvested * typisk Yield' er,
    ved å sammenligne estimert Production mot faktisk Production for
    rader der begge finnes.
    """
    df = df.dropna(subset=['Area harvested', 'Yield', 'Production']).copy()

    # "Typisk" Yield = median på tvers av land, for samme Item/Year
    df['Yield_typical'] = df.groupby(['Item', 'Year'])['Yield'].transform('median')

    # Estimer Production: production (tonnes) = area_harvested (ha) * yield (hg/ha) / 10_000
    df['Production_est'] = df['Area harvested'] * df['Yield_typical'] / 10_000

    df['abs_error'] = (df['Production_est'] - df['Production']).abs()
    df['rel_error'] = df['abs_error'] / df['Production'].replace(0, np.nan)

    print("Feilmetrikk for Production-estimering (median Yield-metode):")
    print(df['rel_error'].describe())
    print(f"\nMedian absolutt relativ feil: {df['rel_error'].median():.2%}")
    print(f"Gjennomsnittlig absolutt relativ feil: {df['rel_error'].mean():.2%}")
    print(f"Andel rader med relativ feil < 10%: {(df['rel_error'] < 0.10).mean():.2%}")
    print(f"Andel rader med relativ feil < 25%: {(df['rel_error'] < 0.25).mean():.2%}")

    return df


#estimation_results = evaluate_yield_estimation(crop1_wide_filtered)


# ============================================================
# 4. Bryt estimeringsfeilen ned per Item, for å se hvilke varer
#    metoden fungerer godt/dårlig for
# ============================================================
def estimation_error_per_item(estimation_df):
    """Median relativ feil i Production-estimering, per Item."""
    summary = (
        estimation_df
        .groupby('Item')['rel_error']
        .median()
        .sort_values(ascending=False)
    )
    print("Items med HØYEST median relativ feil (dårligst egnet for estimering):")
    print(summary.head(10))
    print("\nItems med LAVEST median relativ feil (best egnet for estimering):")
    print(summary.tail(10))
    return summary


#item_error_summary = estimation_error_per_item(estimation_results)

