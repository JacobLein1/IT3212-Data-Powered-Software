import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
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


def plot_item_time_series(df, item, country):
    elements = ["Area harvested", "Yield", "Production"]

    subset = df[(df['Item'] == item) & (df['Area'] == country)].sort_values('Year')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, element in zip(axes, elements):
        ax.plot(subset['Year'], subset[element], marker='o', markersize=2)
        ax.set_title(element)
        ax.set_xlabel("Year")

    axes[0].set_ylabel("Value")
    fig.suptitle(f"{item} — {country}")
    plt.tight_layout()
    plt.show()

#plot_item_time_series(crop1, "Almonds, with shell", "Afghanistan")

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


#plot_all_elements_in_area(crop1, "Afghanistan")

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


#plot_item_all_areas(crop1, "Almonds, with shell")


value_cols = ["Area harvested", "Production", "Yield"]


print(pd.DataFrame.info(crop1))
# Z-score method
#z_scores = np.abs(stats.zscore(crop1['Value'], nan_policy='omit'))
#outliers_z = crop1[z_scores > 3]
#print(f"Z-score outliers: {len(outliers_z)}")



for col in value_cols:
    z_scores = np.abs(
        stats.zscore(crop1_wide[col], nan_policy = "omit")
    )
    outliers_z = crop1_wide[z_scores > 3]
    print(f"{col}: {len(outliers_z)} Z-score outliers")


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


# IQR method
for col in value_cols:
     Q1 = crop1_wide[col].quantile(0.25)
     Q3 = crop1_wide[col].quantile(0.75)
     IQR = Q3 - Q1

     lower = Q1 - 1.5*IQR
     upper = Q3 + 1.5*IQR

     outliers = crop1_wide[
          (crop1_wide[col] < lower) |
          (crop1_wide[col] > upper)
     ]
     print(f"{col}: {len(outliers)} outliers")

# DATA CLEANING!

# remove startup and end-lag rows
value_cols = ['Area harvested', 'Yield', 'Production']
crop1_wide = crop1_wide.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)

is_real = ~(crop1_wide[value_cols].fillna(0) == 0).all(axis=1)
group = [crop1_wide['Area'], crop1_wide['Item']]

seen_real_from_start = is_real.groupby(group).cumsum()
seen_real_from_end = is_real[::-1].groupby(group).cumsum()[::-1]

crop1_wide_trimmed = crop1_wide[
    (seen_real_from_start > 0) & (seen_real_from_end > 0)
].reset_index(drop=True)

print(crop1_wide_trimmed.isna().sum().sum())
print(crop1_wide_trimmed)
# we can compute the corret missing value if two other values of the same row are present:
one_missing_mask = crop1_wide_trimmed[value_cols].isna().sum(axis=1) == 1
for index, row in crop1_wide_trimmed.loc[one_missing_mask].iterrows():
    area_harvested = row["Area harvested"]
    yld = row["Yield"]
    production = row["Production"]


    if area_harvested == 0 or yld == 0 or production == 0:
        continue

    # production (tonnes) = area_harvested (ha) * yield (hg/ha) / 10_000
    # area_harvested = production * 10_000 / yield
    # yield = production * 10_000 / area_harvested
    
    if pd.isna(area_harvested):
        crop1_wide_trimmed.loc[index, "Area harvested"] = (production * 10_000) / yld
    if pd.isna(yld):
        crop1_wide_trimmed.loc[index, "Yield"] = (production * 10_000) / area_harvested
    if pd.isna(production):
        crop1_wide_trimmed.loc[index, "Production"] = (area_harvested * yld) / 10_000


print("CROP1 TRIMMED!")
num_missing_vals = crop1_wide_trimmed.isna().sum().sum()
print(f"Number of missing values: {num_missing_vals}")
num_value_rows = len(crop1_wide_trimmed) * 3 #3 is number of value columns we have in the dataset
num_rows = len(crop1_wide_trimmed)
print(f"Number of rows: {num_rows}")
print("Missing values per column")
print(crop1_wide_trimmed.isna().sum())
print(f"Percentage of values that are missing: {round((num_missing_vals/num_value_rows)*100, 2)}%")
num_missing_vals_rows = crop1_wide_trimmed.isna().any(axis=1).sum()
print(f"Rows with at least one missing value: {num_missing_vals_rows}")
print(f"Percentage of rows with at least one missing value: {round((num_missing_vals_rows/num_rows)*100, 2)}%")
print(crop1_wide_trimmed)

value_cols = ['Area harvested', 'Yield', 'Production']

cols = ['Area harvested', 'Production', 'Yield']

def has_overlap(group):
    non_nan_count = group[cols].notna().sum(axis=1)  # per-row: how many of the 3 are present
    return (non_nan_count >= 2).any()  # True if at least one row has 2+ series present together

overlap_by_group = (
    crop1_wide_trimmed
    .groupby(['Area', 'Item'])
    .apply(has_overlap)
)

to_drop = overlap_by_group[~overlap_by_group].index  # groups with NO overlap anywhere
to_keep = overlap_by_group[overlap_by_group].index

print(f"Dropping {len(to_drop)} Area+Item series with no cross-column overlap")
print(f"Keeping {len(to_keep)} series")

# Filter the dataframe
mask = crop1_wide_trimmed.set_index(['Area', 'Item']).index.isin(to_drop)
crop1_wide_cleaned = crop1_wide_trimmed[~mask].reset_index(drop=True)

print(crop1_wide_cleaned)


# Count NaNs
nan_counts = (
    crop1_wide_cleaned
    .groupby(['Area', 'Item'])[value_cols]
    .apply(lambda g: g.isna().sum().sum())
    .sort_values(ascending=False)
)

# Grab the worst offender
top_area, top_item = nan_counts.index[0]
worst_series = crop1_wide_trimmed[(crop1_wide_trimmed['Area'] == top_area) & (crop1_wide_trimmed['Item'] == top_item)].sort_values('Year')

print(nan_counts)

plot_item_time_series(crop1_wide_cleaned, "Hemp tow waste", "Germany")
plot_item_time_series(crop1_wide_cleaned, "Rubber, natural", "Bolivia (Plurinational State of)")

"""

Calculate average variance, and variance per item/year group, for Yield. Also remove outliers based on percentiles, and print the number of rows before and after filtering.

"""

crop1_wide_clean = crop1_wide_trimmed.dropna(subset=['Yield'])
def filter_within_percentile(df, column, confidence=0.90, by=None):
    tail = (1 - confidence) / 2
    if by:
        lower = df.groupby(by)[column].transform(lambda x: x.quantile(tail))
        upper = df.groupby(by)[column].transform(lambda x: x.quantile(1 - tail))
    else:
        lower = df[column].quantile(tail)
        upper = df[column].quantile(1 - tail)
    return df[(df[column] >= lower) & (df[column] <= upper)]

crop1_wide_filtered = filter_within_percentile(crop1_wide_clean, 'Yield', confidence=0.99, by='Item')

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

