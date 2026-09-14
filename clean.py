import pandas as pd


def clean_data(data_in):
    value_cols = ['Area harvested', 'Yield', 'Production']
    data = data_in.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)

    rows_missing_before = data[value_cols].isna().any(axis=1).sum()

    # we can compute the corret missing value if two other values of the same row are present:
    one_missing_mask = data[value_cols].isna().sum(axis=1) == 1
    for index, row in data.loc[one_missing_mask].iterrows():
        area_harvested = row["Area harvested"]
        yld = row["Yield"]
        production = row["Production"]

        if area_harvested == 0 or yld == 0 or production == 0:
            continue

        # production (tonnes) = area_harvested (ha) * yield (hg/ha) / 10_000
        # area_harvested = production * 10_000 / yield
        # yield = production * 10_000 / area_harvested

        if pd.isna(area_harvested):
            data.loc[index, "Area harvested"] = (production * 10_000) / yld
        if pd.isna(yld):
            data.loc[index, "Yield"] = (production * 10_000) / area_harvested
        if pd.isna(production):
            data.loc[index, "Production"] = (area_harvested * yld) / 10_000

    rows_recovered = (one_missing_mask & data[value_cols].notna().all(axis=1)).sum()

    data_cleaned = data.dropna(subset=value_cols).reset_index(drop=True)
    rows_removed = len(data) - len(data_cleaned)

    total_rows = len(data)
    print("Cleaning statistics:")
    print(f"Rows with missing values: {rows_missing_before} ({round(rows_missing_before / total_rows * 100, 2)}%)")
    print(f"Rows recovered: {rows_recovered} ({round(rows_recovered / total_rows * 100, 2)}%)")
    print(f"Rows removed: {rows_removed} ({round(rows_removed / total_rows * 100, 2)}%)")

    return data_cleaned
