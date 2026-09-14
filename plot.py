import matplotlib.pyplot as plt


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
