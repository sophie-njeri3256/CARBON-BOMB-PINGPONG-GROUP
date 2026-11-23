"""Create the Question 9 country exposure map for the 2025 birth cohort."""

import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pycountry
import xarray as xr
from matplotlib import colors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.question7_cohort_datasets import (
    load_cohort_datasets,
    merge_cohort_datasets,
)


DATA_DIR = PROJECT_ROOT / "Data"
COUNTRY_DATASET = DATA_DIR / "question7_countries_by_birth_year.pkl"
BORDERS_FILE = DATA_DIR / "gdf_country_borders.pkl"
OUTPUT_FILE = PROJECT_ROOT / "figures" / "question9_country_exposure_2025.png"

HAZARD_NAMES = {
    "burntarea": "Wildfire",
    "cropfailedarea": "Crop failure",
    "driedarea": "Drought",
    "floodedarea": "River flood",
    "heatwavedarea": "Heatwave",
    "tropicalcyclonedarea": "Tropical cyclone",
}

COUNTRY_ALIASES = {
    "Bolivia (Plurinational State of)": "BOL",
    "Democratic Republic of the Congo": "COD",
    "China, Hong Kong SAR": "HKG",
    "Iran (Islamic Republic of)": "IRN",
    "Dem. People's Republic of Korea": "PRK",
    "Republic of Korea": "KOR",
    "Kosovo (under UNSC res. 1244)": "XKX",
    "China, Taiwan Province of China": "TWN",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "United States Virgin Islands": "VIR",
    "State of Palestine": "PSE",
}


def country_name_to_iso3(name):
    """Convert a dataset country name to the ISO3 code used by the border file."""
    if name in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[name]
    return pycountry.countries.lookup(name).alpha_3


def load_country_dataset():
    """Load the merged country dataset, or create it from the supplied cohorts."""
    if COUNTRY_DATASET.is_file():
        with COUNTRY_DATASET.open("rb") as file:
            return pickle.load(file)

    cohorts = load_cohort_datasets(DATA_DIR, "countries")
    dataset = merge_cohort_datasets(cohorts, "countries")
    with COUNTRY_DATASET.open("wb") as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def plot_country_exposure_map(dataset):
    """Plot median 2025 additional exposure for six hazards by country."""
    with BORDERS_FILE.open("rb") as file:
        borders = pickle.load(file)

    if not isinstance(borders, gpd.GeoDataFrame):
        raise TypeError("gdf_country_borders.pkl must contain a GeoDataFrame.")

    borders = borders.rename(columns={"abbreviation": "iso3"})

    figure, axes = plt.subplots(
        2, 3, figsize=(18, 10), constrained_layout=True
    )

    for axis, extreme in zip(axes.flat, dataset.coords["extreme"].values):
        medians = (
            dataset["median_val"]
            .sel(birth_years=2025, extreme=extreme)
            .squeeze()
        )

        values = {}
        for country_name in medians.coords["country"].values:
            iso3 = country_name_to_iso3(country_name)
            values[iso3] = float(
                medians.sel(country=country_name).values
            )

        map_data = borders.copy()
        map_data["median_val"] = map_data["iso3"].map(values)
        minimum = map_data["median_val"].min()
        maximum = map_data["median_val"].max()

        if minimum < 0 < maximum:
            cmap = "RdYlBu_r"
            norm = colors.TwoSlopeNorm(
                vmin=minimum, vcenter=0, vmax=maximum
            )
        else:
            cmap = "YlOrRd"
            norm = colors.Normalize(vmin=min(0, minimum), vmax=maximum)

        map_data.plot(
            column="median_val",
            cmap=cmap,
            norm=norm,
            ax=axis,
            edgecolor="white",
            linewidth=0.2,
            missing_kwds={"color": "lightgrey"},
        )
        axis.set_title(HAZARD_NAMES.get(extreme, extreme))
        axis.set_axis_off()

        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axis,
            shrink=0.65,
        )
        colorbar.set_label("Median additional people exposed")

    figure.suptitle(
        "Additional lifetime exposure attributable to Permian Delaware Tight\n"
        "People born in 2025 | median estimate by country",
        fontsize=15,
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT_FILE


if __name__ == "__main__":
    country_dataset = load_country_dataset()
    saved_file = plot_country_exposure_map(country_dataset)
    print(f"Saved Question 9 figure to: {saved_file}")
