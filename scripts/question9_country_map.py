# Short code for Question 9.

import pickle                         # Reads saved Python objects.
import sys                            # Lets the script find project modules.
from pathlib import Path             # Builds file and folder paths.

import geopandas as gpd              # Handles country polygons.
import matplotlib.pyplot as plt      # Creates maps.
import pycountry                     # Converts ordinary country names to ISO3 codes.

ROOT = Path(__file__).resolve().parents[1]  # Finds the project folder.
if str(ROOT) not in sys.path:          # Checks whether the project folder is searchable.
    sys.path.insert(0, str(ROOT))      # Adds the project folder to Python's search list.

from scripts.question7_cohort_datasets import load_cohorts, merge_cohorts  # Reuses Question 7 code.


DATA = ROOT / "Data"                         # Finds the data folder.
COUNTRY_DATA = DATA / "question7_countries_by_birth_year.pkl"  # Merged country Dataset.
BORDERS = DATA / "gdf_country_borders.pkl"  # Country polygons.
OUTPUT = ROOT / "figures" / "question9_country_exposure_2025.png"  # Saved figure.

HAZARDS = {                           # Converts data names into readable titles.
    "burntarea": "Wildfire",
    "cropfailedarea": "Crop failure",
    "driedarea": "Drought",
    "floodedarea": "River flood",
    "heatwavedarea": "Heatwave",
    "tropicalcyclonedarea": "Tropical cyclone",
}

ALIASES = {                           # Handles names that do not match pycountry directly.
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


def to_iso3(name):
    """Convert a country name into the three-letter code used by the borders."""
    if name in ALIASES:               # Checks if the name needs a manual exception.
        return ALIASES[name]          # Returns the manually assigned ISO3 code.
    return pycountry.countries.lookup(name).alpha_3  # Finds an ordinary ISO3 code.


def load_country_data():
    """Load the merged country Dataset or create it from all country cohort files."""
    if COUNTRY_DATA.is_file():        # Checks whether Question 7 already saved the Dataset.
        with COUNTRY_DATA.open("rb") as file:  # Opens the saved Dataset.
            return pickle.load(file)  # Returns the loaded xarray Dataset.

    cohorts = load_cohorts("countries")  # Loads every country cohort file.
    dataset = merge_cohorts(cohorts, "countries")  # Adds the birth_years dimension.
    with COUNTRY_DATA.open("wb") as file:  # Opens the output file.
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)  # Saves it.
    return dataset                    # Returns the newly combined Dataset.


def plot_country_map():
    """Plot median 2025 exposure for each hazard and country."""
    dataset = load_country_data()     # Loads country exposure data.
    with BORDERS.open("rb") as file:  # Opens the country-border file.
        borders = pickle.load(file)   # Loads the GeoDataFrame.
    borders = borders.rename(columns={"abbreviation": "iso3"})  # Uses a clear column name.

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)  # Creates six maps.

    for axis, extreme in zip(axes.flat, dataset["extreme"].values):  # Loops over all hazards.
        medians = dataset["median_val"].sel(  # Selects median values.
            birth_years=2025,
            extreme=extreme,
        ).squeeze()

        values = {}                   # Starts an empty country-code-to-value dictionary.
        for country in medians["country"].values:  # Loops over 183 country names.
            values[to_iso3(country)] = float(medians.sel(country=country).values)  # Converts and stores the median.

        map_data = borders.copy()     # Copies the country map before adding values.
        map_data["median"] = map_data["iso3"].map(values)  # Matches codes and adds exposure values.

        map_data.plot(                # Draws the country choropleth map.
            column="median",
            cmap="RdYlBu_r",
            legend=True,
            ax=axis,
            edgecolor="white",
            linewidth=0.2,
            missing_kwds={"color": "lightgrey"},
        )
        axis.set_title(HAZARDS.get(extreme, extreme))  # Writes the hazard name.
        axis.set_axis_off()           # Removes map axes.

    figure.suptitle("Additional exposure for people born in 2025")  # Adds the main title.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)  # Creates the figures folder.
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")  # Saves a high-resolution PNG.
    plt.close(figure)                 # Frees memory after saving.
    return OUTPUT                     # Returns the final image path.


if __name__ == "__main__":
    saved = plot_country_map()        # Runs the complete Question 9 map.
    print("Saved:", saved)            # Prints the image location.
