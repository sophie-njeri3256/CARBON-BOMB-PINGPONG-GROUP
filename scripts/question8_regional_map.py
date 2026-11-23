# Short code for Question 8.

import pickle                         # Reads saved datasets and border maps.
from pathlib import Path             # Finds project folders and files.

import geopandas as gpd              # Handles geographic country and region polygons.
import matplotlib.pyplot as plt      # Creates the map figure.
import pandas as pd                  # Creates simple tables for medians.


ROOT = Path(__file__).resolve().parents[1]  # Finds the project root.
DATA = ROOT / "Data"                         # Finds the data folder.
REGIONAL_DATA = DATA / "question7_regions_by_birth_year.pkl"  # Merged regional data.
BORDERS = DATA / "gdf_country_borders.pkl"  # Country-border GeoDataFrame.
OUTPUT = ROOT / "figures" / "question8_regional_exposure_2025.png"  # Figure location.

HAZARDS = {                           # Matches data names with beginner-readable titles.
    "burntarea": "Wildfire",
    "cropfailedarea": "Crop failure",
    "driedarea": "Drought",
    "floodedarea": "River flood",
    "heatwavedarea": "Heatwave",
    "tropicalcyclonedarea": "Tropical cyclone",
}


def load_data():
    """Load the merged regional data and dissolve countries into regions."""
    with REGIONAL_DATA.open("rb") as file:  # Opens the regional Dataset file.
        regional = pickle.load(file)         # Loads the xarray Dataset.

    with BORDERS.open("rb") as file:         # Opens the country-border file.
        countries = pickle.load(file)        # Loads the GeoDataFrame.

    regions = countries.dissolve(by="region")  # Joins country shapes with the same region.
    return regional, regions                 # Sends data and regional polygons onward.


def plot_regional_map():
    """Plot the median 2025 exposure for each hazard and region."""
    regional, regions = load_data()          # Loads both required inputs.
    regions = regions.reset_index()          # Adds region names as a normal column.
    regions = regions[regions["region"] != "World"]  # Removes the World aggregate.

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)  # Creates six map panels.

    for axis, extreme in zip(axes.flat, regional["extreme"].values):  # Loops through six hazards.
        values = regional["median_val"].sel(  # Selects median exposure values.
            birth_years=2025,
            extreme=extreme,
        )
        table = pd.DataFrame({               # Converts xarray values into a table.
            "region": regional["region"].values,
            "median_val": values.values.squeeze(),
        })
        map_data = regions.merge(table, on="region")  # Adds exposure values to regional shapes.

        map_data.plot(                       # Draws the regional choropleth map.
            column="median_val",
            cmap="RdYlBu_r",
            legend=True,
            ax=axis,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set_title(HAZARDS.get(extreme, extreme))  # Sets the panel title.
        axis.set_axis_off()                  # Removes latitude and longitude axes.

    figure.suptitle("Additional exposure for people born in 2025")  # Adds one main title.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)  # Creates the figures folder if needed.
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")  # Saves a high-resolution PNG.
    plt.close(figure)                        # Frees memory after saving.
    return OUTPUT                            # Returns the saved image path.


if __name__ == "__main__":
    saved = plot_regional_map()             # Runs the complete Question 8 map.
    print("Saved:", saved)                  # Prints the final image location.
