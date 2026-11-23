# Short code for Questions 5 and 6.

import pickle                         # Reads and writes Python .pkl files.
from pathlib import Path             # Finds files and folders using clear paths.

import matplotlib.pyplot as plt      # Creates charts.
import numpy as np                   # Creates numbers and x-axis positions.
import pandas as pd                  # Creates and sorts tables.


ROOT = Path(__file__).resolve().parents[1]  # Finds the project folder.
DATA = ROOT / "Data" / "ds_carbon_bombs.pkl"  # Location of the carbon-bomb dataset.
FIGURES = ROOT / "figures"                   # Folder for saved charts.


def load_dataset():
    """Load the already-preprocessed carbon-bomb dataset."""
    with DATA.open("rb") as file:      # Opens the file for reading in binary mode.
        dataset = pickle.load(file)    # Converts the .pkl back into a Python object.
    return dataset                     # Sends the xarray Dataset to the caller.


def make_table(dataset):
    """Convert one xarray Dataset into a simple Pandas table."""
    emissions = dataset["potential_emissions"]  # Selects the emission variable.

    table = pd.DataFrame({
        "name": dataset["name"].values,         # Gets every project name.
        "country": dataset["country"].values,   # Gets every country name.
        "fuel": dataset["fuel"].values,         # Gets every fuel type.
        "best": emissions.sel(estimate="best").values,  # Gets the best estimate.
        "low": emissions.sel(estimate="low").values,    # Gets the low estimate.
        "high": emissions.sel(estimate="high").values,  # Gets the high estimate.
    })
    return table                          # Returns the table to plotting functions.


def plot_top_20(dataset):
    """Plot the 20 projects with the largest best estimates."""
    table = make_table(dataset)           # Converts xarray data to a table.
    top = table.nlargest(20, "best")      # Keeps the 20 largest projects.
    top = top.reset_index(drop=True)      # Restarts row numbers at zero.

    x = np.arange(len(top))               # Makes one x position for each project.
    best = top["best"].values             # Gets the bar heights.
    error = np.array([
        best - top["low"].values,         # Distance from best down to low.
        top["high"].values - best,        # Distance from best up to high.
    ])

    figure, axis = plt.subplots(figsize=(18, 8))  # Creates a chart area.
    bars = axis.bar(x, best, yerr=error, capsize=3)  # Draws bars and error bars.
    axis.set_xticks(x)                   # Places ticks below each bar.
    axis.set_xticklabels(top["name"], rotation=75, ha="right")  # Writes project names.
    axis.set_ylabel("Potential emissions (GtCO2)")  # Labels the vertical axis.
    axis.set_title("20 Largest Carbon Bomb Projects")  # Gives the chart a title.
    axis.grid(axis="y", linestyle="--", alpha=0.3)  # Adds faint horizontal lines.
    figure.tight_layout()                # Removes large empty margins.

    output = FIGURES / "top_20_carbon_bombs.png"  # Sets the output file name.
    figure.savefig(output, dpi=300, bbox_inches="tight")  # Saves a high-resolution PNG.
    return figure, top                    # Sends the chart and table to the notebook.


def plot_country_totals(dataset):
    """Plot total best-estimate emissions for each country."""
    table = make_table(dataset)           # Converts xarray data to a table.
    totals = table.groupby("country")["best"].sum()  # Sums emissions by country.
    totals = totals.sort_values().tail(15)  # Keeps the 15 largest countries.

    figure, axis = plt.subplots(figsize=(10, 7))  # Creates a chart area.
    totals.plot(kind="barh", ax=axis)     # Draws a horizontal bar chart.
    axis.set_xlabel("Summed potential emissions (GtCO2)")  # Labels the x-axis.
    axis.set_ylabel("Country")            # Labels the y-axis.
    axis.set_title("Carbon Bomb Emissions by Country")  # Gives the chart a title.
    axis.grid(axis="x", linestyle="--", alpha=0.3)  # Adds faint vertical lines.
    figure.tight_layout()                 # Removes large empty margins.

    output = FIGURES / "carbon_bombs_by_country.png"  # Sets the output file name.
    figure.savefig(output, dpi=300, bbox_inches="tight")  # Saves the PNG.
    return figure, totals                 # Sends the chart and totals to the notebook.


def plot_fuel_totals(dataset):
    """Plot total best-estimate emissions for each fuel type."""
    table = make_table(dataset)           # Converts xarray data to a table.
    totals = table.groupby("fuel")["best"].sum().sort_values()  # Sums by fuel.

    figure, axis = plt.subplots(figsize=(8, 5))  # Creates a chart area.
    totals.plot(kind="bar", ax=axis)      # Draws a vertical bar chart.
    axis.set_xlabel("Fuel type")          # Labels the x-axis.
    axis.set_ylabel("Summed potential emissions (GtCO2)")  # Labels the y-axis.
    axis.set_title("Carbon Bomb Emissions by Fuel")  # Gives the chart a title.
    axis.grid(axis="y", linestyle="--", alpha=0.3)  # Adds faint horizontal lines.
    figure.tight_layout()                 # Removes large empty margins.

    output = FIGURES / "carbon_bombs_by_fuel.png"  # Sets the output file name.
    figure.savefig(output, dpi=300, bbox_inches="tight")  # Saves the PNG.
    return figure, totals                 # Sends the chart and totals to the notebook.


if __name__ == "__main__":
    dataset = load_dataset()               # Loads the saved carbon-bomb Dataset.
    plot_top_20(dataset)                   # Creates the top-20 chart.
    plot_country_totals(dataset)           # Creates the country chart.
    plot_fuel_totals(dataset)              # Creates the fuel chart.
    print("Saved all three Task 5/6 figures to:", FIGURES)  # Reports success.
