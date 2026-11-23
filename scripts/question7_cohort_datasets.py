# Short code for Question 7.

import pickle                         # Reads and saves Python objects.
import re                             # Finds the birth year inside file names.
from pathlib import Path             # Builds paths to files and folders.

import xarray as xr                  # Handles multidimensional datasets.


ROOT = Path(__file__).resolve().parents[1]  # Finds the project root.
DATA = ROOT / "Data"                         # Finds the data folder.
PREFIX = "ds_S2S_LH_MP_CO2_Carbon_Bombs_394_Permian_Delaware_Tight_all_hazards_all_"  # Common file-name text.
YEAR = re.compile(r"_year_(\d{4})\.pkl$")  # Extracts four-digit years from file names.


def find_files(spatial):
    """Find and sort all files for regions or countries."""
    files = []                        # Starts an empty list of (year, path).
    for path in DATA.glob(f"{PREFIX}{spatial}_year_*.pkl"):  # Finds matching files.
        match = YEAR.search(path.name)  # Searches the file name for a birth year.
        if match:                      # Runs only when a year is found.
            files.append((int(match.group(1)), path))  # Stores year and path.
    return sorted(files)               # Sorts files from the oldest to newest year.


def load_cohorts(spatial):
    """Load every cohort file into a dictionary."""
    cohorts = {}                      # Starts an empty year-to-dataset dictionary.
    coordinate = "region" if spatial == "regions" else "country"  # Chooses the spatial coordinate.

    for birth_year, path in find_files(spatial):  # Loops through each found file.
        with path.open("rb") as file:  # Opens the file for binary reading.
            dataset = pickle.load(file)  # Loads the saved xarray Dataset.
        cohorts[birth_year] = dataset  # Stores the Dataset under its birth year.
    return cohorts                     # Returns the complete dictionary.


def merge_cohorts(cohorts, spatial):
    """Join all cohorts and add a birth_years coordinate."""
    years = sorted(cohorts)            # Orders the available birth years.
    datasets = [cohorts[year] for year in years]  # Gets datasets in the same order.

    merged = xr.concat(               # Stacks all cohort datasets together.
        datasets,
        dim=xr.IndexVariable("birth_years", years),
    )
    merged.attrs["spatial_resolution"] = spatial  # Records whether values are regional or national.
    return merged                      # Returns the combined Dataset.


def run_question7():
    """Load, merge, print, and save regional and country datasets."""
    results = {}                       # Starts an empty dictionary for final datasets.

    for spatial in ("regions", "countries"):  # Processes both resolutions.
        cohorts = load_cohorts(spatial)  # Loads all files for this resolution.
        print(f"\n{spatial.title()}: {len(cohorts)} cohorts")  # Prints the cohort count.

        for year, dataset in cohorts.items():  # Checks every loaded cohort.
            print(year, dict(dataset.sizes), list(dataset.data_vars))  # Prints dimensions and variables.

        merged = merge_cohorts(cohorts, spatial)  # Combines all cohorts.
        print(merged)                  # Displays the final merged Dataset.
        results[spatial] = merged      # Stores the merged Dataset.

        output = DATA / f"question7_{spatial}_by_birth_year.pkl"  # Chooses the output file.
        with output.open("wb") as file:  # Opens the output for binary writing.
            pickle.dump(merged, file, protocol=pickle.HIGHEST_PROTOCOL)  # Saves the merged Dataset.
        print("Saved:", output)        # Reports the saved file.

    return results                    # Returns both merged datasets.


if __name__ == "__main__":
    run_question7()                   # Runs the complete Question 7 process.
