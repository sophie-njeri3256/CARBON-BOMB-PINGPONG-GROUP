"""Load and combine the cohort datasets required by assignment Question 7."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
FILE_PREFIX = (
    "ds_S2S_LH_MP_CO2_Carbon_Bombs_394_Permian_Delaware_Tight_"
    "all_hazards_all_"
)
YEAR_PATTERN = re.compile(r"_year_(\d{4})\.pkl$")
SPATIAL_COORDINATES = {"regions": "region", "countries": "country"}


def discover_cohort_files(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    spatial_resolution: str = "regions",
) -> list[tuple[int, Path]]:
    """Find all available cohort files for one spatial resolution."""
    if spatial_resolution not in SPATIAL_COORDINATES:
        options = ", ".join(sorted(SPATIAL_COORDINATES))
        raise ValueError(f"spatial_resolution must be one of: {options}.")

    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    pattern = f"{FILE_PREFIX}{spatial_resolution}_year_*.pkl"
    cohort_files: list[tuple[int, Path]] = []
    for path in directory.glob(pattern):
        match = YEAR_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Could not read the birth year from filename: {path.name}")
        cohort_files.append((int(match.group(1)), path))

    cohort_files.sort(key=lambda item: item[0])
    if not cohort_files:
        raise FileNotFoundError(
            f"No {spatial_resolution} cohort files found in {directory}."
        )

    years = [year for year, _ in cohort_files]
    if len(years) != len(set(years)):
        raise ValueError(f"Duplicate {spatial_resolution} cohort years were found.")
    return cohort_files


def load_cohort_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    spatial_resolution: str = "regions",
) -> dict[int, xr.Dataset]:
    """Load every provided cohort dataset at the requested spatial resolution."""
    loaded: dict[int, xr.Dataset] = {}
    expected_coordinate = SPATIAL_COORDINATES.get(spatial_resolution)
    if expected_coordinate is None:
        options = ", ".join(sorted(SPATIAL_COORDINATES))
        raise ValueError(f"spatial_resolution must be one of: {options}.")

    for birth_year, path in discover_cohort_files(data_dir, spatial_resolution):
        with path.open("rb") as dataset_file:
            dataset = pickle.load(dataset_file)
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"Expected an xarray.Dataset in {path}, got {type(dataset).__name__}.")
        if expected_coordinate not in dataset.coords:
            raise ValueError(
                f"{path.name} has no '{expected_coordinate}' coordinate."
            )
        if "extreme" not in dataset.coords or "median_val" not in dataset.data_vars:
            raise ValueError(
                f"{path.name} must contain an 'extreme' coordinate and 'median_val' variable."
            )
        loaded[birth_year] = dataset

    return loaded


def merge_cohort_datasets(
    datasets_by_year: dict[int, xr.Dataset],
    spatial_resolution: str = "regions",
) -> xr.Dataset:
    """Combine cohort datasets along an integer ``birth_years`` coordinate."""
    if spatial_resolution not in SPATIAL_COORDINATES:
        options = ", ".join(sorted(SPATIAL_COORDINATES))
        raise ValueError(f"spatial_resolution must be one of: {options}.")
    if not datasets_by_year:
        raise ValueError("At least one cohort dataset is required for merging.")

    coordinate = SPATIAL_COORDINATES[spatial_resolution]
    ordered_years = sorted(datasets_by_year)
    ordered_datasets = [datasets_by_year[year] for year in ordered_years]
    reference = ordered_datasets[0]

    for year, dataset in zip(ordered_years[1:], ordered_datasets[1:]):
        if set(dataset.data_vars) != set(reference.data_vars):
            raise ValueError(f"The {year} cohort has a different set of data variables.")
        if set(dataset.dims) != set(reference.dims):
            raise ValueError(f"The {year} cohort has different dimensions.")
        for dim in reference.dims:
            if dataset.sizes[dim] != reference.sizes[dim]:
                raise ValueError(
                    f"The {year} cohort has a different size for dimension '{dim}'."
                )
        for coord in (coordinate, "project", "extreme", "samples"):
            if coord in reference.coords and (
                coord not in dataset.coords
                or not reference.coords[coord].identical(dataset.coords[coord])
            ):
                raise ValueError(
                    f"The {year} cohort has different values for coordinate '{coord}'."
                )

    merged = xr.concat(
        ordered_datasets,
        dim=xr.IndexVariable("birth_years", ordered_years),
        data_vars="all",
        coords="minimal",
        compat="equals",
        join="exact",
    )
    merged.attrs = dict(reference.attrs)
    merged.attrs["birth_years_description"] = (
        "Birth year of each cohort; includes every cohort file available in the source data."
    )
    merged.attrs["spatial_resolution"] = spatial_resolution
    return merged


def run_question7(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    save_merged: bool = True,
) -> dict[str, xr.Dataset]:
    """Load, summarize, and optionally save merged regional and country datasets."""
    results: dict[str, xr.Dataset] = {}
    output_names = {
        "regions": "question7_regions_by_birth_year.pkl",
        "countries": "question7_countries_by_birth_year.pkl",
    }

    for resolution, coordinate in SPATIAL_COORDINATES.items():
        cohort_datasets = load_cohort_datasets(data_dir, resolution)
        print(f"\n{resolution.title()} cohorts ({len(cohort_datasets)} files):")
        for birth_year, dataset in cohort_datasets.items():
            print(
                f"  {birth_year}: dimensions={dict(dataset.sizes)}, "
                f"variables={list(dataset.data_vars)}"
            )

        merged = merge_cohort_datasets(cohort_datasets, resolution)
        results[resolution] = merged
        print(f"\nMerged {resolution} dataset:")
        print(merged)

        if save_merged:
            output_path = Path(data_dir) / output_names[resolution]
            with output_path.open("wb") as output_file:
                pickle.dump(merged, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved merged dataset to: {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and merge all available cohort datasets for Question 7."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the supplied cohort pickle files.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print and return merged datasets without writing merged pickle files.",
    )
    args = parser.parse_args()
    run_question7(args.data_dir, save_merged=not args.no_save)


if __name__ == "__main__":
    main()
