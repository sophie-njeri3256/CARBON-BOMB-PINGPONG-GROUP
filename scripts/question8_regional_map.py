"""Create the Question 8 regional exposure map for the 2025 birth cohort."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm, colors
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.question7_cohort_datasets import (
    DEFAULT_DATA_DIR,
    load_cohort_datasets,
    merge_cohort_datasets,
)


DEFAULT_BORDER_PATH = DEFAULT_DATA_DIR / "gdf_country_borders.pkl"
DEFAULT_REGIONAL_DATASET_PATH = DEFAULT_DATA_DIR / "question7_regions_by_birth_year.pkl"
DEFAULT_FIGURE_PATH = PROJECT_ROOT / "figures" / "question8_regional_exposure_2025.png"
HAZARD_TITLES = {
    "burntarea": "Wildfire",
    "cropfailedarea": "Crop failure",
    "driedarea": "Drought",
    "floodedarea": "River flood",
    "heatwavedarea": "Heatwave",
    "tropicalcyclonedarea": "Tropical cyclone",
}


def load_regional_dataset(
    dataset_path: str | Path = DEFAULT_REGIONAL_DATASET_PATH,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> xr.Dataset:
    """Load Question 7's merged regional dataset, or build it from the supplied cohorts."""
    path = Path(dataset_path)
    if path.is_file():
        with path.open("rb") as dataset_file:
            dataset = pickle.load(dataset_file)
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"Expected an xarray.Dataset in {path}.")
    else:
        cohorts = load_cohort_datasets(data_dir, "regions")
        dataset = merge_cohort_datasets(cohorts, "regions")

    required_coordinates = {"birth_years", "region", "extreme"}
    missing_coordinates = required_coordinates.difference(dataset.coords)
    if missing_coordinates:
        missing = ", ".join(sorted(missing_coordinates))
        raise ValueError(f"Regional dataset is missing coordinates: {missing}")
    if "median_val" not in dataset.data_vars:
        raise ValueError("Regional dataset is missing the 'median_val' variable.")
    return dataset


def create_gdf_regions_border(
    country_borders_path: str | Path = DEFAULT_BORDER_PATH,
) -> gpd.GeoDataFrame:
    """Dissolve country polygons into the eight geographic regions in the input file."""
    path = Path(country_borders_path)
    if not path.is_file():
        raise FileNotFoundError(f"Country borders file not found: {path}")

    with path.open("rb") as borders_file:
        country_borders: Any = pickle.load(borders_file)

    if not isinstance(country_borders, gpd.GeoDataFrame):
        raise TypeError(f"Expected a GeoDataFrame in {path}.")
    if "region" not in country_borders.columns:
        raise ValueError("Country borders GeoDataFrame must contain a 'region' column.")
    if country_borders.crs is None:
        raise ValueError("Country borders GeoDataFrame must have a defined CRS.")
    if country_borders.geometry.isna().any() or country_borders.geometry.is_empty.any():
        raise ValueError("Country borders GeoDataFrame contains missing or empty geometries.")
    if country_borders["region"].isna().any():
        raise ValueError("Country borders GeoDataFrame contains countries with no region.")

    regions = country_borders.dissolve(by="region", as_index=False, sort=True)
    if regions.empty or not regions.geometry.is_valid.all():
        raise ValueError("Dissolving country borders did not produce valid region geometries.")

    return regions


def get_regional_medians(
    regional_dataset: xr.Dataset,
    extreme: str,
    birth_year: int = 2025,
) -> pd.DataFrame:
    """Return one median exposure estimate per region for the requested cohort and hazard."""
    required_coordinates = {"birth_years", "region", "extreme"}
    missing_coordinates = required_coordinates.difference(regional_dataset.coords)
    if missing_coordinates:
        missing = ", ".join(sorted(missing_coordinates))
        raise ValueError(f"Regional dataset is missing coordinates: {missing}")
    if "median_val" not in regional_dataset.data_vars:
        raise ValueError("Regional dataset is missing the 'median_val' variable.")
    if birth_year not in regional_dataset.coords["birth_years"].values:
        raise ValueError(f"Birth year {birth_year} is not available in the regional dataset.")
    if extreme not in regional_dataset.coords["extreme"].values:
        options = ", ".join(map(str, regional_dataset.coords["extreme"].values))
        raise ValueError(f"Unknown extreme '{extreme}'. Available extremes: {options}.")

    medians = regional_dataset["median_val"].sel(
        birth_years=birth_year, extreme=extreme
    ).squeeze(drop=True)
    if medians.dims != ("region",):
        raise ValueError(
            "Selecting a cohort and extreme must leave only the 'region' dimension."
        )

    values = np.asarray(medians.values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Regional median exposure estimates must all be finite.")

    return pd.DataFrame(
        {"region": medians.coords["region"].values, "median_val": values}
    )


def plot_regional_exposure_map(
    regional_dataset: xr.Dataset,
    gdf_regions_border: gpd.GeoDataFrame,
    birth_year: int = 2025,
    output_path: str | Path | None = DEFAULT_FIGURE_PATH,
) -> tuple[Figure, np.ndarray, gpd.GeoDataFrame]:
    """Map 2025 median additional exposure by region for each of the six hazards."""
    if "region" not in gdf_regions_border.columns:
        raise ValueError("gdf_regions_border must contain a 'region' column.")
    if gdf_regions_border.crs is None:
        raise ValueError("gdf_regions_border must have a defined CRS.")
    if not gdf_regions_border["region"].is_unique:
        raise ValueError("gdf_regions_border must contain one geometry per region.")

    dataset_regions = set(map(str, regional_dataset.coords["region"].values))
    border_regions = set(map(str, gdf_regions_border["region"].values))
    non_spatial_regions = {"World"}
    missing_borders = dataset_regions.difference(border_regions, non_spatial_regions)
    if missing_borders:
        missing = ", ".join(sorted(missing_borders))
        raise ValueError(f"No border geometries found for dataset regions: {missing}")
    unexpected_borders = border_regions.difference(dataset_regions)
    if unexpected_borders:
        unexpected = ", ".join(sorted(unexpected_borders))
        raise ValueError(f"Border regions are not present in the dataset: {unexpected}")

    map_regions = gdf_regions_border.loc[
        gdf_regions_border["region"].isin(dataset_regions - non_spatial_regions)
    ].copy()
    map_regions = map_regions.to_crs(epsg=8857)
    extremes = list(map(str, regional_dataset.coords["extreme"].values))
    if len(extremes) != 6:
        raise ValueError(f"Expected six climate extremes, found {len(extremes)}.")

    figure, axes = plt.subplots(
        2, 3, figsize=(17, 10), layout="constrained", squeeze=False
    )
    for axis, extreme in zip(axes.flat, extremes):
        medians = get_regional_medians(regional_dataset, extreme, birth_year)
        region_data = map_regions.merge(
            medians, on="region", how="left", validate="one_to_one", indicator=True
        )
        if (region_data["_merge"] != "both").any() or region_data["median_val"].isna().any():
            raise ValueError(
                f"Some regions have no median exposure estimate for '{extreme}'."
            )
        region_data = region_data.drop(columns="_merge")

        minimum = float(region_data["median_val"].min())
        maximum = float(region_data["median_val"].max())
        if minimum < 0 < maximum:
            normalization = colors.TwoSlopeNorm(
                vmin=minimum, vcenter=0, vmax=maximum
            )
            color_map = "RdYlBu_r"
        else:
            upper = maximum if maximum > 0 else 1.0
            normalization = colors.Normalize(vmin=min(0.0, minimum), vmax=upper)
            color_map = "YlOrRd"

        region_data.plot(
            column="median_val",
            ax=axis,
            cmap=color_map,
            norm=normalization,
            edgecolor="white",
            linewidth=0.45,
        )
        colorbar = figure.colorbar(
            cm.ScalarMappable(norm=normalization, cmap=color_map),
            ax=axis,
            shrink=0.72,
            pad=0.02,
        )
        colorbar.set_label("Median additional people exposed")
        title = HAZARD_TITLES.get(extreme, extreme.replace("_", " ").title())
        axis.set_title(title)
        axis.set_axis_off()

    figure.suptitle(
        f"Additional lifetime exposure attributable to Permian Delaware Tight\n"
        f"People born in {birth_year} | median estimate by region",
        fontsize=15,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")

    return figure, axes, map_regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the Question 8 regional exposure map for the 2025 cohort."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_REGIONAL_DATASET_PATH,
        help="Path to the merged regional birth-cohort dataset from Question 7.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Source data directory, used if the merged Question 7 dataset is absent.",
    )
    parser.add_argument(
        "--borders",
        type=Path,
        default=DEFAULT_BORDER_PATH,
        help="Path to gdf_country_borders.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Path for the high-resolution PNG figure.",
    )
    args = parser.parse_args()

    regional_dataset = load_regional_dataset(args.dataset, args.data_dir)
    gdf_regions_border = create_gdf_regions_border(args.borders)
    figure, _, map_regions = plot_regional_exposure_map(
        regional_dataset,
        gdf_regions_border,
        birth_year=2025,
        output_path=args.output,
    )
    plt.close(figure)
    print(f"Mapped {len(map_regions)} geographic regions.")
    print(f"Saved Question 8 figure to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
