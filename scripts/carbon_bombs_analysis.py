"""Figures for Tasks 5 and 6 of the carbon bombs assignment."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Data" / "ds_carbon_bombs.pkl"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
ESTIMATE_LABELS = ("best", "low", "high")


def load_carbon_bombs_dataset(path: str | Path = DEFAULT_DATASET_PATH):
    """Load the preprocessed xarray Dataset saved by Task 4."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Carbon bombs dataset not found: {dataset_path}")

    with dataset_path.open("rb") as dataset_file:
        dataset = pickle.load(dataset_file)

    required_coordinates = {"name", "country", "fuel", "estimate"}
    missing_coordinates = required_coordinates.difference(dataset.coords)
    if missing_coordinates:
        missing = ", ".join(sorted(missing_coordinates))
        raise ValueError(f"Dataset is missing required coordinates: {missing}")
    if "potential_emissions" not in dataset.data_vars:
        raise ValueError("Dataset is missing the 'potential_emissions' variable.")

    return dataset


def carbon_bombs_to_frame(dataset) -> pd.DataFrame:
    """Convert the project coordinates and emission estimates to one row per project."""
    required_coordinates = {"name", "country", "fuel", "estimate"}
    missing_coordinates = required_coordinates.difference(dataset.coords)
    if missing_coordinates:
        missing = ", ".join(sorted(missing_coordinates))
        raise ValueError(f"Dataset is missing required coordinates: {missing}")
    if "potential_emissions" not in dataset.data_vars:
        raise ValueError("Dataset is missing the 'potential_emissions' variable.")

    estimates = set(dataset.coords["estimate"].values.tolist())
    missing_estimates = set(ESTIMATE_LABELS).difference(estimates)
    if missing_estimates:
        missing = ", ".join(sorted(missing_estimates))
        raise ValueError(f"Dataset is missing required estimate labels: {missing}")

    emissions = dataset["potential_emissions"]
    if not {"name", "estimate"}.issubset(emissions.dims):
        raise ValueError(
            "'potential_emissions' must have both 'name' and 'estimate' dimensions."
        )

    frame = pd.DataFrame(
        {
            "name": dataset.coords["name"].values,
            "country": dataset.coords["country"].values,
            "fuel": dataset.coords["fuel"].values,
            "best": emissions.sel(estimate="best").values,
            "low": emissions.sel(estimate="low").values,
            "high": emissions.sel(estimate="high").values,
        }
    )

    if frame.empty:
        raise ValueError("The carbon bombs dataset contains no projects.")
    if frame[["name", "country", "fuel", *ESTIMATE_LABELS]].isna().any().any():
        raise ValueError("Project names, countries, fuels, and estimates cannot be missing.")
    if not frame["name"].is_unique:
        raise ValueError("Project names must be unique to create project-level figures.")
    if not np.isfinite(frame[["best", "low", "high"]].to_numpy(dtype=float)).all():
        raise ValueError("Emission estimates must all be finite numbers.")
    if (frame["low"] > frame["best"]).any() or (frame["best"] > frame["high"]).any():
        raise ValueError("Each project must satisfy low <= best <= high.")

    return frame


def plot_top_20(
    dataset,
    output_path: str | Path | None = DEFAULT_FIGURES_DIR / "top_20_carbon_bombs.png",
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot the 20 projects with the largest best estimates and their uncertainty ranges."""
    frame = carbon_bombs_to_frame(dataset)
    top_projects = frame.nlargest(20, "best").reset_index(drop=True)
    if top_projects.empty:
        raise ValueError("No projects are available to plot.")

    countries = sorted(top_projects["country"].unique())
    country_colors = {
        country: plt.get_cmap("tab20")(index % 20)
        for index, country in enumerate(countries)
    }
    fuel_hatches = {
        fuel: ("xx" if "coal" in str(fuel).lower() else "//")
        for fuel in sorted(top_projects["fuel"].unique())
    }

    x = np.arange(len(top_projects))
    best = top_projects["best"].to_numpy(dtype=float)
    y_error = np.vstack(
        [
            best - top_projects["low"].to_numpy(dtype=float),
            top_projects["high"].to_numpy(dtype=float) - best,
        ]
    )
    bar_colors = [country_colors[country] for country in top_projects["country"]]

    fig, ax = plt.subplots(figsize=(23, 9))
    bars = ax.bar(
        x,
        best,
        yerr=y_error,
        capsize=3,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
        error_kw={"elinewidth": 1, "ecolor": "black"},
    )
    for bar, fuel in zip(bars, top_projects["fuel"]):
        bar.set_hatch(fuel_hatches[fuel])

    ax.set_xticks(x)
    ax.set_xticklabels(top_projects["name"], rotation=70, ha="right")
    ax.set_ylabel("Potential emissions (GtCO2)")
    ax.set_title("20 Largest Carbon Bomb Projects by Potential Emissions")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    country_handles = [
        Patch(facecolor=country_colors[country], edgecolor="black", label=country)
        for country in countries
    ]
    fig.legend(
        handles=country_handles,
        title="Country",
        loc="upper left",
        bbox_to_anchor=(0.74, 0.9),
        bbox_transform=fig.transFigure,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0,
        fontsize=9,
        frameon=False,
    )
    fuel_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=fuel_hatches[fuel], label=fuel)
        for fuel in sorted(fuel_hatches)
    ]
    fig.legend(
        handles=fuel_handles,
        title="Fuel (hatch)",
        loc="upper left",
        bbox_to_anchor=(0.74, 0.59),
        bbox_transform=fig.transFigure,
        borderaxespad=0,
        fontsize=9,
        frameon=False,
    )
    fig.subplots_adjust(left=0.06, right=0.71, bottom=0.36, top=0.91)

    if output_path is not None:
        _save_figure(fig, output_path)
    return fig, ax, top_projects


def plot_country_totals(
    dataset,
    top_n: int = 15,
    output_path: str | Path | None = DEFAULT_FIGURES_DIR / "carbon_bombs_by_country.png",
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot summed best-estimate emissions for the countries with the largest totals."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    frame = carbon_bombs_to_frame(dataset)
    totals = (
        frame.groupby("country", as_index=False)
        .agg(total_best=("best", "sum"), project_count=("name", "count"))
        .sort_values("total_best", ascending=False)
        .head(top_n)
        .sort_values("total_best", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(totals["country"], totals["total_best"], color="#3977a8")
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xlabel("Summed best-estimate potential emissions (GtCO2)")
    ax.set_ylabel("Country")
    ax.set_title(f"Carbon Bomb Potential Emissions by Country (Top {len(totals)})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)
    return fig, ax, totals.sort_values("total_best", ascending=False).reset_index(drop=True)


def plot_fuel_totals(
    dataset,
    output_path: str | Path | None = DEFAULT_FIGURES_DIR / "carbon_bombs_by_fuel.png",
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Compare summed best-estimate emissions and project counts across fuel types."""
    frame = carbon_bombs_to_frame(dataset)
    totals = (
        frame.groupby("fuel", as_index=False)
        .agg(total_best=("best", "sum"), project_count=("name", "count"))
        .sort_values("total_best", ascending=False)
        .reset_index(drop=True)
    )
    colors = [
        "#d9822b" if "coal" in str(fuel).lower() else "#3977a8"
        for fuel in totals["fuel"]
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(totals["fuel"], totals["total_best"], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    for bar, count in zip(bars, totals["project_count"]):
        ax.annotate(
            f"{count} projects",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Summed best-estimate potential emissions (GtCO2)")
    ax.set_xlabel("Fuel type")
    ax.set_title("Carbon Bomb Potential Emissions by Fuel Type")
    ax.set_ylim(0, totals["total_best"].max() * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)
    return fig, ax, totals


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def create_all_figures(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, pd.DataFrame]:
    """Create and save the Task 5 and Task 6 figures from the saved dataset."""
    dataset = load_carbon_bombs_dataset(dataset_path)
    figures_path = Path(figures_dir)

    top_fig, _, top_projects = plot_top_20(
        dataset, figures_path / "top_20_carbon_bombs.png"
    )
    country_fig, _, country_totals = plot_country_totals(
        dataset, output_path=figures_path / "carbon_bombs_by_country.png"
    )
    fuel_fig, _, fuel_totals = plot_fuel_totals(
        dataset, output_path=figures_path / "carbon_bombs_by_fuel.png"
    )
    plt.close(top_fig)
    plt.close(country_fig)
    plt.close(fuel_fig)

    return {
        "top_20": top_projects,
        "country_totals": country_totals,
        "fuel_totals": fuel_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the carbon bombs figures for assignment Tasks 5 and 6."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the preprocessed ds_carbon_bombs pickle.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where the high-resolution PNG figures are saved.",
    )
    args = parser.parse_args()

    summaries = create_all_figures(args.dataset, args.figures_dir)
    print(f"Saved figures to: {args.figures_dir.resolve()}")
    print("\nTop 20 projects (best estimate, GtCO2):")
    print(summaries["top_20"][["name", "country", "fuel", "best", "low", "high"]].to_string(index=False))
    print("\nCountry totals (best estimate, GtCO2):")
    print(summaries["country_totals"].to_string(index=False))
    print("\nFuel totals (best estimate, GtCO2):")
    print(summaries["fuel_totals"].to_string(index=False))


if __name__ == "__main__":
    main()
