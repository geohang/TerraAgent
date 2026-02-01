"""
Climate Change Simulation Module

This module provides climate simulation functionality based on ClimateBench methodology.
It simulates global temperature anomalies based on emission scenarios and time projections.

Reference: ClimateBench - https://github.com/duncanwp/ClimateBench
Keywords: xarray, NetCDF, Temperature Anomaly, Heatmap
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_warming(year: int, emission_scenario: str) -> plt.Figure:
    """
    Simulate global warming and generate a temperature anomaly heatmap.

    This function simulates global temperature changes based on the input year
    and emission scenario. It incorporates polar amplification effects where
    poles warm approximately 2x faster than equatorial regions.

    Args:
        year: The target year for simulation (e.g., 2030, 2050, 2100).
            Warming is calculated as delta from present (2024).
        emission_scenario: The CO2 emission pathway to simulate.
            Options: "Low" (0.5x modifier), "Medium" (1.0x), "High" (1.5x).

    Returns:
        plt.Figure: A Matplotlib Figure containing the global temperature
            anomaly heatmap with colorbar showing temperature changes in °C.

    Example:
        >>> fig = simulate_warming(2050, "High")
        >>> fig.savefig("climate_output.png")
    """
    # Grid Generation: Create global lat/lon grid (180x360)
    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(-180, 180, 360)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Emission scenario modifiers
    scenario_modifiers = {
        "Low": 0.5,
        "Medium": 1.0,
        "High": 1.5
    }
    modifier = scenario_modifiers.get(emission_scenario, 1.0)

    # Physics Simulation: Calculate warming based on year delta from present
    present_year = 2024
    years_delta = max(0, year - present_year)

    # Base warming rate: ~0.03°C per year (simplified climate sensitivity)
    base_warming = years_delta * 0.03 * modifier

    # Polar Amplification: Poles warm 2x faster than equator
    # Factor ranges from 1.0 at equator to 2.0 at poles
    polar_factor = 1.0 + np.abs(lat_grid) / 90.0

    # Calculate temperature anomaly grid
    temperature_anomaly = base_warming * polar_factor

    # Add some spatial variation for realism (simplified pattern)
    np.random.seed(42)
    spatial_noise = np.random.normal(0, 0.1, temperature_anomaly.shape)
    temperature_anomaly += spatial_noise * base_warming

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

    # Plot heatmap with diverging colormap
    vmax = max(2.0, np.max(temperature_anomaly))
    im = ax.pcolormesh(
        lon, lat, temperature_anomaly,
        cmap='RdYlBu_r',
        vmin=0,
        vmax=vmax,
        shading='auto'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Temperature Anomaly (°C)', fontsize=12)

    # Labels and title
    ax.set_xlabel('Longitude (°)', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title(
        f'Global Temperature Anomaly Projection\n'
        f'Year: {year} | Scenario: {emission_scenario} | '
        f'Global Mean: +{np.mean(temperature_anomaly):.2f}°C',
        fontsize=14,
        fontweight='bold'
    )

    # Add grid lines
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle='--', alpha=0.3, color='white')

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Test the function and save output
    print("Running climate simulation test...")
    fig = simulate_warming(2050, "High")
    fig.savefig("climate_output.png", dpi=150, bbox_inches='tight')
    print("Output saved to climate_output.png")
    plt.show()
