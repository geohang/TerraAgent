"""
Wildfire Risk Assessment Module

This module provides Fire Weather Index (FWI) calculation functionality.
It simulates wildfire risk based on meteorological conditions.

Reference: Fire Weather Index (FWI) System - https://github.com/steidani/FireWeatherIndex
Keywords: numpy, Meteorology, Risk Map, Spatial Grid
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_fwi_risk(
    temperature: float,
    humidity: int,
    wind_speed: float,
    rain_mm: float
) -> plt.Figure:
    """
    Calculate Fire Weather Index risk and generate a risk heatmap.

    This function creates a 50x50 simulation region grid representing a forest
    and calculates wildfire risk based on meteorological inputs using a
    simplified FWI formula.

    Args:
        temperature: Air temperature in degrees Celsius.
            Higher temperatures increase fire risk.
        humidity: Relative humidity as a percentage (0-100).
            Lower humidity increases fire risk.
        wind_speed: Wind speed in km/h.
            Higher wind speeds increase fire spread risk.
        rain_mm: Precipitation in the last 24 hours in millimeters.
            Recent rainfall decreases fire risk.

    Returns:
        plt.Figure: A Matplotlib Figure containing the fire risk heatmap
            using YlOrRd colormap showing danger levels across the region.

    Example:
        >>> fig = calculate_fwi_risk(35.0, 20, 25.0, 0.0)
        >>> fig.savefig("fire_output.png")
    """
    # Spatial Grid: Create 50x50 simulation region
    grid_size = 50
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)

    # Index Calculation: Simplified FWI formula
    # Risk = (Temp × 0.4) - (Humidity × 0.2) + (Wind × 0.5) - (Rain × 2)
    base_risk = (temperature * 0.4) - (humidity * 0.2) + (wind_speed * 0.5) - (rain_mm * 2)

    # Create base risk grid (uniform)
    risk_grid = np.full((grid_size, grid_size), base_risk)

    # Terrain Noise: Add Perlin-like noise to simulate elevation effects
    np.random.seed(42)

    # Generate multi-scale noise for terrain-like variation
    noise = np.zeros((grid_size, grid_size))

    # Large scale features (hills/valleys)
    noise += 0.5 * np.sin(X * 0.1) * np.cos(Y * 0.1)

    # Medium scale features (ridges)
    noise += 0.3 * np.sin(X * 0.2 + Y * 0.15)

    # Small scale random variation
    noise += 0.2 * np.random.randn(grid_size, grid_size)

    # Normalize noise to [-1, 1] range
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1

    # Apply terrain effect (±30% variation)
    terrain_factor = 1.0 + 0.3 * noise
    risk_grid = risk_grid * terrain_factor

    # Ensure non-negative risk values
    risk_grid = np.maximum(risk_grid, 0)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)

    # Plot heatmap with YlOrRd colormap as specified
    vmax = max(50, np.max(risk_grid))
    im = ax.pcolormesh(
        X, Y, risk_grid,
        cmap='YlOrRd',
        vmin=0,
        vmax=vmax,
        shading='auto'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.85)
    cbar.set_label('Fire Weather Index', fontsize=12)

    # Determine risk level label
    mean_risk = np.mean(risk_grid)
    if mean_risk < 10:
        risk_label = "LOW"
    elif mean_risk < 20:
        risk_label = "MODERATE"
    elif mean_risk < 35:
        risk_label = "HIGH"
    elif mean_risk < 50:
        risk_label = "VERY HIGH"
    else:
        risk_label = "EXTREME"

    # Labels and title
    ax.set_xlabel('Grid X (km)', fontsize=11)
    ax.set_ylabel('Grid Y (km)', fontsize=11)
    ax.set_title(
        f'Wildfire Risk Assessment Map\n'
        f'Temp: {temperature}°C | Humidity: {humidity}% | '
        f'Wind: {wind_speed} km/h | Rain: {rain_mm} mm\n'
        f'Mean FWI: {mean_risk:.1f} | Risk Level: {risk_label}',
        fontsize=12,
        fontweight='bold'
    )

    # Add contour lines for risk zones
    contour_levels = [10, 20, 35, 50]
    valid_levels = [l for l in contour_levels if l < vmax]
    if valid_levels:
        contours = ax.contour(X, Y, risk_grid, levels=valid_levels,
                              colors='black', linewidths=0.8, alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

    # Add statistics box
    stats_text = (
        f'Max: {np.max(risk_grid):.1f}\n'
        f'Min: {np.min(risk_grid):.1f}\n'
        f'Std: {np.std(risk_grid):.1f}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Test the function and save output
    print("Running fire risk calculation test...")
    # Hot, dry, windy conditions with no rain
    fig = calculate_fwi_risk(35.0, 15, 30.0, 0.0)
    fig.savefig("fire_output.png", dpi=150, bbox_inches='tight')
    print("Output saved to fire_output.png")
    plt.show()
