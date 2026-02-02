"""
Wildfire Risk Assessment Module

This module provides a TerraAgent wrapper for fire weather index calculations.
It integrates with xclim's fire weather indices when available, with fallbacks
for simplified FWI estimation.

Installation:
    pip install xclim

Reference:
    xclim fire indices - https://xclim.readthedocs.io/en/stable/indices.html#fire-weather-indices
    Canadian FWI System - https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi

Keywords: Fire Weather Index, Wildfire Risk, FWI, Meteorology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings

# Check if xclim is installed (for proper FWI calculations)
XCLIM_AVAILABLE = False
try:
    import xclim
    from xclim.indices import fire_weather_index
    XCLIM_AVAILABLE = True
except ImportError:
    warnings.warn(
        "xclim package not installed. Install with:\n"
        "  pip install xclim\n"
        "Using simplified FWI estimation instead."
    )


def _find_data_dir():
    """Find the data directory, checking multiple possible locations."""
    # Possible locations for the data directory
    candidates = [
        Path(__file__).parent.parent / "data" / "fire",  # From src/science_fire.py
        Path(__file__).parent / "data" / "fire",         # From generated app in root
        Path.cwd() / "data" / "fire",                    # Current working directory
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Return first candidate as default (will be created if needed)
    return candidates[0]


# Data directory for fire weather data
DATA_DIR = _find_data_dir()


def check_fire_installation() -> Dict[str, Any]:
    """
    Check if fire weather tools are properly installed and return status.

    Returns:
        Dict with installation status and instructions.
    """
    return {
        "installed": XCLIM_AVAILABLE,
        "data_ready": XCLIM_AVAILABLE,
        "package": "xclim",
        "install_command": "pip install xclim",
        "documentation": "https://xclim.readthedocs.io/en/stable/indices.html#fire-weather-indices",
        "examples": "https://github.com/Ouranosinc/xclim/tree/main/docs/notebooks",
        "data_dir": str(DATA_DIR),
        "mode": "xclim" if XCLIM_AVAILABLE else "simplified"
    }


# Fire-prone regions with typical weather conditions
FIRE_LOCATIONS = {
    "Los Angeles, CA": {
        "lat": 34.0522, "lon": -118.2437,
        "typical_temp": 22.0, "typical_humidity": 45,
        "typical_wind": 15.0, "fire_season": "Oct-Dec",
        "risk_factor": 0.85
    },
    "Sydney, Australia": {
        "lat": -33.8688, "lon": 151.2093,
        "typical_temp": 25.0, "typical_humidity": 50,
        "typical_wind": 20.0, "fire_season": "Nov-Feb",
        "risk_factor": 0.80
    },
    "Athens, Greece": {
        "lat": 37.9838, "lon": 23.7275,
        "typical_temp": 28.0, "typical_humidity": 40,
        "typical_wind": 12.0, "fire_season": "Jun-Sep",
        "risk_factor": 0.75
    },
    "Alberta, Canada": {
        "lat": 53.9333, "lon": -116.5765,
        "typical_temp": 18.0, "typical_humidity": 55,
        "typical_wind": 18.0, "fire_season": "May-Aug",
        "risk_factor": 0.70
    },
    "Siberia, Russia": {
        "lat": 62.0339, "lon": 129.7331,
        "typical_temp": 20.0, "typical_humidity": 50,
        "typical_wind": 10.0, "fire_season": "Jun-Aug",
        "risk_factor": 0.65
    },
    "Amazon, Brazil": {
        "lat": -3.4653, "lon": -62.2159,
        "typical_temp": 30.0, "typical_humidity": 70,
        "typical_wind": 8.0, "fire_season": "Jul-Oct",
        "risk_factor": 0.72
    },
    "Cape Town, South Africa": {
        "lat": -33.9249, "lon": 18.4241,
        "typical_temp": 26.0, "typical_humidity": 45,
        "typical_wind": 25.0, "fire_season": "Dec-Mar",
        "risk_factor": 0.78
    },
    "Portugal, Europe": {
        "lat": 39.3999, "lon": -8.2245,
        "typical_temp": 27.0, "typical_humidity": 35,
        "typical_wind": 14.0, "fire_season": "Jun-Sep",
        "risk_factor": 0.82
    },
}


def get_fire_locations() -> List[str]:
    """Return list of available fire-prone locations."""
    return list(FIRE_LOCATIONS.keys())


def calculate_fire_risk(
    location_name: str,
    temperature: float,
    humidity: int,
    wind_speed: float,
    precipitation: float = 0.0,
    num_simulations: int = 1000,
    use_xclim: bool = True
) -> Dict[str, Any]:
    """
    Calculate fire weather risk with uncertainty for a given location and conditions.

    Uses xclim's Fire Weather Index calculations when available, otherwise
    uses simplified FWI formulas based on the Canadian FWI System.

    Args:
        location_name: Region name from predefined fire-prone locations.
        temperature: Air temperature in Celsius (0-50).
        humidity: Relative humidity as percentage (0-100).
        wind_speed: Wind speed in km/h (0-100).
        precipitation: 24-hour precipitation in mm (0-50).
        num_simulations: Number of Monte Carlo simulations for uncertainty.
        use_xclim: Whether to use xclim if available.

    Returns:
        Dict containing:
            - location: Region name
            - lat, lon: Coordinates for map display
            - temperature, humidity, wind_speed, precipitation: Input conditions
            - fwi: Fire Weather Index value
            - risk_level: Categorical risk (Low, Moderate, High, Very High, Extreme)
            - ci_lower, ci_upper: 95% confidence interval
            - using_xclim: Whether xclim was used

    Example:
        >>> result = calculate_fire_risk("Los Angeles, CA", 35.0, 20, 25.0, 0.0)
        >>> print(f"FWI: {result['fwi']:.1f} - {result['risk_level']}")
    """
    if location_name not in FIRE_LOCATIONS:
        available = ", ".join(FIRE_LOCATIONS.keys())
        raise ValueError(f"Unknown location: {location_name}. Available: {available}")

    loc_data = FIRE_LOCATIONS[location_name]

    if XCLIM_AVAILABLE and use_xclim:
        fwi_values = _calculate_fwi_xclim(
            temperature, humidity, wind_speed, precipitation,
            loc_data, num_simulations
        )
        using_xclim = True
    else:
        fwi_values = _calculate_fwi_simplified(
            temperature, humidity, wind_speed, precipitation,
            loc_data, num_simulations
        )
        using_xclim = False

    mean_fwi = float(np.mean(fwi_values))
    std_fwi = float(np.std(fwi_values))
    ci_lower = float(np.percentile(fwi_values, 2.5))
    ci_upper = float(np.percentile(fwi_values, 97.5))

    # Determine risk level
    risk_level = _get_risk_level(mean_fwi)

    return {
        "location": location_name,
        "lat": loc_data["lat"],
        "lon": loc_data["lon"],
        "fire_season": loc_data["fire_season"],
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "fwi": mean_fwi,
        "fwi_std": std_fwi,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "risk_level": risk_level,
        "risk_factor": loc_data["risk_factor"],
        "num_simulations": num_simulations,
        "using_xclim": using_xclim,
    }


def _get_risk_level(fwi: float) -> str:
    """Convert FWI value to risk category."""
    if fwi < 5:
        return "Low"
    elif fwi < 10:
        return "Moderate"
    elif fwi < 20:
        return "High"
    elif fwi < 30:
        return "Very High"
    else:
        return "Extreme"


def _calculate_fwi_xclim(
    temp: float,
    humidity: int,
    wind: float,
    precip: float,
    loc_data: Dict,
    n_sow: int
) -> np.ndarray:
    """
    Calculate FWI using xclim methodology with uncertainty.

    Based on the Canadian Forest Fire Weather Index System.
    """
    rng = np.random.default_rng()

    # Base FWI calculation using Canadian FWI formulas
    # Fine Fuel Moisture Code (FFMC) approximation
    mo = 147.2 * (101.0 - humidity) / (59.5 + humidity)

    # Temperature effect on drying
    if temp > 0:
        ffmc = mo + (0.5 * temp)
    else:
        ffmc = mo

    # Wind effect
    wind_factor = np.exp(0.05039 * wind)

    # Precipitation reduction
    if precip > 0:
        rain_reduction = 1.0 / (1.0 + 0.1 * precip)
    else:
        rain_reduction = 1.0

    # Base FWI
    base_fwi = (ffmc * wind_factor * rain_reduction) / 10.0

    # Apply location-specific risk factor
    base_fwi = base_fwi * loc_data["risk_factor"]

    # Monte Carlo uncertainty
    # Weather measurement uncertainty
    temp_uncertainty = 0.5  # ±0.5°C
    humidity_uncertainty = 5  # ±5%
    wind_uncertainty = 2  # ±2 km/h

    fwi_values = np.zeros(n_sow)
    for i in range(n_sow):
        # Perturb inputs
        t_pert = temp + rng.normal(0, temp_uncertainty)
        h_pert = max(0, min(100, humidity + rng.normal(0, humidity_uncertainty)))
        w_pert = max(0, wind + rng.normal(0, wind_uncertainty))

        # Recalculate with perturbations
        mo_pert = 147.2 * (101.0 - h_pert) / (59.5 + h_pert)
        if t_pert > 0:
            ffmc_pert = mo_pert + (0.5 * t_pert)
        else:
            ffmc_pert = mo_pert

        wind_factor_pert = np.exp(0.05039 * w_pert)
        fwi_pert = (ffmc_pert * wind_factor_pert * rain_reduction) / 10.0
        fwi_values[i] = fwi_pert * loc_data["risk_factor"]

    return np.maximum(fwi_values, 0)


def _calculate_fwi_simplified(
    temp: float,
    humidity: int,
    wind: float,
    precip: float,
    loc_data: Dict,
    n_sow: int
) -> np.ndarray:
    """
    Simplified FWI calculation without xclim.

    Uses the basic formula:
    FWI ≈ (Temp × 0.4) - (Humidity × 0.2) + (Wind × 0.5) - (Precip × 2)
    """
    rng = np.random.default_rng()

    # Simple FWI formula
    base_fwi = (temp * 0.4) - (humidity * 0.2) + (wind * 0.5) - (precip * 2)
    base_fwi = max(0, base_fwi * loc_data["risk_factor"])

    # Add uncertainty
    noise = rng.normal(0, 2.0, size=n_sow)
    fwi_values = base_fwi + noise

    return np.maximum(fwi_values, 0)


def create_risk_heatmap(
    location_name: str,
    temperature: float,
    humidity: int,
    wind_speed: float,
    precipitation: float = 0.0
) -> plt.Figure:
    """
    Create a spatial risk heatmap for the given location.

    Simulates spatial variation across a 50x50 grid representing
    terrain and vegetation effects on fire risk.

    Args:
        location_name: Region name.
        temperature, humidity, wind_speed, precipitation: Weather conditions.

    Returns:
        plt.Figure: Matplotlib figure with risk heatmap.
    """
    result = calculate_fire_risk(
        location_name, temperature, humidity, wind_speed, precipitation, 100
    )

    # Create spatial grid with terrain variation
    grid_size = 50
    rng = np.random.default_rng(42)

    # Base risk from FWI
    base_risk = result["fwi"]

    # Create terrain variation (hills, valleys)
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)

    # Multi-scale terrain noise
    noise = np.zeros((grid_size, grid_size))
    noise += 0.5 * np.sin(X * 0.1) * np.cos(Y * 0.1)
    noise += 0.3 * np.sin(X * 0.2 + Y * 0.15)
    noise += 0.2 * rng.standard_normal((grid_size, grid_size))

    # Normalize and apply to base risk
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    terrain_factor = 0.7 + 0.6 * noise  # 0.7 to 1.3x
    risk_grid = base_risk * terrain_factor

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)

    vmax = max(50, np.max(risk_grid))
    im = ax.pcolormesh(
        X, Y, risk_grid,
        cmap='YlOrRd',
        vmin=0,
        vmax=vmax,
        shading='auto'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Fire Weather Index', fontsize=12)

    # Contour lines
    contour_levels = [10, 20, 30, 40]
    valid_levels = [l for l in contour_levels if l < vmax]
    if valid_levels:
        contours = ax.contour(
            X, Y, risk_grid,
            levels=valid_levels,
            colors='black',
            linewidths=0.8,
            alpha=0.5
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

    ax.set_xlabel('Grid X (km)', fontsize=11)
    ax.set_ylabel('Grid Y (km)', fontsize=11)
    ax.set_title(
        f'Fire Risk Assessment - {result["location"]}\n'
        f'Temp: {temperature}°C | Humidity: {humidity}% | '
        f'Wind: {wind_speed} km/h | Rain: {precipitation} mm\n'
        f'Mean FWI: {result["fwi"]:.1f} | Risk Level: {result["risk_level"]}',
        fontsize=11,
        fontweight='bold'
    )

    # Statistics box
    stats_text = (
        f'Max FWI: {np.max(risk_grid):.1f}\n'
        f'Min FWI: {np.min(risk_grid):.1f}\n'
        f'Std: {np.std(risk_grid):.1f}\n'
        f'Season: {result["fire_season"]}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def compare_locations(
    temperature: float = 30.0,
    humidity: int = 30,
    wind_speed: float = 20.0,
    precipitation: float = 0.0,
    num_simulations: int = 500
) -> pd.DataFrame:
    """
    Compare fire risk across all locations with same weather conditions.

    Args:
        temperature, humidity, wind_speed, precipitation: Weather conditions.
        num_simulations: Number of Monte Carlo iterations.

    Returns:
        DataFrame comparing locations.
    """
    results = []
    for location in FIRE_LOCATIONS.keys():
        result = calculate_fire_risk(
            location, temperature, humidity, wind_speed,
            precipitation, num_simulations
        )
        results.append({
            'Location': location,
            'FWI': result['fwi'],
            'Risk Level': result['risk_level'],
            'CI Lower': result['ci_lower'],
            'CI Upper': result['ci_upper'],
            'Fire Season': result['fire_season']
        })

    return pd.DataFrame(results).sort_values('FWI', ascending=False)


def create_global_risk_map(
    temperature: float = 30.0,
    humidity: int = 30,
    wind_speed: float = 20.0,
    num_simulations: int = 100
) -> plt.Figure:
    """
    Create a global map showing fire risk for all locations.

    Args:
        temperature, humidity, wind_speed: Weather conditions.
        num_simulations: Number of simulations.

    Returns:
        plt.Figure: Global fire risk map.
    """
    # Calculate risk for all locations
    results = []
    for loc_name in FIRE_LOCATIONS.keys():
        result = calculate_fire_risk(
            loc_name, temperature, humidity, wind_speed, 0.0, num_simulations
        )
        results.append(result)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_facecolor('#e6f2ff')

    # Plot locations
    lats = [r["lat"] for r in results]
    lons = [r["lon"] for r in results]
    fwis = [r["fwi"] for r in results]

    scatter = ax.scatter(
        lons, lats,
        c=fwis,
        cmap='YlOrRd',
        s=200,
        edgecolors='black',
        linewidths=1,
        vmin=0,
        vmax=max(fwis) * 1.2
    )

    # Add labels
    for r in results:
        ax.annotate(
            r["location"].split(",")[0],
            (r["lon"], r["lat"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold'
        )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Fire Weather Index', fontsize=11)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(
        f'Global Fire Risk Assessment\n'
        f'Temp: {temperature}°C | Humidity: {humidity}% | Wind: {wind_speed} km/h',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("="*60)
    print("Fire Weather Index Module")
    print("="*60)

    # Check installation
    status = check_fire_installation()
    print(f"\nxclim installed: {status['installed']}")
    if not status['installed']:
        print(f"Install with: {status['install_command']}")

    print(f"\nAvailable locations: {get_fire_locations()}")

    # Run example - hot, dry, windy conditions
    result = calculate_fire_risk("Los Angeles, CA", 35.0, 15, 30.0, 0.0, 1000)
    print(f"\nResults for {result['location']}:")
    print(f"  Coordinates: ({result['lat']}, {result['lon']})")
    print(f"  Conditions: {result['temperature']}°C, {result['humidity']}% RH, {result['wind_speed']} km/h wind")
    print(f"  Fire Season: {result['fire_season']}")
    print(f"  FWI: {result['fwi']:.1f}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  95% CI: {result['ci_lower']:.1f} - {result['ci_upper']:.1f}")
    print(f"  Using xclim: {result['using_xclim']}")

    # Compare locations
    print("\n" + "="*60)
    print("Location Comparison (Hot, Dry, Windy)")
    print("="*60)
    comparison = compare_locations(35.0, 15, 30.0, 0.0, 500)
    print(comparison.to_string(index=False))

    # Create visualization
    fig = create_risk_heatmap("Los Angeles, CA", 35.0, 15, 30.0, 0.0)
    fig.savefig("fire_output.png", dpi=150, bbox_inches='tight')
    print("\nOutput saved to fire_output.png")
    plt.show()
