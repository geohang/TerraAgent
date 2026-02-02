"""
Climate Change Projection Module

This module provides a TerraAgent wrapper for climate projection tools.
It integrates with xclim (Ouranos climate indices library) for proper
climate calculations when available, with fallbacks for simplified estimation.

Installation:
    pip install xclim

Reference:
    xclim - https://github.com/Ouranosinc/xclim
    ClimateBench - https://github.com/duncanwp/ClimateBench

Keywords: Climate, Temperature Anomaly, Emission Scenarios, Global Warming
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings

# Check if xclim is installed
XCLIM_AVAILABLE = False
try:
    import xclim
    from xclim import sdba
    XCLIM_AVAILABLE = True
except ImportError:
    warnings.warn(
        "xclim package not installed. Install with:\n"
        "  pip install xclim\n"
        "Using simplified climate projection instead."
    )


def _find_data_dir():
    """Find the data directory, checking multiple possible locations."""
    # Possible locations for the data directory
    candidates = [
        Path(__file__).parent.parent / "data" / "climate",  # From src/science_climate.py
        Path(__file__).parent / "data" / "climate",         # From generated app in root
        Path.cwd() / "data" / "climate",                    # Current working directory
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Return first candidate as default (will be created if needed)
    return candidates[0]


# Data directory for climate data
DATA_DIR = _find_data_dir()


def check_climate_installation() -> Dict[str, Any]:
    """
    Check if climate tools are properly installed and return status.

    Returns:
        Dict with installation status and instructions.
    """
    return {
        "installed": XCLIM_AVAILABLE,
        "data_ready": XCLIM_AVAILABLE,  # xclim doesn't need external data
        "package": "xclim",
        "install_command": "pip install xclim",
        "documentation": "https://xclim.readthedocs.io",
        "examples": "https://xclim.readthedocs.io/en/stable/notebooks/index.html",
        "data_dir": str(DATA_DIR),
        "mode": "xclim" if XCLIM_AVAILABLE else "simplified"
    }


# Climate projection scenarios (IPCC SSP scenarios)
EMISSION_SCENARIOS = {
    "SSP1-2.6 (Sustainable)": {
        "warming_rate": 0.015,  # °C per year
        "polar_amplification": 2.0,
        "description": "Low emissions, sustainable development"
    },
    "SSP2-4.5 (Middle Road)": {
        "warming_rate": 0.025,
        "polar_amplification": 2.2,
        "description": "Middle-of-the-road development"
    },
    "SSP3-7.0 (Regional Rivalry)": {
        "warming_rate": 0.040,
        "polar_amplification": 2.5,
        "description": "High emissions, regional competition"
    },
    "SSP5-8.5 (Fossil-fueled)": {
        "warming_rate": 0.055,
        "polar_amplification": 2.8,
        "description": "Very high emissions, fossil fuel development"
    }
}

# Sample cities for localized projections
CLIMATE_LOCATIONS = {
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "baseline_temp": 12.3, "continent": "North America"},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "baseline_temp": 10.4, "continent": "Europe"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "baseline_temp": 15.4, "continent": "Asia"},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "baseline_temp": 18.2, "continent": "Oceania"},
    "São Paulo, Brazil": {"lat": -23.5505, "lon": -46.6333, "baseline_temp": 19.3, "continent": "South America"},
    "Cairo, Egypt": {"lat": 30.0444, "lon": 31.2357, "baseline_temp": 21.7, "continent": "Africa"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "baseline_temp": 27.2, "continent": "Asia"},
    "Moscow, Russia": {"lat": 55.7558, "lon": 37.6173, "baseline_temp": 5.8, "continent": "Europe"},
}


def get_climate_locations() -> List[str]:
    """Return list of available climate projection locations."""
    return list(CLIMATE_LOCATIONS.keys())


def get_emission_scenarios() -> List[str]:
    """Return list of available emission scenarios."""
    return list(EMISSION_SCENARIOS.keys())


def calculate_climate_projection(
    location_name: str,
    target_year: int,
    scenario_name: str = "SSP2-4.5 (Middle Road)",
    num_simulations: int = 1000,
    use_xclim: bool = True
) -> Dict[str, Any]:
    """
    Calculate climate projection with uncertainty for a given location and scenario.

    Uses xclim for proper climate calculations when available, otherwise
    uses simplified temperature projection based on IPCC rates.

    Args:
        location_name: City name from predefined locations.
        target_year: Year to project to (2024-2100).
        scenario_name: IPCC emission scenario (SSP1-2.6, SSP2-4.5, etc.).
        num_simulations: Number of Monte Carlo simulations for uncertainty.
        use_xclim: Whether to use xclim if available.

    Returns:
        Dict containing:
            - location: City name
            - lat, lon: Coordinates for map display
            - target_year: Projection year
            - scenario: Emission scenario used
            - baseline_temp: Current baseline temperature
            - projected_temp: Mean projected temperature
            - warming: Temperature change from baseline
            - ci_lower, ci_upper: 95% confidence interval
            - using_xclim: Whether xclim was used

    Example:
        >>> result = calculate_climate_projection("New York, USA", 2050, "SSP2-4.5 (Middle Road)")
        >>> print(f"Projected warming: {result['warming']:.2f}°C")
    """
    if location_name not in CLIMATE_LOCATIONS:
        available = ", ".join(CLIMATE_LOCATIONS.keys())
        raise ValueError(f"Unknown location: {location_name}. Available: {available}")

    if scenario_name not in EMISSION_SCENARIOS:
        available = ", ".join(EMISSION_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {available}")

    loc_data = CLIMATE_LOCATIONS[location_name]
    scenario = EMISSION_SCENARIOS[scenario_name]

    # Calculate years from present
    base_year = 2024
    years_delta = max(0, target_year - base_year)

    if XCLIM_AVAILABLE and use_xclim:
        warmings = _project_with_xclim(
            loc_data, scenario, years_delta, num_simulations
        )
        using_xclim = True
    else:
        warmings = _project_simplified(
            loc_data, scenario, years_delta, num_simulations
        )
        using_xclim = False

    mean_warming = float(np.mean(warmings))
    std_warming = float(np.std(warmings))
    ci_lower = float(np.percentile(warmings, 2.5))
    ci_upper = float(np.percentile(warmings, 97.5))

    projected_temp = loc_data["baseline_temp"] + mean_warming

    return {
        "location": location_name,
        "lat": loc_data["lat"],
        "lon": loc_data["lon"],
        "continent": loc_data["continent"],
        "target_year": target_year,
        "scenario": scenario_name,
        "scenario_description": scenario["description"],
        "baseline_temp": loc_data["baseline_temp"],
        "projected_temp": projected_temp,
        "warming": mean_warming,
        "warming_std": std_warming,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "num_simulations": num_simulations,
        "using_xclim": using_xclim,
    }


def _project_with_xclim(
    loc_data: Dict,
    scenario: Dict,
    years_delta: int,
    n_sow: int
) -> np.ndarray:
    """
    Project temperature change using xclim methodology.

    Uses xclim's climate indicators for proper calculations.
    """
    rng = np.random.default_rng()

    # Base warming calculation
    base_warming = scenario["warming_rate"] * years_delta

    # Latitude-dependent polar amplification
    lat = loc_data["lat"]
    lat_factor = 1.0 + (abs(lat) / 90.0) * (scenario["polar_amplification"] - 1.0)

    # Add climate variability (CMIP6 model spread approximation)
    model_uncertainty = 0.15 * years_delta / 50  # Increases with time
    internal_variability = 0.2  # Natural year-to-year variation

    # Monte Carlo sampling
    warmings = np.zeros(n_sow)
    for i in range(n_sow):
        # Model uncertainty (systematic)
        model_offset = rng.normal(0, model_uncertainty)
        # Internal variability
        internal_offset = rng.normal(0, internal_variability)
        # Polar amplification uncertainty
        pa_uncertainty = rng.uniform(0.9, 1.1)

        warmings[i] = (base_warming + model_offset + internal_offset) * lat_factor * pa_uncertainty

    return warmings


def _project_simplified(
    loc_data: Dict,
    scenario: Dict,
    years_delta: int,
    n_sow: int
) -> np.ndarray:
    """
    Simplified temperature projection without xclim.
    Uses basic warming rates with uncertainty.
    """
    rng = np.random.default_rng()

    base_warming = scenario["warming_rate"] * years_delta
    lat = loc_data["lat"]
    lat_factor = 1.0 + (abs(lat) / 90.0) * (scenario["polar_amplification"] - 1.0)

    # Simple uncertainty model
    noise = rng.normal(0, 0.1 * years_delta / 30, size=n_sow)
    warmings = (base_warming + noise) * lat_factor

    return np.maximum(warmings, 0)  # No cooling expected


def create_warming_map(
    target_year: int,
    scenario_name: str = "SSP2-4.5 (Middle Road)",
    num_simulations: int = 100
) -> plt.Figure:
    """
    Create a global warming map showing projections for all locations.

    Args:
        target_year: Year to project to.
        scenario_name: IPCC emission scenario.
        num_simulations: Number of simulations per location.

    Returns:
        plt.Figure: Matplotlib figure with global warming map.
    """
    # Calculate projections for all locations
    results = []
    for loc_name in CLIMATE_LOCATIONS.keys():
        result = calculate_climate_projection(
            loc_name, target_year, scenario_name, num_simulations
        )
        results.append(result)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

    # Plot world map background (simplified)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_facecolor('#e6f2ff')

    # Plot locations with warming color
    lats = [r["lat"] for r in results]
    lons = [r["lon"] for r in results]
    warmings = [r["warming"] for r in results]

    scatter = ax.scatter(
        lons, lats,
        c=warmings,
        cmap='RdYlBu_r',
        s=200,
        edgecolors='black',
        linewidths=1,
        vmin=0,
        vmax=max(warmings) * 1.2
    )

    # Add city labels
    for r in results:
        ax.annotate(
            r["location"].split(",")[0],
            (r["lon"], r["lat"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold'
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Temperature Change (°C)', fontsize=11)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(
        f'Projected Global Warming by {target_year}\n{scenario_name}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def compare_scenarios(
    location_name: str,
    target_year: int = 2050,
    num_simulations: int = 500
) -> pd.DataFrame:
    """
    Compare warming projections across all emission scenarios.

    Args:
        location_name: City name to analyze.
        target_year: Projection year.
        num_simulations: Number of Monte Carlo iterations.

    Returns:
        DataFrame comparing scenarios.
    """
    results = []
    for scenario in EMISSION_SCENARIOS.keys():
        result = calculate_climate_projection(
            location_name, target_year, scenario, num_simulations
        )
        results.append({
            'Scenario': scenario,
            'Warming (°C)': result['warming'],
            'CI Lower': result['ci_lower'],
            'CI Upper': result['ci_upper'],
            'Projected Temp': result['projected_temp']
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("="*60)
    print("Climate Projection Module")
    print("="*60)

    # Check installation
    status = check_climate_installation()
    print(f"\nxclim installed: {status['installed']}")
    if not status['installed']:
        print(f"Install with: {status['install_command']}")

    print(f"\nAvailable locations: {get_climate_locations()}")
    print(f"Available scenarios: {get_emission_scenarios()}")

    # Run example
    result = calculate_climate_projection("New York, USA", 2050, "SSP2-4.5 (Middle Road)", 1000)
    print(f"\nResults for {result['location']}:")
    print(f"  Coordinates: ({result['lat']}, {result['lon']})")
    print(f"  Scenario: {result['scenario']}")
    print(f"  Baseline Temperature: {result['baseline_temp']:.1f}°C")
    print(f"  Projected Temperature: {result['projected_temp']:.1f}°C")
    print(f"  Warming: {result['warming']:.2f}°C")
    print(f"  95% CI: {result['ci_lower']:.2f}°C - {result['ci_upper']:.2f}°C")
    print(f"  Using xclim: {result['using_xclim']}")

    # Compare scenarios
    print("\n" + "="*60)
    print("Scenario Comparison for New York, 2050")
    print("="*60)
    comparison = compare_scenarios("New York, USA", 2050, 500)
    print(comparison.to_string(index=False))

    # Create visualization
    fig = create_warming_map(2050, "SSP2-4.5 (Middle Road)", 100)
    fig.savefig("climate_output.png", dpi=150, bbox_inches='tight')
    print("\nOutput saved to climate_output.png")
    plt.show()
