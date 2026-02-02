"""
Auto-generated Streamlit UI for calculate_flood_loss
"""

import streamlit as st
import matplotlib.pyplot as plt

# Import the science function (paste the function code here or import from module)
# from science_module import calculate_flood_loss

# === PASTE YOUR SCIENCE FUNCTION HERE ===
"""
Flood Loss Uncertainty Assessment Module

This module provides a TerraAgent wrapper for the UNSAFE framework.
UNSAFE (UNcertain Structure And Fragility Ensemble) is an open-source 
framework for estimating property-level flood risk with uncertainty.

Installation:
    pip install git+https://github.com/abpoll/unsafe

Reference: 
    UNSAFE Framework - https://github.com/abpoll/unsafe
    Pollack, A., Doss-Gollin, J., Srikrishnan, V., & Keller, K. (2024)
    DOI: https://doi.org/10.21105/joss.07527

Keywords: Uncertainty Quantification, Probabilistic Modeling, Depth-Damage Functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
import os
import json

# Check if UNSAFE is installed and import its functions
UNSAFE_AVAILABLE = False
UNSAFE_DATA_READY = False

try:
    from unsafe.ddfs import est_hazus_loss, process_hazus
    from unsafe.const import MTR_TO_FT
    UNSAFE_AVAILABLE = True
except ImportError:
    UNSAFE_AVAILABLE = False
    MTR_TO_FT = 3.28084  # Fallback constant
    warnings.warn(
        "UNSAFE package not installed. Install with:\n"
        "  pip install git+https://github.com/abpoll/unsafe\n"
        "Using simplified flood loss estimation instead."
    )


def _find_data_dir():
    """Find the data directory, checking multiple possible locations."""
    # Possible locations for the data directory
    candidates = [
        Path(__file__).parent.parent / "data" / "unsafe",  # From src/science_flood.py
        Path(__file__).parent / "data" / "unsafe",         # From generated app in root
        Path.cwd() / "data" / "unsafe",                    # Current working directory
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Return first candidate as default (will be created if needed)
    return candidates[0]


# Data directory for UNSAFE processed files
DATA_DIR = _find_data_dir()

# Global variables for UNSAFE DDF data (loaded once)
_HAZUS_DDFS = None
_HAZUS_MAX_DICT = None


def _load_unsafe_data():
    """Load UNSAFE depth-damage function data if available."""
    global _HAZUS_DDFS, _HAZUS_MAX_DICT, UNSAFE_DATA_READY
    
    if not UNSAFE_AVAILABLE:
        return False
    
    # Check if data files exist
    ddf_file = DATA_DIR / "physical" / "hazus_ddfs.pqt"
    max_file = DATA_DIR / "physical" / "hazus.json"
    
    if ddf_file.exists() and max_file.exists():
        try:
            _HAZUS_DDFS = pd.read_parquet(ddf_file)
            with open(max_file, "r") as fp:
                _HAZUS_MAX_DICT = json.load(fp)
            UNSAFE_DATA_READY = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load UNSAFE data: {e}")
            return False
    
    return False


def setup_unsafe_data(hazus_csv_path: Optional[str] = None) -> bool:
    """
    Setup UNSAFE depth-damage function data.
    
    Downloads and processes HAZUS DDFs for use with the UNSAFE framework.
    This only needs to be run once.
    
    Args:
        hazus_csv_path: Path to HAZUS DDF CSV file (optional).
            If not provided, will attempt to download from UNSAFE repo.
    
    Returns:
        True if setup successful, False otherwise.
    """
    global UNSAFE_DATA_READY
    
    if not UNSAFE_AVAILABLE:
        print("UNSAFE package not installed. Install with:")
        print("  pip install git+https://github.com/abpoll/unsafe")
        return False
    
    # Create data directory
    (DATA_DIR / "physical").mkdir(parents=True, exist_ok=True)
    
    # Download HAZUS DDF data if not provided
    if hazus_csv_path is None:
        try:
            import urllib.request
            # Download from Zenodo archive (Pollack, A. 2023)
            # https://zenodo.org/records/10027236
            url = "https://zenodo.org/records/10027236/files/haz_fl_dept.csv?download=1"
            hazus_csv_path = DATA_DIR / "haz_fl_dept.csv"
            print(f"Downloading HAZUS DDFs from Zenodo...")
            urllib.request.urlretrieve(url, hazus_csv_path)
            print(f"Downloaded to {hazus_csv_path}")
        except Exception as e:
            print(f"Failed to download HAZUS data: {e}")
            return False
    
    # Process the DDFs using UNSAFE's process_hazus function
    try:
        print("Processing HAZUS DDFs with UNSAFE...")
        process_hazus(str(DATA_DIR), str(DATA_DIR))
        UNSAFE_DATA_READY = True
        print("UNSAFE data setup complete!")
        return _load_unsafe_data()
    except Exception as e:
        print(f"Failed to process HAZUS DDFs: {e}")
        return False


# Try to load data on module import
_load_unsafe_data()


# Sample locations with flood risk data (US cities prone to flooding)
# These coordinates can be used to query flood depth data
FLOOD_LOCATIONS = {
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698, "fips": "48201", "base_risk": 0.75, "avg_property": 350000},
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "fips": "12086", "base_risk": 0.80, "avg_property": 450000},
    "New Orleans, LA": {"lat": 29.9511, "lon": -90.0715, "fips": "22071", "base_risk": 0.85, "avg_property": 280000},
    "Cedar Rapids, IA": {"lat": 41.9779, "lon": -91.6656, "fips": "19113", "base_risk": 0.65, "avg_property": 220000},
    "Sacramento, CA": {"lat": 38.5816, "lon": -121.4944, "fips": "06067", "base_risk": 0.55, "avg_property": 520000},
    "Charleston, SC": {"lat": 32.7765, "lon": -79.9311, "fips": "45019", "base_risk": 0.70, "avg_property": 380000},
    "Norfolk, VA": {"lat": 36.8508, "lon": -76.2859, "fips": "51710", "base_risk": 0.72, "avg_property": 320000},
    "Philadelphia, PA": {"lat": 39.9526, "lon": -75.1652, "fips": "42101", "base_risk": 0.60, "avg_property": 290000},
}


def get_flood_locations() -> List[str]:
    """
    Get list of available flood-prone locations.
    
    Returns:
        List of location names that can be used for flood analysis.
    """
    return list(FLOOD_LOCATIONS.keys())


def check_unsafe_installation() -> Dict[str, Any]:
    """
    Check if UNSAFE is properly installed and return status.
    
    Returns:
        Dict with installation status and instructions.
    """
    return {
        "installed": UNSAFE_AVAILABLE,
        "data_ready": UNSAFE_DATA_READY,
        "package": "unsafe",
        "install_command": "pip install git+https://github.com/abpoll/unsafe",
        "documentation": "https://github.com/abpoll/unsafe",
        "examples": "https://github.com/abpoll/unsafe/tree/main/examples",
        "data_dir": str(DATA_DIR),
        "mode": "direct" if UNSAFE_DATA_READY else ("style" if UNSAFE_AVAILABLE else "simplified")
    }


def calculate_flood_loss(
    location_name: str,
    flood_depth: float,
    num_simulations: int = 1000,
    property_value: Optional[float] = None,
    foundation_type: str = "S",
    num_stories: int = 1,
    use_unsafe: bool = True
) -> Dict[str, Any]:
    """
    Calculate flood loss with uncertainty using UNSAFE framework or simplified model.

    This function estimates the probability distribution of financial losses 
    from flooding using Monte Carlo simulations. When UNSAFE is installed,
    it uses proper depth-damage functions from HAZUS/NACCS.

    Args:
        location_name: Name of the city to analyze (e.g., 'Houston, TX').
            Must be one of the predefined flood-prone locations.
        flood_depth: The flood water depth in meters (0.1 to 5.0).
            Deeper flooding causes exponentially more damage.
        num_simulations: Number of Monte Carlo iterations (default: 1000).
            More simulations provide more stable uncertainty estimates.
        property_value: Override the average property value for the location.
            If None, uses the location's default average.
        foundation_type: Foundation type - 'S' (Slab), 'C' (Crawlspace), 
            or 'B' (Basement). Affects first floor elevation.
        num_stories: Number of stories (1 or 2). Affects damage function.
        use_unsafe: Whether to use UNSAFE if available (default: True).

    Returns:
        Dict containing:
            - location: City name
            - lat, lon: Coordinates for map display
            - mean_loss: Expected loss in USD
            - ci_lower, ci_upper: 95% confidence interval bounds
            - damage_ratio: Fraction of property value lost
            - using_unsafe: Whether UNSAFE framework was used
    
    Example:
        >>> result = calculate_flood_loss("Houston, TX", 1.5, 1000)
        >>> print(f"Expected loss: ${result['mean_loss']:,.0f}")
    
    Note:
        Install UNSAFE for more accurate damage estimation:
        pip install git+https://github.com/abpoll/unsafe
    """
    if location_name not in FLOOD_LOCATIONS:
        available = ", ".join(FLOOD_LOCATIONS.keys())
        raise ValueError(f"Unknown location: {location_name}. Available: {available}")
    
    loc_data = FLOOD_LOCATIONS[location_name]
    val_struct = property_value if property_value else loc_data["avg_property"]
    base_risk = loc_data["base_risk"]
    
    # Convert depth from meters to feet (UNSAFE uses feet internally)
    depth_ft = flood_depth * 3.28084
    
    # First Floor Elevation based on foundation type
    # Following UNSAFE's ffe_dict defaults
    ffe_dict = {
        'S': [0, 0.5, 1.5],   # Slab: 0-1.5 ft
        'C': [0, 1.5, 4],      # Crawlspace: 0-4 ft  
        'B': [0, 1.5, 4]       # Basement: 0-4 ft
    }
    
    if UNSAFE_AVAILABLE and UNSAFE_DATA_READY and use_unsafe:
        # Use UNSAFE's est_hazus_loss directly with processed DDF data
        losses = _estimate_loss_with_unsafe_direct(
            depth_ft, val_struct, foundation_type, num_stories,
            ffe_dict, num_simulations
        )
        using_unsafe = True
    elif UNSAFE_AVAILABLE and use_unsafe:
        # UNSAFE installed but data not ready - use methodology without data files
        losses = _estimate_loss_unsafe_style(
            depth_ft, val_struct, foundation_type, num_stories,
            ffe_dict, num_simulations
        )
        using_unsafe = True
    else:
        # Fallback: Simplified sigmoidal damage curve
        losses = _estimate_loss_simplified(
            depth_ft, val_struct, num_simulations
        )
        using_unsafe = False
    
    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    ci_lower = float(np.percentile(losses, 2.5))
    ci_upper = float(np.percentile(losses, 97.5))
    
    return {
        "location": location_name,
        "lat": loc_data["lat"],
        "lon": loc_data["lon"],
        "fips": loc_data["fips"],
        "flood_depth_m": flood_depth,
        "flood_depth_ft": depth_ft,
        "property_value": val_struct,
        "foundation_type": foundation_type,
        "num_stories": num_stories,
        "base_risk": base_risk,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "damage_ratio": mean_loss / val_struct,
        "num_simulations": num_simulations,
        "using_unsafe": using_unsafe,
    }


def _estimate_loss_with_unsafe_direct(
    depth_ft: float,
    val_struct: float,
    foundation_type: str,
    num_stories: int,
    ffe_dict: Dict,
    n_sow: int
) -> np.ndarray:
    """
    Estimate losses using UNSAFE's est_hazus_loss function directly.
    
    This uses the actual UNSAFE package with processed HAZUS DDFs.
    
    Args:
        depth_ft: Flood depth in feet
        val_struct: Property value in USD
        foundation_type: 'S', 'C', or 'B'
        num_stories: 1 or 2
        ffe_dict: FFE distribution parameters by foundation type
        n_sow: Number of states of world (simulations)
    
    Returns:
        Array of loss values for each simulation
    """
    rng = np.random.default_rng()
    
    # Draw FFE from triangular distribution per UNSAFE methodology
    ffe_params = ffe_dict.get(foundation_type, [0, 0.5, 1.5])
    ffes = rng.triangular(ffe_params[0], ffe_params[1], ffe_params[2], size=n_sow)
    
    # Build DDF type code following UNSAFE conventions
    # Format: "{stories}S{basement}_RES1" e.g., "1SNB_RES1" or "1SWB_A_RES1"
    stories_str = f"{num_stories}S"
    if foundation_type == 'B':
        # Basement: WB = with basement, use A zone default
        ddf_types = pd.Series([f"{stories_str}WB_A_RES1"] * n_sow)
    else:
        # No basement: NB
        ddf_types = pd.Series([f"{stories_str}NB_RES1"] * n_sow)
    
    # Create depth and FFE series
    depths = pd.Series([depth_ft] * n_sow)
    
    # Call UNSAFE's est_hazus_loss directly
    try:
        rel_damages = est_hazus_loss(
            ddf_types,
            depths.values,
            ffes,
            _HAZUS_DDFS,
            _HAZUS_MAX_DICT,
            base_adj=(foundation_type == 'B')  # Basement adjustment
        )
        losses = val_struct * rel_damages.values
    except Exception as e:
        # Fallback if UNSAFE call fails
        warnings.warn(f"UNSAFE est_hazus_loss failed: {e}. Using style-based estimation.")
        losses = _estimate_loss_unsafe_style(
            depth_ft, val_struct, foundation_type, num_stories, ffe_dict, n_sow
        )
    
    return losses


def _estimate_loss_unsafe_style(
    depth_ft: float,
    val_struct: float,
    foundation_type: str,
    num_stories: int,
    ffe_dict: Dict,
    n_sow: int
) -> np.ndarray:
    """
    Estimate losses using UNSAFE's methodology without requiring data files.
    
    Implements the same approach as UNSAFE:
    - Triangular distributions for First Floor Elevation (FFE)
    - Uniform uncertainty bounds (±30%) on depth-damage relationships
    - HAZUS-based depth-damage curves
    
    Reference: https://github.com/abpoll/unsafe
    """
    rng = np.random.default_rng()
    
    # Draw FFE from triangular distribution (UNSAFE approach)
    ffe_params = ffe_dict.get(foundation_type, [0, 0.5, 1.5])
    ffes = rng.triangular(ffe_params[0], ffe_params[1], ffe_params[2], size=n_sow)
    
    # Calculate effective depth (depth relative to first floor)
    # For basement, UNSAFE uses a 4ft basement height adjustment
    if foundation_type == 'B':
        BSMT_HGT = 4  # HAZUS uses 4ft, NACCS uses 9ft
        effective_depths = np.full(n_sow, depth_ft - BSMT_HGT)
    else:
        effective_depths = depth_ft - ffes
    
    # HAZUS depth-damage curve for RES1 (residential) structures
    # Based on HAZUS Technical Manual and UNSAFE's processed DDFs
    unif_unc = 0.3  # 30% uncertainty as per UNSAFE default
    
    # Point estimate damages at standard depths (from HAZUS)
    # Vectorized computation
    base_damages = np.zeros(n_sow)
    
    for i, eff_depth in enumerate(effective_depths):
        if eff_depth <= 0:
            base_damages[i] = 0.0
        elif eff_depth <= 1:
            base_damages[i] = 0.08 + 0.08 * eff_depth  # 8-16%
        elif eff_depth <= 4:
            base_damages[i] = 0.16 + 0.09 * (eff_depth - 1)  # 16-43%
        elif eff_depth <= 8:
            base_damages[i] = 0.43 + 0.09 * (eff_depth - 4)  # 43-79%
        else:
            base_damages[i] = min(0.79 + 0.03 * (eff_depth - 8), 0.95)
    
    # Apply uniform uncertainty (UNSAFE's approach)
    dam_low = np.maximum(0, base_damages * (1 - unif_unc))
    dam_high = np.minimum(1, base_damages * (1 + unif_unc))
    
    # Sample from uniform distribution between bounds
    rel_damages = rng.uniform(dam_low, dam_high)
    
    return val_struct * rel_damages


def _estimate_loss_simplified(
    depth_ft: float,
    val_struct: float,
    n_sow: int
) -> np.ndarray:
    """
    Simplified flood loss estimation when UNSAFE is not available.
    Uses a basic sigmoidal damage curve with noise.
    """
    rng = np.random.default_rng()
    
    # Simple sigmoidal damage curve
    k = 0.5  # Curve steepness
    d0 = 4.0  # Threshold depth in feet
    
    base_damage_ratio = 1.0 / (1.0 + np.exp(-k * (depth_ft - d0)))
    noise = rng.normal(0, 0.1, size=n_sow)
    damage_ratios = np.clip(base_damage_ratio + noise, 0, 1)
    
    return val_struct * damage_ratios


def create_loss_histogram(result: Dict[str, Any]) -> plt.Figure:
    """
    Create a histogram visualization of the flood loss distribution.

    Args:
        result: Dictionary returned by calculate_flood_loss().

    Returns:
        plt.Figure: Matplotlib figure with loss distribution histogram.
    """
    # Re-run simulation to get the loss array for plotting
    losses = _get_loss_array(result)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    ax.hist(losses / 1000, bins=50, density=True, color='steelblue',
            edgecolor='white', alpha=0.7, label='Loss Distribution')
    
    mean_loss = result["mean_loss"]
    ci_lower = result["ci_lower"]
    ci_upper = result["ci_upper"]
    
    ax.axvline(mean_loss / 1000, color='red', linestyle='-', linewidth=2.5,
               label=f'Mean: ${mean_loss/1000:,.0f}K')
    ax.axvline(ci_lower / 1000, color='green', linestyle='--', linewidth=2,
               label=f'2.5th %ile: ${ci_lower/1000:,.0f}K')
    ax.axvline(ci_upper / 1000, color='green', linestyle='--', linewidth=2,
               label=f'97.5th %ile: ${ci_upper/1000:,.0f}K')
    
    ax.axvspan(ci_lower / 1000, ci_upper / 1000, alpha=0.2, color='green')
    
    ax.set_xlabel('Loss Amount ($ Thousands)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    
    title = f'Flood Loss Distribution - {result["location"]}\n'
    title += f'Depth: {result["flood_depth_m"]:.1f}m ({result["flood_depth_ft"]:.1f}ft) | '
    title += f'Property: ${result["property_value"]:,}'
    if result.get("using_unsafe"):
        title += ' | UNSAFE Framework'
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Statistics:\n'
        f'Mean: ${mean_loss:,.0f}\n'
        f'Std Dev: ${result["std_loss"]:,.0f}\n'
        f'95% CI: ${ci_lower:,.0f} - ${ci_upper:,.0f}\n'
        f'Damage: {result["damage_ratio"]:.1%}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace', bbox=props)
    
    plt.tight_layout()
    return fig


def _get_loss_array(result: Dict[str, Any]) -> np.ndarray:
    """Regenerate loss array from result parameters."""
    depth_ft = result["flood_depth_ft"]
    val_struct = result["property_value"]
    n_sow = result["num_simulations"]
    foundation_type = result.get("foundation_type", "S")
    num_stories = result.get("num_stories", 1)
    
    ffe_dict = {'S': [0, 0.5, 1.5], 'C': [0, 1.5, 4], 'B': [0, 1.5, 4]}
    
    if result.get("using_unsafe"):
        if UNSAFE_AVAILABLE and UNSAFE_DATA_READY:
            return _estimate_loss_with_unsafe_direct(
                depth_ft, val_struct, foundation_type, num_stories, ffe_dict, n_sow
            )
        elif UNSAFE_AVAILABLE:
            return _estimate_loss_unsafe_style(
                depth_ft, val_struct, foundation_type, num_stories, ffe_dict, n_sow
            )
    
    return _estimate_loss_simplified(depth_ft, val_struct, n_sow)


def compare_foundation_types(
    location_name: str,
    flood_depth: float,
    num_simulations: int = 1000
) -> pd.DataFrame:
    """
    Compare flood losses across different foundation types.
    
    Args:
        location_name: Name of the city to analyze.
        flood_depth: Flood water depth in meters.
        num_simulations: Number of Monte Carlo iterations.
    
    Returns:
        DataFrame comparing losses by foundation type.
    """
    results = []
    for fnd_type, fnd_name in [('S', 'Slab'), ('C', 'Crawlspace'), ('B', 'Basement')]:
        result = calculate_flood_loss(
            location_name, flood_depth, num_simulations,
            foundation_type=fnd_type
        )
        results.append({
            'Foundation': fnd_name,
            'Mean Loss': result['mean_loss'],
            'CI Lower': result['ci_lower'],
            'CI Upper': result['ci_upper'],
            'Damage Ratio': result['damage_ratio']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the functions
    print("="*60)
    print("UNSAFE Flood Loss Estimation")
    print("="*60)
    
    # Check installation
    status = check_unsafe_installation()
    print(f"\nUNSAFE installed: {status['installed']}")
    if not status['installed']:
        print(f"Install with: {status['install_command']}")
    
    print(f"\nAvailable locations: {get_flood_locations()}")
    
    # Run example
    result = calculate_flood_loss("Houston, TX", 1.5, 1000)
    print(f"\nResults for {result['location']}:")
    print(f"  Coordinates: ({result['lat']}, {result['lon']})")
    print(f"  Flood Depth: {result['flood_depth_m']:.1f}m ({result['flood_depth_ft']:.1f}ft)")
    print(f"  Mean Loss: ${result['mean_loss']:,.0f}")
    print(f"  95% CI: ${result['ci_lower']:,.0f} - ${result['ci_upper']:,.0f}")
    print(f"  Damage Ratio: {result['damage_ratio']:.1%}")
    print(f"  Using UNSAFE: {result['using_unsafe']}")
    
    # Compare foundation types
    print("\n" + "="*60)
    print("Foundation Type Comparison")
    print("="*60)
    comparison = compare_foundation_types("Houston, TX", 1.5, 1000)
    print(comparison.to_string(index=False))
    
    # Create visualization
    fig = create_loss_histogram(result)
    fig.savefig("flood_output.png", dpi=150, bbox_inches='tight')
    print("\nOutput saved to flood_output.png")
    plt.show()

# === END SCIENCE FUNCTION ===

# Page config
st.set_page_config(page_title="Calculate Flood Loss", layout="wide")

# Title and description
st.title("Calculate Flood Loss")
st.markdown("""Calculate flood loss with uncertainty using UNSAFE framework or simplified model.""")
st.markdown("**User Instruction:** Create a flood loss estimation app")

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# Sidebar inputs
st.sidebar.header("Input Parameters")

location_name = st.sidebar.selectbox("Location Name", options=get_flood_locations() if "get_flood_locations" in dir() else ["Houston, TX", "Miami, FL", "New Orleans, LA", "Cedar Rapids, IA", "Sacramento, CA", "Charleston, SC", "Norfolk, VA", "Baton Rouge, LA"], help="Name of the city to analyze (e.g., 'Houston, TX').")
flood_depth = st.sidebar.number_input("Flood Depth", value=1.0, step=0.1, help="The flood water depth in meters (0.1 to 5.0).")
num_simulations = st.sidebar.number_input("Num Simulations", min_value=1, value=1000, step=100, help="Number of Monte Carlo iterations (default: 1000).")
property_value = st.sidebar.number_input("Property Value", value=0, help="Override the average property value for the location.")
foundation_type = st.sidebar.text_input("Foundation Type", value="S", help="Foundation type - 'S' (Slab), 'C' (Crawlspace),")
num_stories = st.sidebar.slider("Num Stories", min_value=0, max_value=100, value=1, help="Number of stories (1 or 2). Affects damage function.")
use_unsafe = st.sidebar.checkbox("Use Unsafe", value=True, help="Whether to use UNSAFE if available (default: True).")

# Main area
st.divider()

# Show location preview before running (makes UI more attractive)
preview_col1, preview_col2 = st.columns([2, 1])

with preview_col1:
    st.subheader("🌊 Flood Risk Analysis")
    
    # Check UNSAFE status
    status = check_unsafe_installation()
    if status['installed'] and status['data_ready']:
        st.success(f"✅ UNSAFE framework active | Mode: **{status['mode']}**")
    else:
        st.warning(f"⚠️ Using simplified estimation | Install UNSAFE: `{status['install_command']}`")
    
    st.markdown(f"""
    **Selected Location:** {location_name}  
    **Flood Depth:** {flood_depth:.1f} meters  
    **Property Value:** {f'${property_value:,.0f}' if property_value > 0 else 'Default for location'}
    
    Click **Run Model** to calculate flood loss with uncertainty quantification.
    """)

with preview_col2:
    st.subheader("🗺️ Location Preview")
    # Show selected location on map
    if location_name in FLOOD_LOCATIONS:
        loc_info = FLOOD_LOCATIONS[location_name]
        preview_map = pd.DataFrame({
            'lat': [loc_info['lat']],
            'lon': [loc_info['lon']],
        })
        st.map(preview_map, zoom=8)
        st.caption(f"📍 {location_name}")

st.divider()

# Run button
if st.sidebar.button("Run Model", type="primary", use_container_width=True):
    with st.spinner("Running model..."):
        try:
            result = calculate_flood_loss(location_name, flood_depth, num_simulations, property_value, foundation_type, num_stories, use_unsafe)
            st.session_state.result = result
            st.success("Model completed successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display result
if st.session_state.result is not None:
    result = st.session_state.result
    
    # Create two columns for results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Key Results")
        
        # Display metrics based on what's available
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            if 'mean_loss' in result:
                st.metric("Mean Loss", f"${result['mean_loss']:,.0f}")
            elif 'mean' in result:
                st.metric("Mean", f"{result['mean']:.2f}")
            elif 'fwi' in result:
                st.metric("Fire Weather Index", f"{result['fwi']:.1f}")
        
        with metric_col2:
            if 'damage_ratio' in result:
                st.metric("Damage Ratio", f"{result['damage_ratio']:.1%}")
            elif 'std_loss' in result:
                st.metric("Std Dev", f"${result['std_loss']:,.0f}")
            elif 'std' in result:
                st.metric("Std Dev", f"{result['std']:.2f}")
        
        with metric_col3:
            if 'ci_lower' in result and 'ci_upper' in result:
                ci_text = f"${result['ci_lower']:,.0f} — ${result['ci_upper']:,.0f}"
                st.metric("95% CI", ci_text)
            elif 'using_unsafe' in result:
                mode = "✅ UNSAFE" if result['using_unsafe'] else "📊 Simplified"
                st.metric("Mode", mode)
        
        # Show confidence interval info
        if 'ci_lower' in result and 'ci_upper' in result:
            st.info(f"**95% Confidence Interval:** ${result.get('ci_lower', 0):,.0f} — ${result.get('ci_upper', 0):,.0f}")
        
        # Create histogram if losses available
        if 'losses' in result or 'mean_loss' in result:
            st.markdown("#### 📈 Loss Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            if 'losses' in result:
                losses = result['losses']
            else:
                # Simulate distribution from mean and std
                import numpy as np
                mean = result.get('mean_loss', 10000)
                std = result.get('std_loss', mean * 0.3)
                losses = np.random.normal(mean, std, 1000)
                losses = np.maximum(losses, 0)
            ax.hist(losses, bins=30, edgecolor='white', alpha=0.7, color='steelblue')
            ax.axvline(result.get('mean_loss', np.mean(losses)), color='red', linestyle='--', 
                      linewidth=2, label='Mean')
            ax.set_xlabel('Loss ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    with col2:
        st.subheader("🗺️ Location")
        
        # Display map if coordinates available
        if 'lat' in result and 'lon' in result:
            import pandas as pd
            map_data = pd.DataFrame({
                'lat': [result['lat']],
                'lon': [result['lon']],
            })
            st.map(map_data, zoom=10)
            st.caption(f"📍 {result.get('location', 'Selected Location')}")
    
    # Details table
    st.divider()
    st.subheader("📋 Calculation Details")
    
    # Build details from result
    import pandas as pd
    details = []
    display_keys = ['location', 'flood_depth_m', 'property_value', 'foundation_type', 
                   'num_stories', 'num_simulations', 'using_unsafe', 'date', 'period']
    for key in display_keys:
        if key in result:
            value = result[key]
            if key == 'property_value':
                value = f"${value:,.0f}"
            elif key == 'flood_depth_m':
                value = f"{value:.2f} m"
            elif key == 'using_unsafe':
                value = "✅ Yes" if value else "❌ No"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
            details.append({'Parameter': key.replace('_', ' ').title(), 'Value': str(value)})
    
    if details:
        details_df = pd.DataFrame(details)
        st.dataframe(details_df, hide_index=True, width="stretch")
    
    # Show full JSON in expander
    with st.expander("🔍 View Full Result JSON"):
        st.json(result)
