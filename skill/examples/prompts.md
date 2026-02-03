# TerraAgent Example Prompts

These example prompts demonstrate how to use TerraAgent with Claude Code.

---

## Integrating a New Scientific Package

### Flood Analysis (UNSAFE)

```
Integrate the UNSAFE package from https://github.com/abpoll/unsafe for flood loss estimation.
Create a wrapper module that calculates property flood damage with uncertainty quantification
using Monte Carlo simulation. Include sample locations in Texas and Florida.
```

### Climate Projections (xclim)

```
Integrate xclim from https://github.com/Ouranosinc/xclim for climate change projections.
Create a wrapper module that projects temperature changes for major cities under different
IPCC SSP emission scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5).
```

### Fire Weather Index

```
Integrate xclim fire weather indices for wildfire risk assessment.
Create a wrapper module that calculates Fire Weather Index (FWI) for fire-prone regions
based on temperature, humidity, wind speed, and precipitation inputs.
```

---

## Generating Streamlit Apps

### Basic App Generation

```
Generate a Streamlit app from src/science_flood.py with:
- Interactive map showing the selected location
- Sidebar inputs for all parameters
- Metrics display for mean loss and confidence intervals
- Histogram showing the loss distribution
```

### Custom Visualization

```
Generate a Streamlit app for src/science_climate.py that includes:
- A global warming map comparing all cities
- Scenario comparison table
- Time series projection chart
- Temperature anomaly metrics
```

---

## Creating New Science Modules

### Earthquake Analysis

```
Create a science module for earthquake analysis using PyGMT.
Include sample fault lines in California, Japan, and Chile.
Calculate seismic hazard with Monte Carlo uncertainty.
Return lat/lon coordinates for map visualization.
```

### Air Quality Analysis

```
Create a science module for air quality analysis.
Include major cities with baseline AQI values.
Calculate projected PM2.5 levels with uncertainty.
Follow the TerraAgent three-tier fallback pattern.
```

### Ocean Acidification

```
Create a science module for ocean acidification projections.
Include coastal monitoring stations as sample locations.
Calculate pH changes under different CO2 scenarios.
Include lat/lon for ocean map visualization.
```

---

## Workflow Automation

### Full Integration Workflow

```
Integrate a new scientific package from GitHub:
1. Clone and install the package
2. Create src/science_{domain}.py wrapper module
3. Generate Streamlit app with rich visualizations
4. Create launcher scripts (.bat and .sh)
5. Update HISTORY.md with integration details
```

### Testing Workflow

```
Test the science_flood module:
1. Check installation status
2. Run calculate_flood_loss for Houston, TX
3. Verify output includes lat, lon, mean_loss, ci_lower, ci_upper
4. Generate and validate Streamlit app code
```

---

## Fixing Common Issues

### Data Directory Issues

```
The science module can't find the data directory.
Check the _find_data_dir() function and verify it searches:
- Path(__file__).parent.parent / "data" / "{domain}"
- Path(__file__).parent / "data" / "{domain}"
- Path.cwd() / "data" / "{domain}"
```

### Package Import Errors

```
The UNSAFE package is not being detected.
Check the import block and verify PACKAGE_AVAILABLE is set correctly.
Run check_installation() to verify status.
```

### Map Not Displaying

```
The Streamlit map is not showing the location.
Verify the calculation function returns 'lat' and 'lon' keys.
Check that LOCATIONS dictionary has valid coordinates.
```
