"""
Flood Loss Uncertainty Assessment Module

This module provides flood risk assessment based on probability theory.
It uses Monte Carlo Simulation to quantify uncertainty in flood damage estimates.

Reference: UNSAFE Framework - https://github.com/abpoll/unsafe
Keywords: Uncertainty Quantification, Probabilistic Modeling, Damage Curves, Histogram
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_loss_uncertainty(
    flood_depth: float,
    property_value: int,
    num_simulations: int = 5000
) -> plt.Figure:
    """
    Calculate flood loss with uncertainty using Monte Carlo simulation.

    This function runs multiple simulations to estimate the probability
    distribution of financial losses from flooding. It uses a sigmoid-like
    depth-damage curve with Gaussian noise to model structural differences.

    Args:
        flood_depth: The flood water depth in meters.
            Deeper flooding causes more damage.
        property_value: The total property value in USD.
            Higher value properties have higher potential losses.
        num_simulations: Number of Monte Carlo iterations to run (default: 5000).
            More simulations provide more stable estimates.

    Returns:
        plt.Figure: A Matplotlib Figure containing a probability distribution
            histogram of potential losses with Mean and 95% CI vertical lines.

    Example:
        >>> fig = calculate_loss_uncertainty(1.5, 500000, 5000)
        >>> fig.savefig("flood_output.png")
    """
    np.random.seed(42)

    # Monte Carlo Setup: Initialize arrays for iterations
    losses = np.zeros(num_simulations)

    # Damage Function: Sigmoid-like depth-damage curve
    # Damage % = 1 / (1 + exp(-k * (depth - d0)))
    # Where k controls steepness, d0 is depth at 50% damage
    k = 2.0  # Steepness parameter
    d0 = 1.5  # Depth at 50% damage (meters)

    for i in range(num_simulations):
        # Base damage ratio from sigmoid curve
        base_damage_ratio = 1.0 / (1.0 + np.exp(-k * (flood_depth - d0)))

        # Uncertainty Injection: Add Gaussian noise to simulate structural differences
        # Noise represents variability in building construction, contents, etc.
        noise = np.random.normal(0, 0.1)  # ~10% standard deviation

        # Calculate damage ratio with noise (clipped to valid range)
        damage_ratio = np.clip(base_damage_ratio + noise, 0, 1)

        # Financial Calc: Loss = PropertyValue × (DamageRatio + Noise)
        losses[i] = property_value * damage_ratio

    # Calculate statistics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    ci_lower = np.percentile(losses, 2.5)
    ci_upper = np.percentile(losses, 97.5)

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # Plot histogram
    n, bins, patches = ax.hist(
        losses / 1000,  # Convert to thousands
        bins=50,
        density=True,
        color='steelblue',
        edgecolor='white',
        alpha=0.7,
        label='Loss Distribution'
    )

    # Mark Mean and 95% CI with vertical lines
    ax.axvline(
        mean_loss / 1000,
        color='red',
        linestyle='-',
        linewidth=2.5,
        label=f'Mean: ${mean_loss/1000:,.0f}K'
    )
    ax.axvline(
        ci_lower / 1000,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'2.5th %ile: ${ci_lower/1000:,.0f}K'
    )
    ax.axvline(
        ci_upper / 1000,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'97.5th %ile: ${ci_upper/1000:,.0f}K'
    )

    # Shade 95% CI region
    ax.axvspan(
        ci_lower / 1000,
        ci_upper / 1000,
        alpha=0.2,
        color='green',
        label='95% Confidence Interval'
    )

    # Labels and title
    ax.set_xlabel('Loss Amount ($ Thousands)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(
        f'Flood Loss Uncertainty Analysis - Monte Carlo Simulation\n'
        f'Flood Depth: {flood_depth}m | Property Value: ${property_value:,} | '
        f'Simulations: {num_simulations:,}',
        fontsize=13,
        fontweight='bold'
    )

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    damage_pct = mean_loss / property_value * 100
    stats_text = (
        f'Statistics:\n'
        f'Mean Loss: ${mean_loss:,.0f}\n'
        f'Std Dev: ${std_loss:,.0f}\n'
        f'95% CI Width: ${(ci_upper - ci_lower):,.0f}\n'
        f'Damage Ratio: {damage_pct:.1f}%'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Test the function and save output
    print("Running flood loss simulation test...")
    fig = calculate_loss_uncertainty(1.5, 500000, 5000)
    fig.savefig("flood_output.png", dpi=150, bbox_inches='tight')
    print("Output saved to flood_output.png")
    plt.show()
