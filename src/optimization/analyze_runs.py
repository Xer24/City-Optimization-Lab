"""
Optimization run analysis and visualization.

Analyzes optimization logs to identify trends, policy impacts, and
best configurations. Generates plots, correlation analysis, and
comprehensive reports.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Dict, Tuple
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ============================================================
# Utility Functions
# ============================================================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_runs(csv_path: str) -> pd.DataFrame:
    """
    Load and validate optimization runs CSV.
    
    Args:
        csv_path: Path to runs CSV file
    
    Returns:
        Cleaned DataFrame with valid runs
    
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If CSV format is invalid
    
    Example:
        >>> df = load_runs("data/opt/policy_runs.csv")
        >>> print(f"Loaded {len(df)} runs")
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find runs CSV at: {csv_path}")
    
    logger.info(f"Loading runs from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if "score" not in df.columns:
        raise ValueError("CSV must contain a 'score' column")
    
    # Remove failed runs
    if "failed" in df.columns:
        n_failed = df["failed"].fillna(0).astype(int).sum()
        if n_failed > 0:
            logger.warning(f"Removing {n_failed} failed runs")
        df = df[df["failed"].fillna(0).astype(int) == 0].copy()
    
    # Coerce score to numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    n_invalid = df["score"].isna().sum()
    if n_invalid > 0:
        logger.warning(f"Removing {n_invalid} runs with invalid scores")
    df = df.dropna(subset=["score"]).copy()
    
    # Ensure run_id exists
    if "run_id" not in df.columns:
        logger.info("Creating run_id column")
        df["run_id"] = range(1, len(df) + 1)
    
    # Sort by run_id
    df = df.sort_values("run_id").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} valid runs")
    return df


# ============================================================
# Analysis Functions
# ============================================================

def best_so_far_curve(df: pd.DataFrame) -> pd.Series:
    """
    Compute best-so-far objective over trials.
    
    Args:
        df: DataFrame with 'score' column
    
    Returns:
        Series of cumulative minimum scores
    """
    return df["score"].cummin()


def guess_policy_cols(df: pd.DataFrame) -> List[str]:
    """
    Identify policy parameter columns in DataFrame.
    
    Args:
        df: Runs DataFrame
    
    Returns:
        List of column names that appear to be policy parameters
    
    Note:
        Uses heuristics: known names, numeric dtype, reasonable value ranges
    """
    # Prefer known policy parameter names
    known = [
        "transit_frequency_mult",
        "congestion_toll",
        "road_capacity_scale",
        "signal_green_ratio",  # For future use
    ]
    
    policy_cols = [c for c in known if c in df.columns]
    
    if policy_cols:
        logger.info(f"Found policy columns: {policy_cols}")
        return policy_cols
    
    # Fallback: guess from numeric columns
    exclude = {
        "run_id", "utc_time", "base_seed", "seed", "score", "failed",
        "avg_travel_time", "total_energy", "mean_congestion",
        "congestion_p95", "total_emissions", "avg_energy",
    }
    
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    logger.info(f"Guessed policy columns: {numeric_cols}")
    return numeric_cols


def compute_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for optimization runs.
    
    Args:
        df: Runs DataFrame
    
    Returns:
        Dictionary with statistics
    """
    scores = df["score"]
    
    return {
        "n_runs": len(df),
        "best_score": float(scores.min()),
        "worst_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "median_score": float(scores.median()),
        "std_score": float(scores.std()),
        "improvement": float(scores.iloc[0] - scores.min()) if len(scores) > 0 else 0.0,
    }


# ============================================================
# Visualization Functions
# ============================================================

def save_best_so_far_plot(
    df: pd.DataFrame,
    out_png: str,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot best-so-far objective over trials.
    
    Args:
        df: Runs DataFrame
        out_png: Output PNG file path
        figsize: Figure size (width, height)
    """
    best_so_far = best_so_far_curve(df)
    
    plt.figure(figsize=figsize)
    plt.plot(df["run_id"], best_so_far, linewidth=2, color="darkblue")
    plt.scatter(df["run_id"], df["score"], alpha=0.3, s=10, color="lightblue")
    
    plt.xlabel("Run ID", fontsize=12)
    plt.ylabel("Best-so-far Score (lower is better)", fontsize=12)
    plt.title("Optimization Progress", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved best-so-far plot to {out_png}")


def save_histogram(
    df: pd.DataFrame,
    out_png: str,
    bins: int = 40,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot histogram of objective scores.
    
    Args:
        df: Runs DataFrame
        out_png: Output PNG file path
        bins: Number of histogram bins
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    scores = df["score"].values  # This is already a numpy array
    plt.hist(scores, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    
    # Add vertical lines for statistics
    plt.axvline(float(scores.min()), color="green", linestyle="--", linewidth=2, label="Best")
    plt.axvline(float(np.median(scores)), color="orange", linestyle="--", linewidth=2, label="Median")
    plt.axvline(float(scores.mean()), color="red", linestyle="--", linewidth=2, label="Mean")
    
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Score Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved histogram to {out_png}")


def save_knob_relationships(
    df: pd.DataFrame,
    policy_cols: List[str],
    out_dir: str,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Create scatter plots of policy parameters vs score.
    
    For each policy parameter, creates:
    - Scatter plot of parameter value vs score
    - Binned mean curve showing trend
    
    Args:
        df: Runs DataFrame
        policy_cols: List of policy parameter columns
        out_dir: Output directory for plots
        figsize: Figure size
    """
    ensure_dir(out_dir)
    
    for col in policy_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
        
        # Convert to numeric
        x = pd.to_numeric(df[col], errors="coerce")
        y = df["score"]
        
        # Filter valid data
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        
        if len(x) < 5:
            logger.warning(f"Too few valid points for {col}, skipping")
            continue
        
        plt.figure(figsize=figsize)
        
        # Scatter plot - x and y are already pandas Series
        plt.scatter(x.to_numpy(), y.to_numpy(), s=20, alpha=0.4, color="steelblue")
        
        # Binned mean curve
        try:
            n_bins = min(20, max(5, int(len(x) ** 0.5)))
            x_bins = pd.qcut(x, q=n_bins, duplicates="drop")
            grouped = pd.DataFrame({"x": x, "y": y, "bin": x_bins}).groupby(
                "bin", observed=True
            )
            x_mean = grouped["x"].mean()
            y_mean = grouped["y"].mean()
            plt.plot(
                x_mean.to_numpy(),
                y_mean.to_numpy(),
                color="darkred",
                linewidth=2,
                marker="o",
                label="Binned Mean",
            )
            plt.legend()
        except Exception as e:
            logger.debug(f"Could not add binned mean for {col}: {e}")
        
        plt.xlabel(col.replace("_", " ").title(), fontsize=12)
        plt.ylabel("Score (lower is better)", fontsize=12)
        plt.title(f"Policy Impact: {col}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(out_dir, f"knob_{col}.png")
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()
        
        logger.debug(f"Saved {col} plot to {output_path}")
    
    logger.info(f"Saved {len(policy_cols)} knob relationship plots to {out_dir}")


def save_correlation_heatmap(
    df: pd.DataFrame,
    policy_cols: List[str],
    out_png: str,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Create correlation heatmap for policy parameters and score.
    
    Args:
        df: Runs DataFrame
        policy_cols: List of policy parameter columns
        out_png: Output PNG file path
        figsize: Figure size
    """
    # Select relevant columns
    cols_to_plot = [c for c in policy_cols if c in df.columns] + ["score"]
    subset = df[cols_to_plot].select_dtypes(include=[np.number])
    
    if len(subset.columns) < 2:
        logger.warning("Not enough numeric columns for correlation heatmap")
        return
    
    # Compute correlation matrix
    corr_matrix = subset.corr()
    
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    im = plt.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation", fontsize=12)
    
    # Set ticks and labels
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Annotate cells with correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = plt.text(
                j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=10
            )
    
    plt.title("Parameter Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved correlation heatmap to {out_png}")


def correlation_table(
    df: pd.DataFrame,
    policy_cols: List[str],
) -> pd.DataFrame:
    """
    Compute correlation between policy parameters and score.
    
    Args:
        df: Runs DataFrame
        policy_cols: List of policy parameter columns
    
    Returns:
        DataFrame with correlations, sorted by absolute value
    """
    rows = []
    
    for col in policy_cols:
        if col not in df.columns:
            continue
        
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df["score"], errors="coerce")
        
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        
        corr = x[mask].corr(y[mask])
        rows.append({
            "parameter": col,
            "correlation": float(corr),
            "abs_correlation": abs(float(corr)),
        })
    
    result = pd.DataFrame(rows)
    
    if len(result) > 0:
        result = result.sort_values("abs_correlation", ascending=False)
    
    return result


# ============================================================
# Report Generation
# ============================================================

def write_report_md(
    df: pd.DataFrame,
    policy_cols: List[str],
    out_md: str,
    best_json_path: Optional[str],
    figs: Dict[str, str],
) -> None:
    """
    Generate markdown analysis report.
    
    Args:
        df: Runs DataFrame
        policy_cols: List of policy parameter columns
        out_md: Output markdown file path
        best_json_path: Path to best policy JSON (optional)
        figs: Dictionary of figure paths
    """
    stats = compute_statistics(df)
    best_row = df.loc[df["score"].idxmin()]
    
    lines = []
    
    # Header
    lines.append("# Optimization Run Analysis\n")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary statistics
    lines.append("## Summary Statistics\n")
    lines.append(f"- **Total runs**: {stats['n_runs']}")
    lines.append(f"- **Best score**: {stats['best_score']:.6g} (run_id={int(best_row['run_id'])})")
    lines.append(f"- **Worst score**: {stats['worst_score']:.6g}")
    lines.append(f"- **Mean score**: {stats['mean_score']:.6g}")
    lines.append(f"- **Median score**: {stats['median_score']:.6g}")
    lines.append(f"- **Std deviation**: {stats['std_score']:.6g}")
    lines.append(f"- **Total improvement**: {stats['improvement']:.6g} ({100.0 * stats['improvement'] / df['score'].iloc[0]:.1f}%)\n")
    
    if best_json_path and os.path.exists(best_json_path):
        lines.append(f"- **Best policy JSON**: `{best_json_path}`\n")
    
    # Plots
    lines.append("## Visualizations\n")
    lines.append(f"- **Best-so-far curve**: `{figs.get('best_so_far', 'N/A')}`")
    lines.append(f"- **Score histogram**: `{figs.get('hist', 'N/A')}`")
    lines.append(f"- **Correlation heatmap**: `{figs.get('corr_heatmap', 'N/A')}`")
    lines.append(f"- **Parameter relationship plots**: `{figs.get('knobs_dir', 'N/A')}`\n")
    
    # Best policy parameters
    lines.append("## Best Policy Parameters\n")
    lines.append("```json")
    best_policy = {k: float(best_row[k]) for k in policy_cols if k in best_row.index}
    import json
    lines.append(json.dumps(best_policy, indent=2))
    lines.append("```\n")
    
    # Correlation analysis
    corr_df = correlation_table(df, policy_cols)
    if len(corr_df) > 0:
        lines.append("## Parameter Correlations with Score\n")
        lines.append("Pearson correlation coefficients (negative = reduces score):\n")
        corr_display = corr_df[["parameter", "correlation"]].copy()
        corr_display.columns = ["Parameter", "Correlation"]
        lines.append(corr_display.to_markdown(index=False))
        lines.append("\n")
    
    # Notes
    lines.append("## Notes\n")
    lines.append("- **Lower scores are better**")
    lines.append("- Correlation does not prove causation; use as directional guidance")
    lines.append("- Negative correlation means parameter reduces score (improvement)")
    lines.append("- Positive correlation means parameter increases score (degradation)\n")
    
    # Write file
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Saved analysis report to {out_md}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze optimization runs: plots + correlations + report"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default="data/opt/policy_runs.csv",
        help="Runs CSV produced by optimizer (default: data/opt/policy_runs.csv)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/opt/analysis",
        help="Output directory for plots/report (default: data/opt/analysis)"
    )
    parser.add_argument(
        "--best-json",
        type=str,
        default="data/opt/best_policy.json",
        help="Best policy JSON path (optional)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count (default: 40)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for run analysis."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load runs
    try:
        df = load_runs(args.csv)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except ValueError as e:
        logger.error(f"Invalid CSV format: {e}")
        return
    
    # Identify policy columns
    policy_cols = guess_policy_cols(df)
    
    if not policy_cols:
        logger.warning("No policy columns identified")
    
    # Create output directory
    ensure_dir(args.out)
    
    # Generate plots
    logger.info("Generating visualizations...")
    
    best_so_far_png = os.path.join(args.out, "best_so_far.png")
    hist_png = os.path.join(args.out, "score_hist.png")
    corr_heatmap_png = os.path.join(args.out, "correlation_heatmap.png")
    knobs_dir = os.path.join(args.out, "knobs")
    
    save_best_so_far_plot(df, best_so_far_png)
    save_histogram(df, hist_png, bins=args.bins)
    save_knob_relationships(df, policy_cols, knobs_dir)
    save_correlation_heatmap(df, policy_cols, corr_heatmap_png)
    
    # Generate report
    logger.info("Generating analysis report...")
    report_md = os.path.join(args.out, "report.md")
    write_report_md(
        df=df,
        policy_cols=policy_cols,
        out_md=report_md,
        best_json_path=args.best_json,
        figs={
            "best_so_far": best_so_far_png,
            "hist": hist_png,
            "corr_heatmap": corr_heatmap_png,
            "knobs_dir": knobs_dir,
        },
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Analyzed {len(df)} optimization runs")
    print(f"\nOutput files:")
    print(f"  - Best-so-far plot: {best_so_far_png}")
    print(f"  - Score histogram: {hist_png}")
    print(f"  - Correlation heatmap: {corr_heatmap_png}")
    print(f"  - Parameter plots: {knobs_dir}/")
    print(f"  - Analysis report: {report_md}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()