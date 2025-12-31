from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_runs(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find runs CSV at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic sanity
    if "score" not in df.columns:
        raise ValueError("CSV must contain a 'score' column.")

    # Drop failed runs if present
    if "failed" in df.columns:
        df = df[df["failed"].fillna(0).astype(int) == 0].copy()

    # Coerce score numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()

    # Run id (if missing, make one)
    if "run_id" not in df.columns:
        df["run_id"] = range(1, len(df) + 1)

    df = df.sort_values("run_id").reset_index(drop=True)
    return df


def best_so_far_curve(df: pd.DataFrame) -> pd.Series:
    return df["score"].cummin()


def guess_policy_cols(df: pd.DataFrame) -> List[str]:
    # Prefer known names, fallback to float columns excluding obvious outputs.
    known = ["transit_frequency_mult", "congestion_toll", "road_capacity_scale", "signal_green_ratio"]
    policy_cols = [c for c in known if c in df.columns]

    if policy_cols:
        return policy_cols

    exclude = {
        "run_id", "utc_time", "base_seed", "seed", "score", "failed",
        "avg_travel_time", "total_energy", "mean_congestion", "congestion_p95", "total_emissions",
        "avg_energy"
    }

    # Numeric candidates
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    # Heuristic: policy columns usually have small-ish unique counts and not too many zeros
    # (We keep it simple)
    return numeric_cols


def save_best_so_far_plot(df: pd.DataFrame, out_png: str) -> None:
    bs = best_so_far_curve(df)
    plt.figure()
    plt.plot(df["run_id"], bs)
    plt.xlabel("run_id")
    plt.ylabel("best-so-far score (lower is better)")
    plt.title("Best-so-far objective over trials")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_histogram(df: pd.DataFrame, out_png: str, bins: int = 40) -> None:
    plt.figure()
    plt.hist(df["score"].values, bins=bins)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title("Score histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_knob_relationships(df: pd.DataFrame, policy_cols: List[str], out_dir: str) -> None:
    """
    Creates one plot per policy knob:
    - scatter knob vs score
    - also plots a simple binned mean curve for readability
    """
    ensure_dir(out_dir)

    for col in policy_cols:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        y = df["score"]

        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]

        if len(x) < 5:
            continue

        plt.figure()
        plt.scatter(x.values, y.values, s=12, alpha=0.35)
        plt.xlabel(col)
        plt.ylabel("score")
        plt.title(f"{col} vs score")

        # Binned mean line (no seaborn)
        try:
            bins = min(20, max(5, int(len(x) ** 0.5)))
            x_bins = pd.qcut(x, q=bins, duplicates="drop")
            grouped = pd.DataFrame({"x": x, "y": y, "bin": x_bins}).groupby("bin", observed=True)
            x_mean = grouped["x"].mean()
            y_mean = grouped["y"].mean()
            plt.plot(x_mean.values, y_mean.values)
        except Exception:
            pass

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"knob_{col}.png"), dpi=180)
        plt.close()


def correlation_table(df: pd.DataFrame, policy_cols: List[str]) -> pd.DataFrame:
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
        rows.append({"knob": col, "pearson_corr_with_score": float(corr)})
    out = pd.DataFrame(rows).sort_values("pearson_corr_with_score")
    return out


def write_report_md(
    df: pd.DataFrame,
    policy_cols: List[str],
    out_md: str,
    best_json_path: Optional[str],
    figs: dict,
) -> None:
    best_row = df.loc[df["score"].idxmin()]
    lines = []

    lines.append("# Optimization Run Analysis\n")
    lines.append(f"- Runs analyzed: **{len(df)}**\n")
    lines.append(f"- Best score: **{best_row['score']:.6g}** (run_id={int(best_row['run_id'])})\n")

    if best_json_path and os.path.exists(best_json_path):
        lines.append(f"- Best policy JSON: `{best_json_path}`\n")

    lines.append("\n## Plots\n")
    lines.append(f"- Best-so-far curve: `{figs['best_so_far']}`\n")
    lines.append(f"- Score histogram: `{figs['hist']}`\n")
    lines.append(f"- Knob relationship plots folder: `{figs['knobs_dir']}`\n")

    lines.append("\n## Best policy parameters (from best run row)\n")
    lines.append("```json\n")
    best_policy = {k: float(best_row[k]) for k in policy_cols if k in best_row.index}
    lines.append(pd.Series(best_policy).to_json(indent=2))
    lines.append("\n```\n")

    corr_df = correlation_table(df, policy_cols)
    if len(corr_df) > 0:
        lines.append("\n## Knob correlations with score (Pearson)\n")
        lines.append(corr_df.to_markdown(index=False))
        lines.append("\n")

    lines.append("\n## Notes\n")
    lines.append("- Lower score is better.\n")
    lines.append("- Correlation does not prove causation; use as a directional hint.\n")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze optimization runs: plots + knob relationships + report.")
    p.add_argument("--csv", type=str, default="data/opt/policy_runs.csv", help="Runs CSV produced by optimizer.")
    p.add_argument("--out", type=str, default="data/opt/analysis", help="Output directory for plots/report.")
    p.add_argument("--best-json", type=str, default="data/opt/best_policy.json", help="Optional best policy json path.")
    p.add_argument("--bins", type=int, default=40, help="Histogram bin count.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    df = load_runs(args.csv)
    policy_cols = guess_policy_cols(df)

    best_so_far_png = os.path.join(args.out, "best_so_far.png")
    hist_png = os.path.join(args.out, "score_hist.png")
    knobs_dir = os.path.join(args.out, "knobs")

    save_best_so_far_plot(df, best_so_far_png)
    save_histogram(df, hist_png, bins=args.bins)
    save_knob_relationships(df, policy_cols, knobs_dir)

    report_md = os.path.join(args.out, "report.md")
    write_report_md(
        df=df,
        policy_cols=policy_cols,
        out_md=report_md,
        best_json_path=args.best_json,
        figs={"best_so_far": best_so_far_png, "hist": hist_png, "knobs_dir": knobs_dir},
    )

    print("Wrote:")
    print(" -", best_so_far_png)
    print(" -", hist_png)
    print(" -", knobs_dir)
    print(" -", report_md)


if __name__ == "__main__":
    main()
