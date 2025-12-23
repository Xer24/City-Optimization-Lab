# src/csv_export.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data.db import get_conn


def export_ticks(run_id: int, out_path: str) -> None:
    """Export global tick series (congestion + energy_usage) for one run."""
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
        SELECT r.run_id, r.run_label, r.started_at, r.sim_name,
        t.tick, t.congestion, t.energy_usage
        FROM runs r
        JOIN ticks t ON t.run_id = r.run_id
        WHERE r.run_id = ?
        ORDER BY t.tick

            """,
            conn,
            params=(run_id,)
        )

    df.to_csv(out_path, index=False)
    print(f"Wrote CSV -> {out_path}")


def export_run_summaries(out_path: str) -> None:
    """Export one-row-per-run summary table."""
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                r.run_id,
                r.run_label,
                r.started_at,
                r.sim_name,
                s.n_ticks,
                s.n_edges,
                s.congestion_mean,
                s.congestion_max,
                s.energy_sum,
                s.energy_mean,
                s.last_tick
            FROM runs r
            JOIN run_summary s ON s.run_id = r.run_id
            ORDER BY r.run_id DESC
            """,
            conn
        )

    df.to_csv(out_path, index=False)
    print(f"Wrote CSV -> {out_path}")


def export_edge_tick(run_id: int, tick: int, out_path: str) -> None:
    """Export congestion on all edges for a single tick (good for mapping/heatmap)."""
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                run_id,
                tick,
                edge_id,
                congestion
            FROM edge_ticks
            WHERE run_id = ? AND tick = ?
            ORDER BY edge_id
            """,
            conn,
            params=(run_id, tick)
        )

    df.to_csv(out_path, index=False)
    print(f"Wrote CSV -> {out_path}")


def export_edge_series(run_id: int, edge_id: int, out_path: str) -> None:
    """Export congestion over time for one edge."""
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                run_id,
                tick,
                edge_id,
                congestion
            FROM edge_ticks
            WHERE run_id = ? AND edge_id = ?
            ORDER BY tick
            """,
            conn,
            params=(run_id, edge_id)
        )

    df.to_csv(out_path, index=False)
    print(f"Wrote CSV -> {out_path}")


if __name__ == "__main__":
    # Common defaults to test:
    export_ticks(run_id=1, out_path="data/run_1_ticks.csv")
    export_run_summaries(out_path="data/run_summaries.csv")

    # Uncomment if you want edge exports:
    # export_edge_tick(run_id=1, tick=0, out_path="data/run_1_edges_tick0.csv")
    # export_edge_series(run_id=1, edge_id=12, out_path="data/run_1_edge12_series.csv")
