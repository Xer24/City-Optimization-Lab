import sqlite3
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

DB_PATH = Path(__file__).resolve().parent / "sim.db"


# ----------------------------
# Connection helpers
# ----------------------------
def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def tune_for_bulk(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")


def chunked(seq: List[tuple], chunk_size: int) -> Iterable[List[tuple]]:
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


# ----------------------------
# Schema
# ----------------------------
def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            sim_name TEXT NOT NULL,
            run_number INTEGER NOT NULL,
            run_label TEXT NOT NULL,
            params_json TEXT,
            UNIQUE(sim_name, run_number),
            UNIQUE(run_label)
        );
        """)

        # Tick-level globals (NOW: 3 congestion columns)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            car_congestion REAL NOT NULL,
            ped_congestion REAL NOT NULL,
            transit_congestion REAL NOT NULL,
            energy_usage REAL NOT NULL,
            PRIMARY KEY (run_id, tick),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        # Edge-level flows by mode
        conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_ticks_mode (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            edge_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            flow REAL NOT NULL,
            PRIMARY KEY (run_id, tick, edge_id, mode),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        # Summary per run (we summarize car_congestion only for now)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS run_summary (
            run_id INTEGER PRIMARY KEY,
            n_ticks INTEGER NOT NULL,
            n_edges INTEGER,
            car_congestion_mean REAL,
            car_congestion_max REAL,
            energy_sum REAL,
            energy_mean REAL,
            last_tick INTEGER,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_run ON ticks(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_mode_run ON edge_ticks_mode(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_mode_edge ON edge_ticks_mode(run_id, edge_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_mode_tick ON edge_ticks_mode(run_id, tick);")


# ----------------------------
# Run creation
# ----------------------------
def create_run(conn: sqlite3.Connection, sim_name: str, params: dict) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(MAX(run_number), 0) + 1 FROM runs WHERE sim_name = ?",
        (sim_name,)
    )
    run_number = cur.fetchone()[0]
    run_label = f"{sim_name}-{run_number:04d}"

    cur.execute(
        """
        INSERT INTO runs (started_at, sim_name, run_number, run_label, params_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.now(timezone.utc).isoformat(), sim_name, run_number, run_label, json.dumps(params))
    )
    return cur.lastrowid


# ----------------------------
# Inserts
# ----------------------------
# ticks: (tick, car_cong, ped_cong, transit_cong, energy_usage)
TickRow = Tuple[int, float, float, float, float]

# edge-mode: (tick, edge_id, mode, flow)
EdgeModeRow = Tuple[int, int, str, float]


def insert_ticks(conn: sqlite3.Connection, run_id: int, tick_rows: List[TickRow]) -> None:
    conn.executemany(
        """
        INSERT INTO ticks
        (run_id, tick, car_congestion, ped_congestion, transit_congestion, energy_usage)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(run_id, *row) for row in tick_rows]
    )


def insert_edge_ticks_mode(
    conn: sqlite3.Connection,
    run_id: int,
    edge_mode_rows: List[EdgeModeRow],
    chunk_size: int = 50_000
) -> None:
    sql = """
    INSERT INTO edge_ticks_mode (run_id, tick, edge_id, mode, flow)
    VALUES (?, ?, ?, ?, ?)
    """
    for batch in chunked(edge_mode_rows, chunk_size):
        conn.executemany(sql, [(run_id, *row) for row in batch])


# ----------------------------
# Summary
# ----------------------------
def update_run_summary(conn: sqlite3.Connection, run_id: int) -> None:
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) AS n_ticks,
            AVG(car_congestion) AS car_congestion_mean,
            MAX(car_congestion) AS car_congestion_max,
            SUM(energy_usage) AS energy_sum,
            AVG(energy_usage) AS energy_mean,
            MAX(tick) AS last_tick
        FROM ticks
        WHERE run_id = ?
    """, (run_id,))
    n_ticks, car_mean, car_max, e_sum, e_mean, last_tick = cur.fetchone()

    cur.execute("SELECT COUNT(DISTINCT edge_id) FROM edge_ticks_mode WHERE run_id = ?", (run_id,))
    n_edges = cur.fetchone()[0]

    conn.execute("""
        INSERT INTO run_summary
        (run_id, n_ticks, n_edges, car_congestion_mean, car_congestion_max, energy_sum, energy_mean, last_tick)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            n_ticks=excluded.n_ticks,
            n_edges=excluded.n_edges,
            car_congestion_mean=excluded.car_congestion_mean,
            car_congestion_max=excluded.car_congestion_max,
            energy_sum=excluded.energy_sum,
            energy_mean=excluded.energy_mean,
            last_tick=excluded.last_tick
    """, (run_id, n_ticks or 0, n_edges, car_mean, car_max, e_sum, e_mean, last_tick))


# ----------------------------
# One-call save API
# ----------------------------
def save_run(
    sim_name: str,
    params: dict,
    tick_rows: List[TickRow],
    edge_mode_rows: Optional[List[EdgeModeRow]] = None,
    edge_chunk_size: int = 50_000
) -> int:
    edge_mode_rows = edge_mode_rows or []

    with get_conn() as conn:
        tune_for_bulk(conn)
        conn.execute("BEGIN;")
        try:
            run_id = create_run(conn, sim_name, params)
            insert_ticks(conn, run_id, tick_rows)
            if edge_mode_rows:
                insert_edge_ticks_mode(conn, run_id, edge_mode_rows, chunk_size=edge_chunk_size)
            update_run_summary(conn, run_id)
            conn.commit()
            return run_id
        except Exception:
            conn.rollback()
            raise


if __name__ == "__main__":
    init_db()
    print(f"DB ready at: {DB_PATH}")
