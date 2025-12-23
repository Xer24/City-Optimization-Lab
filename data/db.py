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
    """
    Safe speed-ups for big inserts.
    WAL helps with write performance and reduces locking.
    """
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")


def chunked(seq: List[tuple], chunk_size: int) -> Iterable[List[tuple]]:
    """
    Chunk a list into batches for executemany to avoid huge memory spikes.
    """
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

        # Global values per tick
        conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            congestion REAL NOT NULL,
            energy_usage REAL NOT NULL,
            PRIMARY KEY (run_id, tick),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        # Congestion per road segment (edge) per tick
        conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_ticks (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            edge_id INTEGER NOT NULL,
            congestion REAL NOT NULL,
            PRIMARY KEY (run_id, tick, edge_id),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        # Summary stats per run
        conn.execute("""
        CREATE TABLE IF NOT EXISTS run_summary (
            run_id INTEGER PRIMARY KEY,
            n_ticks INTEGER NOT NULL,
            n_edges INTEGER,
            congestion_mean REAL,
            congestion_max REAL,
            energy_sum REAL,
            energy_mean REAL,
            last_tick INTEGER,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        # Indexes for speed
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_run ON ticks(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_ticks_run ON edge_ticks(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_ticks_edge ON edge_ticks(run_id, edge_id);")


# ----------------------------
# Run creation (auto increment naming)
# ----------------------------
def create_run(conn: sqlite3.Connection, sim_name: str, params: dict) -> int:
    """
    Creates a run with run_number incrementing per sim_name, e.g. city_sim-0007
    """
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
# tick_rows format: [(tick, congestion, energy_usage), ...]
TickRow = Tuple[int, float, float]

# edge_rows format: [(tick, edge_id, congestion), ...]
EdgeRow = Tuple[int, int, float]


def insert_ticks(conn: sqlite3.Connection, run_id: int, tick_rows: List[TickRow]) -> None:
    conn.executemany(
        "INSERT INTO ticks (run_id, tick, congestion, energy_usage) VALUES (?, ?, ?, ?)",
        [(run_id, *row) for row in tick_rows]
    )


def insert_edge_ticks(
    conn: sqlite3.Connection,
    run_id: int,
    edge_rows: List[EdgeRow],
    chunk_size: int = 50_000
) -> None:
    """
    Chunked insert for potentially huge (tick x edges) data.
    """
    sql = "INSERT INTO edge_ticks (run_id, tick, edge_id, congestion) VALUES (?, ?, ?, ?)"
    for batch in chunked(edge_rows, chunk_size):
        conn.executemany(sql, [(run_id, *row) for row in batch])


# ----------------------------
# Summary
# ----------------------------
def update_run_summary(conn: sqlite3.Connection, run_id: int) -> None:
    cur = conn.cursor()

    # Global tick summary
    cur.execute("""
        SELECT
            COUNT(*) AS n_ticks,
            AVG(congestion) AS congestion_mean,
            MAX(congestion) AS congestion_max,
            SUM(energy_usage) AS energy_sum,
            AVG(energy_usage) AS energy_mean,
            MAX(tick) AS last_tick
        FROM ticks
        WHERE run_id = ?
    """, (run_id,))
    n_ticks, cong_mean, cong_max, e_sum, e_mean, last_tick = cur.fetchone()

    # How many distinct edges we stored for this run
    cur.execute("SELECT COUNT(DISTINCT edge_id) FROM edge_ticks WHERE run_id = ?", (run_id,))
    n_edges = cur.fetchone()[0]

    conn.execute("""
        INSERT INTO run_summary
        (run_id, n_ticks, n_edges, congestion_mean, congestion_max, energy_sum, energy_mean, last_tick)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            n_ticks=excluded.n_ticks,
            n_edges=excluded.n_edges,
            congestion_mean=excluded.congestion_mean,
            congestion_max=excluded.congestion_max,
            energy_sum=excluded.energy_sum,
            energy_mean=excluded.energy_mean,
            last_tick=excluded.last_tick
    """, (run_id, n_ticks or 0, n_edges, cong_mean, cong_max, e_sum, e_mean, last_tick))


# ----------------------------
# One-call "save everything" API
# ----------------------------
def save_run(
    sim_name: str,
    params: dict,
    tick_rows: List[TickRow],
    edge_rows: Optional[List[EdgeRow]] = None,
    edge_chunk_size: int = 50_000
) -> int:
    """
    Save one full simulation run.

    tick_rows: [(tick, congestion, energy_usage), ...]
    edge_rows: [(tick, edge_id, congestion), ...]  (optional but recommended)
    """
    edge_rows = edge_rows or []

    with get_conn() as conn:
        tune_for_bulk(conn)
        conn.execute("BEGIN;")
        try:
            run_id = create_run(conn, sim_name, params)
            insert_ticks(conn, run_id, tick_rows)
            if edge_rows:
                insert_edge_ticks(conn, run_id, edge_rows, chunk_size=edge_chunk_size)
            update_run_summary(conn, run_id)
            conn.commit()
            return run_id
        except Exception:
            conn.rollback()
            raise


if __name__ == "__main__":
    init_db()
    print(f"DB ready at: {DB_PATH}")
