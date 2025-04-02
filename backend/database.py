import sqlite3
import logging
from typing import Optional, Dict, Any
import datetime  # For timestamps

DATABASE_FILE = "jobs.db"
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_db_connection() -> sqlite3.Connection:
    # Added timeout to help with concurrent access.
    conn = sqlite3.connect(DATABASE_FILE, detect_types=sqlite3.PARSE_DECLTYPES, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Create table with all required columns (including started_at, finished_at, results_dois_path)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'RUNNING', 'STARTED', 'COMPLETED', 'FAILED')),
                query TEXT,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                error_message TEXT,
                results_data_path TEXT,
                results_graph_path TEXT,
                results_embeddings_path TEXT,
                results_clusters_path TEXT,
                results_umap_path TEXT,
                results_dois_path TEXT
            )
        """)
        conn.commit()
        log.info("Database table 'jobs' initialized successfully.")
    except sqlite3.Error as e:
        log.error(f"Database error during initialization: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def add_job(job_id: str, query: str) -> bool:
    sql = "INSERT INTO jobs (job_id, status, query, created_at) VALUES (?, ?, ?, ?)"
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        created_time = datetime.datetime.now(datetime.timezone.utc)
        cursor.execute(sql, (job_id, 'PENDING', query, created_time))
        conn.commit()
        log.info(f"Added job {job_id} with status PENDING.")
        return True
    except sqlite3.Error as e:
        log.error(f"Database error adding job {job_id}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def update_job_status(job_id: str, status: str, error_message: Optional[str] = None,
                      set_started: bool = False, set_finished: bool = False) -> bool:
    updates = ["status = ?", "error_message = ?"]
    params = [status, error_message if error_message else None]
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if set_started:
        updates.append("started_at = ?")
        params.append(now_utc)
    if set_finished:
        updates.append("finished_at = ?")
        params.append(now_utc)
    params.append(job_id)
    sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        conn.commit()
        if cursor.rowcount > 0:
            log.info(f"Updated job {job_id} status to {status}.")
            return True
        else:
            if status not in ['PENDING', 'STARTED']:
                log.warning(f"Job {job_id} not found for status update to {status}.")
            return False
    except sqlite3.Error as e:
        log.error(f"Database error updating status for job {job_id}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT status, error_message FROM jobs WHERE job_id = ?"
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (job_id,))
        job = cursor.fetchone()
        return dict(job) if job else None
    except sqlite3.Error as e:
        log.error(f"Database error getting status for job {job_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def store_results_path(job_id: str,
                       results_data_path: Optional[str] = None,
                       results_graph_path: Optional[str] = None,
                       results_embeddings_path: Optional[str] = None,
                       results_umap_path: Optional[str] = None,
                       results_clusters_path: Optional[str] = None,
                       results_dois_path: Optional[str] = None) -> bool:
    updates = []
    params = []
    if results_data_path:
        updates.append("results_data_path = ?")
        params.append(results_data_path)
    if results_graph_path:
        updates.append("results_graph_path = ?")
        params.append(results_graph_path)
    if results_embeddings_path:
        updates.append("results_embeddings_path = ?")
        params.append(results_embeddings_path)
    if results_umap_path:
        updates.append("results_umap_path = ?")
        params.append(results_umap_path)
    if results_clusters_path:
        updates.append("results_clusters_path = ?")
        params.append(results_clusters_path)
    if results_dois_path:
        updates.append("results_dois_path = ?")
        params.append(results_dois_path)
    if not updates:
        log.warning(f"No valid result paths provided to store for job {job_id}")
        return False
    params.append(job_id)
    sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        conn.commit()
        if cursor.rowcount > 0:
            log.info(f"Stored {len(updates)} result path(s) for job {job_id}")
            return True
        else:
            log.warning(f"Job {job_id} not found for storing result paths.")
            return False
    except sqlite3.Error as e:
        log.error(f"DB error storing result paths for job {job_id}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_results_path(job_id: str) -> Optional[Dict[str, Optional[str]]]:
    sql = """
        SELECT results_data_path, results_graph_path, results_embeddings_path,
               results_umap_path, results_clusters_path, results_dois_path
        FROM jobs WHERE job_id = ?
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (job_id,))
        result = cursor.fetchone()
        return dict(result) if result else None
    except sqlite3.Error as e:
        log.error(f"DB error getting results paths for job {job_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

# Initialize database on module load.
init_db()
