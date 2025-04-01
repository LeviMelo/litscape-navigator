# backend/database.py
import sqlite3
import logging
from typing import Optional, Dict, Any
import datetime  # Import datetime for timestamp

DATABASE_FILE = "jobs.db"
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a database connection."""
    # Ensure isolation level allows concurrent reads/writes if needed, though Celery usually uses separate processes
    conn = sqlite3.connect(DATABASE_FILE, detect_types=sqlite3.PARSE_DECLTYPES, timeout=10) # Added timeout
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the jobs table if it doesn't exist."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Added column 'results_dois_path'
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'RUNNING', 'STARTED', 'COMPLETED', 'FAILED')), -- Added STARTED
                query TEXT,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP, -- Optional: track when worker picked it up
                finished_at TIMESTAMP, -- Optional: track when finished
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
    """Adds a new job with PENDING status and current timestamp."""
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

def update_job_status(job_id: str, status: str, error_message: Optional[str] = None, set_started: bool = False, set_finished: bool = False) -> bool:
    """Updates the status and optionally error message, started_at, finished_at of a job."""
    updates = ["status = ?", "error_message = ?"]
    params = [status, error_message if error_message else None]
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if set_started:
        updates.append("started_at = ?")
        params.append(now_utc)
    if set_finished:
        updates.append("finished_at = ?")
        params.append(now_utc)

    params.append(job_id) # For WHERE clause
    sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        conn.commit()
        updated_rows = cursor.rowcount
        if updated_rows > 0:
            log.info(f"Updated job {job_id} status to {status}.")
            return True
        else:
            # Don't log warning if status is PENDING/STARTED, task might not exist yet in DB when signal fires
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
    """Gets the status and error message (if any) of a job."""
    # Include started_at, finished_at if needed by frontend later
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

# ***** This is the corrected version from previous step *****
def store_results_path(job_id: str,
                       results_data_path: Optional[str] = None,
                       results_graph_path: Optional[str] = None,
                       results_embeddings_path: Optional[str] = None,
                       results_umap_path: Optional[str] = None,
                       results_clusters_path: Optional[str] = None,
                       results_dois_path: Optional[str] = None
                       ) -> bool:
    """Stores the paths to the various result files for a completed job."""
    updates = []
    params = []

    # Build the SET clause and parameter list dynamically
    if results_data_path: updates.append("results_data_path = ?"); params.append(results_data_path)
    if results_graph_path: updates.append("results_graph_path = ?"); params.append(results_graph_path)
    if results_embeddings_path: updates.append("results_embeddings_path = ?"); params.append(results_embeddings_path)
    if results_umap_path: updates.append("results_umap_path = ?"); params.append(results_umap_path)
    if results_clusters_path: updates.append("results_clusters_path = ?"); params.append(results_clusters_path)
    if results_dois_path: updates.append("results_dois_path = ?"); params.append(results_dois_path) # Add update for dois path

    if not updates:
        log.warning(f"No valid result paths provided to store for job {job_id}")
        return False

    params.append(job_id) # Add job_id for the WHERE clause
    sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        conn.commit()
        updated_rows = cursor.rowcount
        if updated_rows > 0:
            log.info(f"Stored {len(updates)} result path(s) for job {job_id}")
            return True
        else:
            log.warning(f"Job {job_id} not found for storing result paths.")
            return False
    except sqlite3.Error as e:
        log.error(f"DB error storing result paths for job {job_id}: {e}", exc_info=True)
        return False
    finally:
        if conn: conn.close()

def get_results_path(job_id: str) -> Optional[Dict[str, Optional[str]]]:
    """Gets all stored result file paths for a job."""
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
        if conn: conn.close()

# Initialize the database when the module is first imported.
init_db()