# backend/database.py
import sqlite3
import logging
from typing import Optional, Dict, Any
import datetime # Import datetime for timestamp

DATABASE_FILE = "jobs.db" # Database file will be created in the backend folder
log = logging.getLogger(__name__) # Get logger instance

# Configure logging if not already configured elsewhere (e.g., in main.py)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a database connection."""
    conn = sqlite3.connect(DATABASE_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    # Use Row factory to access columns by name
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the jobs table if it doesn't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Create table: job_id (text, primary key), status (text),
        # query (text), created_at (timestamp), and paths for result files (nullable)
        # Added error_message field
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
                query TEXT,
                created_at TIMESTAMP NOT NULL,
                error_message TEXT,
                results_data_path TEXT,
                results_graph_path TEXT,
                results_embeddings_path TEXT,
                results_clusters_path TEXT,
                results_umap_path TEXT
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
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Use current UTC time
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

def update_job_status(job_id: str, status: str, error_message: Optional[str] = None) -> bool:
    """Updates the status and optionally error message of a job."""
    sql = "UPDATE jobs SET status = ?, error_message = ? WHERE job_id = ?"
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Ensure error_message is explicitly None if not provided
        err_msg = error_message if error_message else None
        cursor.execute(sql, (status, err_msg, job_id))
        conn.commit()
        updated_rows = cursor.rowcount
        if updated_rows > 0:
            log.info(f"Updated job {job_id} status to {status}.")
            return True
        else:
            log.warning(f"Job {job_id} not found for status update.")
            return False
    except sqlite3.Error as e:
        log.error(f"Database error updating status for job {job_id}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Gets the status and error message (if any) of a job."""
    sql = "SELECT status, error_message FROM jobs WHERE job_id = ?"
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (job_id,))
        job = cursor.fetchone()
        if job:
            return dict(job) # Convert row object to dictionary
        else:
            log.warning(f"Job {job_id} not found when querying status.")
            return None
    except sqlite3.Error as e:
        log.error(f"Database error getting status for job {job_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def store_results_path(job_id: str, data_path: Optional[str] = None, graph_path: Optional[str] = None) -> bool:
    """Stores the path(s) to the results file(s) for a job. Expand later."""
    # This function will grow as we add more result file types
    if not data_path and not graph_path:
        log.warning(f"No paths provided to store for job {job_id}")
        return False

    updates = []
    params = []
    if data_path:
        updates.append("results_data_path = ?")
        params.append(data_path)
    if graph_path:
        updates.append("results_graph_path = ?")
        params.append(graph_path)
    # Add more paths here later (embeddings, umap, clusters)

    params.append(job_id) # Add job_id for the WHERE clause
    sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        conn.commit()
        updated_rows = cursor.rowcount
        if updated_rows > 0:
            log.info(f"Stored results path(s) for job {job_id}")
            return True
        else:
            log.warning(f"Job {job_id} not found for storing results path(s).")
            return False
    except sqlite3.Error as e:
        log.error(f"Database error storing results path(s) for job {job_id}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_results_path(job_id: str) -> Optional[Dict[str, Optional[str]]]:
    """Gets the path(s) to the results file(s) for a job."""
    # Select all potential result paths
    sql = """
        SELECT results_data_path, results_graph_path, results_embeddings_path,
               results_clusters_path, results_umap_path
        FROM jobs WHERE job_id = ?
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (job_id,))
        result = cursor.fetchone()
        return dict(result) if result else None
    except sqlite3.Error as e:
        log.error(f"Database error getting results path(s) for job {job_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

# --- Initialize the database when the module is first imported ---
# This ensures the table exists before other functions try to use it.
init_db()