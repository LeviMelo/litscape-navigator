# backend/main.py
from fastapi import FastAPI, HTTPException, status as HttpStatus
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, Any, Dict
import uuid
import json
import os
import pickle # To potentially load graph later
import networkx as nx # To potentially load graph later

# Import NEW Celery task instance and database functions
from celery_worker import celery_app, run_full_pipeline # Using the new task name
import database

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
log = logging.getLogger(__name__)

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 50 # Default max results

class JobResponse(BaseModel):
    job_id: str

class StatusResponse(BaseModel):
    job_id: str
    status: str # PENDING, RUNNING, COMPLETED, FAILED
    error_message: Optional[str] = None

class GraphInfo(BaseModel):
    nodes: Optional[int] = None
    edges: Optional[int] = None
    graph_file_exists: bool = False

class ResultsResponse(BaseModel):
    # Use Dict[str, Any] for merged_data initially for flexibility
    merged_data: Dict[str, Any]
    graph_info: GraphInfo

# --- FastAPI App Instance & CORS ---
app = FastAPI(title="LitScape Navigator API", version="0.2.0")
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Endpoints ---
@app.get("/")
def read_index():
    return {"message": "Welcome to LitScape Navigator API v0.2"}

@app.post("/api/searches", response_model=JobResponse, status_code=HttpStatus.HTTP_202_ACCEPTED)
async def start_search_job(search_request: SearchRequest):
    """Initiates the full background data processing pipeline."""
    job_id = str(uuid.uuid4())
    log.info(f"Received request start FULL PIPELINE job {job_id} query: '{search_request.query}'")
    if not database.add_job(job_id, search_request.query):
         log.error(f"Failed add job {job_id} to DB."); raise HTTPException(status_code=500, detail="DB error.")
    try:
        # Call the correct task
        run_full_pipeline.delay(job_id=job_id, query=search_request.query, max_results=search_request.max_results)
        log.info(f"FULL PIPELINE task job {job_id} sent to Celery.")
    except Exception as e:
         log.error(f"Failed send task job {job_id}: {e}", exc_info=True)
         database.update_job_status(job_id, 'FAILED', error_message=f"Queue fail: {e}")
         raise HTTPException(status_code=500, detail=f"Queue task fail: {e}")
    return JobResponse(job_id=job_id)

@app.get("/api/searches/{job_id}/status", response_model=StatusResponse)
async def get_search_status(job_id: str):
    """Retrieves the current status of the background job."""
    log.debug(f"Request status job {job_id}")
    status_info = database.get_job_status(job_id)
    if status_info is None: raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return StatusResponse(job_id=job_id, status=status_info['status'], error_message=status_info.get('error_message'))

@app.get("/api/searches/{job_id}/results", response_model=ResultsResponse)
async def get_search_results(job_id: str):
    """Retrieves the merged data and basic graph info for a COMPLETED job."""
    log.info(f"Request results job {job_id}")
    status_info = database.get_job_status(job_id)

    if status_info is None: raise HTTPException(status_code=404, detail="Job not found")
    status = status_info['status']
    if status != 'COMPLETED':
        error_detail = f"Job is {status}." + (f" Error: {status_info.get('error_message')}" if status == 'FAILED' else "")
        raise HTTPException(status_code=400, detail=error_detail) # 400 Bad Request if not ready/failed

    result_paths = database.get_results_path(job_id)
    if not result_paths: raise HTTPException(status_code=500, detail="Paths missing.")

    data_path = result_paths.get('results_data_path')
    graph_path = result_paths.get('results_graph_path')
    if not data_path: raise HTTPException(status_code=500, detail="Data path missing.")

    log.info(f"Loading data job {job_id} from: {data_path}")
    if not os.path.exists(data_path): raise HTTPException(status_code=500, detail="Data file not found.")

    graph_info_data = {"graph_file_exists": False, "nodes": None, "edges": None}
    if graph_path and os.path.exists(graph_path):
        graph_info_data["graph_file_exists"] = True
        # Optionally load graph here to get counts if needed, but can be slow
        # try:
        #     with open(graph_path, 'rb') as gf: G = pickle.load(gf)
        #     graph_info_data["nodes"] = G.number_of_nodes()
        #     graph_info_data["edges"] = G.number_of_edges()
        # except Exception as ge: log.error(f"Err loading graph {graph_path}: {ge}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f: merged_data_content = json.load(f)
        log.info(f"Loaded data job {job_id}")
        return ResultsResponse(merged_data=merged_data_content, graph_info=GraphInfo(**graph_info_data))
    except Exception as e:
        log.error(f"Failed load/process results job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed load results.")