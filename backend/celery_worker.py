# backend/celery_worker.py
from celery import Celery
from celery.signals import task_prerun, task_failure # Import signals
from celery.exceptions import SoftTimeLimitExceeded
import time
import logging
import os
import json
import pickle
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import traceback

# --- ML/Processing Imports ---
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# --- Our Modules ---
from pubmed_fetcher import search_pubmed_sync, fetch_pubmed_citing_pmids
from crossref_fetcher import fetch_crossref_references
from opencitations_fetcher import fetch_opencitations_citations
import database

# --- Logging Setup ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(name)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# --- Global Variables / Constants ---
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Celery Configuration ---
REDIS_URL = 'redis://localhost:6379/0'
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer='json', accept_content=['json'], result_serializer='json',
    timezone='UTC', enable_utc=True, broker_connection_retry_on_startup=True,
    task_time_limit=1800, task_soft_time_limit=1740,
    task_track_started=True, # Ensure STARTED state is tracked
    task_acks_late=True, # Acknowledge task only after completion/failure
    worker_prefetch_multiplier=1 # Process one task at a time per worker process (good for long tasks)
)
# --- End Celery Configuration ---


# --- Celery Signal Handlers ---
# These run in the worker process context

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Update DB status to STARTED when task begins execution."""
    job_id = kwargs.get('job_id') if kwargs else None
    if job_id:
        log.info(f"[Job {job_id} / Task {task_id}] Signal: task_prerun. Updating status to STARTED.")
        # Use set_started=True
        database.update_job_status(job_id, 'STARTED', set_started=True)
    else:
        log.warning(f"[Task {task_id}] task_prerun signal fired but no job_id found in kwargs.")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **extra):
    """Update DB status to FAILED immediately when a task raises an exception."""
    job_id = kwargs.get('job_id') if kwargs else None
    if job_id:
        error_str = f"{type(exception).__name__}: {exception}"
        log.error(f"[Job {job_id} / Task {task_id}] Signal: task_failure. Error: {error_str}. Updating status to FAILED.")
         # Use set_finished=True
        database.update_job_status(job_id, 'FAILED', error_message=error_str, set_finished=True)
    else:
        log.error(f"[Task {task_id}] task_failure signal fired but no job_id found. Error: {exception}")

# --- End Celery Signal Handlers ---


# --- Helper Functions (Merge, Graph, Embed, UMAP, Cluster - Keep as corrected previously) ---
def merge_pipeline_data(pubmed_articles: List[Dict[str, Any]], crossref_data: Dict[str, Dict[str, Any]], opencitations_data: Dict[str, Dict[str, Any]], pubmed_citing_data: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    # ... (Body from previous correct version) ...
    merged_data: Dict[str, Dict[str, Any]] = {}
    dois_in_pubmed_set = {a['doi'] for a in pubmed_articles if a.get('doi')}
    pmid_to_doi_map = {a['pmid']: a['doi'] for a in pubmed_articles if a.get('pmid') and a.get('doi')}
    log.info(f"Merge Start: PubMed={len(pubmed_articles)}, CR={len(crossref_data)}, OC={len(opencitations_data)}, PM_Cite={len(pubmed_citing_data)}")
    processed_count, skipped_count = 0, 0
    for article in pubmed_articles:
        doi, pmid = article.get("doi"), article.get("pmid")
        if not doi: skipped_count += 1; continue
        if not article.get('title') or article.get('title')=='No Title' or not article.get('abstract') or article.get('abstract')=='No Abstract': skipped_count += 1; log.debug(f"Skip {doi}: missing title/abs."); continue
        cr_info = crossref_data.get(doi, {"references": [], "reference_count": 0, "is_referenced_by_count": 0})
        oc_info = opencitations_data.get(doi, {"citations": [], "citation_count": 0})
        pmid_cited_by_pmids = pubmed_citing_data.get(pmid, []) if pmid else []
        pmid_cited_by_dois = [pmid_to_doi_map[p] for p in pmid_cited_by_pmids if p in pmid_to_doi_map]
        all_citations_dois = set(oc_info.get("citations", [])) | set(pmid_cited_by_dois)
        valid_citations = [c for c in all_citations_dois if c in dois_in_pubmed_set]
        valid_references = [r for r in cr_info.get("references", []) if r in dois_in_pubmed_set]
        merged_data[doi] = {
            "doi": doi, "pmid": pmid, "title": article.get("title", "N/A"),
            "abstract": article.get("abstract", "N/A"), "year": article.get("year", "Unknown"),
            "mesh_terms": article.get("mesh_terms", []), "references": valid_references,
            "citations": valid_citations, "reference_count_raw_cr": cr_info.get("reference_count", 0),
            "citation_count_raw_oc": oc_info.get("citation_count", 0),
            "citation_count_raw_cr": cr_info.get("is_referenced_by_count", 0),
            "citation_count_raw_pm": len(pmid_cited_by_pmids), "source": "pubmed",
        }; processed_count += 1
    log.info(f"Merge End. Processed: {processed_count}, Skipped: {skipped_count}. Final: {len(merged_data)}")
    return merged_data

def create_citation_graph(merged_data: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
    # ... (Body from previous correct version) ...
    if not merged_data: log.warning("create_graph empty data."); return nx.DiGraph()
    log.info(f"Create Graph Start: {len(merged_data)} articles...")
    G = nx.DiGraph(); valid_dois = set(merged_data.keys())
    for doi, data in merged_data.items(): G.add_node(doi, **data)
    edge_ref_count, edge_cite_count = 0, 0
    for doi, data in tqdm(merged_data.items(), desc="Build Graph Edges", ncols=80, mininterval=1.0):
        for ref_doi in data.get("references", []):
            if ref_doi in valid_dois and ref_doi != doi and not G.has_edge(ref_doi, doi): G.add_edge(ref_doi, doi, type="reference"); edge_ref_count += 1
        for cite_doi in data.get("citations", []):
            if cite_doi in valid_dois and cite_doi != doi and not G.has_edge(doi, cite_doi): G.add_edge(doi, cite_doi, type="citation"); edge_cite_count += 1
    log.info(f"Create Graph End. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} ({edge_ref_count} ref, {edge_cite_count} cite)")
    num_isolates = len(list(nx.isolates(G))); log.warning(f"Graph isolates: {num_isolates}") if num_isolates > 0 else None
    return G

def generate_embeddings(merged_data: Dict[str, Dict[str, Any]], model_name: str = MODEL_NAME, batch_size: int = 32) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
     # ... (Body from previous correct version) ...
    if not merged_data: log.warning("generate_embeddings empty data."); return None, None
    log.info(f"Generating embeddings model '{model_name}'..."); start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Check for CUDA at runtime
    try: model = SentenceTransformer(model_name, device=device); log.info(f"Model loaded onto {device.upper()}.")
    except Exception as e_load: log.error(f"Failed load ST model '{model_name}' on {device.upper()}: {e_load}", exc_info=True); return None, None
    dois = list(merged_data.keys())
    texts = [(merged_data[d].get('title', '') or '') + ' ' + (merged_data[d].get('abstract', '') or '') for d in dois]
    log.info(f"Prepared {len(texts)} texts.")
    try:
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        log.info(f"Embeddings ({embeddings.shape}) took {time.time() - start_time:.1f}s.")
        if not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != len(dois): raise ValueError("Output shape mismatch")
        return dois, embeddings
    except Exception as e: log.error(f"Error during ST encode: {e}", exc_info=True); return None, None

def perform_umap_reduction(embeddings: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'cosine') -> Optional[np.ndarray]:
    # ... (Body from previous correct version) ...
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] < n_components: log.warning(f"Invalid UMAP input shape {embeddings.shape if embeddings is not None else 'None'}"); return None
    log.info(f"Performing UMAP (n={n_neighbors}, min_d={min_dist}, metric='{metric}')..."); start_time = time.time()
    try:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42, n_jobs=1)
        umap_results = reducer.fit_transform(embeddings)
        log.info(f"UMAP ({umap_results.shape}) took {time.time() - start_time:.1f}s.")
        return umap_results
    except Exception as e: log.error(f"Error during UMAP: {e}", exc_info=True); return None

def perform_clustering(embeddings: np.ndarray, method: str = 'agglomerative_ward', n_clusters: Optional[int] = None, distance_threshold: Optional[float] = 2.5) -> Optional[np.ndarray]:
    # ... (Body from previous correct version) ...
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] < 2: log.warning(f"Invalid Cluster input shape {embeddings.shape if embeddings is not None else 'None'}"); return None
    log.info(f"Performing clustering method '{method}'..."); start_time = time.time()
    cluster_labels = None
    try:
        if method == 'agglomerative_ward':
            if distance_threshold is None and n_clusters is None: raise ValueError("Agglomerative requires n_clusters or distance_threshold")
            n_c, d_t = (None, distance_threshold) if distance_threshold is not None else (n_clusters, None)
            log.info(f"Agglomerative Ward (n={n_c}, d_t={d_t})")
            clusterer = AgglomerativeClustering(n_clusters=n_c, linkage='ward', metric='euclidean', distance_threshold=d_t)
            cluster_labels = clusterer.fit_predict(embeddings)
        else: raise NotImplementedError(f"Clustering method '{method}' not implemented.")
        if cluster_labels is not None:
             num_found = len(set(cluster_labels) - {-1}); noise = np.sum(cluster_labels == -1)
             log.info(f"Clustering took {time.time() - start_time:.1f}s. Found {num_found} clusters (+ {noise} noise).")
             return cluster_labels
        else: raise ValueError("Clustering returned None.")
    except Exception as e: log.error(f"Error during clustering ({method}): {e}", exc_info=True); return None
# --- End Semantic Analysis Helper Functions ---


# --- Main Celery Task Definition ---
@celery_app.task(name='tasks.run_full_pipeline', bind=True, acks_late=True) # Removed time limits for debugging
def run_full_pipeline(self, job_id: str, query: str, max_results: int):
    """Main pipeline task with improved error handling."""
    task_id = self.request.id
    log.info(f"[Job {job_id} / Task {task_id}] Task STARTED signal should fire now.")
    # Note: DB status might already be 'STARTED' due to task_prerun signal

    # Initialize vars
    # ... (same initializations as before) ...
    pubmed_articles, crossref_results, opencitations_results = [], {}, {}
    pubmed_citing_results, merged_data = {}, {}
    graph: Optional[nx.DiGraph] = None; ordered_dois: Optional[List[str]] = None
    embeddings: Optional[np.ndarray] = None; umap_coords: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None; error_msg: Optional[str] = None
    final_status: str = 'FAILED'
    results_data_path, results_graph_path = None, None
    results_embeddings_path, results_umap_path, results_clusters_path = None, None, None
    results_dois_path = None


    try:
        # --- Pipeline Steps ---
        # Step 1: PubMed
        log.info(f"[Job {job_id}] Step 1: PubMed..."); start_time = time.time(); pubmed_articles = search_pubmed_sync(query, max_results); log.info(f"PubMed ({len(pubmed_articles)}) took {time.time() - start_time:.1f}s.")
        if not pubmed_articles: raise ValueError("PubMed empty.")
        dois_found = [a['doi'] for a in pubmed_articles if a.get('doi')]; pmids_found = [a['pmid'] for a in pubmed_articles if a.get('pmid')];
        if not dois_found: raise ValueError("No DOIs found.")

        # Step 2: CrossRef
        log.info(f"[Job {job_id}] Step 2: CrossRef..."); start_time = time.time(); crossref_results = fetch_crossref_references(dois_found); log.info(f"CrossRef took {time.time() - start_time:.1f}s.")
        # Step 3: OpenCitations
        log.info(f"[Job {job_id}] Step 3: OpenCitations..."); start_time = time.time(); opencitations_results = fetch_opencitations_citations(dois_found); log.info(f"OpenCitations took {time.time() - start_time:.1f}s.")
        # Step 4: PubMed Citing
        if pmids_found: log.info(f"[Job {job_id}] Step 4: PubMed Citing..."); start_time = time.time(); pubmed_citing_results = fetch_pubmed_citing_pmids(pmids_found); log.info(f"PubMed Citing took {time.time() - start_time:.1f}s.")
        else: log.info(f"[Job {job_id}] Step 4: Skip PubMed Citing.")

        # Step 5: Merge
        log.info(f"[Job {job_id}] Step 5: Merging..."); start_time = time.time(); merged_data = merge_pipeline_data(pubmed_articles, crossref_results, opencitations_results, pubmed_citing_results); log.info(f"Merging ({len(merged_data)}) took {time.time() - start_time:.1f}s.")
        if not merged_data: raise ValueError("Merge empty.")

        # Step 6: Graph
        log.info(f"[Job {job_id}] Step 6: Graph..."); start_time = time.time(); graph = create_citation_graph(merged_data); log.info(f"Graph ({graph.number_of_nodes()}N/{graph.number_of_edges()}E) took {time.time() - start_time:.1f}s.")

        # Step 7: Embeddings
        log.info(f"[Job {job_id}] Step 7: Embeddings..."); start_time = time.time(); ordered_dois, embeddings = generate_embeddings(merged_data); log.info(f"Embeddings took {time.time() - start_time:.1f}s.")
        if ordered_dois is None or embeddings is None: raise ValueError("Embed fail.")

        # Step 8: UMAP
        log.info(f"[Job {job_id}] Step 8: UMAP..."); start_time = time.time(); umap_coords = perform_umap_reduction(embeddings); log.info(f"UMAP took {time.time() - start_time:.1f}s.")
        if umap_coords is None: raise ValueError("UMAP fail.")

        # Step 9: Clustering
        log.info(f"[Job {job_id}] Step 9: Clustering..."); start_time = time.time(); cluster_labels = perform_clustering(embeddings, method='agglomerative_ward', distance_threshold=2.5); log.info(f"Clustering took {time.time() - start_time:.1f}s.")
        if cluster_labels is None: raise ValueError("Cluster fail.")

        # Step 10: Save Results
        log.info(f"[Job {job_id}] Step 10: Saving results..."); start_time = time.time()
        # Define paths
        results_data_path = os.path.join(RESULTS_DIR, f"{job_id}_merged_data.json")
        results_graph_path = os.path.join(RESULTS_DIR, f"{job_id}_graph.pkl")
        results_embeddings_path = os.path.join(RESULTS_DIR, f"{job_id}_embeddings.npy")
        results_umap_path = os.path.join(RESULTS_DIR, f"{job_id}_umap_coords.npy")
        results_clusters_path = os.path.join(RESULTS_DIR, f"{job_id}_clusters.json")
        results_dois_path = os.path.join(RESULTS_DIR, f"{job_id}_dois_order.json")
        # Save files
        with open(results_data_path, 'w', encoding='utf-8') as f: json.dump(merged_data, f, ensure_ascii=False)
        with open(results_graph_path, 'wb') as f: pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(results_embeddings_path, embeddings)
        np.save(results_umap_path, umap_coords)
        with open(results_clusters_path, 'w', encoding='utf-8') as f: json.dump(cluster_labels.tolist(), f)
        with open(results_dois_path, 'w', encoding='utf-8') as f: json.dump(ordered_dois, f)
        log.info(f"[Job {job_id}] Saving results took {time.time() - start_time:.1f}s.")

        final_status = 'COMPLETED' # If all steps successful

    except SoftTimeLimitExceeded:
         log.error(f"[Job {job_id}] Task soft time limit exceeded!")
         error_msg = "Processing timed out."
         final_status = 'FAILED'
         # Signal handler should catch this too, but update here just in case
         database.update_job_status(job_id, final_status, error_message=error_msg, set_finished=True)


    except Exception as e:
        log.error(f"[Job {job_id}] Pipeline Error: {e}", exc_info=True)
        step_info = "Unknown Step"; lineno = -1
        try:
            tb = traceback.extract_tb(e.__traceback__)
            if tb: lineno = tb[-1].lineno
            # Crude line number checks - ADJUST THESE IF CODE STRUCTURE CHANGES
            if 300 <= lineno < 330: step_info = "Data Fetch/Merge/Graph"
            elif 330 <= lineno < 340: step_info = "Embeddings"
            elif 340 <= lineno < 345: step_info = "UMAP"
            elif 345 <= lineno < 350: step_info = "Clustering"
            elif 350 <= lineno < 370: step_info = "Saving Results"
        except Exception: pass
        error_msg = f"Pipeline fail at ~{step_info} (line ~{lineno}): {type(e).__name__}: {e}"
        final_status = 'FAILED'
        # Signal handler should catch this, no need to update DB here typically

    # --- Post-Execution DB Update (Only if NOT caught by failure signal) ---
    # This block might not be reached if task_failure signal handles the update first
    # or if a hard time limit kills the process.

    # Check current status before potentially overwriting FAILED status from signal
    current_db_status_info = database.get_job_status(job_id)
    current_db_status = current_db_status_info.get('status') if current_db_status_info else 'UNKNOWN'

    if current_db_status != 'FAILED': # Don't overwrite if signal already marked as failed
        log.info(f"[Job {job_id}] Task function finished. Attempting final DB update status='{final_status}'.")
        paths_to_store = {
            'results_data_path': results_data_path, 'results_graph_path': results_graph_path,
            'results_embeddings_path': results_embeddings_path, 'results_umap_path': results_umap_path,
            'results_clusters_path': results_clusters_path, 'results_dois_path': results_dois_path
        }
        valid_paths = {k: v for k, v in paths_to_store.items() if v is not None}

        path_update_success = True
        if final_status == 'COMPLETED':
            if not database.store_results_path(job_id, **valid_paths):
                log.error(f"[Job {job_id}] CRITICAL: Completed but fail store paths!");
                path_update_success = False
                final_status = 'FAILED' # Downgrade status
                error_msg = (error_msg or "") + " DB path store fail."

        # Update final status, making sure FAILED status sticks if path saving failed
        # Pass set_finished=True
        if not database.update_job_status(job_id, final_status, error_message=error_msg, set_finished=True):
             log.error(f"[Job {job_id}] CRITICAL: Fail update final status '{final_status}' in finally block!")
        else:
             log.info(f"[Job {job_id}] Final status '{final_status}' updated via finally block.")
    else:
         log.info(f"[Job {job_id}] Task function finished, but status already set to FAILED (likely by signal). Skipping final update.")


    # Return summary (useful for Celery result backend)
    return { "job_id": job_id, "task_id": task_id, "final_status": final_status, "error": error_msg, **valid_paths }
# --- End Main Celery Task ---