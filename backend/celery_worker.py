from celery import Celery
from celery.signals import task_prerun, task_failure
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

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from pubmed_fetcher import search_pubmed_sync, fetch_pubmed_citing_pmids
from crossref_fetcher import fetch_crossref_references
from opencitations_fetcher import fetch_opencitations_citations
import database

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(name)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REDIS_URL = 'redis://localhost:6379/0'
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_time_limit=1800,
    task_soft_time_limit=1740,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1
)

# --- Signal Handlers ---
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    job_id = kwargs.get('job_id') if kwargs else None
    if job_id:
        log.info(f"[Job {job_id} / Task {task_id}] task_prerun signal: updating status to STARTED.")
        database.update_job_status(job_id, 'STARTED', set_started=True)
    else:
        log.warning(f"[Task {task_id}] task_prerun signal fired but no job_id found.")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **extra):
    job_id = kwargs.get('job_id') if kwargs else None
    if job_id:
        error_str = f"{type(exception).__name__}: {exception}"
        log.error(f"[Job {job_id} / Task {task_id}] task_failure signal: {error_str}. Updating status to FAILED.")
        database.update_job_status(job_id, 'FAILED', error_message=error_str, set_finished=True)
    else:
        log.error(f"[Task {task_id}] task_failure signal fired but no job_id found. Error: {exception}")

# --- Helper Functions ---
def merge_pipeline_data(pubmed_articles: List[Dict[str, Any]],
                        crossref_data: Dict[str, Dict[str, Any]],
                        opencitations_data: Dict[str, Dict[str, Any]],
                        pubmed_citing_data: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    merged_data: Dict[str, Dict[str, Any]] = {}
    dois_in_pubmed_set = {a['doi'] for a in pubmed_articles if a.get('doi')}
    pmid_to_doi_map = {a['pmid']: a['doi'] for a in pubmed_articles if a.get('pmid') and a.get('doi')}
    log.info(f"Merge Start: PubMed={len(pubmed_articles)}, CR={len(crossref_data)}, OC={len(opencitations_data)}, PM_Cite={len(pubmed_citing_data)}")
    processed_count, skipped_count = 0, 0
    for article in pubmed_articles:
        doi, pmid = article.get("doi"), article.get("pmid")
        if not doi:
            skipped_count += 1
            continue
        if not article.get('title') or article.get('title')=='No Title' or not article.get('abstract') or article.get('abstract')=='No Abstract':
            skipped_count += 1
            log.debug(f"Skip {doi}: missing title/abs.")
            continue
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
        }
        processed_count += 1
    log.info(f"Merge End. Processed: {processed_count}, Skipped: {skipped_count}. Final: {len(merged_data)}")
    return merged_data

def create_citation_graph(merged_data: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
    if not merged_data:
        log.warning("create_citation_graph: empty data.")
        return nx.DiGraph()
    log.info(f"Create Graph Start: {len(merged_data)} articles...")
    G = nx.DiGraph()
    valid_dois = set(merged_data.keys())
    for doi, data in merged_data.items():
        G.add_node(doi, **data)
    edge_ref_count, edge_cite_count = 0, 0
    for doi, data in tqdm(merged_data.items(), desc="Build Graph Edges", ncols=80, mininterval=1.0):
        for ref_doi in data.get("references", []):
            if ref_doi in valid_dois and ref_doi != doi and not G.has_edge(ref_doi, doi):
                G.add_edge(ref_doi, doi, type="reference")
                edge_ref_count += 1
        for cite_doi in data.get("citations", []):
            if cite_doi in valid_dois and cite_doi != doi and not G.has_edge(doi, cite_doi):
                G.add_edge(doi, cite_doi, type="citation")
                edge_cite_count += 1
    log.info(f"Create Graph End. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} ({edge_ref_count} ref, {edge_cite_count} cite)")
    num_isolates = len(list(nx.isolates(G)))
    if num_isolates > 0:
        log.warning(f"Graph isolates: {num_isolates}")
    return G

def generate_embeddings(merged_data: Dict[str, Dict[str, Any]],
                        model_name: str = MODEL_NAME, batch_size: int = 32) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    if not merged_data:
        log.warning("generate_embeddings: empty data.")
        return None, None
    log.info(f"Generating embeddings model '{model_name}'...")
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = SentenceTransformer(model_name, device=device)
        log.info(f"Model loaded onto {device.upper()}.")
    except Exception as e_load:
        log.error(f"Failed to load SentenceTransformer model '{model_name}' on {device.upper()}: {e_load}", exc_info=True)
        return None, None
    dois = list(merged_data.keys())
    texts = [(merged_data[d].get('title', '') or '') + ' ' + (merged_data[d].get('abstract', '') or '') for d in dois]
    log.info(f"Prepared {len(texts)} texts.")
    try:
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        log.info(f"Embeddings ({embeddings.shape}) took {time.time() - start_time:.1f}s.")
        if not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != len(dois):
            raise ValueError("Output shape mismatch")
        return dois, embeddings
    except Exception as e:
        log.error(f"Error during SentenceTransformer encoding: {e}", exc_info=True)
        return None, None

def perform_umap_reduction(embeddings: np.ndarray,
                           n_components: int = 2, n_neighbors: int = 15,
                           min_dist: float = 0.1, metric: str = 'cosine') -> Optional[np.ndarray]:
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] < n_components:
        log.warning(f"perform_umap_reduction: Invalid input shape {embeddings.shape if embeddings is not None else 'None'}")
        return None
    log.info(f"Performing UMAP (n={n_neighbors}, min_d={min_dist}, metric='{metric}')...")
    start_time = time.time()
    try:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric=metric, random_state=42, n_jobs=1)
        umap_results = reducer.fit_transform(embeddings)
        log.info(f"UMAP ({umap_results.shape}) took {time.time() - start_time:.1f}s.")
        return umap_results
    except Exception as e:
        log.error(f"Error during UMAP reduction: {e}", exc_info=True)
        return None

def perform_clustering(embeddings: np.ndarray,
                       method: str = 'agglomerative_ward',
                       n_clusters: Optional[int] = None,
                       distance_threshold: Optional[float] = 2.5) -> Optional[np.ndarray]:
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] < 2:
        log.warning(f"perform_clustering: Invalid input shape {embeddings.shape if embeddings is not None else 'None'}")
        return None
    log.info(f"Performing clustering method '{method}'...")
    start_time = time.time()
    cluster_labels = None
    try:
        if method == 'agglomerative_ward':
            if distance_threshold is None and n_clusters is None:
                raise ValueError("Agglomerative clustering requires n_clusters or distance_threshold")
            n_c, d_t = (None, distance_threshold) if distance_threshold is not None else (n_clusters, None)
            log.info(f"Agglomerative Ward (n={n_c}, d_t={d_t})")
            clusterer = AgglomerativeClustering(n_clusters=n_c, linkage='ward', metric='euclidean', distance_threshold=d_t)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise NotImplementedError(f"Clustering method '{method}' is not implemented.")
        if cluster_labels is not None:
            num_found = len(set(cluster_labels) - {-1})
            noise = np.sum(cluster_labels == -1)
            log.info(f"Clustering took {time.time() - start_time:.1f}s. Found {num_found} clusters (+ {noise} noise).")
            return cluster_labels
        else:
            raise ValueError("Clustering returned None.")
    except Exception as e:
        log.error(f"Error during clustering ({method}): {e}", exc_info=True)
        return None

# --- Main Celery Task ---
@celery_app.task(name='tasks.run_full_pipeline', bind=True, acks_late=True)
def run_full_pipeline(self, job_id: str, query: str, max_results: int):
    task_id = self.request.id
    log.info(f"[Job {job_id} / Task {task_id}] Starting full pipeline. Query='{query}', MaxResults={max_results}")
    database.update_job_status(job_id, 'RUNNING')
    pubmed_articles, crossref_results, opencitations_results = [], {}, {}
    pubmed_citing_results, merged_data = {}, {}
    graph: Optional[nx.DiGraph] = None
    ordered_dois: Optional[List[str]] = None
    embeddings: Optional[np.ndarray] = None
    umap_coords: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    error_msg: Optional[str] = None
    final_status: str = 'FAILED'
    results_data_path = results_graph_path = None
    results_embeddings_path = results_umap_path = results_clusters_path = None
    results_dois_path = None

    try:
        log.info(f"[Job {job_id}] Step 1: PubMed...")
        start_time = time.time()
        pubmed_articles = search_pubmed_sync(query, max_results)
        log.info(f"PubMed ({len(pubmed_articles)}) took {time.time() - start_time:.1f}s.")
        if not pubmed_articles:
            raise ValueError("PubMed returned empty result.")
        dois_found = [a['doi'] for a in pubmed_articles if a.get('doi')]
        pmids_found = [a['pmid'] for a in pubmed_articles if a.get('pmid')]
        if not dois_found:
            raise ValueError("No DOIs found in PubMed results.")

        log.info(f"[Job {job_id}] Step 2: CrossRef...")
        start_time = time.time()
        crossref_results = fetch_crossref_references(dois_found)
        log.info(f"CrossRef took {time.time() - start_time:.1f}s.")

        log.info(f"[Job {job_id}] Step 3: OpenCitations...")
        start_time = time.time()
        opencitations_results = fetch_opencitations_citations(dois_found)
        log.info(f"OpenCitations took {time.time() - start_time:.1f}s.")

        if pmids_found:
            log.info(f"[Job {job_id}] Step 4: PubMed Citing...")
            start_time = time.time()
            pubmed_citing_results = fetch_pubmed_citing_pmids(pmids_found)
            log.info(f"PubMed Citing took {time.time() - start_time:.1f}s.")
        else:
            log.info(f"[Job {job_id}] Step 4: Skipping PubMed Citing (no PMIDs).")

        log.info(f"[Job {job_id}] Step 5: Merging...");
        start_time = time.time()
        merged_data = merge_pipeline_data(pubmed_articles, crossref_results, opencitations_results, pubmed_citing_results)
        log.info(f"Merging ({len(merged_data)}) took {time.time() - start_time:.1f}s.");
        if not merged_data:
            raise ValueError("Merge produced empty result.")

        log.info(f"[Job {job_id}] Step 6: Graph...");
        start_time = time.time()
        graph = create_citation_graph(merged_data)
        log.info(f"Graph ({graph.number_of_nodes()} nodes/{graph.number_of_edges()} edges) took {time.time() - start_time:.1f}s.");

        log.info(f"[Job {job_id}] Step 7: Embeddings...");
        start_time = time.time();
        ordered_dois, embeddings = generate_embeddings(merged_data)
        log.info(f"Embeddings took {time.time() - start_time:.1f}s.");
        if ordered_dois is None or embeddings is None:
            raise ValueError("Embeddings generation failed.")

        log.info(f"[Job {job_id}] Step 8: UMAP...");
        start_time = time.time();
        umap_coords = perform_umap_reduction(embeddings)
        log.info(f"UMAP took {time.time() - start_time:.1f}s.");
        if umap_coords is None:
            raise ValueError("UMAP reduction failed.")

        log.info(f"[Job {job_id}] Step 9: Clustering...");
        start_time = time.time();
        cluster_labels = perform_clustering(embeddings, method='agglomerative_ward', distance_threshold=2.5)
        log.info(f"Clustering took {time.time() - start_time:.1f}s.");
        if cluster_labels is None:
            raise ValueError("Clustering failed.")

        log.info(f"[Job {job_id}] Step 10: Saving results...");
        start_time = time.time();
        results_data_path = os.path.join(RESULTS_DIR, f"{job_id}_merged_data.json")
        results_graph_path = os.path.join(RESULTS_DIR, f"{job_id}_graph.pkl")
        results_embeddings_path = os.path.join(RESULTS_DIR, f"{job_id}_embeddings.npy")
        results_umap_path = os.path.join(RESULTS_DIR, f"{job_id}_umap_coords.npy")
        results_clusters_path = os.path.join(RESULTS_DIR, f"{job_id}_clusters.json")
        results_dois_path = os.path.join(RESULTS_DIR, f"{job_id}_dois_order.json")
        with open(results_data_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False)
        with open(results_graph_path, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(results_embeddings_path, embeddings)
        np.save(results_umap_path, umap_coords)
        with open(results_clusters_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_labels.tolist(), f)
        with open(results_dois_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_dois, f)
        log.info(f"[Job {job_id}] Saving results took {time.time() - start_time:.1f}s.");
        final_status = 'COMPLETED'
    except SoftTimeLimitExceeded:
        log.error(f"[Job {job_id}] Soft time limit exceeded.");
        error_msg = "Processing timed out."
        final_status = 'FAILED'
        database.update_job_status(job_id, final_status, error_message=error_msg, set_finished=True)
    except Exception as e:
        log.error(f"[Job {job_id}] Pipeline Error: {e}", exc_info=True)
        try:
            tb = traceback.extract_tb(e.__traceback__)
            lineno = tb[-1].lineno if tb else -1
        except Exception:
            lineno = -1
        error_msg = f"Pipeline error at line ~{lineno}: {type(e).__name__}: {e}"
        final_status = 'FAILED'
    finally:
        current_db_status_info = database.get_job_status(job_id)
        current_db_status = current_db_status_info.get('status') if current_db_status_info else 'UNKNOWN'
        if current_db_status != 'FAILED':
            log.info(f"[Job {job_id}] Final DB update with status '{final_status}'.")
            paths_to_store = {
                'results_data_path': results_data_path,
                'results_graph_path': results_graph_path,
                'results_embeddings_path': results_embeddings_path,
                'results_umap_path': results_umap_path,
                'results_clusters_path': results_clusters_path,
                'results_dois_path': results_dois_path
            }
            valid_paths = {k: v for k, v in paths_to_store.items() if v is not None}
            if final_status == 'COMPLETED':
                if not database.store_results_path(job_id, **valid_paths):
                    log.error(f"[Job {job_id}] Failed to store result paths. Downgrading status to FAILED.")
                    final_status = 'FAILED'
                    error_msg = (error_msg or "") + " DB path store failure."
            if not database.update_job_status(job_id, final_status, error_message=error_msg, set_finished=True):
                log.error(f"[Job {job_id}] Failed final DB update with status '{final_status}'.")
            else:
                log.info(f"[Job {job_id}] Final status updated to '{final_status}'.")
        else:
            log.info(f"[Job {job_id}] Skipping final DB update; status already FAILED.")
    return {
        "job_id": job_id,
        "task_id": task_id,
        "final_status": final_status,
        "error": error_msg,
        **(valid_paths if 'valid_paths' in locals() else {})
    }
