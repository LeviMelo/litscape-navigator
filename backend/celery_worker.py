# backend/celery_worker.py
from celery import Celery
import time
import logging
import os
import json
import pickle # To save the NetworkX graph object
import networkx as nx # Import networkx
from typing import Dict, List, Any, Optional # Type hints
from tqdm import tqdm # Added tqdm for graph building progress

# Import our data fetching and processing modules
from pubmed_fetcher import search_pubmed_sync, fetch_pubmed_citing_pmids
from crossref_fetcher import fetch_crossref_references
from opencitations_fetcher import fetch_opencitations_citations
import database # Our database helper module

# --- Logging Setup ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers(): # Add handler if not configured globally
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# --- Celery Configuration ---
REDIS_URL = 'redis://localhost:6379/0'
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_time_limit=1200, # Increased time limit (20 minutes)
    task_soft_time_limit=1140 # Soft limit (19 minutes)
)
# --- End Celery Configuration ---


# --- Helper: Merge Data Function ---
def merge_pipeline_data(pubmed_articles: List[Dict],
                           crossref_data: Dict[str, Dict[str, Any]],
                           opencitations_data: Dict[str, Dict[str, Any]],
                           pubmed_citing_data: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Merges data from different sources into a single dictionary keyed by DOI."""
    merged_data: Dict[str, Dict[str, Any]] = {}
    # Create lookups for quick checking
    dois_in_pubmed_set = {a['doi'] for a in pubmed_articles if a.get('doi')}
    pmid_to_doi_map = {a['pmid']: a['doi'] for a in pubmed_articles if a.get('pmid') and a.get('doi')}

    log.info(f"Starting merge. PubMed articles: {len(pubmed_articles)}, CrossRef DOIs: {len(crossref_data)}, OC DOIs: {len(opencitations_data)}, PubMed CitedIn PMIDs: {len(pubmed_citing_data)}")

    processed_count = 0
    skipped_count = 0

    for article in pubmed_articles:
        doi = article.get("doi")
        pmid = article.get("pmid")

        # Skip articles if they lack a DOI (essential key)
        if not doi:
            skipped_count += 1
            log.debug(f"Skipping article with missing DOI (PMID: {pmid}, Title: {article.get('title', '')[:30]}...).")
            continue

        # Skip articles if they lack basic content (adjust criteria as needed)
        if not article.get('title') or article.get('title') == 'No Title' or \
           not article.get('abstract') or article.get('abstract') == 'No Abstract':
            skipped_count += 1
            log.debug(f"Skipping article {doi} due to missing title or abstract.")
            continue

        # Get data from other sources, providing safe defaults
        cr_info = crossref_data.get(doi, {"references": [], "reference_count": 0, "is_referenced_by_count": 0})
        oc_info = opencitations_data.get(doi, {"citations": [], "citation_count": 0})
        pmid_cited_by_pmids = pubmed_citing_data.get(pmid, []) if pmid else []

        # Convert citing PMIDs to DOIs only if the citing paper is also in our initial pubmed set
        pmid_cited_by_dois = [pmid_to_doi_map[p] for p in pmid_cited_by_pmids if p in pmid_to_doi_map]

        # Combine citation lists (OpenCitations DOIs + PubMed-derived DOIs)
        all_citations_dois = set(oc_info.get("citations", [])) | set(pmid_cited_by_dois)
        # Filter: Keep only citations that are themselves present in our initial PubMed result set
        valid_citations_in_corpus = [c_doi for c_doi in all_citations_dois if c_doi in dois_in_pubmed_set]

        # Filter references similarly: Keep only references present in our initial PubMed result set
        valid_references_in_corpus = [r_doi for r_doi in cr_info.get("references", []) if r_doi in dois_in_pubmed_set]

        # Construct the final merged record
        merged_data[doi] = {
            "doi": doi,
            "pmid": pmid,
            "title": article.get("title", "N/A"),
            "abstract": article.get("abstract", "N/A"),
            "year": article.get("year", "Unknown"),
            "mesh_terms": article.get("mesh_terms", []),
            # Use the filtered lists for graph consistency
            "references": valid_references_in_corpus,
            "citations": valid_citations_in_corpus,
            # Store raw counts separately for potential analysis
            "reference_count_raw_cr": cr_info.get("reference_count", 0),
            "citation_count_raw_oc": oc_info.get("citation_count", 0),
            "citation_count_raw_cr": cr_info.get("is_referenced_by_count", 0), # CrossRef cited-by
            "citation_count_raw_pm": len(pmid_cited_by_pmids),
            # Source tracking
            "source": "pubmed", # Indicate primary source
        }
        processed_count += 1

    log.info(f"Merge complete. Processed: {processed_count}, Skipped: {skipped_count}. Final merged count: {len(merged_data)}")
    return merged_data
# --- End Merge Data Function ---


# --- Helper: Create Graph Function ---
def create_citation_graph(merged_data: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
    """Creates a directed NetworkX graph from the merged article data."""
    if not merged_data:
        log.warning("create_citation_graph called with empty merged_data.")
        return nx.DiGraph()

    log.info(f"Creating NetworkX DiGraph from {len(merged_data)} articles...")
    G = nx.DiGraph()
    valid_dois_in_merged = set(merged_data.keys()) # DOIs that made it through merging

    # Add nodes first, attaching all metadata
    for doi, data in merged_data.items():
        # Exclude potentially large lists from node attributes if memory is a concern
        # node_attrs = {k: v for k, v in data.items() if k not in ['references', 'citations', 'abstract', 'mesh_terms']}
        # G.add_node(doi, **node_attrs)
        G.add_node(doi, **data) # Add all data for now

    edge_ref_count = 0
    edge_cite_count = 0
    # Add edges based on the filtered 'references' and 'citations' lists
    for doi, data in tqdm(merged_data.items(), desc="Building Graph Edges", ncols=80):
        # Add 'reference' edges: referenced_doi -> current_doi
        for ref_doi in data.get("references", []): # Uses the pre-filtered list
            if ref_doi in valid_dois_in_merged and ref_doi != doi:
                if not G.has_edge(ref_doi, doi):
                    G.add_edge(ref_doi, doi, type="reference")
                    edge_ref_count += 1

        # Add 'citation' edges: current_doi -> citing_doi
        for cite_doi in data.get("citations", []): # Uses the pre-filtered list
            if cite_doi in valid_dois_in_merged and cite_doi != doi:
                if not G.has_edge(doi, cite_doi):
                    G.add_edge(doi, cite_doi, type="citation")
                    edge_cite_count += 1

    log.info(f"Graph created. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} ({edge_ref_count} ref, {edge_cite_count} cite)")
    num_isolates = len(list(nx.isolates(G)))
    if num_isolates > 0: log.warning(f"Graph contains {num_isolates} isolated nodes.")
    # Optional: Check strongly/weakly connected components
    # if not nx.is_weakly_connected(G.to_undirected()): log.warning("Graph is not weakly connected.")

    return G
# --- End Create Graph Function ---


# --- Celery Task Definition ---
@celery_app.task(name='tasks.run_full_pipeline', bind=True) # bind=True allows access to self (the task instance)
def run_full_pipeline(self, job_id: str, query: str, max_results: int):
    """
    The background task that performs the full data pipeline.
    (Orchestration logic is identical to the version provided in Phase 3, Step 4)
    """
    task_id = self.request.id # Get Celery's internal task ID
    log.info(f"[Job {job_id} / Task {task_id}] Received task. Starting FULL pipeline for query: '{query}', max_results={max_results}")
    database.update_job_status(job_id, 'RUNNING')

    pubmed_articles: List[Dict] = []
    crossref_results: Dict = {}
    opencitations_results: Dict = {}
    pubmed_citing_results: Dict = {}
    merged_data: Dict = {}
    graph: Optional[nx.DiGraph] = None
    error_msg: Optional[str] = None
    final_status: str = 'FAILED'
    results_data_path: Optional[str] = None
    results_graph_path: Optional[str] = None

    try:
        # Step 1: PubMed Search
        log.info(f"[Job {job_id}] Step 1: Fetching PubMed data...")
        start_time = time.time()
        pubmed_articles = search_pubmed_sync(query, max_results)
        log.info(f"[Job {job_id}] PubMed fetch took {time.time() - start_time:.2f}s. Found {len(pubmed_articles)} articles.")
        if not pubmed_articles: raise ValueError("No articles found in initial PubMed search.")

        dois_found = [a['doi'] for a in pubmed_articles if a.get('doi')]
        pmids_found = [a['pmid'] for a in pubmed_articles if a.get('pmid')]
        log.info(f"[Job {job_id}] Extracted {len(dois_found)} DOIs and {len(pmids_found)} PMIDs.")
        if not dois_found: raise ValueError("No DOIs found in PubMed results to proceed.")

        # Step 2: CrossRef Fetch
        log.info(f"[Job {job_id}] Step 2: Fetching CrossRef references...")
        start_time = time.time(); crossref_results = fetch_crossref_references(dois_found, max_workers=20,base_delay=0.5,mailto="levi4328@gmail.com");
        log.info(f"[Job {job_id}] CrossRef fetch took {time.time() - start_time:.2f}s.")

        # Step 3: OpenCitations Fetch
        log.info(f"[Job {job_id}] Step 3: Fetching OpenCitations citations..."); start_time = time.time()
        opencitations_results = fetch_opencitations_citations(dois_found);
        log.info(f"[Job {job_id}] OpenCitations fetch took {time.time() - start_time:.2f}s.")

        # Step 4: PubMed Citing PMIDs Fetch
        if pmids_found:
             log.info(f"[Job {job_id}] Step 4: Fetching PubMed citing PMIDs..."); start_time = time.time()
             pubmed_citing_results = fetch_pubmed_citing_pmids(pmids_found);
             log.info(f"[Job {job_id}] PubMed citing fetch took {time.time() - start_time:.2f}s.")
        else: log.info(f"[Job {job_id}] Step 4: Skipping PubMed citing fetch (no PMIDs).")

        # Step 5: Merge Data
        log.info(f"[Job {job_id}] Step 5: Merging all fetched data..."); start_time = time.time()
        merged_data = merge_pipeline_data(pubmed_articles, crossref_results, opencitations_results, pubmed_citing_results);
        log.info(f"[Job {job_id}] Data merging took {time.time() - start_time:.2f}s. Final article count: {len(merged_data)}")
        if not merged_data: raise ValueError("Data merging resulted in an empty dataset.")

        # Step 6: Create Graph
        log.info(f"[Job {job_id}] Step 6: Creating citation graph..."); start_time = time.time()
        graph = create_citation_graph(merged_data);
        log.info(f"[Job {job_id}] Graph creation took {time.time() - start_time:.2f}s.")

        # Step 7: Save Results
        log.info(f"[Job {job_id}] Step 7: Saving merged data and graph..."); start_time = time.time()
        results_dir = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(results_dir, exist_ok=True)
        results_data_path = os.path.join(results_dir, f"{job_id}_merged_data.json")
        results_graph_path = os.path.join(results_dir, f"{job_id}_graph.pkl")

        with open(results_data_path, 'w', encoding='utf-8') as f: json.dump(merged_data, f, ensure_ascii=False)
        log.info(f"[Job {job_id}] Merged data saved to {results_data_path}")
        with open(results_graph_path, 'wb') as f: pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol
        log.info(f"[Job {job_id}] Graph saved to {results_graph_path}")
        log.info(f"[Job {job_id}] Saving results took {time.time() - start_time:.2f}s.")

        final_status = 'COMPLETED'

    except Exception as e:
        log.error(f"[Job {job_id}] FULL PIPELINE ERROR: {e}", exc_info=True)
        error_msg = f"Pipeline failed: {type(e).__name__}: {e}"
        final_status = 'FAILED'

    finally:
        # Step 8: Update Final DB Status & Paths
        log.info(f"[Job {job_id}] Attempting to update final status to '{final_status}' in DB.")
        path_update_success = True
        if final_status == 'COMPLETED':
            if not database.store_results_path(job_id, data_path=results_data_path, graph_path=results_graph_path):
                log.error(f"[Job {job_id}] CRITICAL: Completed but failed store result paths!")
                path_update_success = False
                final_status = 'FAILED' # Downgrade status if path storing fails
                error_msg = (error_msg or "") + " Failed to store result paths in database."

        if not database.update_job_status(job_id, final_status, error_message=error_msg):
             log.error(f"[Job {job_id}] CRITICAL: Failed to update final status '{final_status}' in database!")
        else:
             log.info(f"[Job {job_id}] Pipeline finished. Final status '{final_status}' updated in DB.")

    # Return summary for Celery's result backend
    return {
        "job_id": job_id,
        "task_id": task_id,
        "final_status": final_status,
        "results_data_file": results_data_path,
        "results_graph_file": results_graph_path,
        "error": error_msg
    }
# --- End Celery Task ---