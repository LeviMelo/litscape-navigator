# backend/graph_builder.py
import networkx as nx
import logging
from typing import List, Dict, Any, Tuple, Optional

log = logging.getLogger(__name__)

def merge_all_data(
    pubmed_articles: List[Dict[str, Any]],
    crossref_data: Dict[str, Dict[str, Any]],
    opencitations_data: Dict[str, Dict[str, Any]],
    pubmed_citations: Dict[str, List[str]],
    job_id: str = "N/A"
) -> Dict[str, Dict[str, Any]]:
    """
    Merges data from different sources into a single dictionary keyed by DOI.
    Handles potential missing DOIs or data from sources.

    Args:
        pubmed_articles: List of dicts from search_pubmed_sync.
        crossref_data: Dict DOI -> {'references': [...], 'reference_count': int}.
        opencitations_data: Dict DOI -> {'citations': [...], 'citation_count': int}.
        pubmed_citations: Dict PMID -> list of citing PMIDs.

    Returns:
        Dictionary where keys are DOIs and values are dictionaries containing
        all merged metadata for that article.
    """
    log.info(f"[Job {job_id}] Starting data merging process.")
    merged_data: Dict[str, Dict[str, Any]] = {}
    pmid_to_doi_map: Dict[str, str] = {}

    # First pass: Populate merged_data with PubMed info and build PMID->DOI map
    for article in pubmed_articles:
        doi = article.get("doi")
        pmid = article.get("pmid")

        if not doi:
            log.debug(f"[Job {job_id}] Skipping PubMed article (PMID: {pmid}) due to missing DOI during merge.")
            continue

        # Store the primary data from PubMed
        merged_data[doi] = {
            "doi": doi,
            "pmid": pmid,
            "title": article.get("title", "No Title"),
            "abstract": article.get("abstract", "No Abstract"),
            "mesh_terms": article.get("mesh_terms", []),
            "year": article.get("year", "Unknown"),
            "references": [], # Initialize citation/reference lists
            "reference_count": 0,
            "citations_crossref": [], # Keep sources separate initially if needed
            "citation_count_crossref": 0, # CrossRef doesn't provide citing DOIs directly
            "citations_opencitations": [],
            "citation_count_opencitations": 0,
            "citations_pubmed_pmids": [], # Citing PMIDs from PubMed ELink
            "citations_pubmed_dois": [] # Will be filled later if citing PMIDs map to DOIs in our set
        }
        # Build map for later lookup
        if pmid:
            pmid_to_doi_map[pmid] = doi

    log.info(f"[Job {job_id}] Initial merge complete with {len(merged_data)} articles (keyed by DOI). PMID->DOI map size: {len(pmid_to_doi_map)}")

    # Second pass: Add CrossRef references
    for doi, cr_info in crossref_data.items():
        if doi in merged_data:
            ref_list = cr_info.get("references", [])
            ref_count = cr_info.get("reference_count", 0)
            if ref_count == -1: # Check for error indicator
                 log.warning(f"[Job {job_id}] CrossRef data for DOI {doi} indicates fetch error. Skipping refs.")
                 merged_data[doi]["reference_count"] = -1 # Propagate error indicator
            else:
                 merged_data[doi]["references"] = ref_list
                 merged_data[doi]["reference_count"] = ref_count
        else:
             log.debug(f"[Job {job_id}] DOI {doi} from CrossRef not found in primary PubMed DOI list.")


    # Third pass: Add OpenCitations citations
    for doi, oc_info in opencitations_data.items():
        if doi in merged_data:
            cite_list = oc_info.get("citations", [])
            cite_count = oc_info.get("citation_count", 0)
            if cite_count == -1: # Check for error indicator
                 log.warning(f"[Job {job_id}] OpenCitations data for DOI {doi} indicates fetch error. Skipping citations.")
                 merged_data[doi]["citation_count_opencitations"] = -1
            else:
                 merged_data[doi]["citations_opencitations"] = cite_list
                 merged_data[doi]["citation_count_opencitations"] = cite_count
        else:
             log.debug(f"[Job {job_id}] DOI {doi} from OpenCitations not found in primary PubMed DOI list.")


    # Fourth pass: Add PubMed citing PMIDs and resolve to DOIs within our set
    for pmid, citing_pmids in pubmed_citations.items():
         doi = pmid_to_doi_map.get(pmid)
         if doi and doi in merged_data:
              # Handle potential error indicators from fetcher
              if citing_pmids and isinstance(citing_pmids[0], str) and citing_pmids[0].startswith("ERROR_"):
                   log.warning(f"[Job {job_id}] PubMed citing PMID fetch for PMID {pmid} (DOI {doi}) indicates error: {citing_pmids[0]}. Skipping.")
                   # Optionally mark this article as having a fetch error for this source
                   merged_data[doi]["citations_pubmed_pmids"] = ["FETCH_ERROR"]
                   merged_data[doi]["citations_pubmed_dois"] = ["FETCH_ERROR"]
                   continue

              merged_data[doi]["citations_pubmed_pmids"] = citing_pmids
              # Resolve citing PMIDs to DOIs *that are also in our dataset*
              resolved_dois = []
              for citing_pmid in citing_pmids:
                   citing_doi = pmid_to_doi_map.get(citing_pmid)
                   if citing_doi and citing_doi in merged_data: # Crucial: only link if citing DOI is in our set
                       resolved_dois.append(citing_doi)
              merged_data[doi]["citations_pubmed_dois"] = resolved_dois
         # else: PMID not in our initial set's map, ignore.


    log.info(f"[Job {job_id}] Data merging finished. Final dataset size: {len(merged_data)} articles.")
    return merged_data


def create_citation_graph(
    merged_data: Dict[str, Dict[str, Any]],
    job_id: str = "N/A"
) -> nx.DiGraph:
    """
    Creates a directed graph from the merged article data.
    Nodes are DOIs. Edges represent citations or references.

    Edge Types:
    - 'reference': ref_doi -> doi (article `doi` cites `ref_doi`)
    - 'citation_oc': citing_doi -> doi (OpenCitations says `citing_doi` cites `doi`)
    - 'citation_pm': citing_doi -> doi (PubMed ELink says `citing_doi` cites `doi`)

    Returns:
        A NetworkX DiGraph object.
    """
    log.info(f"[Job {job_id}] Starting graph construction.")
    graph = nx.DiGraph()

    # Add nodes with metadata from merged_data
    for doi, data in merged_data.items():
        if not doi: continue # Skip if somehow DOI is missing here
        # Select attributes to store on the node
        node_attrs = {
            "pmid": data.get("pmid"),
            "title": data.get("title", "No Title"),
            "year": data.get("year", "Unknown"),
            "mesh_terms": data.get("mesh_terms", []),
            "abstract": data.get("abstract", "No Abstract"), # Storing abstract might make graph large
            "reference_count": data.get("reference_count", 0),
            "citation_count_oc": data.get("citation_count_opencitations", 0),
            # Add other counts or flags if needed
        }
        # Filter out attributes with None values if desired
        # node_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        graph.add_node(doi, **node_attrs)

    log.info(f"[Job {job_id}] Added {graph.number_of_nodes()} nodes to the graph.")

    # Add edges based on different sources
    edge_count = 0
    for doi, data in merged_data.items():
        if not doi: continue

        # 1. Reference Edges (Article `doi` cites `ref_doi`)
        # Direction: ref_doi -> doi
        for ref_doi in data.get("references", []):
            if ref_doi and ref_doi in graph: # Check if referenced DOI is in our graph
                if not graph.has_edge(ref_doi, doi):
                     graph.add_edge(ref_doi, doi, type="reference")
                     edge_count += 1
                # else: Edge might already exist from another source, handle later if needed

        # 2. OpenCitations Citation Edges (`citing_doi` cites article `doi`)
        # Direction: citing_doi -> doi
        for citing_doi_oc in data.get("citations_opencitations", []):
             if citing_doi_oc and citing_doi_oc in graph:
                  if not graph.has_edge(citing_doi_oc, doi):
                       graph.add_edge(citing_doi_oc, doi, type="citation_oc")
                       edge_count += 1
                  # else: Maybe add weight or merge types if edge exists? For now, ignore.

        # 3. PubMed Citation Edges (`citing_doi` cites article `doi`)
        # Direction: citing_doi -> doi
        # Using the DOIs we resolved earlier that are present in our dataset
        for citing_doi_pm in data.get("citations_pubmed_dois", []):
             if citing_doi_pm and citing_doi_pm in graph:
                 if not graph.has_edge(citing_doi_pm, doi):
                       # Check if it exists as 'citation_oc' already
                       if graph.has_edge(citing_doi_pm, doi, key=None): # Check if *any* edge exists
                            existing_data = graph.get_edge_data(citing_doi_pm, doi)
                            # If existing type is reference, this is weird (cycle?) log it.
                            if existing_data.get('type') == 'reference':
                                 log.warning(f"[Job {job_id}] Potential citation conflict/cycle? PM cites {doi} ({citing_doi_pm}), but reference edge exists.")
                            # If existing type is oc, maybe merge? For now, let pm override or add second edge? Let's just add type.
                            existing_types = existing_data.get('type', '').split(',')
                            if 'citation_pm' not in existing_types:
                                 graph[citing_doi_pm][doi]['type'] = existing_data.get('type','') + ',citation_pm'
                       else:
                           graph.add_edge(citing_doi_pm, doi, type="citation_pm")
                           edge_count += 1


    log.info(f"[Job {job_id}] Graph construction finished. Added {edge_count} new edges. Total edges: {graph.number_of_edges()}")

    # Optional: Check for cycles (citations should ideally form a DAG)
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            log.warning(f"[Job {job_id}] Found {len(cycles)} cycle(s) in the citation graph. Example cycle: {cycles[0][:5]}...")
            # Decide how to handle cycles if necessary (e.g., remove edges)
    except Exception as cycle_error:
        log.error(f"[Job {job_id}] Error checking for cycles: {cycle_error}")


    return graph