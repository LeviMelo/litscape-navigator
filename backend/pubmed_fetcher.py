# backend/pubmed_fetcher.py
import requests
import xml.etree.ElementTree as ET
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed # Keep if search_pubmed becomes threaded
from tqdm import tqdm
from typing import List, Dict, Optional, Set, Any # Added Any
import json

# Configure logging for this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers(): # Add handler if not configured globally
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# --- PubMed Search Function ---
# Using a slightly enhanced version that fetches more fields needed for merge
def search_pubmed_sync(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Synchronously searches PubMed, fetches article metadata.

    Returns:
      List of dictionaries, each containing:
      'doi', 'pmid', 'title', 'abstract', 'year', 'mesh_terms'.
      Fields might be None or empty lists if not found.
    """
    log.info(f"Starting PubMed search for: '{query}' (max_results: {max_results})")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&usehistory=y&retmode=json"
    articles: List[Dict[str, Any]] = []
    fetch_chunk_size = 200 # How many articles to fetch details for at once

    try:
        # Step 1: Search for PMIDs and get history server info
        search_response = requests.get(search_url, timeout=20)
        search_response.raise_for_status()
        search_data = search_response.json()

        esearchresult = search_data.get("esearchresult", {})
        id_list = esearchresult.get("idlist", [])
        count = int(esearchresult.get("count", 0))
        webenv = esearchresult.get("webenv")
        query_key = esearchresult.get("querykey")

        if not id_list or not webenv or not query_key or count == 0:
            log.info("No articles found or history server info missing.")
            return []

        actual_fetch_count = min(count, max_results) # Don't try to fetch more than found or requested
        log.info(f"Found {count} PMIDs. Will fetch details for {actual_fetch_count} using history server.")

        # Step 2: Fetch details in chunks using history server
        for retstart in tqdm(range(0, actual_fetch_count, fetch_chunk_size), desc="Fetching PubMed Details"):
            fetch_url = (f"{base_url}efetch.fcgi?db=pubmed&query_key={query_key}&WebEnv={webenv}"
                         f"&retstart={retstart}&retmax={fetch_chunk_size}&retmode=xml")

            try:
                fetch_response = requests.get(fetch_url, timeout=45) # Longer timeout for potentially large fetches
                fetch_response.raise_for_status()

                # Step 3: Parse XML chunk
                tree = ET.fromstring(fetch_response.content)
                for article_data in tree.findall(".//PubmedArticle"):
                    try:
                        doi: Optional[str] = None
                        pmid: Optional[str] = None
                        title: Optional[str] = "No Title"
                        abstract: Optional[str] = "No Abstract"
                        year: Optional[str] = "Unknown"
                        mesh_terms: List[str] = []

                        # Extract PMID
                        pmid_elem = article_data.find(".//MedlineCitation/PMID")
                        if pmid_elem is not None: pmid = pmid_elem.text

                        # Extract DOI
                        doi_elem = article_data.find(".//PubmedData/ArticleIdList/ArticleId[@IdType='doi']")
                        if doi_elem is not None: doi = doi_elem.text

                        # Extract Title
                        title_elem = article_data.find(".//ArticleTitle")
                        if title_elem is not None: title = title_elem.text if title_elem.text else "No Title"

                        # Extract Abstract (handle multiple AbstractText elements)
                        abstract_parts = []
                        for ab_elem in article_data.findall(".//Abstract/AbstractText"):
                            if ab_elem.text:
                                label = ab_elem.get("Label")
                                text_part = f"{label}: {ab_elem.text}" if label else ab_elem.text
                                abstract_parts.append(text_part)
                        if abstract_parts: abstract = "\n".join(abstract_parts)

                        # Extract Year
                        pubdate = article_data.find(".//Article/Journal/JournalIssue/PubDate")
                        if pubdate is not None:
                            year_elem = pubdate.find("Year")
                            if year_elem is not None and year_elem.text:
                                year = year_elem.text
                            else: # Try MedlineDate format e.g., "2023 Oct-Dec" or "2023 Spring"
                                medline_date_elem = pubdate.find("MedlineDate")
                                if medline_date_elem is not None and medline_date_elem.text:
                                    year_match = medline_date_elem.text.strip()[:4] # Take first 4 chars
                                    if year_match.isdigit(): year = year_match

                        # Extract MeSH Terms
                        mesh_list = article_data.findall(".//MeshHeadingList/MeshHeading")
                        for mesh_heading in mesh_list:
                            desc_elem = mesh_heading.find("DescriptorName")
                            if desc_elem is not None and desc_elem.text:
                                mesh_terms.append(desc_elem.text)
                                # Optionally add qualifiers too
                                # qual_list = mesh_heading.findall("QualifierName")
                                # for qual_elem in qual_list:
                                #    if qual_elem is not None and qual_elem.text:
                                #         mesh_terms.append(f"{desc_elem.text}/{qual_elem.text}")

                        # Append if we have at least a PMID or DOI
                        if pmid or doi:
                            articles.append({
                                "pmid": pmid,
                                "doi": doi,
                                "title": title,
                                "abstract": abstract,
                                "year": year,
                                "mesh_terms": mesh_terms
                            })
                        else:
                             log.debug("Skipping article record with no PMID or DOI.")

                    except Exception as parse_err:
                        log.warning(f"Error parsing individual PubMed article (PMID: {pmid}): {parse_err}", exc_info=False)
                        continue # Skip this article, continue with others in chunk

            except requests.exceptions.RequestException as fetch_err:
                 log.error(f"Error fetching PubMed details chunk (retstart {retstart}): {fetch_err}. Skipping chunk.", exc_info=False)
                 continue # Skip this chunk, try next one
            except ET.ParseError as xml_err:
                 log.error(f"Error parsing PubMed details XML chunk (retstart {retstart}): {xml_err}. Skipping chunk.", exc_info=False)
                 continue

        log.info(f"PubMed search and fetch complete. Retrieved details for {len(articles)} articles.")
        return articles

    except requests.exceptions.RequestException as e:
        log.error(f"Initial PubMed search request failed: {e}", exc_info=True)
        return []
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse initial PubMed search JSON response: {e}", exc_info=True)
        return []
    except Exception as e:
        log.error(f"An unexpected error occurred during PubMed search: {e}", exc_info=True)
        return []

# --- PubMed Citing PMIDs Fetch Function ---
def fetch_pubmed_citing_pmids(pmids: List[str],
                              chunk_size: int = 150,
                              max_retries: int = 3,
                              base_delay: float = 2.0) -> Dict[str, List[str]]:
    """
    Fetches PMIDs that cite the given list of input PMIDs using PubMed eLink.
    (Code is identical to the version provided in the previous response for Phase 3, Step 3)
    """
    if not pmids:
        return {}

    log.info(f"Fetching PubMed citing PMIDs for {len(pmids)} input PMIDs (chunk size: {chunk_size})")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    # Initialize dict with empty lists for all input PMIDs
    pmid_citations: Dict[str, List[str]] = {pmid: [] for pmid in pmids}

    chunks = [pmids[i : i + chunk_size] for i in range(0, len(pmids), chunk_size)]

    for chunk in tqdm(chunks, desc="Fetching Citing PMIDs", ncols=80):
        if not chunk: continue

        params = {
            "dbfrom": "pubmed", "db": "pubmed", "cmd": "neighbor",
            "linkname": "pubmed_pubmed_citedin", "id": ",".join(chunk), "retmode": "xml",
        }
        attempt = 0
        success = False
        while attempt <= max_retries and not success:
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 429: raise requests.exceptions.RequestException("Rate limited (429)")
                response.raise_for_status()
                root = ET.fromstring(response.content)
                processed_in_chunk: Set[str] = set()

                for linkset in root.findall("LinkSet"):
                    id_list_elem = linkset.find("IdList")
                    if id_list_elem is None: continue
                    input_pmid_list = [id_elem.text for id_elem in id_list_elem.findall("Id") if id_elem.text]
                    if not input_pmid_list: continue

                    citing_pmids_for_input = []
                    linksetdb_list = linkset.findall("LinkSetDb")
                    for linksetdb in linksetdb_list:
                        link_name = linksetdb.findtext("LinkName")
                        if link_name == "pubmed_pubmed_citedin":
                            for link in linksetdb.findall("Link"):
                                cited_id = link.findtext("Id")
                                if cited_id: citing_pmids_for_input.append(cited_id)
                            break # Found citedin

                    for input_pmid in input_pmid_list:
                        if input_pmid in pmid_citations:
                            current_set = set(pmid_citations[input_pmid])
                            current_set.update(citing_pmids_for_input)
                            pmid_citations[input_pmid] = list(current_set)
                            processed_in_chunk.add(input_pmid)

                missing_in_chunk = set(chunk) - processed_in_chunk
                if missing_in_chunk: log.warning(f"Response missing LinkSet for PMIDs: {missing_in_chunk}")
                success = True # Mark chunk success

            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt > max_retries: log.error(f"Failed citing fetch chunk ({chunk[0]}) retries: {e}. Skip.", exc_info=False); break
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                log.warning(f"PubMed citing error chunk ({chunk[0]}): {e}. Retry {attempt}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            except ET.ParseError as e: log.error(f"Failed parse citing XML chunk ({chunk[0]}): {e}. Skip.", exc_info=False); break
            except Exception as e: log.error(f"Unexpected citing error chunk ({chunk[0]}): {e}. Skip.", exc_info=True); break

    total_found = sum(len(v) for v in pmid_citations.values())
    log.info(f"PubMed citing PMID fetch complete. Found {total_found} citation links.")
    return pmid_citations