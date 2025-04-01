# backend/opencitations_fetcher.py
import requests
import time
import random
import logging
import json # Added for JSONDecodeError handling
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Configure logging for this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers(): # Add handler if not configured globally
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


def fetch_opencitations_citations(dois: List[str],
                                  max_retries: int = 2, # Keep retries low for OC
                                  base_delay: float = 5.0, # Keep delay higher
                                  max_workers: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Fetches citation data (papers citing the input DOIs) from OpenCitations COCI API.

    Args:
        dois: List of DOIs to fetch citation data for.
        max_retries: Maximum number of retries for failed requests per DOI.
        base_delay: Base delay for exponential backoff on retries.
        max_workers: Maximum number of concurrent threads for fetching.

    Returns:
        A dictionary mapping input DOIs to their OpenCitations data
        (e.g., {'doi': {'citations': [...], 'citation_count': N}}).
        Returns empty lists/zero counts for DOIs not found or failed.
    """
    if not dois:
        log.info("fetch_opencitations_citations called with empty DOI list.")
        return {}

    log.info(f"Fetching OpenCitations data for {len(dois)} DOIs (workers: {max_workers})")
    all_opencitations_data: Dict[str, Dict[str, Any]] = {}
    base_url = "https://opencitations.net/index/coci/api/v1/citations/"
    processed_count = 0

    # --- Worker Function ---
    def process_single_doi(doi_to_process: str) -> Optional[Dict[str, Any]]:
        """Fetches and processes data for a single DOI with retries."""
        if not doi_to_process:
             return None

        default_result = {"citations": [], "citation_count": 0}
        # URL-encode the DOI to handle special characters safely
        try:
            encoded_doi = requests.utils.quote(doi_to_process, safe='') # Encode fully
            url = f"{base_url}{encoded_doi}"
        except Exception as e:
             log.error(f"Failed to URL-encode DOI {doi_to_process}: {e}. Skipping.")
             return default_result

        attempt = 0
        last_exception = None

        while attempt <= max_retries:
            try:
                response = requests.get(url, timeout=45) # Increased timeout for OC API

                # 200 OK: Check content
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Expecting a list of citation objects
                        if isinstance(data, list) and data: # Check if list and not empty
                            citations = [
                                item.get("citing") for item in data
                                if isinstance(item, dict) and item.get("citing") # Ensure item is dict and 'citing' exists
                            ]
                            # Filter out potential None values if get returns None
                            citations = [c for c in citations if c]
                            return {"citations": citations, "citation_count": len(citations)}
                        else:
                            # Empty list means no citations found for this DOI
                            log.debug(f"DOI {doi_to_process}: No citation data in OpenCitations response (empty list or non-list).")
                            return default_result
                    except json.JSONDecodeError:
                        # Handle cases where 200 OK is returned but body is not valid JSON
                        log.warning(f"OpenCitations returned non-JSON 200 OK for DOI {doi_to_process}. Treating as no citations found.")
                        return default_result

                # 404 Not Found: Treat as successfully processed (no data)
                elif response.status_code == 404:
                    log.debug(f"DOI {doi_to_process} not found in OpenCitations (404).")
                    return default_result

                # Other HTTP errors -> Raise to trigger retry
                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e: # Includes HTTPError, ConnectionError, Timeout
                last_exception = e
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 0

                # Decide whether to retry based on error type/status
                should_retry = isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) or \
                               (status_code in [429, 500, 502, 503, 504])

                if should_retry:
                    attempt += 1
                    if attempt > max_retries:
                        log.error(f"OpenCitations give up for {doi_to_process} after {max_retries} retries ({type(e).__name__}).")
                        return default_result
                    delay = base_delay * (2**attempt) + random.uniform(0, 2)
                    log.warning(f"OpenCitations {type(e).__name__} (status: {status_code}) for {doi_to_process}. Retrying in {delay:.1f}s... ({attempt}/{max_retries})")
                    time.sleep(delay)
                else: # Non-retryable request errors
                    log.error(f"OpenCitations non-retryable request error for {doi_to_process}: {e}. Giving up.", exc_info=False)
                    return default_result

            except Exception as e: # Catch any other unexpected errors
                last_exception = e
                log.error(f"OpenCitations unexpected error for {doi_to_process}: {e}. Giving up.", exc_info=True)
                return default_result

        # If loop finishes without returning
        log.error(f"OpenCitations ultimately failed for {doi_to_process} after {max_retries+1} attempts. Last error: {last_exception}")
        return default_result
    # --- End Worker Function ---

    # --- Execute Workers ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        valid_dois = [doi for doi in dois if doi]
        if not valid_dois:
            log.warning("No valid DOIs provided to OpenCitations fetcher.")
            return {}
        future_to_doi = {executor.submit(process_single_doi, doi): doi for doi in valid_dois}

        for future in tqdm(as_completed(future_to_doi), total=len(valid_dois), desc="Fetching OpenCitations", unit="doi", ncols=80):
            doi = future_to_doi[future]
            try:
                result = future.result()
                all_opencitations_data[doi] = result if result is not None else {"citations": [], "citation_count": 0}
                processed_count += 1
            except Exception as e:
                log.error(f"Error retrieving OpenCitations result future for DOI {doi}: {e}", exc_info=True)
                all_opencitations_data[doi] = {"citations": [], "citation_count": 0}

    # --- Final Log ---
    log.info(f"OpenCitations data fetching complete. Processed {processed_count}/{len(valid_dois)} valid DOIs.")
    missed_dois = set(valid_dois) - set(all_opencitations_data.keys())
    if missed_dois:
         log.warning(f"These DOIs were submitted but missing from final OpenCitations results: {missed_dois}")

    return all_opencitations_data