# backend/crossref_fetcher.py
import requests
import time
import random
import logging
from habanero import Crossref
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Configure logging for this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set default level for this logger
# Add handler if logging is not configured globally (e.g., by FastAPI/Uvicorn)
# This ensures logs appear when running fetchers independently if needed.
if not log.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


# Suppress excessive logging from habanero library itself
logging.getLogger("habanero.crossref").setLevel(logging.WARNING)
logging.getLogger("habanero.request").setLevel(logging.WARNING)


def fetch_crossref_references(dois: List[str],
                              mailto: str = "litscape.dev@example.com", # Replace with a real email if deploying
                              max_retries: int = 3,
                              base_delay: float = 2.0,
                              batch_size: int = 20, # Note: habanero might handle internal batching differently
                              max_workers: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Fetches reference and citation count data from CrossRef for a list of DOIs.

    Args:
        dois: List of DOIs to fetch data for.
        mailto: Email address for Crossref politeness pool.
        max_retries: Maximum number of retries for failed requests per DOI.
        base_delay: Base delay for exponential backoff on retries.
        batch_size: Conceptual batch size (actual parallelism depends on max_workers).
        max_workers: Maximum number of concurrent threads for fetching.

    Returns:
        A dictionary mapping input DOIs to their Crossref data
        (e.g., {'doi': {'references': [...], 'reference_count': N, 'is_referenced_by_count': M}}).
        Returns empty lists/zero counts for DOIs not found or failed.
    """
    if not dois:
        log.info("fetch_crossref_references called with empty DOI list.")
        return {}

    log.info(f"Fetching CrossRef data for {len(dois)} DOIs (workers: {max_workers})")
    # Initialize Crossref client with mailto address
    try:
        cr = Crossref(mailto=mailto)
    except Exception as e:
        log.error(f"Failed to initialize Crossref client: {e}", exc_info=True)
        # Return empty dict for all DOIs if client fails
        return {doi: {"references": [], "reference_count": 0, "is_referenced_by_count": 0} for doi in dois}

    all_crossref_data: Dict[str, Dict[str, Any]] = {}
    processed_count = 0

    # --- Worker Function ---
    def process_single_doi(doi_to_process: str) -> Optional[Dict[str, Any]]:
        """Fetches and processes data for a single DOI with retries."""
        nonlocal cr # Access the outer scope Crossref client instance
        if not doi_to_process: # Skip empty DOIs
            return None

        default_result = {"references": [], "reference_count": 0, "is_referenced_by_count": 0}
        attempt = 0
        last_exception = None

        while attempt <= max_retries:
            try:
                # Use habanero's works() method to fetch data for the DOI
                works_result = cr.works(ids=doi_to_process)

                # Check the structure of the response carefully
                if works_result and isinstance(works_result, dict) and \
                   'message' in works_result and isinstance(works_result['message'], dict):

                    item = works_result['message']
                    references = []
                    # Ensure 'reference' exists, is a list, and iterate safely
                    if item.get("reference") and isinstance(item["reference"], list):
                        references = [
                            ref.get("DOI")
                            for ref in item["reference"]
                            if isinstance(ref, dict) and ref.get("DOI") # Check ref is dict and DOI exists
                        ]

                    return {
                        "references": references,
                        "reference_count": item.get("reference-count", 0),
                        "is_referenced_by_count": item.get("is-referenced-by-count", 0)
                    }
                else:
                    # Handle cases where DOI is valid but no message data returned
                    log.debug(f"DOI {doi_to_process}: No 'message' data found in Crossref response.")
                    return default_result # Return default structure

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response is not None else 0
                if status_code == 404:
                    log.debug(f"DOI {doi_to_process} not found in Crossref (404).")
                    return default_result # Treat 404 as successfully processed (no data)
                elif status_code in [429, 500, 502, 503, 504]: # Rate limit or server errors -> Retry
                    attempt += 1
                    if attempt > max_retries:
                        log.error(f"CrossRef give up for {doi_to_process} after {max_retries} retries (HTTP {status_code}).")
                        return default_result
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    log.warning(f"CrossRef {status_code} for {doi_to_process}. Retrying in {delay:.1f}s... ({attempt}/{max_retries})")
                    time.sleep(delay)
                else: # Other HTTP errors -> Give up immediately
                    log.error(f"CrossRef non-retryable HTTP error {status_code} for {doi_to_process}: {e}. Giving up.")
                    return default_result

            except requests.exceptions.RequestException as e: # Connection errors, timeouts -> Retry
                 last_exception = e
                 attempt += 1
                 if attempt > max_retries:
                     log.error(f"CrossRef connection/timeout error for {doi_to_process} after {max_retries} retries: {e}. Giving up.", exc_info=False)
                     return default_result
                 delay = base_delay * (2**attempt) + random.uniform(0, 1)
                 log.warning(f"CrossRef connection/timeout for {doi_to_process}: {e}. Retrying in {delay:.1f}s... ({attempt}/{max_retries})")
                 time.sleep(delay)

            except Exception as e: # Catch any other unexpected errors during processing
                last_exception = e
                log.error(f"CrossRef unexpected error for {doi_to_process}: {e}. Giving up.", exc_info=True)
                return default_result # Give up on unexpected errors

        # If loop finishes without returning (should only happen on max retries)
        log.error(f"CrossRef ultimately failed for {doi_to_process} after {max_retries+1} attempts. Last error: {last_exception}")
        return default_result
    # --- End Worker Function ---

    # --- Execute Workers ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create future tasks for all valid DOIs
        valid_dois = [doi for doi in dois if doi] # Filter out potential None/empty strings
        if not valid_dois:
             log.warning("No valid DOIs provided after filtering.")
             return {}
        future_to_doi = {executor.submit(process_single_doi, doi): doi for doi in valid_dois}

        # Process results as they complete, with progress bar
        for future in tqdm(as_completed(future_to_doi), total=len(valid_dois), desc="Fetching CrossRef Refs", unit="doi", ncols=80):
            doi = future_to_doi[future]
            try:
                result = future.result()
                # Ensure we always store a result, even if None was returned unexpectedly
                all_crossref_data[doi] = result if result is not None else {"references": [], "reference_count": 0, "is_referenced_by_count": 0}
                processed_count += 1
            except Exception as e:
                # Log error if fetching the future's result fails
                log.error(f"Error retrieving CrossRef result future for DOI {doi}: {e}", exc_info=True)
                all_crossref_data[doi] = {"references": [], "reference_count": 0, "is_referenced_by_count": 0} # Store default on error

    # --- Final Log ---
    log.info(f"CrossRef data fetching complete. Processed {processed_count}/{len(valid_dois)} valid DOIs.")
    # Check if any DOIs were completely missed (shouldn't happen with current logic)
    missed_dois = set(valid_dois) - set(all_crossref_data.keys())
    if missed_dois:
         log.warning(f"These DOIs were submitted but missing from final CrossRef results: {missed_dois}")

    return all_crossref_data