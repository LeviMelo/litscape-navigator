// frontend/src/App.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';

// --- Interfaces ---
// Represents the structure of article data stored in merged_data.json
interface Article {
  doi: string | null;
  pmid: string | null;
  title: string | null;
  abstract: string | null;
  year: string | null;
  mesh_terms?: string[]; // Optional
  references?: string[]; // List of DOIs it references
  citations?: string[]; // List of DOIs that cite it (within corpus)
  reference_count_raw_cr?: number;
  citation_count_raw_oc?: number;
  citation_count_raw_cr?: number;
  citation_count_raw_pm?: number;
  source?: string;
}

// Response from POST /api/searches
interface StartJobResponse {
  job_id: string;
}

// Response from GET /api/searches/{job_id}/status
interface JobStatusResponse {
  job_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  error_message?: string | null;
}

// Structure for graph info returned by results endpoint
interface GraphInfo {
  nodes?: number | null;
  edges?: number | null;
  graph_file_exists: boolean;
}

// Response from GET /api/searches/{job_id}/results
interface ResultsResponse {
  merged_data: { [doi: string]: Article }; // Articles keyed by DOI
  graph_info: GraphInfo;
}
// --- End Interfaces ---


function App() {
  // --- State Variables ---
  const [query, setQuery] = useState<string>('myostatin inhibitor');
  const [articles, setArticles] = useState<Article[]>([]); // Display results
  const [isLoading, setIsLoading] = useState<boolean>(false); // For start/fetch results ops
  const [isPolling, setIsPolling] = useState<boolean>(false); // True while job is PENDING/RUNNING
  const [error, setError] = useState<string | null>(null); // Error display
  const [jobId, setJobId] = useState<string | null>(null); // Active job ID
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null); // Status object
  const [graphInfo, setGraphInfo] = useState<GraphInfo | null>(null); // Info about graph file

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  // --- End State Variables ---


  // --- Helper Functions ---
  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
      setIsPolling(false);
      console.log("Polling stopped.");
    }
  }, []);

  const fetchResults = useCallback(async (completedJobId: string) => {
    if (!completedJobId) return;
    console.log(`Fetching results job: ${completedJobId}`);
    setIsLoading(true); setError(null); setGraphInfo(null);
    try {
      const response = await fetch(`http://localhost:8000/api/searches/${completedJobId}/results`);
      if (!response.ok) {
         let detail = 'Unknown error fetching results.'; try { const data = await response.json(); detail = data.detail || detail; } catch(e){}
         throw new Error(`HTTP ${response.status}: ${detail}`);
      }
      const data: ResultsResponse = await response.json();
      const articlesArray = data.merged_data ? Object.values(data.merged_data) : [];
      // Sort articles by year (descending) if year exists, otherwise keep original order
      articlesArray.sort((a, b) => {
          const yearA = parseInt(a.year || '0');
          const yearB = parseInt(b.year || '0');
          if (yearA && yearB) return yearB - yearA; // Descending numeric sort
          if (yearA) return -1; // Articles with year come first
          if (yearB) return 1;
          return 0; // Keep original order if neither has a year
      });
      setArticles(articlesArray);
      setGraphInfo(data.graph_info || null);
      console.log(`Fetched ${articlesArray.length} articles. Graph?: ${data.graph_info?.graph_file_exists}`);
    } catch (e: unknown) {
      console.error("Fetch results error: ", e);
      const message = e instanceof Error ? e.message : "Unknown fetch results error.";
      setError(`Fetch Results Error: ${message}`); setArticles([]); setGraphInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, []); // fetchResults now depends on nothing external

  const checkJobStatus = useCallback(async (currentJobId: string) => {
    if (!currentJobId) { stopPolling(); return; };
    console.log(`Polling status job: ${currentJobId}`);
    try {
      const response = await fetch(`http://localhost:8000/api/searches/${currentJobId}/status`);
      if (response.status === 404) { setError(`Job ${currentJobId} not found.`); stopPolling(); setJobId(null); setJobStatus(null); setIsLoading(false); return; }
      if (!response.ok) { let detail = 'Status check fail.'; try { const data = await response.json(); detail = data.detail || detail; } catch(e){} throw new Error(`HTTP ${response.status}: ${detail}`); }
      const data: JobStatusResponse = await response.json();
      setJobStatus(data); console.log("Job status:", data.status);
      if (data.status === 'COMPLETED' || data.status === 'FAILED') {
        stopPolling();
        if (data.status === 'COMPLETED') { setError(null); await fetchResults(currentJobId); }
        else { setError(`Job failed: ${data.error_message || 'Unknown worker error'}`); setArticles([]); setGraphInfo(null); } // Clear graph info on fail too
        setIsLoading(false); // Ensure loading stops on terminal state
      } else { setIsPolling(true); setIsLoading(true); } // Keep loading while running
    } catch (e: unknown) { const message = e instanceof Error ? e.message : "Unknown status check error."; setError(`Status Check Error: ${message}. Stop polling.`); stopPolling(); setJobStatus(null); setIsLoading(false); }
  }, [fetchResults, stopPolling]); // Keep dependencies

  const handleSearch = async () => {
    if (!query.trim()) { setError("Enter query."); return; }
    stopPolling(); setIsLoading(true); setError(null); setArticles([]); setJobStatus(null); setJobId(null); setGraphInfo(null); // Clear graph info
    console.log("Starting full pipeline job...");
    try {
      const response = await fetch('http://localhost:8000/api/searches', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ query: query, max_results: 50 }) }); // Using 50 max_results here
      if (!response.ok) { let detail = 'Fail start job.'; try { const data = await response.json(); detail = data.detail || detail; } catch(e){} throw new Error(`HTTP ${response.status}: ${detail}`); }
      const data: StartJobResponse = await response.json();
      const newJobId = data.job_id; setJobId(newJobId); setJobStatus({ job_id: newJobId, status: 'PENDING' }); setIsPolling(true); console.log(`Job ID: ${newJobId}. Polling...`);
      pollingIntervalRef.current = setInterval(() => { checkJobStatus(newJobId); }, 3000); // Poll every 3s
      setTimeout(() => checkJobStatus(newJobId), 500); // Initial check sooner
    } catch (e: unknown) { const message = e instanceof Error ? e.message : "Unknown start error."; setError(`Start Error: ${message}`); setIsLoading(false); setIsPolling(false); setJobStatus(null); setJobId(null); }
  };

  // Cleanup Effect for polling
  useEffect(() => { return () => { stopPolling(); }; }, [stopPolling]);
  // --- End Helper Functions & Effects ---


  // --- Render JSX ---
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-blue-50 flex flex-col items-center pt-10 pb-20 p-4 font-sans">
      {/* Header */}
      <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold text-blue-800 mb-2 tracking-tight">LitScape Navigator</h1>
          <p className="text-lg text-gray-600">Explore Literature Semantically</p>
      </header>

      {/* Search Input Section */}
      <section className="bg-white p-6 rounded-lg shadow-xl w-full max-w-3xl mb-8 border border-gray-200">
         <label htmlFor="search-query" className="block text-md font-semibold text-gray-800 mb-2">
            Enter PubMed Search Query:
         </label>
         <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-3">
            <input
                id="search-query"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., myostatin inhibitor mechanism"
                className="flex-grow p-3 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
                disabled={isLoading || isPolling}
            />
            <button
                onClick={handleSearch}
                disabled={isLoading || isPolling}
                className={`px-6 py-3 rounded-md text-white font-bold transition duration-150 ease-in-out whitespace-nowrap shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                    (isLoading || isPolling)
                    ? 'bg-gray-400 cursor-not-allowed animate-pulse' // Added pulse animation
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
            >
                {/* Dynamic button text */}
                {isPolling ? 'Processing...' : (isLoading ? 'Starting...' : 'Search')}
            </button>
         </div>
         {/* Job Info Area */}
         <div className="mt-4 text-xs text-gray-500 space-y-1">
             {jobId && <div>Job ID: <code className="bg-gray-200 px-1.5 py-0.5 rounded font-mono">{jobId}</code></div>}
             {jobStatus &&
                <div>Status:
                    <span className={`font-semibold uppercase px-2 py-0.5 rounded-full ml-1 ${
                        jobStatus.status === 'FAILED' ? 'text-red-800 bg-red-100' :
                        jobStatus.status === 'COMPLETED' ? 'text-green-800 bg-green-100' :
                        jobStatus.status === 'RUNNING' ? 'text-indigo-800 bg-indigo-100 animate-pulse' : // Pulse for running
                        'text-gray-800 bg-gray-100' // Pending
                    }`}>
                        {jobStatus.status}
                    </span>
                </div>
             }
         </div>
         {/* Error Display Area */}
         {error && <p className="mt-4 text-sm text-red-700 bg-red-100 p-3 rounded-md border border-red-300 shadow-sm">{error}</p>}
      </section>

      {/* Results Section */}
      <section className="bg-white p-6 rounded-lg shadow-xl w-full max-w-3xl border border-gray-200">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800 border-b border-gray-200 pb-3">Results</h2>
        {/* Loading/Idle/Empty States */}
        {!jobId && !isLoading && <p className="text-gray-500 italic">Enter a query and click Search to begin.</p>}
        {(isLoading || isPolling) && jobStatus?.status !== 'COMPLETED' && jobStatus?.status !== 'FAILED' &&
            <div className="flex items-center justify-center text-gray-500 py-4">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing search, please wait... This may take several minutes.
            </div>
        }
        {jobStatus?.status === 'COMPLETED' && articles.length === 0 && <p className="text-gray-500 italic">Search completed, but no articles were found or processed successfully for this query.</p>}

        {/* Display Results List */}
        {jobStatus?.status === 'COMPLETED' && articles.length > 0 && (
            <>
                <div className="text-sm text-gray-600 mb-4 flex justify-between items-center">
                    <span>Displaying {articles.length} articles (sorted by year descending).</span>
                    <span className={`text-xs font-medium px-2 py-1 rounded-full ${graphInfo?.graph_file_exists ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'}`}>
                        Graph File: {graphInfo?.graph_file_exists ? 'Generated' : 'Not Generated'}
                    </span>
                </div>
                <ul className="space-y-4 max-h-[60vh] overflow-y-auto pr-2"> {/* Scrollable list */}
                    {articles.map((article) => (
                    <li key={article.doi || article.pmid} className="text-sm border rounded-md p-4 shadow-sm bg-white hover:shadow-md transition-shadow duration-150 ease-in-out">
                        {/* Title and Year */}
                        <div className="flex justify-between items-start mb-1">
                            <h3 className="font-semibold text-base text-gray-800 mr-2">{article.title || 'No Title Available'}</h3>
                            <span className="text-xs font-medium text-gray-500 whitespace-nowrap bg-gray-100 px-1.5 py-0.5 rounded">{article.year || 'N/A'}</span>
                        </div>
                        {/* IDs and Links */}
                        <div className="text-xs text-blue-600 mt-1.5 space-x-2">
                            {article.doi &&
                                <a href={`https://doi.org/${article.doi}`} target="_blank" rel="noopener noreferrer" className="hover:underline">
                                    DOI: {article.doi}
                                </a>
                            }
                            {article.doi && article.pmid && <span>|</span>}
                            {article.pmid &&
                                <a href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}/`} target="_blank" rel="noopener noreferrer" className="hover:underline">
                                    PMID: {article.pmid}
                                </a>
                            }
                            {(!article.doi && !article.pmid) && <span className="text-gray-400">No ID</span>}
                        </div>
                        {/* MeSH Terms */}
                        {article.mesh_terms && article.mesh_terms.length > 0 && (
                            <div className="mt-2 text-xs text-gray-600">
                                <span className="font-medium">MeSH: </span>
                                {article.mesh_terms.slice(0, 6).join('; ')}{article.mesh_terms.length > 6 ? '...' : ''}
                            </div>
                        )}
                        {/* Optional: Abstract Snippet (if needed later) */}
                        {/* <p className="text-xs text-gray-700 mt-2 leading-relaxed line-clamp-2">
                            {article.abstract ? article.abstract.substring(0, 150) + '...' : 'No abstract available.'}
                        </p> */}
                    </li>
                    ))}
                </ul>
            </>
        )}
         {/* Error message displayed in the search section */}
      </section>
    </div>
  );
}

export default App;