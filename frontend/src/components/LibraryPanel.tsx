import { useState, useEffect } from 'react'

const API_BASE_URL = import.meta.env.VITE_RAG_BASE_URL || 'http://localhost:8000'

interface Paper {
  paper_id: string
  title: string
  authors?: string[]
  year?: number
  venue?: string
  arxiv_id?: string
  num_chunks?: number
  file_path?: string
}

interface PapersResponse {
  papers: Paper[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export default function LibraryPanel() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [total, setTotal] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)
  const itemsPerPage = 10

  useEffect(() => {
    loadPapers()
  }, [currentPage, searchQuery])

  const loadPapers = async () => {
    setLoading(true)
    setError(null)

    try {
      const offset = (currentPage - 1) * itemsPerPage
      const params = new URLSearchParams({
        limit: itemsPerPage.toString(),
        offset: offset.toString(),
      })

      if (searchQuery) {
        params.append('search', searchQuery)
      }

      const res = await fetch(`${API_BASE_URL}/papers?${params}`)

      if (!res.ok) {
        throw new Error(`Failed to load papers: ${res.status}`)
      }

      const data: PapersResponse = await res.json()
      setPapers(data.papers)
      setTotal(data.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load papers')
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    setCurrentPage(1) // Reset to first page on new search
  }

  const totalPages = Math.ceil(total / itemsPerPage)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Paper Library</h2>
            <p className="text-sm text-gray-500 mt-1">
              {total} papers indexed
            </p>
          </div>
          <button
            onClick={loadPapers}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
          >
            {loading ? '🔄 Loading...' : '🔄 Refresh'}
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search papers by title or author..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-full px-4 py-3 pl-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <span className="absolute left-3 top-3 text-xl">🔍</span>
          {searchQuery && (
            <button
              onClick={() => handleSearch('')}
              className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <span className="text-red-600 text-xl mr-3">❌</span>
            <div>
              <h3 className="text-red-800 font-medium">Error</h3>
              <p className="text-red-600 text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Papers List */}
      {loading && papers.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="text-gray-600 mt-4">Loading papers...</p>
        </div>
      ) : papers.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <div className="text-6xl mb-4">📚</div>
          <p className="text-gray-600">
            {searchQuery
              ? `No papers found matching "${searchQuery}"`
              : 'No papers found in the library'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {papers.map((paper) => (
            <div
              key={paper.paper_id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => setSelectedPaper(paper)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  {/* Title */}
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {paper.title}
                  </h3>

                  {/* Authors */}
                  {paper.authors && paper.authors.length > 0 && (
                    <p className="text-sm text-gray-600 mb-2">
                      👥 {paper.authors.slice(0, 3).join(', ')}
                      {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                    </p>
                  )}

                  {/* Metadata */}
                  <div className="flex flex-wrap gap-3 text-sm text-gray-500">
                    {paper.year && (
                      <span className="flex items-center">
                        📅 {paper.year}
                      </span>
                    )}
                    {paper.venue && (
                      <span className="flex items-center">
                        🏛️ {paper.venue}
                      </span>
                    )}
                    {paper.arxiv_id && (
                      <span className="flex items-center">
                        📄 arXiv:{paper.arxiv_id}
                      </span>
                    )}
                    {paper.num_chunks !== undefined && paper.num_chunks > 0 && (
                      <span className="flex items-center">
                        🧩 {paper.num_chunks} chunks
                      </span>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="ml-4 flex flex-col space-y-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedPaper(paper)
                    }}
                    className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                  >
                    View Details
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Showing {(currentPage - 1) * itemsPerPage + 1}-
              {Math.min(currentPage * itemsPerPage, total)} of {total} papers
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                ← Previous
              </button>

              <div className="flex items-center space-x-1">
                {[...Array(Math.min(5, totalPages))].map((_, i) => {
                  let pageNum
                  if (totalPages <= 5) {
                    pageNum = i + 1
                  } else if (currentPage <= 3) {
                    pageNum = i + 1
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i
                  } else {
                    pageNum = currentPage - 2 + i
                  }

                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`px-3 py-1 rounded ${
                        currentPage === pageNum
                          ? 'bg-blue-600 text-white'
                          : 'border border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      {pageNum}
                    </button>
                  )
                })}
              </div>

              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Paper Detail Modal */}
      {selectedPaper && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          onClick={() => setSelectedPaper(null)}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900 flex-1">
                  {selectedPaper.title}
                </h2>
                <button
                  onClick={() => setSelectedPaper(null)}
                  className="text-gray-400 hover:text-gray-600 text-2xl ml-4"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                {selectedPaper.authors && selectedPaper.authors.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">Authors</h3>
                    <p className="text-sm text-gray-600">
                      {selectedPaper.authors.join(', ')}
                    </p>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  {selectedPaper.year && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-700 mb-1">Year</h3>
                      <p className="text-sm text-gray-600">{selectedPaper.year}</p>
                    </div>
                  )}

                  {selectedPaper.venue && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-700 mb-1">Venue</h3>
                      <p className="text-sm text-gray-600">{selectedPaper.venue}</p>
                    </div>
                  )}

                  {selectedPaper.arxiv_id && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-700 mb-1">ArXiv ID</h3>
                      <p className="text-sm text-gray-600">
                        <a
                          href={`https://arxiv.org/abs/${selectedPaper.arxiv_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          {selectedPaper.arxiv_id}
                        </a>
                      </p>
                    </div>
                  )}

                  {selectedPaper.num_chunks !== undefined && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-700 mb-1">Chunks</h3>
                      <p className="text-sm text-gray-600">{selectedPaper.num_chunks}</p>
                    </div>
                  )}
                </div>

                {selectedPaper.file_path && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-700 mb-1">File Path</h3>
                    <p className="text-sm text-gray-600 font-mono bg-gray-50 p-2 rounded">
                      {selectedPaper.file_path}
                    </p>
                  </div>
                )}

                <div className="pt-4 border-t border-gray-200">
                  <button
                    onClick={() => {
                      // Navigate to Ask panel with pre-filled question
                      setSelectedPaper(null)
                      // You can implement this by passing a callback or using context/state management
                    }}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Ask Question About This Paper
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
