import { useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_RAG_BASE_URL || 'http://localhost:8000'

interface IndexMetrics {
  retrieved_n?: number
  kept_n?: number
  latency_ms?: number
  total_documents?: number
  message?: string
}

export default function IngestPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [uploadResult, setUploadResult] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<IndexMetrics | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      // Validate file type
      const validTypes = ['application/pdf', 'text/plain']
      if (!validTypes.includes(file.type)) {
        setError('Please select a PDF or TXT file')
        return
      }
      setSelectedFile(file)
      setError(null)
      setUploadResult(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first')
      return
    }

    setUploading(true)
    setError(null)
    setUploadResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      // Note: This endpoint needs to be implemented in the backend
      const res = await fetch(`${API_BASE_URL}/papers/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Upload failed: ${res.status}`)
      }

      const data = await res.json()
      setUploadResult(
        data.message || `Successfully uploaded: ${selectedFile.name}`
      )
      setSelectedFile(null)

      // Reset file input
      const fileInput = document.getElementById('file-upload') as HTMLInputElement
      if (fileInput) fileInput.value = ''
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleRefreshIndex = async () => {
    setRefreshing(true)
    setError(null)
    setMetrics(null)

    try {
      // Note: This endpoint needs to be implemented in the backend
      const res = await fetch(`${API_BASE_URL}/index/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!res.ok) {
        throw new Error(`Refresh failed: ${res.status}`)
      }

      const data = await res.json()
      setMetrics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Refresh failed')
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Paper Card */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Upload Paper
        </h2>

        <div className="space-y-4">
          {/* File Upload Area */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
            <input
              id="file-upload"
              type="file"
              accept=".pdf,.txt"
              onChange={handleFileSelect}
              className="hidden"
              disabled={uploading}
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center"
            >
              <div className="text-5xl mb-4">📄</div>
              <p className="text-sm text-gray-600 mb-2">
                Click to select or drag and drop
              </p>
              <p className="text-xs text-gray-500">PDF or TXT files</p>
            </label>
          </div>

          {/* Selected File Display */}
          {selectedFile && (
            <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">📎</span>
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  setSelectedFile(null)
                  const fileInput = document.getElementById('file-upload') as HTMLInputElement
                  if (fileInput) fileInput.value = ''
                }}
                className="text-red-600 hover:text-red-800"
                disabled={uploading}
              >
                ✕
              </button>
            </div>
          )}

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className={`
              w-full py-3 px-6 rounded-lg font-medium text-white
              transition-all duration-200
              ${
                !selectedFile || uploading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700 active:bg-green-800'
              }
            `}
          >
            {uploading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Uploading...
              </span>
            ) : (
              '📤 Upload Paper'
            )}
          </button>

          {/* Upload Success */}
          {uploadResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-start">
                <span className="text-green-600 text-xl mr-3">✅</span>
                <div>
                  <h3 className="text-green-800 font-medium">Success</h3>
                  <p className="text-green-600 text-sm mt-1">{uploadResult}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Refresh Index Card */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Index Management
        </h2>

        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Refresh the vector database index to include newly uploaded papers.
            This process may take a few minutes depending on the number of new
            documents.
          </p>

          <button
            onClick={handleRefreshIndex}
            disabled={refreshing}
            className={`
              w-full py-3 px-6 rounded-lg font-medium text-white
              transition-all duration-200
              ${
                refreshing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
              }
            `}
          >
            {refreshing ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Refreshing Index...
              </span>
            ) : (
              '🔄 Refresh Index'
            )}
          </button>

          {/* Metrics Display */}
          {metrics && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-blue-900 font-medium mb-3">Index Metrics</h3>
              <div className="grid grid-cols-2 gap-4">
                {metrics.retrieved_n !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {metrics.retrieved_n}
                    </p>
                    <p className="text-xs text-gray-600">Retrieved</p>
                  </div>
                )}
                {metrics.kept_n !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {metrics.kept_n}
                    </p>
                    <p className="text-xs text-gray-600">Kept</p>
                  </div>
                )}
                {metrics.latency_ms !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {metrics.latency_ms}ms
                    </p>
                    <p className="text-xs text-gray-600">Latency</p>
                  </div>
                )}
                {metrics.total_documents !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {metrics.total_documents}
                    </p>
                    <p className="text-xs text-gray-600">Total Docs</p>
                  </div>
                )}
              </div>
              {metrics.message && (
                <p className="text-sm text-blue-700 mt-3">{metrics.message}</p>
              )}
            </div>
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
    </div>
  )
}
