import { useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_RAG_BASE_URL || 'http://localhost:8000'

interface HealthResponse {
  retrieved_n?: number
  kept_n?: number
  citations?: number
  latency_ms?: number
  trace_id?: string
  status?: string
  timestamp?: string
  [key: string]: any
}

export default function DebugPanel() {
  const [loading, setLoading] = useState(false)
  const [healthData, setHealthData] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<HealthResponse[]>([])

  const checkHealth = async () => {
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE_URL}/health/rag`)

      if (!res.ok) {
        throw new Error(`Health check failed: ${res.status}`)
      }

      const data = await res.json()
      const timestampedData = {
        ...data,
        timestamp: new Date().toLocaleTimeString(),
      }

      setHealthData(timestampedData)
      setHistory((prev) => [timestampedData, ...prev].slice(0, 10))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed')
    } finally {
      setLoading(false)
    }
  }

  const clearHistory = () => {
    setHistory([])
  }

  return (
    <div className="space-y-6">
      {/* Health Check Card */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          System Health Check
        </h2>

        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Check the current status of the RAG system, including retrieval
            metrics and performance indicators.
          </p>

          <button
            onClick={checkHealth}
            disabled={loading}
            className={`
              w-full py-3 px-6 rounded-lg font-medium text-white
              transition-all duration-200
              ${
                loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
              }
            `}
          >
            {loading ? (
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
                Checking...
              </span>
            ) : (
              '🔍 Check Health'
            )}
          </button>
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

      {/* Latest Health Data */}
      {healthData && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Latest Health Status
            </h2>
            {healthData.timestamp && (
              <span className="text-xs text-gray-500">
                {healthData.timestamp}
              </span>
            )}
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {healthData.retrieved_n !== undefined && (
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <p className="text-3xl font-bold text-blue-600">
                  {healthData.retrieved_n}
                </p>
                <p className="text-xs text-gray-600 mt-1">Retrieved</p>
              </div>
            )}
            {healthData.kept_n !== undefined && (
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <p className="text-3xl font-bold text-green-600">
                  {healthData.kept_n}
                </p>
                <p className="text-xs text-gray-600 mt-1">Kept</p>
              </div>
            )}
            {healthData.citations !== undefined && (
              <div className="bg-purple-50 rounded-lg p-4 text-center">
                <p className="text-3xl font-bold text-purple-600">
                  {healthData.citations}
                </p>
                <p className="text-xs text-gray-600 mt-1">Citations</p>
              </div>
            )}
            {healthData.latency_ms !== undefined && (
              <div className="bg-yellow-50 rounded-lg p-4 text-center">
                <p className="text-3xl font-bold text-yellow-600">
                  {healthData.latency_ms}
                </p>
                <p className="text-xs text-gray-600 mt-1">Latency (ms)</p>
              </div>
            )}
          </div>

          {/* Trace ID */}
          {healthData.trace_id && (
            <div className="bg-gray-50 rounded-lg p-3 mb-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Trace ID:</span>
                <code className="text-xs font-mono text-gray-800 bg-white px-2 py-1 rounded">
                  {healthData.trace_id}
                </code>
              </div>
            </div>
          )}

          {/* Status Indicator */}
          {healthData.status && (
            <div className="flex items-center space-x-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  healthData.status === 'healthy' ||
                  healthData.status === 'ok' ||
                  healthData.status === 'success'
                    ? 'bg-green-500'
                    : 'bg-yellow-500'
                }`}
              />
              <span className="text-sm text-gray-700">
                Status: <strong>{healthData.status}</strong>
              </span>
            </div>
          )}

          {/* Raw JSON */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-900">
              Show raw JSON
            </summary>
            <div className="mt-2 bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-green-400 text-xs font-mono">
                {JSON.stringify(healthData, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Health Check History
            </h2>
            <button
              onClick={clearHistory}
              className="text-sm text-red-600 hover:text-red-800"
            >
              Clear History
            </button>
          </div>

          <div className="space-y-2">
            {history.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <span className="text-xs text-gray-500 w-20">
                    {item.timestamp}
                  </span>
                  {item.retrieved_n !== undefined && (
                    <span className="text-xs">
                      Retrieved: <strong>{item.retrieved_n}</strong>
                    </span>
                  )}
                  {item.kept_n !== undefined && (
                    <span className="text-xs">
                      Kept: <strong>{item.kept_n}</strong>
                    </span>
                  )}
                  {item.latency_ms !== undefined && (
                    <span className="text-xs">
                      Latency: <strong>{item.latency_ms}ms</strong>
                    </span>
                  )}
                </div>
                {item.trace_id && (
                  <code className="text-xs font-mono text-gray-500">
                    {item.trace_id.slice(0, 8)}...
                  </code>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* System Information */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          System Information
        </h2>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">API Base URL:</span>
            <code className="text-gray-900 font-mono text-xs">
              {API_BASE_URL}
            </code>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Dashboard Version:</span>
            <span className="text-gray-900 font-medium">2.0.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Last Check:</span>
            <span className="text-gray-900">
              {healthData?.timestamp || 'Never'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
