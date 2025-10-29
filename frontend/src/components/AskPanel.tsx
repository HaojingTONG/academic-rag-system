import { useState, useEffect } from 'react'
import ProcessingSteps from './ProcessingSteps'

const API_BASE_URL = import.meta.env.VITE_RAG_BASE_URL || 'http://localhost:8000'

interface QueryResponse {
  answer: string
  sources: Array<{
    index: number
    content: string
    title?: string
    score: number
    metadata: Record<string, any>
  }>
  num_sources: number
  success: boolean
  metadata?: Record<string, any>
}

export default function AskPanel() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showRawJSON, setShowRawJSON] = useState(false)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking')

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection()
  }, [])

  const checkBackendConnection = async () => {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000)

      const res = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (res.ok) {
        setBackendStatus('online')
      } else {
        setBackendStatus('offline')
      }
    } catch (err) {
      setBackendStatus('offline')
    }
  }

  const handleAsk = async () => {
    if (!query.trim()) {
      setError('请输入问题')
      return
    }

    // Check backend status first
    if (backendStatus === 'offline') {
      setError('后端服务器未连接，请先启动后端服务')
      return
    }

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      // Add timeout (60 seconds)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 60000)

      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: query,
          top_k: topK,
        }),
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(`HTTP ${res.status}: ${errorText || res.statusText}`)
      }

      const data = await res.json()
      setResponse(data)
      setBackendStatus('online')
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('请求超时（60秒）- 后端可能正在加载模型或处理复杂查询，请稍后再试')
        } else if (err.message.includes('Failed to fetch')) {
          setError('无法连接到后端服务器 - 请确保后端正在运行（python app/main.py）')
          setBackendStatus('offline')
        } else {
          setError(err.message)
        }
      } else {
        setError('发生未知错误')
      }
    } finally {
      setLoading(false)
    }
  }

  const renderAnswer = (text: string) => {
    // Simple citation rendering - replace [N] with superscript
    const regex = /\[(\d+)\]/g
    const rendered = text.replace(
      regex,
      '<sup class="text-blue-600 font-bold">[$1]</sup>'
    )
    return rendered
  }

  return (
    <div className="space-y-6">
      {/* Backend Status Banner */}
      {backendStatus === 'offline' && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <span className="text-2xl">⚠️</span>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-red-800">
                后端服务器未连接
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>请在项目根目录运行以下命令启动后端：</p>
                <code className="block mt-2 bg-gray-900 text-green-400 px-3 py-2 rounded text-xs">
                  python app/main.py
                </code>
                <p className="mt-2">或者使用：</p>
                <code className="block mt-1 bg-gray-900 text-green-400 px-3 py-2 rounded text-xs">
                  make serve
                </code>
              </div>
              <div className="mt-3">
                <button
                  onClick={checkBackendConnection}
                  className="text-sm font-medium text-red-800 hover:text-red-900 underline"
                >
                  重新检查连接
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {backendStatus === 'online' && (
        <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded-r-lg">
          <div className="flex items-center">
            <span className="text-xl mr-2">✓</span>
            <p className="text-sm font-medium text-green-800">
              后端服务器已连接 - {API_BASE_URL}
            </p>
          </div>
        </div>
      )}

      {/* Query Input Card */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Ask a Question
        </h2>

        <div className="space-y-4">
          {/* Query Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Question
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What is the Transformer architecture?"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
              disabled={loading}
            />
          </div>

          {/* Parameters */}
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Top K (Number of sources)
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                min={1}
                max={20}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loading}
              />
            </div>

            <div className="flex items-center space-x-2 pt-7">
              <input
                type="checkbox"
                id="showRaw"
                checked={showRawJSON}
                onChange={(e) => setShowRawJSON(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="showRaw" className="text-sm text-gray-700">
                Show raw JSON
              </label>
            </div>
          </div>

          {/* Ask Button */}
          <button
            onClick={handleAsk}
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
                Processing...
              </span>
            ) : (
              '🔍 Ask'
            )}
          </button>
        </div>
      </div>

      {/* Processing Steps Visualization */}
      {loading && <ProcessingSteps isProcessing={loading} error={error} />}

      {/* Error Display */}
      {error && !loading && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <span className="text-red-600 text-xl mr-3">❌</span>
            <div className="flex-1">
              <h3 className="text-red-800 font-medium">查询失败</h3>
              <p className="text-red-600 text-sm mt-1">{error}</p>

              {backendStatus === 'offline' && (
                <div className="mt-3 space-y-2">
                  <p className="text-sm text-red-700 font-medium">快速解决方案：</p>
                  <ol className="text-sm text-red-700 list-decimal ml-5 space-y-1">
                    <li>打开新终端窗口</li>
                    <li>进入项目目录：<code className="bg-red-100 px-1 rounded">cd ~/Desktop/academic-rag-system</code></li>
                    <li>运行：<code className="bg-red-100 px-1 rounded">python app/main.py</code></li>
                    <li>等待看到 "✅ RAG System API ready!"</li>
                    <li>点击上方的 "重新检查连接" 按钮</li>
                  </ol>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div className="space-y-4">
          {/* Answer Card */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Answer</h2>
              {response.metadata && (
                <span className="text-xs text-gray-500">
                  {response.metadata.total_time && `${response.metadata.total_time.toFixed(2)}s`}
                </span>
              )}
            </div>
            <div
              className="prose prose-sm max-w-none text-gray-700 leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: renderAnswer(response.answer),
              }}
            />
          </div>

          {/* Retrieved Sources */}
          {response.sources && response.sources.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Retrieved Sources ({response.sources.length})
              </h2>
              <div className="space-y-3">
                {response.sources.map((source, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                          [{source.index}]
                        </span>
                        {source.title && (
                          <span className="text-xs font-medium text-gray-700">
                            {source.title}
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-gray-500">
                        Score: {source.score.toFixed(4)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 line-clamp-3">
                      {source.content}
                    </p>
                    {source.metadata && Object.keys(source.metadata).length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {Object.entries(source.metadata).map(([key, value]) => (
                          <span
                            key={key}
                            className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded"
                          >
                            {key}: {String(value)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Raw JSON */}
          {showRawJSON && (
            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-green-400 text-xs font-mono">
                {JSON.stringify(response, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
