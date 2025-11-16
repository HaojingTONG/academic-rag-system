import { useState, useEffect } from 'react'

const API_BASE_URL = import.meta.env.VITE_RAG_BASE_URL || 'http://localhost:8000'

interface RAGConfig {
  top_k: number
  use_hybrid_retrieval: boolean
  vector_weight: number
  bm25_weight: number
  enable_reranking: boolean
  enable_diversification: boolean
  enable_filtering: boolean
  llm_model: string
  llm_temperature: number
  max_tokens: number
  include_sources: boolean
  max_context_length: number
  timeout: number
}

export default function SettingsPanel() {
  const [config, setConfig] = useState<Partial<RAGConfig>>({})
  const [originalConfig, setOriginalConfig] = useState<Partial<RAGConfig>>({})
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  useEffect(() => {
    // Check if config has changed
    setHasChanges(JSON.stringify(config) !== JSON.stringify(originalConfig))
  }, [config, originalConfig])

  const loadConfig = async () => {
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE_URL}/config`)

      if (!res.ok) {
        throw new Error(`Failed to load config: ${res.status}`)
      }

      const data = await res.json()
      const loadedConfig = data.config || {}

      setConfig(loadedConfig)
      setOriginalConfig(loadedConfig)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
    } finally {
      setLoading(false)
    }
  }

  const saveConfig = async () => {
    setSaving(true)
    setError(null)
    setSuccess(null)

    try {
      // Only send changed fields
      const updates: Partial<RAGConfig> = {}
      Object.keys(config).forEach((key) => {
        if (config[key as keyof RAGConfig] !== originalConfig[key as keyof RAGConfig]) {
          updates[key as keyof RAGConfig] = config[key as keyof RAGConfig] as any
        }
      })

      const res = await fetch(`${API_BASE_URL}/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      })

      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.detail || `Failed to save config: ${res.status}`)
      }

      const data = await res.json()
      setSuccess(data.message || 'Configuration saved successfully')
      setOriginalConfig(config)

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration')
    } finally {
      setSaving(false)
    }
  }

  const resetConfig = async () => {
    if (!confirm('Are you sure you want to reset all settings to defaults?')) {
      return
    }

    setSaving(true)
    setError(null)
    setSuccess(null)

    try {
      const res = await fetch(`${API_BASE_URL}/config/reset`, {
        method: 'POST',
      })

      if (!res.ok) {
        throw new Error(`Failed to reset config: ${res.status}`)
      }

      const data = await res.json()
      const resetConfig = data.config || {}

      setConfig(resetConfig)
      setOriginalConfig(resetConfig)
      setSuccess('Configuration reset to defaults')

      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset configuration')
    } finally {
      setSaving(false)
    }
  }

  const updateConfig = (key: keyof RAGConfig, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <p className="text-gray-600 mt-4">Loading configuration...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">System Settings</h2>
        <p className="text-sm text-gray-500">
          Configure RAG system parameters. Changes apply immediately without restart.
        </p>
      </div>

      {/* Success Message */}
      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <span className="text-green-600 text-xl mr-3">✅</span>
            <p className="text-green-800 font-medium">{success}</p>
          </div>
        </div>
      )}

      {/* Error Message */}
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

      {/* Retrieval Settings */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-md font-semibold text-gray-900 mb-4">🔍 Retrieval Settings</h3>

        <div className="space-y-4">
          {/* Top K */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Top K (Number of documents)
            </label>
            <input
              type="number"
              value={config.top_k || 5}
              onChange={(e) => updateConfig('top_k', parseInt(e.target.value))}
              min={1}
              max={20}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">
              Number of documents to retrieve (1-20)
            </p>
          </div>

          {/* Vector Weight */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Vector Search Weight: {(config.vector_weight || 0.7).toFixed(2)}
            </label>
            <input
              type="range"
              value={config.vector_weight || 0.7}
              onChange={(e) => updateConfig('vector_weight', parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.05}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Weight for semantic vector search (0.0-1.0)
            </p>
          </div>

          {/* BM25 Weight */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              BM25 Keyword Weight: {(config.bm25_weight || 0.3).toFixed(2)}
            </label>
            <input
              type="range"
              value={config.bm25_weight || 0.3}
              onChange={(e) => updateConfig('bm25_weight', parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.05}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Weight for keyword-based search (0.0-1.0)
            </p>
          </div>

          {/* Enable Reranking */}
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="enable_reranking"
              checked={config.enable_reranking !== false}
              onChange={(e) => updateConfig('enable_reranking', e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="enable_reranking" className="text-sm font-medium text-gray-700">
              Enable Reranking (bge-reranker-large)
            </label>
          </div>

          {/* Enable Diversification */}
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="enable_diversification"
              checked={config.enable_diversification !== false}
              onChange={(e) => updateConfig('enable_diversification', e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="enable_diversification" className="text-sm font-medium text-gray-700">
              Enable MMR Diversification
            </label>
          </div>
        </div>
      </div>

      {/* Generation Settings */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-md font-semibold text-gray-900 mb-4">🤖 Generation Settings</h3>

        <div className="space-y-4">
          {/* LLM Model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              LLM Model
            </label>
            <input
              type="text"
              value={config.llm_model || 'gpt-4o-mini'}
              onChange={(e) => updateConfig('llm_model', e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g., gpt-4o-mini, llama3.1:8b"
            />
            <p className="text-xs text-gray-500 mt-1">
              OpenAI model name or Ollama model
            </p>
          </div>

          {/* Temperature */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Temperature: {(config.llm_temperature || 0.1).toFixed(2)}
            </label>
            <input
              type="range"
              value={config.llm_temperature || 0.1}
              onChange={(e) => updateConfig('llm_temperature', parseFloat(e.target.value))}
              min={0}
              max={2}
              step={0.1}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Higher = more creative, Lower = more focused (0.0-2.0)
            </p>
          </div>

          {/* Max Tokens */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Tokens
            </label>
            <input
              type="number"
              value={config.max_tokens || 2000}
              onChange={(e) => updateConfig('max_tokens', parseInt(e.target.value))}
              min={100}
              max={4000}
              step={100}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">
              Maximum length of generated answer (100-4000)
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between space-x-4">
          <button
            onClick={resetConfig}
            disabled={saving}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
          >
            🔄 Reset to Defaults
          </button>

          <div className="flex items-center space-x-3">
            {hasChanges && (
              <span className="text-sm text-amber-600 font-medium">
                ⚠️ Unsaved changes
              </span>
            )}
            <button
              onClick={loadConfig}
              disabled={saving}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={saveConfig}
              disabled={!hasChanges || saving}
              className={`px-6 py-3 rounded-lg font-medium text-white transition-colors ${
                hasChanges && !saving
                  ? 'bg-blue-600 hover:bg-blue-700'
                  : 'bg-gray-400 cursor-not-allowed'
              }`}
            >
              {saving ? (
                <span className="flex items-center">
                  <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
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
                  Saving...
                </span>
              ) : (
                '💾 Save Changes'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Info Card */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <span className="text-blue-600 text-xl mr-3">ℹ️</span>
          <div>
            <h3 className="text-blue-900 font-medium mb-1">Hot Reload</h3>
            <p className="text-blue-700 text-sm">
              Configuration changes apply immediately without restarting the backend.
              The current query session will use the new settings.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
