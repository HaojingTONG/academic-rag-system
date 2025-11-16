import { useState } from 'react'
import AskPanel from './components/AskPanel'
import IngestPanel from './components/IngestPanel'
import DebugPanel from './components/DebugPanel'
import LibraryPanel from './components/LibraryPanel'
import SettingsPanel from './components/SettingsPanel'

type Tab = 'ask' | 'library' | 'settings' | 'ingest' | 'debug'

export default function RAGDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('ask')

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'ask', label: 'Ask', icon: '🔍' },
    { id: 'library', label: 'Library', icon: '📚' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
    { id: 'ingest', label: 'Ingest', icon: '📥' },
    { id: 'debug', label: 'Debug', icon: '🔧' },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="text-3xl">📚</div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Academic RAG System
                </h1>
                <p className="text-sm text-gray-500">
                  Research Assistant Dashboard
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8" aria-label="Tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2
                  transition-colors duration-200
                  ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <span className="text-xl">{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'ask' && <AskPanel />}
        {activeTab === 'library' && <LibraryPanel />}
        {activeTab === 'settings' && <SettingsPanel />}
        {activeTab === 'ingest' && <IngestPanel />}
        {activeTab === 'debug' && <DebugPanel />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-center text-sm text-gray-500">
            Academic RAG System v2.0 | Built with React + Tailwind CSS
          </p>
        </div>
      </footer>
    </div>
  )
}
