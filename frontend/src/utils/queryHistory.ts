// Query history management using localStorage

export interface QueryHistoryItem {
  id: string
  question: string
  timestamp: number
  answer?: string
  num_sources?: number
}

const HISTORY_KEY = 'rag_query_history'
const MAX_HISTORY_ITEMS = 50

export function getQueryHistory(): QueryHistoryItem[] {
  try {
    const stored = localStorage.getItem(HISTORY_KEY)
    if (!stored) return []
    return JSON.parse(stored)
  } catch {
    return []
  }
}

export function addToHistory(item: Omit<QueryHistoryItem, 'id' | 'timestamp'>): void {
  try {
    const history = getQueryHistory()

    const newItem: QueryHistoryItem = {
      ...item,
      id: crypto.randomUUID(),
      timestamp: Date.now(),
    }

    // Add to beginning
    history.unshift(newItem)

    // Keep only last N items
    const trimmed = history.slice(0, MAX_HISTORY_ITEMS)

    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed))
  } catch (error) {
    console.error('Failed to save query history:', error)
  }
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(HISTORY_KEY)
  } catch (error) {
    console.error('Failed to clear history:', error)
  }
}

export function getRecentQueries(limit: number = 5): string[] {
  const history = getQueryHistory()
  return history
    .slice(0, limit)
    .map(item => item.question)
}
