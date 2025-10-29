import { useEffect, useState } from 'react'

interface Step {
  id: string
  label: string
  status: 'pending' | 'active' | 'completed' | 'error'
  duration?: number
}

interface ProcessingStepsProps {
  isProcessing: boolean
  currentStep?: string
  error?: string | null
}

export default function ProcessingSteps({ isProcessing, currentStep, error }: ProcessingStepsProps) {
  const [steps, setSteps] = useState<Step[]>([
    { id: 'connecting', label: '连接到后端服务器', status: 'pending' },
    { id: 'retrieving', label: '从向量数据库检索相关文档', status: 'pending' },
    { id: 'reranking', label: '对检索结果进行重排序', status: 'pending' },
    { id: 'generating', label: '使用 LLM 生成答案', status: 'pending' },
    { id: 'formatting', label: '格式化响应结果', status: 'pending' },
  ])

  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (isProcessing) {
      setStartTime(Date.now())

      // Simulate step progression
      const timers: NodeJS.Timeout[] = []

      // Step 1: Connecting (immediate)
      timers.push(setTimeout(() => updateStep('connecting', 'active'), 0))
      timers.push(setTimeout(() => updateStep('connecting', 'completed'), 300))

      // Step 2: Retrieving
      timers.push(setTimeout(() => updateStep('retrieving', 'active'), 300))
      timers.push(setTimeout(() => updateStep('retrieving', 'completed'), 1500))

      // Step 3: Reranking
      timers.push(setTimeout(() => updateStep('reranking', 'active'), 1500))
      timers.push(setTimeout(() => updateStep('reranking', 'completed'), 2500))

      // Step 4: Generating
      timers.push(setTimeout(() => updateStep('generating', 'active'), 2500))
      // This will complete when actual response arrives

      return () => timers.forEach(timer => clearTimeout(timer))
    } else {
      // Reset when not processing
      setSteps(prev => prev.map(step => ({ ...step, status: 'pending' })))
      setStartTime(null)
      setElapsed(0)
    }
  }, [isProcessing])

  useEffect(() => {
    if (startTime && isProcessing) {
      const interval = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTime) / 1000))
      }, 100)

      return () => clearInterval(interval)
    }
  }, [startTime, isProcessing])

  useEffect(() => {
    if (error) {
      // Mark current active step as error
      setSteps(prev => prev.map(step =>
        step.status === 'active' ? { ...step, status: 'error' } : step
      ))
    }
  }, [error])

  const updateStep = (id: string, status: Step['status']) => {
    setSteps(prev => prev.map(step =>
      step.id === id ? { ...step, status } : step
    ))
  }

  const getStepIcon = (status: Step['status']) => {
    switch (status) {
      case 'completed':
        return '✓'
      case 'active':
        return '⟳'
      case 'error':
        return '✗'
      default:
        return '○'
    }
  }

  const getStepColor = (status: Step['status']) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'active':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-gray-400 bg-gray-50 border-gray-200'
    }
  }

  if (!isProcessing) return null

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          处理进度
        </h3>
        <div className="flex items-center space-x-2">
          <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-600">
            {elapsed}s
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={`flex items-start space-x-3 p-3 rounded-lg border transition-all duration-300 ${getStepColor(step.status)}`}
          >
            <div className="flex-shrink-0 mt-0.5">
              <span className={`inline-flex items-center justify-center h-6 w-6 rounded-full font-bold text-sm ${
                step.status === 'active' ? 'animate-spin' : ''
              }`}>
                {getStepIcon(step.status)}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <p className={`text-sm font-medium ${
                  step.status === 'pending' ? 'text-gray-500' : ''
                }`}>
                  {index + 1}. {step.label}
                </p>
                {step.status === 'active' && (
                  <div className="flex space-x-1">
                    <div className="h-1.5 w-1.5 bg-blue-600 rounded-full animate-bounce"></div>
                    <div className="h-1.5 w-1.5 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="h-1.5 w-1.5 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                )}
              </div>
              {step.status === 'error' && error && (
                <p className="text-xs text-red-600 mt-1">
                  {error}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      {elapsed > 10 && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <span className="text-yellow-600 text-lg">⚠️</span>
            <div>
              <p className="text-sm font-medium text-yellow-800">
                处理时间较长
              </p>
              <p className="text-xs text-yellow-700 mt-1">
                这可能是因为：
              </p>
              <ul className="text-xs text-yellow-700 mt-1 ml-4 list-disc space-y-1">
                <li>后端服务器未运行或响应缓慢</li>
                <li>首次查询需要加载模型（可能需要 1-2 分钟）</li>
                <li>查询的文档数量较多</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {elapsed > 30 && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <span className="text-red-600 text-lg">⚠️</span>
            <div>
              <p className="text-sm font-medium text-red-800">
                超时警告
              </p>
              <p className="text-xs text-red-700 mt-1">
                请检查后端服务器是否正常运行：
              </p>
              <code className="text-xs bg-gray-900 text-green-400 px-2 py-1 rounded mt-2 block">
                python app/main.py
              </code>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
