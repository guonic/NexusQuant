/**
 * DTW Labeling Entry Page
 * 
 * Entry point for DTW labeling tool.
 * User can input CSV filename or select from list.
 */

import React, { useState } from 'react'
import DTWLabelingView from './DTWLabelingView'

export default function DTWLabelingEntry() {
  const [filename, setFilename] = useState('')
  const [showView, setShowView] = useState(false)

  const handleStart = () => {
    if (!filename.trim()) {
      alert('请输入 CSV 文件名')
      return
    }
    setShowView(true)
  }

  if (showView) {
    return <DTWLabelingView csvFile={filename} />
  }

  return (
    <div className="flex flex-col h-screen bg-eidos-surface">
      <div className="p-8 max-w-2xl mx-auto w-full">
        <h1 className="text-2xl font-bold text-white mb-4">DTW 标注工具</h1>
        <p className="text-eidos-muted mb-6">
          请输入 DTW 命中结果 CSV 文件名（例如: dtw_hits_600487.SH_2024-01-01_2024-12-31.csv）
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            placeholder="dtw_hits_600487.SH_2024-01-01_2024-12-31.csv"
            className="flex-1 px-4 py-2 bg-eidos-muted/20 border border-eidos-muted/30 rounded text-white placeholder-eidos-muted"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleStart()
              }
            }}
          />
          <button
            onClick={handleStart}
            className="px-6 py-2 bg-eidos-accent text-white rounded hover:bg-eidos-accent/80"
          >
            开始标注
          </button>
        </div>
        <p className="text-xs text-eidos-muted mt-4">
          提示: CSV 文件应位于 outputs/ 目录下，或使用绝对路径
        </p>
      </div>
    </div>
  )
}
