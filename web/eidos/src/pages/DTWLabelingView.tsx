/**
 * DTW Labeling View
 * 
 * Main page for labeling DTW hits from CSV files.
 * Features:
 * - Paginated table of DTW hits
 * - Click row to open annotation modal
 * - K-line chart with MA5, MA10
 * - Highlight matched interval
 * - Click to annotate platform/breakout dates
 * - Positive/negative label
 * - Save annotation back to CSV
 */

import React, { useState, useEffect, useRef } from 'react'
import { getDTWHits, type DTWHitRow } from '@/services/api'
import DTWLabelingModal from '@/components/dtw/DTWLabelingModal'

interface DTWLabelingViewProps {
  csvFile: string
}

export default function DTWLabelingView({ csvFile }: DTWLabelingViewProps) {
  const [hits, setHits] = useState<DTWHitRow[]>([])
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [total, setTotal] = useState(0)
  const [totalPages, setTotalPages] = useState(0)
  const [loading, setLoading] = useState(false)
  const [selectedHit, setSelectedHit] = useState<DTWHitRow | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [targetRowIndex, setTargetRowIndex] = useState<number | null>(null)
  const tableBodyRef = useRef<HTMLTableSectionElement>(null)

  useEffect(() => {
    loadHits()
  }, [csvFile, page, pageSize])

  useEffect(() => {
    // Scroll to target row after data loads
    if (targetRowIndex !== null && !loading && hits.length > 0) {
      // Use setTimeout to ensure DOM is fully rendered
      setTimeout(() => {
        if (tableBodyRef.current) {
          const rows = tableBodyRef.current.querySelectorAll('tr')
          if (rows[targetRowIndex]) {
            // Scroll to make the target row appear at position 5 (index 4)
            const targetRow = rows[targetRowIndex] as HTMLTableRowElement
            const tableContainer = tableBodyRef.current.closest('.flex-1.overflow-auto')
            if (tableContainer) {
              const rowHeight = targetRow.offsetHeight || 40 // Default row height
              const targetScrollTop = targetRow.offsetTop - rowHeight * 4 // Position at 5th row (4 rows above)
              tableContainer.scrollTo({
                top: Math.max(0, targetScrollTop),
                behavior: 'smooth',
              })
            }
            setTargetRowIndex(null)
          }
        }
      }, 50)
    }
  }, [hits, targetRowIndex, loading])

  const loadHits = async () => {
    try {
      setLoading(true)
      const response = await getDTWHits(csvFile, page, pageSize)
      setHits(response.hits)
      setTotal(response.total)
      setTotalPages(response.total_pages)
    } catch (error) {
      console.error('Failed to load DTW hits:', error)
    } finally {
      setLoading(false)
    }
  }

  const findNextUnlabeledRow = async (startPage: number, currentTotalPages: number): Promise<{ page: number; rowIndex: number } | null> => {
    // Search from current page onwards
    for (let p = startPage; p <= currentTotalPages; p++) {
      const response = await getDTWHits(csvFile, p, pageSize)
      const unlabeledIndex = response.hits.findIndex((h) => !h.label || h.label.trim() === '')
      if (unlabeledIndex >= 0) {
        return { page: p, rowIndex: unlabeledIndex }
      }
    }
    // If not found, search from page 1 to startPage - 1
    for (let p = 1; p < startPage; p++) {
      const response = await getDTWHits(csvFile, p, pageSize)
      const unlabeledIndex = response.hits.findIndex((h) => !h.label || h.label.trim() === '')
      if (unlabeledIndex >= 0) {
        return { page: p, rowIndex: unlabeledIndex }
      }
    }
    return null
  }

  const handleRowClick = (hit: DTWHitRow) => {
    setSelectedHit(hit)
    setIsModalOpen(true)
  }

  const handleModalClose = async () => {
    setIsModalOpen(false)
    const currentPage = page
    setSelectedHit(null)
    
    // Reload hits to refresh labels and get updated totalPages
    const response = await getDTWHits(csvFile, page, pageSize)
    setHits(response.hits)
    setTotal(response.total)
    setTotalPages(response.total_pages)
    
    // Find next unlabeled row (use updated totalPages)
    const nextUnlabeled = await findNextUnlabeledRow(currentPage, response.total_pages)
    if (nextUnlabeled) {
      // Switch to the page containing the next unlabeled row
      if (nextUnlabeled.page !== currentPage) {
        setPage(nextUnlabeled.page)
        // Wait for page to load, then set target row index
        setTimeout(() => {
          setTargetRowIndex(nextUnlabeled.rowIndex)
        }, 200)
      } else {
        // Same page, just scroll to the row
        setTimeout(() => {
          setTargetRowIndex(nextUnlabeled.rowIndex)
        }, 100)
      }
    }
  }

  const getLabelBadge = (label: string | null | undefined) => {
    if (!label) return null
    const isPositive = label.toLowerCase() === 'positive'
    return (
      <span
        className={`px-2 py-1 rounded text-xs font-semibold ${
          isPositive
            ? 'bg-green-500/20 text-green-400'
            : 'bg-red-500/20 text-red-400'
        }`}
      >
        {label}
      </span>
    )
  }

  return (
    <div className="flex flex-col h-full bg-eidos-surface">
      {/* Header */}
      <div className="p-4 border-b border-eidos-muted/20">
        <h1 className="text-xl font-bold text-white">DTW 标注工具</h1>
        <p className="text-sm text-eidos-muted mt-1">文件: {csvFile}</p>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-eidos-muted">加载中...</div>
          </div>
        ) : (
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-eidos-surface/95 backdrop-blur-sm border-b border-eidos-muted/20">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  模板
                </th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  股票代码
                </th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  开始日期
                </th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  结束日期
                </th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  分数
                </th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  命中数
                </th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-eidos-muted uppercase">
                  标签
                </th>
              </tr>
            </thead>
            <tbody ref={tableBodyRef}>
              {hits.map((hit, idx) => {
                return (
                  <tr
                    key={idx}
                    onClick={() => handleRowClick(hit)}
                    className="border-b border-eidos-muted/10 hover:bg-eidos-muted/10 cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-2 text-sm text-white">{hit.template}</td>
                    <td className="px-4 py-2 text-sm text-white">{hit.ts_code}</td>
                    <td className="px-4 py-2 text-sm text-white">{hit.start_date}</td>
                    <td className="px-4 py-2 text-sm text-white">{hit.end_date}</td>
                    <td className="px-4 py-2 text-sm text-eidos-accent">{hit.score.toFixed(4)}</td>
                    <td className="px-4 py-2 text-sm text-white">{hit.hit_count || 1}</td>
                    <td className="px-4 py-2 text-sm">{getLabelBadge(hit.label)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      <div className="p-4 border-t border-eidos-muted/20 flex items-center justify-between">
        <div className="text-sm text-eidos-muted">
          共 {total} 条，第 {page} / {totalPages} 页
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1 bg-eidos-muted/20 text-white rounded hover:bg-eidos-muted/30 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            上一页
          </button>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-3 py-1 bg-eidos-muted/20 text-white rounded hover:bg-eidos-muted/30 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            下一页
          </button>
        </div>
        <select
          value={pageSize}
          onChange={(e) => {
            setPageSize(Number(e.target.value))
            setPage(1)
          }}
          className="px-2 py-1 bg-eidos-muted/20 text-white rounded border border-eidos-muted/30"
        >
          <option value={25}>25 / 页</option>
          <option value={50}>50 / 页</option>
          <option value={100}>100 / 页</option>
          <option value={200}>200 / 页</option>
          <option value={500}>500 / 页</option>
        </select>
      </div>

      {/* Modal */}
      {isModalOpen && selectedHit && (
        <DTWLabelingModal
          hit={selectedHit}
          csvFile={csvFile}
          onClose={handleModalClose}
        />
      )}
    </div>
  )
}
