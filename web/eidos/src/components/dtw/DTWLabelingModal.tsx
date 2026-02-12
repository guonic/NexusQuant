/**
 * DTW Labeling Modal
 * 
 * Modal for annotating a single DTW hit:
 * - Display K-line chart with MA5, MA10
 * - Highlight matched interval
 * - Click to annotate platform start/end and breakout date
 * - Positive/negative label selection
 * - Save annotation back to CSV
 */

import React, { useState, useEffect, useRef } from 'react'
import { StockChart } from '../stock/StockChart'
import type { KlineData, IndicatorData, IndicatorConfig, BacktestOverlay } from '../stock/StockChart.types'
import { getDTWKline, saveDTWAnnotation, type DTWHitRow } from '@/services/api'

interface DTWLabelingModalProps {
  hit: DTWHitRow
  csvFile: string
  onClose: () => void
}

export default function DTWLabelingModal({ hit, csvFile, onClose }: DTWLabelingModalProps) {
  const [klineData, setKlineData] = useState<KlineData[]>([])
  const [indicatorData, setIndicatorData] = useState<IndicatorData>({})
  const [loading, setLoading] = useState(true)
  const [platformStart, setPlatformStart] = useState<string>(hit.platform_start || hit.start_date)
  const [platformEnd, setPlatformEnd] = useState<string>(hit.platform_end || hit.end_date)
  const [breakoutDate, setBreakoutDate] = useState<string>(hit.breakout_date || hit.end_date)
  const [label, setLabel] = useState<string>(hit.label || '')
  const [notes, setNotes] = useState<string>(hit.notes || '')
  const [saving, setSaving] = useState(false)
  const [annotationMode, setAnnotationMode] = useState<boolean>(false)
  const [clickCount, setClickCount] = useState(0)
  const platformStartRef = useRef<string>(hit.platform_start || hit.start_date)
  const platformEndRef = useRef<string>(hit.platform_end || hit.end_date)

  useEffect(() => {
    loadKlineData()
  }, [hit])

  const loadKlineData = async () => {
    try {
      setLoading(true)
      // Always extend 5 bars on each side
      const response = await getDTWKline(hit.ts_code, hit.start_date, hit.end_date, 5, 5)
      setKlineData(response.kline_data)
      setIndicatorData({
        ma5: response.indicators?.ma5 || [],
        ma10: response.indicators?.ma10 || [],
      })
    } catch (error) {
      console.error('Failed to load K-line data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleKlineClick = async (date: string) => {
    if (!annotationMode) return

    if (clickCount === 0) {
      // First click: platform start
      setPlatformStart(date)
      platformStartRef.current = date
      setClickCount(1)
    } else if (clickCount === 1) {
      // Second click: platform end
      setPlatformEnd(date)
      platformEndRef.current = date
      setClickCount(2)
    } else if (clickCount === 2) {
      // Third click: breakout date - save and close
      setBreakoutDate(date)
      setAnnotationMode(false)
      setClickCount(0)
      
      // Save as positive sample using ref values (always up-to-date)
      try {
        setSaving(true)
        const rowIndex = hit.start_index ?? 0
        await saveDTWAnnotation({
          csv_file: csvFile,
          row_index: rowIndex,
          label: 'positive',
          platform_start: platformStartRef.current,
          platform_end: platformEndRef.current,
          breakout_date: date,
          notes: notes || undefined,
        })
        onClose()
      } catch (error) {
        console.error('Failed to save annotation:', error)
        alert('保存失败，请重试')
      } finally {
        setSaving(false)
      }
    }
  }

  const handleStartAnnotation = () => {
    setAnnotationMode(true)
    setClickCount(0)
    // Reset to default values
    const defaultStart = hit.start_date
    const defaultEnd = hit.end_date
    setPlatformStart(defaultStart)
    setPlatformEnd(defaultEnd)
    setBreakoutDate(defaultEnd)
    platformStartRef.current = defaultStart
    platformEndRef.current = defaultEnd
  }


  const handleClose = async () => {
    // Auto save as negative sample when closing
    try {
      setSaving(true)
      const rowIndex = hit.start_index ?? 0
      await saveDTWAnnotation({
        csv_file: csvFile,
        row_index: rowIndex,
        label: 'negative',
        platform_start: platformStart,
        platform_end: platformEnd,
        breakout_date: breakoutDate,
        notes: notes || undefined,
      })
      onClose()
    } catch (error) {
      console.error('Failed to save annotation:', error)
      alert('保存失败，请重试')
    } finally {
      setSaving(false)
    }
  }

  const handleSave = async () => {
    if (!label) {
      alert('请选择标签（正样本/负样本）')
      return
    }

    try {
      setSaving(true)
      // Use the row index from hit (set by backend)
      const rowIndex = hit.start_index ?? 0
      await saveDTWAnnotation({
        csv_file: csvFile,
        row_index: rowIndex,
        label,
        platform_start: platformStart,
        platform_end: platformEnd,
        breakout_date: breakoutDate,
        notes: notes || undefined,
      })
      onClose()
    } catch (error) {
      console.error('Failed to save annotation:', error)
      alert('保存失败，请重试')
    } finally {
      setSaving(false)
    }
  }

  const indicators: IndicatorConfig = {
    ma5: true,
    ma10: true,
    ma20: false,
    ma30: false,
    ma60: false,
    ma120: false,
    ema: false,
    wma: false,
    rsi: false,
    kdj: false,
    cci: false,
    wr: false,
    obv: false,
    macd: false,
    dmi: false,
    bollinger: false,
    envelope: false,
    atr: false,
    bbw: false,
    vol: false,
    vwap: false,
  }

  // Create overlay for highlighting matched interval
  const backtestOverlay: BacktestOverlay = {
    backtestStart: hit.start_date,
    backtestEnd: hit.end_date,
  }

  // Get available dates for dropdowns
  const availableDates = klineData.map((k) => k.date).filter(Boolean)

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-eidos-surface rounded-lg p-8">
          <div className="text-eidos-muted">加载 K 线数据中...</div>
        </div>
      </div>
    )
  }

  const handleBackgroundClick = () => {
    if (!annotationMode) {
      handleClose()
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={handleBackgroundClick}>
      <div
        className="bg-eidos-surface rounded-lg w-[95vw] h-[90vh] flex flex-col border border-eidos-muted/20"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-4 border-b border-eidos-muted/20 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">标注样本</h2>
            <p className="text-sm text-eidos-muted mt-1">
              {hit.ts_code} | {hit.start_date} ~ {hit.end_date} | 分数: {hit.score.toFixed(4)}
            </p>
          </div>
          <button
            onClick={handleClose}
            disabled={saving}
            className="text-eidos-muted hover:text-white transition-colors disabled:opacity-50"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Chart */}
          <div className="flex-1 min-h-0 relative">
            <StockChart
              symbol={hit.ts_code}
              klineData={klineData}
              indicatorData={indicatorData}
              indicators={indicators}
              timeInterval="1d"
              embedded={true}
              className="h-full"
              backtestOverlay={backtestOverlay}
              clickable={annotationMode}
              onKlineClick={handleKlineClick}
              initialVisibleWindow={{ start: 0, end: 100 }} // Show all data by default
            />
            {annotationMode && (
              <div className="absolute top-4 left-4 bg-eidos-accent/90 text-white px-4 py-2 rounded shadow-lg z-10">
                {clickCount === 0 && '第1步：请点击 K 线选择平台开始日期'}
                {clickCount === 1 && '第2步：请点击 K 线选择平台结束日期'}
                {clickCount === 2 && '第3步：请点击 K 线选择突破日期（完成后自动保存为正样本）'}
              </div>
            )}
          </div>

          {/* Annotation Panel */}
          <div className="p-4 border-t border-eidos-muted/20 bg-eidos-surface/50">
            <div className="grid grid-cols-2 gap-4">
              {/* Left Column: Date Selection */}
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-semibold text-eidos-muted mb-2">
                    日期标注（匹配区间: {hit.start_date} ~ {hit.end_date}）
                  </label>
                  <div className="space-y-2">
                    <div>
                      <label className="block text-xs text-eidos-muted mb-1">平台开始</label>
                      <select
                        value={platformStart}
                        onChange={(e) => setPlatformStart(e.target.value)}
                        className="w-full px-3 py-1 bg-eidos-muted/20 border border-eidos-muted/30 rounded text-white text-sm"
                      >
                        {availableDates.map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-eidos-muted mb-1">平台结束</label>
                      <select
                        value={platformEnd}
                        onChange={(e) => setPlatformEnd(e.target.value)}
                        className="w-full px-3 py-1 bg-eidos-muted/20 border border-eidos-muted/30 rounded text-white text-sm"
                      >
                        {availableDates.map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-eidos-muted mb-1">突破日期</label>
                      <select
                        value={breakoutDate}
                        onChange={(e) => setBreakoutDate(e.target.value)}
                        className="w-full px-3 py-1 bg-eidos-muted/20 border border-eidos-muted/30 rounded text-white text-sm"
                      >
                        {availableDates.map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Label & Notes */}
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-semibold text-eidos-muted mb-2">标签</label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setLabel('positive')}
                      className={`px-4 py-2 rounded text-sm font-semibold ${
                        label === 'positive'
                          ? 'bg-green-500 text-white'
                          : 'bg-eidos-muted/20 text-white hover:bg-eidos-muted/30'
                      }`}
                    >
                      正样本
                    </button>
                    <button
                      onClick={() => setLabel('negative')}
                      className={`px-4 py-2 rounded text-sm font-semibold ${
                        label === 'negative'
                          ? 'bg-red-500 text-white'
                          : 'bg-eidos-muted/20 text-white hover:bg-eidos-muted/30'
                      }`}
                    >
                      负样本
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-eidos-muted mb-2">备注</label>
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    className="w-full px-3 py-2 bg-eidos-muted/20 border border-eidos-muted/30 rounded text-white text-sm resize-none"
                    rows={3}
                    placeholder="可选备注..."
                  />
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="mt-4 flex justify-between items-center">
              <button
                onClick={handleStartAnnotation}
                disabled={saving || annotationMode}
                className={`px-4 py-2 rounded text-sm font-semibold ${
                  annotationMode
                    ? 'bg-eidos-accent text-white'
                    : 'bg-green-500 text-white hover:bg-green-600'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {annotationMode ? '标注中...（点击3次K线）' : '开始标注（点击3次K线）'}
              </button>
              <div className="flex gap-2">
                <button
                  onClick={handleClose}
                  disabled={saving || annotationMode}
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {saving ? '保存中...' : '标注为负样本并关闭'}
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving || !label || annotationMode}
                  className="px-4 py-2 bg-eidos-accent text-white rounded hover:bg-eidos-accent/80 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {saving ? '保存中...' : '手动保存'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
