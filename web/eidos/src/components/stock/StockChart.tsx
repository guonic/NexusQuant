/**
 * Standard Stock Chart Component
 * 
 * A reusable stock analysis panel that supports:
 * - K-line chart with volume
 * - Multiple technical indicators
 * - Time interval selection (1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M)
 * - Panel resize (minimize/maximize)
 * - Window dragging/scrolling
 */

import React, { useState, useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { StockChartProps, IndicatorConfig, IndicatorData, TimeInterval, KlineData } from './StockChart.types'
import { DEFAULT_INDICATORS, INDICATOR_CATEGORIES, INDICATOR_LABELS } from './StockChart.config'

export function StockChart({
  symbol,
  klineData,
  indicatorData = {},
  indicators: externalIndicators,
  onIndicatorsChange,
  timeInterval = '1d',
  onTimeIntervalChange,
  embedded = false,
  className = '',
  style,
  backtestOverlay,
  onKlineClick,
  clickable = false,
  initialVisibleWindow,
}: StockChartProps) {
  // Internal state
  const [indicators, setIndicators] = useState<IndicatorConfig>(
    externalIndicators || DEFAULT_INDICATORS
  )
  const [isMinimized, setIsMinimized] = useState(false)
  const defaultWindow = initialVisibleWindow || { start: 70, end: 100 }
  const [visibleWindow, setVisibleWindow] = useState<{ start: number; end: number }>(defaultWindow)
  
  // Refs
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<echarts.ECharts | null>(null)
  const resizeHandlerRef = useRef<(() => void) | null>(null)
  const dataZoomHandlerRef = useRef<((params: any) => void) | null>(null)
  const visibleWindowRef = useRef<{ start: number; end: number }>(defaultWindow) // 保存窗口范围，避免指标变化时丢失
  const containerClickHandlerRef = useRef<((event: MouseEvent) => void) | null>(null)
  const echartsClickHandledRef = useRef<boolean>(false)
  const lastClickTimeRef = useRef<number>(0)
  
  // Update indicators when external prop changes
  useEffect(() => {
    if (externalIndicators) {
      setIndicators(externalIndicators)
    }
  }, [externalIndicators])
  
  // Notify parent when indicators change
  const handleIndicatorToggle = (key: keyof IndicatorConfig) => {
    const newIndicators = { ...indicators, [key]: !indicators[key] }
    setIndicators(newIndicators)
    onIndicatorsChange?.(newIndicators)
  }
  
  // Handle dataZoom change (from ECharts slider or inside zoom)
  const handleDataZoom = (params: any) => {
    // 忽略来自 setOption 的触发（通过检查是否与当前 ref 值相同）
    let newStart: number | undefined
    let newEnd: number | undefined
    
    if (params.batch) {
      const zoom = params.batch[0]
      if (zoom && zoom.start !== undefined && zoom.end !== undefined) {
        newStart = zoom.start
        newEnd = zoom.end
      }
    } else if (params.start !== undefined && params.end !== undefined) {
      newStart = params.start
      newEnd = params.end
    }
    
    // 只有当值真正改变时才更新（避免 setOption 触发的无效更新）
    if (newStart !== undefined && newEnd !== undefined) {
      const threshold = 0.1 // 允许的误差范围
      const startDiff = Math.abs(visibleWindowRef.current.start - newStart)
      const endDiff = Math.abs(visibleWindowRef.current.end - newEnd)
      
      // 只有当变化超过阈值时才更新（说明是用户操作，而不是 setOption 的重置）
      if (startDiff > threshold || endDiff > threshold) {
        const newWindow = { start: newStart, end: newEnd }
        visibleWindowRef.current = newWindow
        setVisibleWindow(newWindow)
      }
    }
  }
  
  // Sync visibleWindowRef when visibleWindow changes from external source
  // But don't sync when indicators change - keep the current window range
  useEffect(() => {
    // Only sync if the change is meaningful (not just a re-render)
    // This ensures that when indicators change, we keep the current window range
    if (Math.abs(visibleWindowRef.current.start - visibleWindow.start) > 0.1 ||
        Math.abs(visibleWindowRef.current.end - visibleWindow.end) > 0.1) {
      visibleWindowRef.current = visibleWindow
    }
  }, [visibleWindow])
  
  // Render chart
  useEffect(() => {
    if (!chartRef.current || klineData.length === 0 || isMinimized) return
    
    // Cleanup
    if (resizeHandlerRef.current) {
      window.removeEventListener('resize', resizeHandlerRef.current)
      resizeHandlerRef.current = null
    }
    
    if (chartInstanceRef.current) {
      try {
        if (!chartInstanceRef.current.isDisposed()) {
          chartInstanceRef.current.dispose()
        }
      } catch (e) {
        // Ignore
      }
      chartInstanceRef.current = null
    }
    
    // Create chart
    let chart: echarts.ECharts
    try {
      chart = echarts.init(chartRef.current, 'dark')
      chartInstanceRef.current = chart
    } catch (error) {
      console.error('Error initializing ECharts:', error)
      return
    }
    
    // Prepare data
    const dates = klineData.map((d) => d.date)
    const klineValues = klineData.map((d) => [d.open, d.close, d.low, d.high])
    const volumes = klineData.map((d) => d.volume)
    
    // Calculate date display interval based on visible data density
    const calculateDateInterval = () => {
      const totalDays = dates.length
      const visibleDays = Math.ceil((visibleWindowRef.current.end - visibleWindowRef.current.start) / 100 * totalDays)
      
      if (visibleDays <= 30) {
        return 1 // 显示每天
      } else if (visibleDays <= 90) {
        return 5 // 显示每周（约每5天）
      } else if (visibleDays <= 180) {
        return 10 // 显示每两周（约每10天）
      } else if (visibleDays <= 365) {
        return 20 // 显示每月（约每20天）
      } else {
        return 60 // 显示每季度（约每60天）
      }
    }
    
    const dateInterval = calculateDateInterval()
    
    // Format date label based on visible range
    const formatDateLabel = (value: string, index: number) => {
      try {
        const date = new Date(value)
        if (isNaN(date.getTime())) return value
        
        const visibleDays = Math.ceil((visibleWindowRef.current.end - visibleWindowRef.current.start) / 100 * dates.length)
        
        if (visibleDays <= 30) {
          return `${date.getMonth() + 1}-${date.getDate()}` // 月-日
        } else if (visibleDays <= 90) {
          return `${date.getMonth() + 1}-${date.getDate()}` // 月-日
        } else if (visibleDays <= 180) {
          return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}` // 年-月-日
        } else if (visibleDays <= 365) {
          return `${date.getFullYear()}-${date.getMonth() + 1}` // 年-月
        } else {
          return `${date.getFullYear()}-${date.getMonth() + 1}` // 年-月
        }
      } catch {
        return value
      }
    }
    
    // Build grids, axes, and series
    const grids: any[] = [
      { left: '10%', right: '8%', top: '10%', height: '55%' }, // K-line
      { left: '10%', right: '8%', top: '68%', height: '12%' }, // Volume
    ]
    
    const xAxes: any[] = [
      {
        type: 'category',
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        splitLine: { show: false },
        min: 'dataMin',
        max: 'dataMax',
        axisLabel: { 
          color: '#8B949E', 
          fontSize: 8, 
          rotate: -45,
          interval: (index: number) => index % dateInterval === 0,
          formatter: formatDateLabel,
        },
      },
      {
        type: 'category',
        gridIndex: 1,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { 
          show: true,
          color: '#8B949E', 
          fontSize: 8, 
          rotate: -45,
          interval: (index: number) => index % dateInterval === 0,
          formatter: formatDateLabel,
        },
        min: 'dataMin',
        max: 'dataMax',
      },
    ]
    
    const yAxes: any[] = [
      {
        scale: true,
        splitArea: {
          show: true,
          areaStyle: {
            color: ['rgba(139, 148, 158, 0.05)', 'rgba(139, 148, 158, 0.02)'],
          },
        },
        axisLabel: { color: '#8B949E', fontSize: 9 },
      },
      {
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      },
    ]
    
    // Prepare backtest overlay data
    let backtestStartIdx = -1
    let backtestEndIdx = -1
    let buyMarkers: any[] = []
    let sellMarkers: any[] = []
    
    if (backtestOverlay) {
      // Find backtest period indices
      if (backtestOverlay.backtestStart && backtestOverlay.backtestEnd) {
        const normalizeDate = (dateStr: string | null | undefined): string | null => {
          if (!dateStr) return null
          return dateStr.split('T')[0]
        }
        
        const normalizedStart = normalizeDate(backtestOverlay.backtestStart)
        const normalizedEnd = normalizeDate(backtestOverlay.backtestEnd)
        
        backtestStartIdx = normalizedStart
          ? dates.findIndex(d => {
              const normalizedD = normalizeDate(d)
              return normalizedD && normalizedD >= normalizedStart
            })
          : -1
        
        backtestEndIdx = dates.length
        if (normalizedEnd) {
          for (let i = dates.length - 1; i >= 0; i--) {
            const normalizedD = normalizeDate(dates[i])
            if (normalizedD && normalizedD <= normalizedEnd) {
              backtestEndIdx = i + 1
              break
            }
          }
        }
      }
      
      // Prepare trade markers
      if (backtestOverlay.tradeMarkers) {
        const normalizeDate = (dateStr: string | null | undefined): string | null => {
          if (!dateStr) return null
          return dateStr.split('T')[0] // 统一格式为 YYYY-MM-DD
        }
        
        backtestOverlay.tradeMarkers.forEach((marker) => {
          const normalizedMarkerDate = normalizeDate(marker.date)
          if (!normalizedMarkerDate) return
          
          // 尝试多种匹配方式
          let dateIndex = dates.findIndex((d) => {
            const normalizedD = normalizeDate(d)
            return normalizedD === normalizedMarkerDate
          })
          
          // 如果找不到精确匹配，尝试模糊匹配（忽略时间部分）
          if (dateIndex < 0) {
            dateIndex = dates.findIndex((d) => {
              const normalizedD = normalizeDate(d)
              return normalizedD && normalizedD.startsWith(normalizedMarkerDate.substring(0, 10))
            })
          }
          
          if (dateIndex >= 0 && marker.price && !isNaN(marker.price)) {
            const markerData = [dateIndex, marker.price]
            if (marker.direction === 1) {
              buyMarkers.push(markerData)
            } else {
              sellMarkers.push(markerData)
            }
          }
        })
        
        // 调试日志 - 显示未匹配的标记
        if (backtestOverlay.tradeMarkers.length > 0) {
          const unmatched = backtestOverlay.tradeMarkers.filter((marker) => {
            const normalizedMarkerDate = normalizeDate(marker.date)
            if (!normalizedMarkerDate) return true
            const dateIndex = dates.findIndex((d) => {
              const normalizedD = normalizeDate(d)
              return normalizedD === normalizedMarkerDate
            })
            return dateIndex < 0
          })
          
          console.log('Trade markers:', {
            total: backtestOverlay.tradeMarkers.length,
            buy: buyMarkers.length,
            sell: sellMarkers.length,
            matched: buyMarkers.length + sellMarkers.length,
            unmatched: unmatched.length,
            unmatchedSample: unmatched.slice(0, 5),
            datesSample: dates.slice(0, 5),
            markersSample: backtestOverlay.tradeMarkers.slice(0, 5),
          })
        }
      }
    }
    
    const hasBacktestPeriod = backtestStartIdx >= 0 && backtestEndIdx > backtestStartIdx
    
    const series: any[] = [
      {
        name: 'K线',
        type: 'candlestick',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: klineValues,
        barWidth: '70%', // K线宽度（更宽）
        barCategoryGap: '1%', // 极小间距
        itemStyle: {
          color: '#FF3D00', // 红涨（国内习惯）
          color0: '#00F2FF', // 绿跌（国内习惯）
          borderColor: null,
          borderColor0: null,
        },
        markArea: hasBacktestPeriod ? {
          silent: false,
          itemStyle: {
            color: 'rgba(197, 160, 89, 0.2)',
            borderColor: 'rgba(197, 160, 89, 0.5)',
            borderWidth: 1,
          },
          data: [
            [
              {
                name: '回测开始',
                xAxis: backtestStartIdx,
              },
              {
                name: '回测结束',
                xAxis: Math.min(backtestEndIdx - 1, dates.length - 1),
              },
            ],
          ],
          label: {
            show: true,
            position: 'insideTop',
            formatter: '回测区间',
            color: '#C5A059',
            fontSize: 12,
            fontWeight: 'bold',
            backgroundColor: 'rgba(197, 160, 89, 0.4)',
            padding: [4, 8],
            borderRadius: 4,
          },
        } : undefined,
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: volumes,
        barWidth: '70%', // 成交量柱宽度，与K线保持一致
        barCategoryGap: '1%', // 极小间距
        itemStyle: {
          color: (params: any) => {
            const idx = params.dataIndex
            // 红涨绿跌（国内习惯）
            return idx > 0 && klineData[idx].close >= klineData[idx - 1].close
              ? '#FF3D00' // 上涨：红色
              : '#00F2FF' // 下跌：绿色
          },
        },
      },
    ]
    
    // Add moving averages (overlay on K-line)
    const maColors: Record<string, string> = {
      ma5: '#FFD700',
      ma10: '#00F2FF',
      ma20: '#FF6B9D',
      ma30: '#C5A059',
      ma60: '#9B59B6',
      ma120: '#E67E22',
    }
    
    if (indicators.ma5 && indicatorData.ma5) {
      series.push({
        name: 'MA5',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ma5.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: maColors.ma5, width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.ma10 && indicatorData.ma10) {
      series.push({
        name: 'MA10',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ma10.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: maColors.ma10, width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.ma20 && indicatorData.ma20) {
      series.push({
        name: 'MA20',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ma20.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: maColors.ma20, width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.ma30 && indicatorData.ma30) {
      series.push({
        name: 'MA30',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ma30.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: maColors.ma30, width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.ma60 && indicatorData.ma60) {
      series.push({
        name: 'MA60',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ma60.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: maColors.ma60, width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.ma120 && indicatorData.ma120) {
      series.push({
        name: 'MA120',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ma120.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: maColors.ma120, width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.ema && indicatorData.ema) {
      series.push({
        name: 'EMA',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.ema.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: '#00CED1', width: 1.5 },
        symbol: 'none',
      })
    }
    
    if (indicators.wma && indicatorData.wma) {
      series.push({
        name: 'WMA',
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: indicatorData.wma.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: '#FFA500', width: 1.5 },
        symbol: 'none',
      })
    }
    
    // Add Bollinger Bands (overlay on K-line)
    if (indicators.bollinger && indicatorData.bollinger) {
      series.push(
        {
          name: '布林上轨',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: indicatorData.bollinger.upper.map(v => v !== null && !isNaN(v) ? v : null),
          lineStyle: { color: '#C5A059', width: 1, type: 'dashed' },
          symbol: 'none',
        },
        {
          name: '布林中轨',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: indicatorData.bollinger.middle.map(v => v !== null && !isNaN(v) ? v : null),
          lineStyle: { color: '#C5A059', width: 1, type: 'dashed' },
          symbol: 'none',
        },
        {
          name: '布林下轨',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: indicatorData.bollinger.lower.map(v => v !== null && !isNaN(v) ? v : null),
          lineStyle: { color: '#C5A059', width: 1, type: 'dashed' },
          symbol: 'none',
        }
      )
    }
    
    // Add MACD subplot
    let currentTop = 83
    let gridIndex = 2
    if (indicators.macd && indicatorData.macd) {
      grids.push({
        left: '10%',
        right: '8%',
        top: `${currentTop}%`,
        height: '8%',
      })
      xAxes.push({
        type: 'category',
        gridIndex: gridIndex,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      })
      yAxes.push({
        scale: true,
        gridIndex: gridIndex,
        splitNumber: 2,
        axisLabel: { color: '#8B949E', fontSize: 9 },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      })
      
      series.push(
        {
          name: 'MACD',
          type: 'line',
          xAxisIndex: gridIndex,
          yAxisIndex: gridIndex,
          data: indicatorData.macd.macd.map(v => v !== null && !isNaN(v) ? v : null),
          lineStyle: { color: '#00F2FF', width: 1.5 },
          symbol: 'none',
        },
        {
          name: 'Signal',
          type: 'line',
          xAxisIndex: gridIndex,
          yAxisIndex: gridIndex,
          data: indicatorData.macd.signal.map(v => v !== null && !isNaN(v) ? v : null),
          lineStyle: { color: '#FFD700', width: 1.5 },
          symbol: 'none',
        },
        {
          name: 'Histogram',
          type: 'bar',
          xAxisIndex: gridIndex,
          yAxisIndex: gridIndex,
          data: indicatorData.macd.histogram.map(v => v !== null && !isNaN(v) ? v : null),
          itemStyle: {
            color: (params: any) => {
              const val = params.value
              return val >= 0 ? '#00F2FF' : '#FF3D00'
            },
          },
        }
      )
      currentTop += 10
      gridIndex++
    }
    
    // Add RSI subplot
    if (indicators.rsi && indicatorData.rsi) {
      grids.push({
        left: '10%',
        right: '8%',
        top: `${currentTop}%`,
        height: '8%',
      })
      xAxes.push({
        type: 'category',
        gridIndex: gridIndex,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      })
      yAxes.push({
        scale: true,
        gridIndex: gridIndex,
        min: 0,
        max: 100,
        splitNumber: 4,
        axisLabel: { color: '#8B949E', fontSize: 9 },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: {
          show: true,
          lineStyle: { color: '#8B949E', opacity: 0.2 },
        },
      })
      
      series.push({
        name: 'RSI',
        type: 'line',
        xAxisIndex: gridIndex,
        yAxisIndex: gridIndex,
        data: indicatorData.rsi.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: '#C5A059', width: 1.5 },
        symbol: 'none',
        markLine: {
          data: [
            { yAxis: 70, name: '超买' },
            { yAxis: 30, name: '超卖' },
            { yAxis: 50, name: '中线' },
          ],
          lineStyle: { color: '#8B949E', opacity: 0.3, type: 'dashed' },
        },
      })
      currentTop += 10
      gridIndex++
    }
    
    // Add ATR subplot
    if (indicators.atr && indicatorData.atr) {
      grids.push({
        left: '10%',
        right: '8%',
        top: `${currentTop}%`,
        height: '8%',
      })
      xAxes.push({
        type: 'category',
        gridIndex: gridIndex,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      })
      yAxes.push({
        scale: true,
        gridIndex: gridIndex,
        splitNumber: 2,
        axisLabel: { color: '#8B949E', fontSize: 9 },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      })
      
      series.push({
        name: 'ATR',
        type: 'line',
        xAxisIndex: gridIndex,
        yAxisIndex: gridIndex,
        data: indicatorData.atr.map(v => v !== null && !isNaN(v) ? v : null),
        lineStyle: { color: '#8B949E', width: 1.5 },
        symbol: 'none',
        areaStyle: { color: 'rgba(139, 148, 158, 0.2)' },
      })
      currentTop += 10
      gridIndex++
    }
    
    // Add buy/sell markers as scatter series
    if (buyMarkers.length > 0) {
      series.push({
        name: '买入',
        type: 'scatter',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: buyMarkers,
        symbol: 'triangle',
        symbolSize: 12,
        itemStyle: {
          color: '#FF3D00', // 买入：红色（国内习惯）
        },
        label: {
          show: true,
          position: 'bottom',
          formatter: '买',
          color: '#FF3D00', // 买入：红色（国内习惯）
          fontSize: 10,
          fontWeight: 'bold',
        },
        zlevel: 10, // 确保显示在K线上方
      })
    }
    
    if (sellMarkers.length > 0) {
      series.push({
        name: '卖出',
        type: 'scatter',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: sellMarkers,
        symbol: 'triangle',
        symbolRotate: 180, // 倒三角
        symbolSize: 12,
        itemStyle: {
          color: '#00F2FF', // 卖出：绿色（国内习惯）
        },
        label: {
          show: true,
          position: 'top',
          formatter: '卖',
          color: '#00F2FF', // 卖出：绿色（国内习惯）
          fontSize: 10,
          fontWeight: 'bold',
        },
        zlevel: 10, // 确保显示在K线上方
      })
    }
    
    const legendData = series.map(s => s.name)
    
    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      legend: {
        data: legendData,
        textStyle: { color: '#8B949E', fontSize: 8 },
        top: 0,
        type: 'scroll',
        itemWidth: 10,
        itemHeight: 6,
        itemGap: 4,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        backgroundColor: '#161B22',
        borderColor: '#C5A059',
        textStyle: { color: '#8B949E' },
      },
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: xAxes.map((_, i) => i),
          start: visibleWindowRef.current.start, // 使用ref保存的值，避免指标变化时重置
          end: visibleWindowRef.current.end,
          zoomOnMouseWheel: 'ctrl', // Ctrl + 滚轮缩放（macOS触控板：Command + 双指缩放）
          moveOnMouseMove: true, // 鼠标拖动平移
          moveOnMouseWheel: true, // 滚轮左右移动（同花顺逻辑）
          preventDefaultMouseMove: true,
        },
        {
          type: 'slider',
          xAxisIndex: xAxes.map((_, i) => i),
          start: visibleWindowRef.current.start, // 使用ref保存的值
          end: visibleWindowRef.current.end,
          bottom: 2,
          height: 12,
          handleStyle: {
            color: '#C5A059',
          },
          dataBackground: {
            areaStyle: {
              color: 'rgba(197, 160, 89, 0.2)',
            },
          },
          selectedDataBackground: {
            areaStyle: {
              color: 'rgba(197, 160, 89, 0.4)',
            },
          },
          borderColor: '#C5A059',
          textStyle: {
            color: '#8B949E',
            fontSize: 10,
          },
          // 支持拖动滑块两端来调整窗口大小
          brushSelect: true,
        },
      ],
      series,
    }
    
    // 在设置 option 前，先保存当前的 dataZoom 状态（如果图表已存在）
    if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed()) {
      try {
        const currentOption = chartInstanceRef.current.getOption()
        // getOption() 可能返回数组或对象，需要正确处理
        const optionObj = Array.isArray(currentOption) ? currentOption[0] : currentOption
        
        if (optionObj && typeof optionObj === 'object' && 'dataZoom' in optionObj) {
          const currentDataZoom = (optionObj as any).dataZoom
          
          if (Array.isArray(currentDataZoom) && currentDataZoom.length > 0) {
            const sliderZoom = currentDataZoom.find((dz: any) => dz && dz.type === 'slider')
            if (sliderZoom && sliderZoom.start !== undefined && sliderZoom.end !== undefined) {
              // 使用当前图表的 dataZoom 值，而不是 ref 的值（更准确）
              const preservedWindow = { start: sliderZoom.start, end: sliderZoom.end }
              visibleWindowRef.current = preservedWindow
              
              // 更新 option 中的 dataZoom
              option.dataZoom = option.dataZoom.map((dz: any) => ({
                ...dz,
                start: preservedWindow.start,
                end: preservedWindow.end,
              }))
            }
          }
        }
      } catch (e) {
        // 如果获取失败，使用 ref 的值（静默处理，不输出警告）
        // console.warn('Failed to preserve dataZoom:', e)
      }
    }
    
    chart.setOption(option, { notMerge: false })
    
    // Handle dataZoom events
    chart.off('dataZoom')
    chart.on('dataZoom', handleDataZoom)
    dataZoomHandlerRef.current = handleDataZoom
    
    // Handle click events if clickable
    if (clickable && onKlineClick) {
      chart.off('click')
      chart.on('click', (params: any) => {
        echartsClickHandledRef.current = false
        let dateIndex = -1
        
        // Case 1: Click on candlestick series (body or shadows)
        if (params.componentType === 'series' && params.seriesType === 'candlestick') {
          dateIndex = params.dataIndex
          echartsClickHandledRef.current = true
        }
        // Case 2: Click on xAxis or grid - get value from params
        else if (params.componentType === 'xAxis' || params.componentType === 'grid') {
          const clickedValue = params.value
          if (clickedValue !== undefined && clickedValue !== null) {
            if (typeof clickedValue === 'number') {
              dateIndex = Math.round(clickedValue)
            } else if (typeof clickedValue === 'string') {
              dateIndex = dates.findIndex(d => d === clickedValue || d.split('T')[0] === clickedValue.split('T')[0])
            }
            if (dateIndex >= 0) {
              echartsClickHandledRef.current = true
            }
          }
        }
        // Case 3: Try to get dataIndex from any params
        else if (params.dataIndex !== undefined && params.dataIndex !== null) {
          dateIndex = params.dataIndex
          echartsClickHandledRef.current = true
        }
        // Case 4: Try to get value from params
        else if (params.value !== undefined && params.value !== null) {
          const clickedValue = params.value
          if (typeof clickedValue === 'number') {
            dateIndex = Math.round(clickedValue)
          } else if (typeof clickedValue === 'string') {
            dateIndex = dates.findIndex(d => d === clickedValue || d.split('T')[0] === clickedValue.split('T')[0])
          }
          if (dateIndex >= 0) {
            echartsClickHandledRef.current = true
          }
        }
        
        // Trigger callback if we found a valid date index
        if (dateIndex >= 0 && dateIndex < dates.length) {
          const now = Date.now()
          // Prevent duplicate calls within 100ms
          if (now - lastClickTimeRef.current > 100) {
            lastClickTimeRef.current = now
            onKlineClick(dates[dateIndex])
          }
        }
      })
      
      // Also listen to click events on the chart container as fallback
      // This catches clicks that ECharts didn't handle (e.g., on shadows, empty areas)
      if (chartRef.current) {
        containerClickHandlerRef.current = (event: MouseEvent) => {
          // Skip if ECharts already handled it
          if (echartsClickHandledRef.current) {
            echartsClickHandledRef.current = false // Reset for next click
            return
          }
          
          if (!chartInstanceRef.current || !chartRef.current) return
          
          const now = Date.now()
          // Prevent duplicate calls within 100ms
          if (now - lastClickTimeRef.current < 100) {
            return
          }
          
          try {
            const rect = chartRef.current.getBoundingClientRect()
            const x = event.clientX - rect.left
            const y = event.clientY - rect.top
            
            // Convert pixel coordinates to data coordinates
            const pointInPixel = [x, y]
            const pointInGrid = chartInstanceRef.current.convertFromPixel('grid', pointInPixel)
            
            if (pointInGrid && Array.isArray(pointInGrid) && pointInGrid.length >= 2) {
              const xValue = pointInGrid[0]
              if (typeof xValue === 'number' && !isNaN(xValue) && isFinite(xValue)) {
                const dateIndex = Math.max(0, Math.min(dates.length - 1, Math.round(xValue)))
                if (dateIndex >= 0 && dateIndex < dates.length) {
                  lastClickTimeRef.current = now
                  onKlineClick(dates[dateIndex])
                }
              }
            }
          } catch (e) {
            // Ignore conversion errors
            console.warn('Failed to convert pixel to data coordinates:', e)
          }
        }
        
        chartRef.current.addEventListener('click', containerClickHandlerRef.current)
      }
    }
    
    // Handle resize
    const handleResize = () => {
      if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed()) {
        try {
          chartInstanceRef.current.resize()
        } catch (e) {
          console.warn('Chart resize error:', e)
        }
      }
    }
    
    resizeHandlerRef.current = handleResize
    window.addEventListener('resize', handleResize)
    
    // Cleanup
    return () => {
      if (chartInstanceRef.current && dataZoomHandlerRef.current) {
        chartInstanceRef.current.off('dataZoom', dataZoomHandlerRef.current)
      }
      if (resizeHandlerRef.current) {
        window.removeEventListener('resize', resizeHandlerRef.current)
        resizeHandlerRef.current = null
      }
      // Remove container click handler if it exists
      if (chartRef.current && containerClickHandlerRef.current) {
        chartRef.current.removeEventListener('click', containerClickHandlerRef.current)
        containerClickHandlerRef.current = null
      }
      if (chartInstanceRef.current) {
        try {
          if (!chartInstanceRef.current.isDisposed()) {
            chartInstanceRef.current.dispose()
          }
        } catch (e) {
          // Ignore
        }
        chartInstanceRef.current = null
      }
    }
  }, [klineData, indicatorData, indicators, isMinimized, backtestOverlay?.backtestStart, backtestOverlay?.backtestEnd, backtestOverlay?.tradeMarkers, clickable, onKlineClick]) // 移除 visibleWindow 依赖，使用 ref 保持窗口范围
  
  const timeIntervals: { value: TimeInterval; label: string }[] = [
    { value: '1m', label: '1分钟' },
    { value: '5m', label: '5分钟' },
    { value: '15m', label: '15分钟' },
    { value: '30m', label: '30分钟' },
    { value: '60m', label: '60分钟' },
    { value: '1d', label: '日线' },
    { value: '1w', label: '周线' },
    { value: '1M', label: '月线' },
  ]
  
  if (isMinimized) {
    return (
      <div className={`bg-eidos-surface rounded-lg border border-eidos-muted/20 ${className}`} style={style}>
        <div className="flex items-center justify-between p-2">
          <span className="text-sm text-eidos-gold font-semibold">{symbol}</span>
          <button
            onClick={() => setIsMinimized(false)}
            className="text-eidos-muted hover:text-white transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
        </div>
      </div>
    )
  }
  
  return (
    <div className={`bg-eidos-surface rounded-lg border border-eidos-muted/20 flex flex-col ${className}`} style={style}>
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-eidos-muted/20">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-eidos-gold">{symbol}</h3>
          {onTimeIntervalChange ? (
            <select
              value={timeInterval}
              onChange={(e) => onTimeIntervalChange(e.target.value as TimeInterval)}
              className="bg-eidos-surface border border-eidos-muted/30 rounded px-1.5 py-0.5 text-[10px] text-eidos-muted focus:outline-none focus:border-eidos-gold"
            >
              {timeIntervals.map(interval => (
                <option key={interval.value} value={interval.value}>{interval.label}</option>
              ))}
            </select>
          ) : (
            <span className="text-[10px] text-eidos-muted px-1.5 py-0.5">
              {timeIntervals.find(i => i.value === timeInterval)?.label || timeInterval}
            </span>
          )}
        </div>
        <button
          onClick={() => setIsMinimized(true)}
          className="text-eidos-muted hover:text-white transition-colors"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>
      
      {/* Indicator Selection */}
      <div className="px-2 py-0.5 border-b border-eidos-muted/20 bg-eidos-surface/30">
        <div className="flex items-center gap-1 flex-wrap text-[9px]">
          <span className="text-eidos-muted font-medium">技术指标:</span>
          {Object.entries(INDICATOR_CATEGORIES).map(([categoryKey, category]) => (
            <div key={categoryKey} className="flex items-center gap-0.5">
              <span className="text-eidos-muted/50">{category.label}:</span>
              {category.indicators.map(indicatorKey => (
                <label key={indicatorKey} className="flex items-center gap-0.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={indicators[indicatorKey]}
                    onChange={() => handleIndicatorToggle(indicatorKey)}
                    className="w-2 h-2 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
                  />
                  <span className="text-eidos-muted">{INDICATOR_LABELS[indicatorKey]}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
      </div>
      
      
      {/* Chart */}
      <div className="p-1.5 flex-1" style={{ minHeight: '300px' }}>
        <div
          ref={chartRef}
          className="w-full"
          style={{ minHeight: '300px', height: '300px', width: '100%' }}
        />
        {klineData.length === 0 && (
          <div className="flex items-center justify-center h-[400px] text-eidos-muted text-sm">
            暂无K线数据
          </div>
        )}
      </div>
    </div>
  )
}

