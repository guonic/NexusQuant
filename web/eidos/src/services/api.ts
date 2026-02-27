import axios from 'axios'
import type {
  Experiment,
  LedgerEntry,
  Trade,
  PerformanceMetrics,
  TradeStats,
  BacktestReport,
  ReportConfig,
} from '@/types/eidos'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * 获取所有实验列表
 */
export async function getExperiments(): Promise<Experiment[]> {
  const response = await api.get<Experiment[]>('/experiments')
  return response.data
}

/**
 * 获取单个实验信息
 */
export async function getExperiment(expId: string): Promise<Experiment> {
  const response = await api.get<Experiment>(`/experiments/${expId}`)
  return response.data
}

/**
 * 获取实验的账户流水
 */
export async function getLedger(expId: string): Promise<LedgerEntry[]> {
  const response = await api.get<LedgerEntry[]>(`/experiments/${expId}/ledger`)
  return response.data
}

/**
 * 获取实验的交易记录
 */
export async function getTrades(
  expId: string,
  params?: { symbol?: string; start_date?: string; end_date?: string }
): Promise<Trade[]> {
  const response = await api.get<Trade[]>(`/experiments/${expId}/trades`, { params })
  return response.data
}

/**
 * 获取实验的性能指标
 */
export async function getPerformanceMetrics(expId: string): Promise<PerformanceMetrics> {
  const response = await api.get<PerformanceMetrics>(`/experiments/${expId}/metrics`)
  return response.data
}

/**
 * 获取实验的交易统计
 */
export async function getTradeStats(expId: string): Promise<TradeStats> {
  const response = await api.get<TradeStats>(`/experiments/${expId}/trade-stats`)
  return response.data
}

/**
 * 获取股票的K线数据
 */
export interface KlineData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface KlineResponse {
  kline_data: KlineData[]
  indicators?: {
    macd?: { macd: (number | null)[]; signal: (number | null)[]; histogram: (number | null)[] }
    rsi?: (number | null)[]
    bollinger?: { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] }
    atr?: (number | null)[]
    ma5?: (number | null)[]
    ma10?: (number | null)[]
    ma20?: (number | null)[]
    ma30?: (number | null)[]
  }
  backtest_start?: string
  backtest_end?: string
}

export async function getStockKline(
  expId: string,
  symbol: string,
  startDate?: string,
  endDate?: string,
  indicators?: string[]
): Promise<KlineResponse> {
  const response = await api.get<KlineResponse>(`/experiments/${expId}/kline/${symbol}`, {
    params: {
      start_date: startDate,
      end_date: endDate,
      indicators: indicators?.join(','),
    },
  })
  return response.data
}

/**
 * 获取完整的回测报告
 */
export async function getBacktestReport(
  expId: string,
  config?: ReportConfig
): Promise<BacktestReport> {
  const params: Record<string, string> = {}

  if (config?.format) {
    params.format = config.format
  }
  if (config?.categories) {
    params.categories = config.categories.join(',')
  }
  if (config?.metrics) {
    params.metrics = config.metrics.join(',')
  }

  const response = await api.get<BacktestReport>(`/experiments/${expId}/report`, { params })
  return response.data
}

/**
 * 导出报告（HTML/Markdown）
 */
export async function exportReport(expId: string, format: 'html' | 'markdown'): Promise<Blob> {
  const response = await api.get(`/experiments/${expId}/report`, {
    params: { format },
    responseType: 'blob',
  })
  return response.data
}

/**
 * DTW Labeling API
 */

export interface DTWHitRow {
  template: string
  ts_code: string
  start_date: string
  end_date: string
  score: number
  hit_count?: number
  start_index?: number
  end_index?: number
  label?: string | null
  platform_start?: string | null
  platform_end?: string | null
  breakout_date?: string | null
  notes?: string | null
}

export interface DTWHitsPageResponse {
  hits: DTWHitRow[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface DTWKlineResponse {
  kline_data: KlineData[]
  indicators?: {
    ma5?: (number | null)[]
    ma10?: (number | null)[]
  }
  match_start: string
  match_end: string
}

export interface DTWAnnotationRequest {
  csv_file: string
  row_index: number
  label: string
  platform_start?: string
  platform_end?: string
  breakout_date?: string
  notes?: string
}

/**
 * Get paginated DTW hits from CSV
 */
export async function getDTWHits(
  filename: string,
  page: number = 1,
  pageSize: number = 50
): Promise<DTWHitsPageResponse> {
  const response = await api.get<DTWHitsPageResponse>(`/dtw/labeling/csv/${encodeURIComponent(filename)}`, {
    params: { page, page_size: pageSize },
  })
  return response.data
}

/**
 * Get K-line data for DTW labeling
 */
export async function getDTWKline(
  symbol: string,
  startDate: string,
  endDate: string,
  extendLeftBars: number = 5,
  extendRightBars: number = 5
): Promise<DTWKlineResponse> {
  const response = await api.get<DTWKlineResponse>(`/dtw/labeling/kline/${symbol}`, {
    params: {
      start_date: startDate,
      end_date: endDate,
      extend_left_bars: extendLeftBars,
      extend_right_bars: extendRightBars,
    },
  })
  return response.data
}

/**
 * Save DTW annotation
 */
export async function saveDTWAnnotation(request: DTWAnnotationRequest): Promise<{ success: boolean; message: string }> {
  const response = await api.post(`/dtw/labeling/annotate`, request)
  return response.data
}

export default api

