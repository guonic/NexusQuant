/**
 * Types for StockChart component
 */

export type TimeInterval = '1m' | '5m' | '15m' | '30m' | '60m' | '1d' | '1w' | '1M'

export interface KlineData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface IndicatorConfig {
  // 趋势类
  ma5: boolean
  ma10: boolean
  ma20: boolean
  ma30: boolean
  ma60: boolean
  ma120: boolean
  ema: boolean
  wma: boolean
  
  // 震荡类
  rsi: boolean
  kdj: boolean
  cci: boolean
  wr: boolean
  obv: boolean
  
  // 趋势+震荡
  macd: boolean
  dmi: boolean
  
  // 通道类
  bollinger: boolean
  envelope: boolean
  
  // 波动类
  atr: boolean
  bbw: boolean
  
  // 成交量类
  vol: boolean
  vwap: boolean
}

export interface IndicatorData {
  // 趋势类
  ma5?: (number | null)[]
  ma10?: (number | null)[]
  ma20?: (number | null)[]
  ma30?: (number | null)[]
  ma60?: (number | null)[]
  ma120?: (number | null)[]
  ema?: (number | null)[]
  wma?: (number | null)[]
  
  // 震荡类
  rsi?: (number | null)[]
  kdj?: { k: (number | null)[]; d: (number | null)[]; j: (number | null)[] }
  cci?: (number | null)[]
  wr?: (number | null)[]
  obv?: (number | null)[]
  
  // 趋势+震荡
  macd?: { macd: (number | null)[]; signal: (number | null)[]; histogram: (number | null)[] }
  dmi?: { pdi: (number | null)[]; mdi: (number | null)[]; adx: (number | null)[]; adxr: (number | null)[] }
  
  // 通道类
  bollinger?: { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] }
  envelope?: { upper: (number | null)[]; lower: (number | null)[] }
  
  // 波动类
  atr?: (number | null)[]
  bbw?: (number | null)[]
  
  // 成交量类
  vol?: (number | null)[]
  vwap?: (number | null)[]
}

export interface TradeMarker {
  date: string
  price: number
  direction: 1 | -1 // 1 for buy, -1 for sell
}

export interface BacktestOverlay {
  backtestStart?: string | null
  backtestEnd?: string | null
  tradeMarkers?: TradeMarker[]
}

export interface StockChartProps {
  symbol: string
  klineData: KlineData[]
  indicatorData?: IndicatorData
  indicators?: IndicatorConfig
  onIndicatorsChange?: (indicators: IndicatorConfig) => void
  timeInterval?: TimeInterval
  onTimeIntervalChange?: (interval: TimeInterval) => void
  embedded?: boolean
  className?: string
  style?: React.CSSProperties
  backtestOverlay?: BacktestOverlay // Optional backtest overlay
  onKlineClick?: (date: string) => void // Callback when K-line is clicked
  clickable?: boolean // Enable click interaction
  initialVisibleWindow?: { start: number; end: number } // Initial visible window (0-100)
}

