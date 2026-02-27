#!/usr/bin/env python3
"""
Run market regime analysis on XAUUSD M1 CSV.

Two parts:
  1. State labeling (上行/下行/中继): pluggable via --labeler (ma | overlap).
  2. Trading signal labeling: pattern match (下中下上等) + breakout validation.

Usage:
    python examples/market_regime_xauusd.py
    python examples/market_regime_xauusd.py --csv /path/to/xauusd-m1-bid-2025-12-01-2025-12-31.csv
    python examples/market_regime_xauusd.py --labeler overlap --overlap-threshold 0.6 --plot --out result.html
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nq.trading.selector.teapot.market_regime import (
    REGIME_LABELERS,
    add_validated_breakout_signals,
    adjust_relay_boundaries,
    get_regime_segment_table,
)


def load_xauusd_csv(path: Path) -> pd.DataFrame:
    """Load XAUUSD M1 CSV; expect columns: timestamp, open, high, low, close."""
    df = pd.read_csv(path)
    # timestamp is in milliseconds
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    elif "datetime" not in df.columns:
        raise ValueError("CSV must have 'timestamp' or 'datetime' column")
    # normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _inject_yaxis_autorange_script(html_path: Path, div_id: str = "market-regime-plot") -> None:
    """Inject JS: poll layout.xaxis.range and set y-axis so data span is 80% of total (blank 20%)."""
    js = '''<script>
(function() {
  var divId = "''' + div_id + '''";
  function getGd() {
    var el = document.getElementById(divId);
    if (el) return el;
    return document.querySelector('.plotly-graph-div');
  }
  function run() {
    var gd = getGd();
    var log = (typeof console !== 'undefined' && console.log) ? function() { console.log.apply(console, ['[market-regime]'].concat(Array.prototype.slice.call(arguments))); } : function() {};
    log('run()', 'gd=', !!gd, 'divId=', divId);
    if (!gd) { log('retry 150ms: gd not found'); setTimeout(run, 150); return; }
    if (typeof Plotly === 'undefined') { log('retry 150ms: Plotly undefined'); setTimeout(run, 150); return; }
    if (!gd.data || gd.data.length === 0) { log('retry 150ms: gd.data empty', gd.data ? gd.data.length : 0); setTimeout(run, 150); return; }
    log('init ok', 'traces=', gd.data.length, 'on relayout + fallback 1.5s');

    function toTime(v) {
      if (v == null) return 0;
      if (typeof v === 'number' && isFinite(v)) {
        if (v > 1e12) return v;
        if (v > 1e9 && v < 1e12) return v * 1000;
        return v;
      }
      try {
        if (v instanceof Date) return v.getTime();
        var s = String(v);
        var t = new Date(s).getTime();
        if (!isNaN(t)) return t;
        t = new Date(s.replace(' ', 'T')).getTime();
        return isNaN(t) ? 0 : t;
      } catch(e) { log('toTime fail', v, e); return 0; }
    }
    function getXRange() {
      var xr = (gd._fullLayout && gd._fullLayout.xaxis && gd._fullLayout.xaxis.range) || (gd.layout && gd.layout.xaxis && gd.layout.xaxis.range);
      if (xr && xr.length === 2) {
        var t0 = toTime(xr[0]), t1 = toTime(xr[1]);
        if (t0 && t1) {
          var from = (gd._fullLayout && gd._fullLayout.xaxis && gd._fullLayout.xaxis.range) ? '_fullLayout' : 'layout';
          return { r: [t0, t1], from: from, xr: xr };
        }
        log('getXRange: xr invalid t0/t1', t0, t1, xr);
      } else { log('getXRange: no layout range', 'has _fullLayout.xaxis', !!(gd._fullLayout && gd._fullLayout.xaxis), 'has layout.xaxis', !!(gd.layout && gd.layout.xaxis)); }
      var xMin = Infinity, xMax = -Infinity;
      for (var i = 0; i < gd.data.length; i++) {
        var xs = gd.data[i].x;
        if (!xs) continue;
        for (var k = 0; k < xs.length; k++) {
          var tx = toTime(xs[k]);
          if (tx) { xMin = Math.min(xMin, tx); xMax = Math.max(xMax, tx); }
        }
      }
      if (xMin === Infinity) { log('getXRange: no x in data'); return null; }
      log('getXRange: fallback from data', [new Date(xMin).toISOString(), new Date(xMax).toISOString()]);
      return { r: [xMin, xMax], from: 'data', xr: null };
    }
    var lastX0 = null, lastX1 = null;
    var skipNames = { 'regime': 1, 'entry long': 1 };
    function applyYRange() {
      var out = getXRange();
      if (!out) { log('applyYRange: getXRange null'); return; }
      var r = out.r, x0 = r[0], x1 = r[1];
      if (lastX0 !== null && lastX1 !== null && Math.abs(lastX0 - x0) < 1000 && Math.abs(lastX1 - x1) < 1000) {
        log('applyYRange: skip range unchanged');
        return;
      }
      lastX0 = x0; lastX1 = x1;
      log('applyYRange: x from', out.from, 'x0=', x0, 'x1=', x1);
      var yMin = Infinity, yMax = -Infinity;
      var y3Min = Infinity, y3Max = -Infinity;
      var pointsInWindow = 0;
      var fullData = gd._fullData || gd.data;
      for (var i = 0; i < fullData.length; i++) {
        var t = fullData[i], xs = t.x, ys = t.y;
        if (!xs) continue;
        var onY2 = t.yaxis && String(t.yaxis).indexOf('y2') !== -1;
        var onY3 = t.yaxis && String(t.yaxis).indexOf('y3') !== -1;
        var useMainY = !onY2 && !onY3;
        var skip = useMainY && t.name && skipNames[t.name];
        var highs = t.high, lows = t.low;
        var isCandle = t.type === 'candlestick' && highs && lows;
        for (var j = 0; j < xs.length; j++) {
          var tx = toTime(xs[j]);
          if (!tx && xs[j] != null) continue;
          if (tx >= x0 && tx <= x1) {
            pointsInWindow++;
            if (isCandle) {
              var h = Number(highs[j]), l = Number(lows[j]);
              if (!isNaN(h) && useMainY && !skip) { yMin = Math.min(yMin, l); yMax = Math.max(yMax, h); }
            } else if (ys) {
              var v = Number(ys[j]);
              if (!isNaN(v)) {
                if (onY3) { y3Min = Math.min(y3Min, v); y3Max = Math.max(y3Max, v); }
                else if (useMainY && !skip) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
              }
            }
          }
        }
      }
      if (pointsInWindow === 0) {
        var d0 = fullData[0];
        var sx = d0 && d0.x && d0.x[0];
        log('applyYRange: no points in window, fallback to full y', 'x[0]=', sx, 'toTime=', toTime(sx), 'x0=', x0, 'x1=', x1);
        for (var i = 0; i < fullData.length; i++) {
          var t = fullData[i];
          var onY2 = t.yaxis && String(t.yaxis).indexOf('y2') !== -1;
          var onY3 = t.yaxis && String(t.yaxis).indexOf('y3') !== -1;
          var useMainY = !onY2 && !onY3;
          if (useMainY && t.name && skipNames[t.name]) continue;
          var isCandle = t.type === 'candlestick' && t.high && t.low;
          if (isCandle && useMainY) {
            for (var j = 0; j < t.high.length; j++) {
              var h = Number(t.high[j]), l = Number(t.low[j]);
              if (!isNaN(h)) yMax = Math.max(yMax, h);
              if (!isNaN(l)) yMin = Math.min(yMin, l);
            }
          }
          var ys = t.y;
          if (!ys) continue;
          for (var j = 0; j < ys.length; j++) {
            var v = Number(ys[j]);
            if (!isNaN(v)) {
              if (onY3) { y3Min = Math.min(y3Min, v); y3Max = Math.max(y3Max, v); }
              else if (useMainY) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
            }
          }
        }
      }
      var upd = {};
      if (yMin !== Infinity && yMax !== -Infinity && isFinite(yMin) && isFinite(yMax) && yMax > yMin) {
        var span = Math.max(yMax - yMin, 0.5);
        var pad = span * 0.125;
        var yLo = yMin - pad, yHi = yMax + pad;
        if (isFinite(yLo) && isFinite(yHi) && yHi > yLo) {
          upd['yaxis.range'] = [yLo, yHi];
          upd['yaxis.autorange'] = false;
          log('applyYRange: relayout yaxis', 'range', [yLo, yHi]);
        } else { log('applyYRange: y range invalid', yLo, yHi); }
      } else { log('applyYRange: no y in window', 'yMin', yMin, 'yMax', yMax); }
      if (y3Min !== Infinity && y3Max !== -Infinity && isFinite(y3Min) && isFinite(y3Max) && y3Max > y3Min) {
        var s3 = Math.max(y3Max - y3Min, 0.01);
        var tLo = y3Min - s3/2, tHi = y3Max + s3/2;
        if (isFinite(tLo) && isFinite(tHi) && tHi > tLo) {
          upd['yaxis3.range'] = [tLo, tHi];
          upd['yaxis3.autorange'] = false;
          log('applyYRange: relayout yaxis3', 'range', [tLo, tHi]);
        }
      }
      upd['yaxis2.range'] = [-1.5, 1.5];
      upd['yaxis2.autorange'] = false;
      if (Object.keys(upd).length) {
        var xr = (gd._fullLayout && gd._fullLayout.xaxis && gd._fullLayout.xaxis.range) || (gd.layout && gd.layout.xaxis && gd.layout.xaxis.range);
        if (xr && xr.length === 2) { upd['xaxis.range'] = [xr[0], xr[1]]; }
        try { Plotly.relayout(gd, upd); log('applyYRange: Plotly.relayout done'); } catch (e) { log('applyYRange: Plotly.relayout error', e); }
      } else { log('applyYRange: upd empty, no relayout'); }
    }
    applyYRange();
    if (gd.on) {
      gd.on('plotly_relayout', function() { log('event relayout'); setTimeout(applyYRange, 0); });
      gd.on('plotly_afterplot', function() { log('event afterplot'); applyYRange(); });
      log('bound relayout + afterplot');
    } else { log('gd.on missing, no event bind'); }
    setInterval(applyYRange, 1500);
  }
  if (document.readyState === 'complete') run();
  else window.addEventListener('load', run);
})();
</script>
'''
    html_path = Path(html_path)
    text = html_path.read_text(encoding="utf-8")
    if "setInterval(applyYRange" in text:
        return
    text = text.replace("</body>", js + "\n</body>")
    html_path.write_text(text, encoding="utf-8")


def _slice_data_for_json(df: pd.DataFrame) -> dict:
    """Build dict of arrays for crosshair slice display; NaN -> null in JSON."""
    cols = ["datetime", "open", "high", "low", "close", "regime", "bundle_tightness"]
    if "volume" in df.columns:
        cols.append("volume")
    out = {}
    for c in cols:
        if c == "datetime":
            out[c] = pd.to_datetime(df[c]).astype(str).tolist()
        else:
            out[c] = [None if pd.isna(v) else (int(v) if c == "regime" else float(v)) for v in df[c]]
    if "last_relay_max" in df.columns:
        out["last_relay_max"] = [None if pd.isna(v) else float(v) for v in df["last_relay_max"]]
    if "pattern_2_relay_gid" in df.columns:
        out["pattern_2_relay_gid"] = [None if pd.isna(v) else int(v) for v in df["pattern_2_relay_gid"]]
    if "entry_signal_long" in df.columns:
        out["entry_signal_long"] = [bool(v) for v in df["entry_signal_long"]]
    if "_pattern_match" in df.columns:
        out["_pattern_match"] = [bool(v) if not pd.isna(v) else False for v in df["_pattern_match"]]
    if "_price_down_ok" in df.columns:
        out["_price_down_ok"] = [bool(v) if not pd.isna(v) else False for v in df["_price_down_ok"]]
    if "_price_up_ok" in df.columns:
        out["_price_up_ok"] = [bool(v) if not pd.isna(v) else False for v in df["_price_up_ok"]]
    if "_pattern_ok" in df.columns:
        out["_pattern_ok"] = [bool(v) if not pd.isna(v) else False for v in df["_pattern_ok"]]
    if "_thresh" in df.columns:
        out["_thresh"] = [None if pd.isna(v) else float(v) for v in df["_thresh"]]
    if "relay_start_adjusted" in df.columns:
        out["relay_start_adjusted"] = [bool(v) for v in df["relay_start_adjusted"]]
    if "relay_end_adjusted" in df.columns:
        out["relay_end_adjusted"] = [bool(v) for v in df["relay_end_adjusted"]]
    if "relay_solid_box_center" in df.columns:
        out["relay_solid_box_center"] = [None if pd.isna(v) else float(v) for v in df["relay_solid_box_center"]]
    if "relay_logical_box_center" in df.columns:
        out["relay_logical_box_center"] = [None if pd.isna(v) else float(v) for v in df["relay_logical_box_center"]]
    return out


def _inject_crosshair_and_slice(html_path: Path, df: pd.DataFrame, div_id: str = "market-regime-plot") -> None:
    """Inject crosshair (vertical dashed line) and slice data tooltip (HLOCV + regime + tightness)."""
    slice_data = _slice_data_for_json(df)
    slice_json = json.dumps(slice_data, ensure_ascii=False)
    js = """<script>
window.MARKET_REGIME_SLICE_DATA = """ + slice_json + """;
(function() {
  var divId = """ + json.dumps(div_id) + """;
  function getGd() {
    var el = document.getElementById(divId);
    return el || document.querySelector('.plotly-graph-div');
  }
  function run() {
    var gd = getGd();
    if (!gd || typeof Plotly === 'undefined') { setTimeout(run, 100); return; }
    var data = window.MARKET_REGIME_SLICE_DATA;
    if (!data || !data.datetime) return;
    var crosshairV = { type: 'line', x0: null, x1: null, y0: 0, y1: 1, yref: 'paper',
      line: { dash: 'dash', color: 'rgba(200,200,200,0.9)', width: 1 } };
    var crosshairH = { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: null, y1: null, yref: 'y',
      line: { dash: 'dash', color: 'rgba(200,200,200,0.7)', width: 1 } };
    var tip = document.createElement('div');
    tip.id = 'market-regime-slice-tip';
    tip.style.cssText = 'position:absolute;right:12px;top:80px;min-width:180px;padding:8px 10px;background:rgba(30,30,30,0.95);border:1px solid #555;border-radius:6px;font:12px monospace;color:#ddd;z-index:1000;pointer-events:none;';
    gd.parentNode.style.position = 'relative';
    gd.parentNode.appendChild(tip);
    function fmt(v) { return v == null || v === '' ? '-' : (typeof v === 'number' && (v | 0) === v ? v : Number(v).toFixed(4)); }
    function showSlice(idx) {
      if (idx < 0 || idx >= data.datetime.length) { tip.style.display = 'none'; return; }
      var r = data.regime[idx], rStr = r === 1 ? '上行' : (r === -1 ? '下降' : '中继');
      var html = '<div style="margin-bottom:4px;color:#888;">' + (data.datetime[idx] || '') + '</div>';
      html += 'O ' + fmt(data.open[idx]) + ' &nbsp; H ' + fmt(data.high[idx]) + '<br/>';
      html += 'L ' + fmt(data.low[idx]) + ' &nbsp; C ' + fmt(data.close[idx]);
      if (data.volume != null) html += '<br/>V ' + fmt(data.volume[idx]);
      html += '<br/><span style="color:#8f8;">' + rStr + '</span> &nbsp; tight ' + fmt(data.bundle_tightness[idx]);
      if (data.relay_start_adjusted != null && data.relay_start_adjusted[idx]) {
        html += ' &nbsp; <span style="color:#0ff;">[修正起始]</span>';
      }
      if (data.relay_end_adjusted != null && data.relay_end_adjusted[idx]) {
        html += ' &nbsp; <span style="color:#f0f;">[修正结束]</span>';
      }
      if (data.relay_solid_box_center != null && data.relay_solid_box_center[idx] != null) {
        html += '<br/><span style="color:#888;font-size:11px;">solid中心 ' + fmt(data.relay_solid_box_center[idx]) + '</span>';
      }
      if (data.relay_logical_box_center != null && data.relay_logical_box_center[idx] != null) {
        html += ' &nbsp; <span style="color:#888;font-size:11px;">logical中心 ' + fmt(data.relay_logical_box_center[idx]) + '</span>';
      }
      if (data.last_relay_max != null) {
        var lrm = data.last_relay_max[idx], c = data.close[idx];
        var th = data._thresh != null ? data._thresh[idx] : null;
        var hasThreshold = lrm != null && typeof lrm === 'number';
        html += '<br/><span style="color:#888;font-size:11px;">2.中继高 ' + fmt(lrm) + (th != null ? ' 门限 ' + fmt(th) : '') + '</span>';
        if (data.pattern_2_relay_gid != null && data.pattern_2_relay_gid[idx] != null) {
          html += ' &nbsp; <span style="color:#88a;">2.中继段 gid=' + data.pattern_2_relay_gid[idx] + '</span>';
        }
        if (data.entry_signal_long != null && data.entry_signal_long[idx]) {
          html += ' &nbsp; <span style="color:#8f8;">✓信号</span>';
        } else if (hasThreshold && r === 1) {
          if (th != null && c != null && c <= th) html += ' &nbsp; <span style="color:#a66;">C未突破门限</span>';
          else if (th == null && c != null && lrm != null && c <= lrm) html += ' &nbsp; <span style="color:#a66;">C未突破</span>';
          else html += ' &nbsp; <span style="color:#a66;">非首破/ma2斜率</span>';
        } else if (!hasThreshold && r === 1) {
          var reason = [];
          if (data._pattern_match != null && !data._pattern_match[idx]) {
            reason.push('形态未匹配');
          } else if (data._pattern_match != null && data._pattern_match[idx]) {
            if (data._price_down_ok != null && !data._price_down_ok[idx]) reason.push('下跌价格判定未通过');
            if (data._price_up_ok != null && !data._price_up_ok[idx]) reason.push('上涨价格判定未通过');
            if (data._pattern_ok != null && !data._pattern_ok[idx] && reason.length === 0) reason.push('价格判定未通过');
          }
          if (reason.length > 0) {
            html += ' &nbsp; <span style="color:#a66;">无命中: ' + reason.join(', ') + '</span>';
          } else {
            html += ' &nbsp; <span style="color:#a66;">无命中(下中下上/下中下上中上/下中下中上)</span>';
          }
        }
      }
      tip.innerHTML = html;
      tip.style.display = 'block';
    }
    function moveCrosshair(xVal, closeVal) {
      crosshairV.x0 = crosshairV.x1 = xVal;
      var shapes = [crosshairV];
      if (closeVal != null && typeof closeVal === 'number' && isFinite(closeVal)) {
        crosshairH.y0 = crosshairH.y1 = closeVal;
        shapes.push(crosshairH);
      }
      Plotly.relayout(gd, { shapes: shapes });
    }
    gd.on('plotly_hover', function(ev) {
      if (!ev || !ev.points || !ev.points.length) return;
      var pt = ev.points[0];
      var idx = pt.pointIndex;
      if (typeof idx !== 'number' || idx < 0) return;
      var c = data.close != null && data.close[idx] != null ? Number(data.close[idx]) : null;
      var o = data.open != null && data.open[idx] != null ? Number(data.open[idx]) : null;
      var lineY = (c != null && o != null) ? Math.max(c, o) : (c != null ? c : null);
      moveCrosshair(pt.x, lineY);
      showSlice(idx);
    });
    gd.on('plotly_unhover', function() {
      Plotly.relayout(gd, { shapes: [] });
      tip.style.display = 'none';
    });
  }
  if (document.readyState === 'complete') run();
  else window.addEventListener('load', run);
})();
</script>
"""
    html_path = Path(html_path)
    text = html_path.read_text(encoding="utf-8")
    if "MARKET_REGIME_SLICE_DATA" in text:
        return
    text = text.replace("</body>", js + "\n</body>")
    html_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Market regime analysis on XAUUSD M1")
    parser.add_argument(
        "--csv",
        type=str,
        default="/Users/guonic/.duke/download/download/xauusd-m1-bid-2025-12-01-2025-12-31.csv",
        help="Path to XAUUSD M1 CSV (timestamp, open, high, low, close)",
    )
    parser.add_argument(
        "--tightness",
        type=float,
        default=0.1,
        help="Bundle tightness threshold for relay (default 0.1)",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=0.02,
        help="MA5 slope filter for trend (default 0.02)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate Plotly HTML chart with regime background",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="market_regime_result.html",
        help="Output HTML path when --plot is set",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Use only first N rows (0 = all)",
    )
    parser.add_argument(
        "--break-offset",
        type=float,
        default=0.05,
        help="Breakout confirmation offset in price units (default 0.05 for XAUUSD)",
    )
    parser.add_argument(
        "--no-ma2-slope",
        action="store_true",
        help="Disable MA2 slope filter for entry signal",
    )
    parser.add_argument(
        "--debug-segments",
        action="store_true",
        help="Export regime segment table (group_id, start_time, end_time, high_max) and pattern_2_relay_gid to verify 2.中继 对应哪一段",
    )
    parser.add_argument(
        "--use-relay-high",
        action="store_true",
        help="Use K-line high of 2.中继 (incl. wicks) as threshold; default uses body max(open,close)",
    )
    parser.add_argument(
        "--labeler",
        type=str,
        choices=list(REGIME_LABELERS.keys()),
        default="ma",
        help="Regime labeler: ma (Mid+MA2/MA3/MA5+bundle_tightness) or overlap (K-line body overlap+color switch)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.6,
        help="Body overlap ratio threshold for overlap labeler (default 0.6); >0.7 strong relay",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = load_xauusd_csv(csv_path)
    if args.head > 0:
        df = df.head(args.head)

    # Part 1: state labeling (pluggable)
    labeler_fn = REGIME_LABELERS[args.labeler]
    if args.labeler == "ma":
        df = labeler_fn(
            df,
            high_col="high",
            low_col="low",
            threshold_tightness=args.tightness,
            slope_filter=args.slope,
        )
    else:
        df = labeler_fn(
            df,
            overlap_threshold=args.overlap_threshold,
            containment_threshold=0.7,
            expand_left=True,
            max_expand_left_bars=1,
            min_trend_bars=3,
            min_trend_bars_force=4,
            max_range_expand_ratio=0.2,
            max_center_drift_ratio=0.5,
        )

    # Part 2: trading signal labeling (pattern match + breakout)
    debug_2_relay = args.debug_segments or args.plot
    df = add_validated_breakout_signals(
        df,
        break_offset=args.break_offset,
        use_relay_high=args.use_relay_high,
        require_ma2_slope=not args.no_ma2_slope,
        debug_2_relay=debug_2_relay,
    )
    # 修正中继段的起始和结束时间点（基于斜率变化和绝对中心）
    df = adjust_relay_boundaries(df)

    # Summary stats
    n = len(df)
    r1 = (df["regime"] == 1).sum()
    r0 = (df["regime"] == 0).sum()
    rm1 = (df["regime"] == -1).sum()
    breakout = df["breakout_signal"].sum() if "breakout_signal" in df.columns else 0
    exit_l = df["exit_long"].sum() if "exit_long" in df.columns else 0
    exit_s = df["exit_short"].sum() if "exit_short" in df.columns else 0

    print("Market regime analysis")
    print(f"Labeler: {args.labeler}")
    print(f"File: {csv_path}")
    print(f"Bars: {n}")
    print(f"Regime: long(1)={r1}, relay(0)={r0}, short(-1)={rm1}")
    print(f"Breakout signals: {breakout}")
    print(f"Exit long: {exit_l}, Exit short: {exit_s}")
    entry_long = df["entry_signal_long"].sum()
    print(f"Validated entry long (break relay-after-down): {entry_long}")

    if debug_2_relay:
        seg_table = get_regime_segment_table(
            df, regime_col="regime", datetime_col="datetime", high_col="high"
        )
        out_path = Path(args.out)
        seg_path = out_path.parent / (out_path.stem + "_segments.csv")
        seg_table.to_csv(seg_path, index=False)
        print(f"  Segment table (查 2.中继 对应时段): {seg_path}")
        print("  每行: group_id, regime, regime_name, start_time, end_time, bar_count, high_max.")
        print("  信号行上的 pattern_2_relay_gid 即门限所用的中继段 group_id，在表中可对出 start_time/end_time。")

    if df["regime"].nunique() == 1 and args.plot:
        only_val = int(df["regime"].iloc[0])
        name = {1: "上行", 0: "中继", -1: "下降"}[only_val]
        print(f"  [Plot] 整段 regime 均为 {name}({only_val})，中间状态图会显示为一根直线；可缩小 --head 或调小 --slope 以看到状态切换")

    if args.plot:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Install plotly to use --plot: pip install plotly", file=sys.stderr)
            sys.exit(1)

        x = df["datetime"] if "datetime" in df.columns else df.index

        # K 线区域略大，状态栏与 tightness 留足空间，避免标题/曲线重叠
        h1 = 0.52
        remainder = 1.0 - h1
        h2 = remainder * 0.5
        h3 = remainder * 0.5
        row_heights = [round(h1, 3), round(h2, 3), round(h3, 3)]
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=row_heights,
            subplot_titles=("Price (close)", "上行 / 中继 / 下降", "bundle_tightness"),
        )

        # K 线（蜡烛图）打底
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="K线",
                increasing_line_color="lime",
                decreasing_line_color="orangered",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["close"], name="close", line=dict(color="white", width=1)),
            row=1,
            col=1,
        )
        if "ma2" in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df["ma2"], name="MA2", line=dict(color="gold", width=1), opacity=0.8),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=df["ma3"], name="MA3", line=dict(color="orange", width=1), opacity=0.8),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=df["ma5"], name="MA5", line=dict(color="cyan", width=1), opacity=0.8),
                row=1,
                col=1,
            )
        # Row 2: 上行(1)/中继(0)/下降(-1) — 细线阶梯；信号标在状态栏
        regime = df["regime"].astype(int)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=regime.tolist(),
                name="上行/中继/下降",
                mode="lines",
                line=dict(color="lime", width=1.5, shape="hv"),
                legendgroup="regime",
            ),
            row=2,
            col=1,
        )
        mask_long = df["entry_signal_long"]
        if mask_long.any():
            fig.add_trace(
                go.Scatter(
                    x=x[mask_long],
                    y=[1] * mask_long.sum(),
                    name="entry long",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="lime", line=dict(width=1, color="white")),
                ),
                row=2,
                col=1,
            )
        # 修正后的中继段边界点标记
        if "relay_start_adjusted" in df.columns:
            mask_start = df["relay_start_adjusted"]
            if mask_start.any():
                fig.add_trace(
                    go.Scatter(
                        x=x[mask_start],
                        y=df.loc[mask_start, "close"].tolist(),
                        name="relay start (adjusted)",
                        mode="markers",
                        marker=dict(symbol="square", size=8, color="cyan", line=dict(width=1, color="white")),
                    ),
                    row=1,
                    col=1,
                )
        if "relay_end_adjusted" in df.columns:
            mask_end = df["relay_end_adjusted"]
            if mask_end.any():
                fig.add_trace(
                    go.Scatter(
                        x=x[mask_end],
                        y=df.loc[mask_end, "close"].tolist(),
                        name="relay end (adjusted)",
                        mode="markers",
                        marker=dict(symbol="diamond", size=8, color="magenta", line=dict(width=1, color="white")),
                    ),
                    row=1,
                    col=1,
                )
        # Row 3: 仅 bundle_tightness，确保无其它线
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bundle_tightness"].tolist(),
                name="bundle_tightness",
                mode="lines",
                line=dict(color="orange", width=1),
                legendgroup="tightness",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=720,
            margin=dict(b=60, t=56),
            xaxis_title="Time",
            showlegend=True,
            dragmode="zoom",
            xaxis=dict(
                type="date",
                autorange=True,
                fixedrange=False,
                rangeslider=dict(visible=True, thickness=0.04, bgcolor="rgba(80,80,80,0.3)"),
            ),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(
            title_text="状态",
            row=2,
            col=1,
            tickvals=[-1, 0, 1],
            ticktext=["下降", "中继", "上行"],
            range=[-1.5, 1.5],
            autorange=False,
        )
        fig.update_yaxes(title_text="Tightness", row=3, col=1)
        if df["regime"].nunique() == 1:
            only_val = int(df["regime"].iloc[0])
            name = {1: "上行", 0: "中继", -1: "下降"}[only_val]
            hint = "缩小时间范围或调小 --slope 可见阶梯" if args.labeler == "ma" else "缩小时间范围可见阶梯"
            fig.add_annotation(
                text=f"本段状态恒定({name})，故为一直线；{hint}",
                xref="paper",
                yref="y2 domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=11, color="#888"),
            )

        # Initial y-axis: 有数据 range 占总数 80%，上下留白各 10%
        y_cols = [df["close"]]
        for c in ["ma2", "ma3", "ma5"]:
            if c in df.columns:
                y_cols.append(df[c])
        y_price = pd.concat(y_cols)
        y_lo, y_hi = float(y_price.min()), float(y_price.max())
        span = max(y_hi - y_lo, 0.5)
        margin = span * 0.125
        t_lo = float(df["bundle_tightness"].min())
        t_hi = float(df["bundle_tightness"].max())
        t_span = max(t_hi - t_lo, 0.01)
        fig.update_layout(
            yaxis=dict(range=[y_lo - margin, y_hi + margin], autorange=False),
            yaxis2=dict(range=[-1.5, 1.5], autorange=False),
            yaxis3=dict(range=[t_lo - t_span / 2, t_hi + t_span / 2], autorange=False),
        )

        out_path = Path(args.out)
        fig.write_html(
            str(out_path),
            div_id="market-regime-plot",
            config=dict(
                scrollZoom=True,
                displayModeBar=True,
                modeBarButtonsToAdd=["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
            ),
        )
        _inject_yaxis_autorange_script(out_path, div_id="market-regime-plot")
        df["_thresh"] = df["last_relay_max"] + args.break_offset
        _inject_crosshair_and_slice(out_path, df, div_id="market-regime-plot")
        print(f"Chart saved: {out_path}")
        print("  Open this HTML in browser; y-axis: data range = 80% of total (blank 20%).")
        print("  After zoom/pan, y-axis recalculates from the current window; reopen HTML if range looks wrong.")
        print("  Hover on chart to see slice: 2.中继高, and why no signal (C未突破 / 形态未命中).")
        print("  Search tree patterns: 1.下中下上 2.下中下上中上 3.下中下中上; no signal if not first break / ma2_slope<=0 / regime!=1.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
