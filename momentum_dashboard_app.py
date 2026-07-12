"""
Leveraged ETF Momentum System - Baseline Version
================================================
Parameters:
  - Fast Momentum Lookback: 11 Days (Weight: 0.75)
  - Slow Momentum Lookback: 48 Days (Weight: 0.25)
  - Universe: TQQQ, SOXL, FNGU, UPRO
  - Regime Filters: SMA55, Hurst Ensemble, Credit Veto (HYG/IEF), VIX Volatility
  - Risk Management: Dynamic Position Sizing (ATR), Trailing Stops
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "yfinance", "curl_cffi"], check=False)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings, time

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── SYSTEM CONFIGURATION ──────────────────────────────────────────────────
START_DATE = "1998-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

TRADING_COST = 0.0025  # 0.25% Spread + Slippage + Fee
TRACKING_ERR = 0.0012  # Daily Tracking Error Noise
ER_COST = {"TQQQ": 0.0086, "SOXL": 0.0095, "FNGU": 0.0095, "UPRO": 0.0091}
ETF_LIST = ["TQQQ", "SOXL", "FNGU", "UPRO"]

# Baseline Momentum Setup
MOM_LOOKBACKS = [11, 48]
MOM_WEIGHTS   = [0.75, 0.25]

REQUIRED_TICKERS = {
    "QQQ": "QQQ", "NDX": "^NDX", "SPY": "SPY", "SOXX": "SOXX", "SMH": "SMH",
    "VIX": "^VIX", "TQQQ": "TQQQ", "SOXL": "SOXL", "UPRO": "UPRO",
    "HYG": "HYG", "IEF": "IEF",
    "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL",
    "AMZN": "AMZN", "NVDA": "NVDA", "META": "META", "TSLA": "TSLA",
}

# ── 1. DATA DOWNLOADING & PREPARATION ─────────────────────────────────────
print("1/4: Lade und bereinige Marktdaten...")
close_data, high_data, low_data = {}, {}, {}

def yf_download_retry(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df is not None and not df.empty: 
                return df
        except Exception: 
            pass
        time.sleep(1)
    return None

for name, ticker in REQUIRED_TICKERS.items():
    df = yf_download_retry(ticker, START_DATE, END_DATE)
    if df is not None and not df.empty:
        close_data[name] = df["Close"].squeeze()
        if "High" in df: high_data[name] = df["High"].squeeze()
        if "Low"  in df: low_data[name]  = df["Low"].squeeze()

RC = pd.DataFrame(close_data).ffill()
RH = pd.DataFrame(high_data).ffill()
RL = pd.DataFrame(low_data).ffill()

# History Extension QQQ via NDX
if "NDX" in RC and "QQQ" in RC:
    fvi = RC["QQQ"].first_valid_index()
    if fvi:
        ratio = RC["QQQ"].loc[fvi] / RC["NDX"].loc[fvi]
        RC["QQQ"] = RC["QQQ"].combine_first(RC["NDX"] * ratio)

# History Extension Semi Sector
RC["SEMI"] = RC.get("SMH", RC["QQQ"])
if "SOXX" in RC and "SMH" in RC:
    fvi = RC["SOXX"].first_valid_index()
    if fvi:
        ratio = RC["SOXX"].loc[fvi] / RC["SMH"].loc[fvi]
        RC["SEMI"] = RC["SOXX"].combine_first(RC["SMH"] * ratio)

mag7 = [c for c in ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA"] if c in RC]
mag7p = (1 + RC[mag7].pct_change().mean(axis=1)).cumprod() * 100

def synth3x(base, er, real=None, real_from=None, seed=0):
    np.random.seed(seed)
    r = base.pct_change().fillna(0)
    lev = 3 * r - er / 252 + np.random.normal(0, TRACKING_ERR, len(r))
    s = pd.Series((1 + lev).cumprod() * 100, index=base.index)
    if real is not None and real_from is not None:
        ra = real.reindex(s.index).ffill()
        fvi = ra.first_valid_index()
        if fvi and fvi in s.index: 
            s *= ra.loc[fvi] / s.loc[fvi]
        mask = ra.notna() & (ra.index >= pd.Timestamp(real_from))
        s[mask] = ra[mask]
    return s

def reb_flags(idx, wd=3): # Rebalance Wednesday
    f = pd.Series(False, index=idx)
    wg = {}
    for d in idx:
        k = (d.year, d.isocalendar()[1])
        if k not in wg: wg[k] = []
        wg[k].append(d)
    for k, wdl in wg.items():
        ww = [d.weekday() for d in wdl]
        if wd in ww: 
            f[wdl[ww.index(wd)]] = True
        else:
            later = [d for d in wdl if d.weekday() > wd]
            if later: f[later[0]] = True
            elif wdl: f[wdl[-1]] = True
    return f

ETFS = pd.DataFrame({
    "TQQQ": synth3x(RC["QQQ"], ER_COST["TQQQ"], RC.get("TQQQ"), "2010-02-11", 1),
    "SOXL": synth3x(RC["SEMI"], ER_COST["SOXL"], RC.get("SOXL"), "2010-03-03", 2),
    "FNGU": synth3x(mag7p, ER_COST["FNGU"], seed=3),
    "UPRO": synth3x(RC["SPY"], ER_COST["UPRO"], RC.get("UPRO"), "2009-06-25", 4),
})
VIX_S = RC.get("VIX", pd.Series(15.0, index=RC.index))
DF    = pd.DataFrame({"QQQ": RC["QQQ"], "VIX": VIX_S}).join(ETFS, how="inner")
DO    = reb_flags(DF.index, 3)
N     = len(DF)

qqqp = DF["QQQ"].values
qqqh_raw = RH.get("QQQ", RC["QQQ"]).reindex(DF.index).ffill().values
qqql_raw = RL.get("QQQ", RC["QQQ"]).reindex(DF.index).ffill().values
qqqh = np.where(np.isnan(qqqh_raw), qqqp, qqqh_raw)
qqql = np.where(np.isnan(qqql_raw), qqqp, qqql_raw)

# ── 2. REGIME & INDICATOR CALCULATIONS ────────────────────────────────────
print("2/4: Berechne Filter & Indikatoren...")

# Credit Signal Veto
CREDIT = (RC["HYG"] / RC["IEF"]).reindex(DF.index).ffill() if ("HYG" in RC and "IEF" in RC) else pd.Series(1.0, index=DF.index)
cma = CREDIT.rolling(50).mean()
credit_arr = (CREDIT < cma * (1 - 0.025)).fillna(False).values

# Hurst Exponent Ensemble (Choppiness)
log_ret = np.log(DF["QQQ"] / DF["QQQ"].shift(1)).fillna(0).values
def hurst_rs(x):
    n = len(x)
    if n < 20: return 0.5
    lags = np.unique(np.linspace(8, n // 2, 6).astype(int))
    rs_pts = []
    for lag in lags:
        if lag < 2: continue
        nseg = n // lag
        rs_seg = []
        for k in range(nseg):
            seg = x[k*lag:(k+1)*lag]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std()
            if S > 0: rs_seg.append(R / S)
        if rs_seg: rs_pts.append((lag, np.mean(rs_seg)))
    if len(rs_pts) < 2: return 0.5
    lx = np.log([p[0] for p in rs_pts])
    ly = np.log([p[1] for p in rs_pts])
    return float(np.clip(np.polyfit(lx, ly, 1)[0], 0.0, 1.0))

h1 = np.nan_to_num(pd.Series(log_ret).rolling(100).apply(hurst_rs, raw=True).values, nan=0.5)
h2 = np.nan_to_num(pd.Series(log_ret).rolling(30).apply(hurst_rs, raw=True).values, nan=0.5)
chop = (h1 < 0.50) & (h2 < 0.46)

vix_arr = DF["VIX"].fillna(15.0).values
etf_arr = {e: DF[e].values for e in ETF_LIST if e in DF.columns}
do_arr  = DO.reindex(DF.index).values.astype(bool)

# Trend & Volatility (SMA55 + ATR)
cs = np.concatenate([[0.0], np.cumsum(qqqp)])
sma = np.full(N, np.nan)
sma[54:] = (cs[55:] - cs[:N-54]) / 55

pc = np.concatenate([[qqqp[0]], qqqp[:-1]])
tr = np.maximum(qqqh - qqql, np.maximum(np.abs(qqqh - pc), np.abs(qqql - pc)))
al = 2.0 / 15.0
atr = np.zeros(N)
atr[0] = tr[0]
for i in range(1, N): atr[i] = al * tr[i] + (1 - al) * atr[i-1]
atr_pct = np.where(qqqp > 0, atr / qqqp * 100.0, 2.0)

assets = dict(etf_arr)
assets["QQQ1X"] = qqqp

# ── 3. CORE SYSTEM SIMULATION ─────────────────────────────────────────────
print("3/4: Führe Simulation aus...")

def get_momentum(e, i):
    arr = assets.get(e)
    if arr is None: return -99.0
    s = 0.0
    for w_, lb_ in zip(MOM_WEIGHTS, MOM_LOOKBACKS):
        j = max(0, i - lb_)
        p0 = arr[j]; p1 = arr[i]
        if p0 > 0: s += w_ * (p1 / p0 - 1)
    return s

def dyn_position(i):
    a = atr_pct[i]
    return min(1.0, 0.022 / (a / 100.0)) if a > 0 else 1.0

def exit_stop_pct(i):
    a = atr_pct[i]
    if a < 1.8: return 0.21
    if a > 3.0: return 0.09
    return 0.14

def get_vix_regime(i, vx):
    a = atr_pct[i]
    if a > 5.5: return 0
    if a > 2.5:
        if vx >= 40: return 1
        if vx >= 28: return 2
        if vx >= 20: return 3
    return 4

def get_best_asset(i, soxl_stopped_out):
    cands = [e for e in ETF_LIST if not (e == "SOXL" and soxl_stopped_out)]
    if not cands: cands = ["TQQQ"]
    sc = {e: get_momentum(e, i) for e in cands}
    neg = all(x < 0 for x in sc.values())
    rnk = sorted(sc.items(), key=lambda x: x[1], reverse=True)
    return rnk, sc, neg

port = 1.0; pos = {}; cur = None; trail = {}
phase = "BEAR"; bear_entry_px = None; dip_bought = False; soxl_stop = None
portfolio_history = np.ones(N)

def apply_trade(new_pos):
    global port, pos, cur
    turnover = sum(abs(new_pos.get(k, 0) - pos.get(k, 0)) for k in set(pos) | set(new_pos))
    port *= (1 - turnover * TRADING_COST * 0.5)
    pos = {k: w for k, w in new_pos.items() if abs(w) > 1e-4}
    cur = max(pos, key=pos.get) if pos else None

for i in range(N):
    q = qqqp[i]; sm = sma[i]; vx = vix_arr[i] if not np.isnan(vix_arr[i]) else 15.0

    if i > 0:
        port *= 1.0 + sum(pos.get(e, 0) * (assets[e][i] / assets[e][i-1] - 1)
                          for e in pos if e in assets and assets[e][i-1] > 0
                          and abs(pos.get(e, 0)) > 1e-5)
    portfolio_history[i] = max(port, 0.001)

    soxl_sl_active = False
    if soxl_stop and "SOXL" in assets:
        cp = assets["SOXL"][i]
        if cp > 0:
            if cp / soxl_stop - 1 >= 0.08: soxl_stop = None
            else: soxl_sl_active = True

    px_bull = (q >= sm) if not np.isnan(sm) else True
    bull = px_bull and not bool(credit_arr[i])

    if phase != "BEAR" and not bull:
        bear_entry_px = q; dip_bought = False; trail = {}; apply_trade({}); phase = "BEAR"

    if phase == "BEAR":
        if not bull and bear_entry_px and not dip_bought:
            if q / bear_entry_px - 1 <= -0.42:
                apply_trade({"TQQQ": 1.00}); dip_bought = True
        if bull:
            ps = dyn_position(i)
            rnk, sc, neg = get_best_asset(i, soxl_sl_active)
            best_e = rnk[0][0]
            apply_trade({best_e: min(1.0, ps * (0.5 if neg else 1.0))})
            phase = "BULL"; trail = {best_e: assets.get(best_e, portfolio_history)[i]}; dip_bought = False
        continue

    phase = "BULL"; ps = dyn_position(i); sp = exit_stop_pct(i)

    if chop[i]:
        target = {"QQQ1X": min(1.0, ps)}
        if target != pos: apply_trade(target)
        continue

    reg = get_vix_regime(i, vx)
    if reg == 0:
        target = {"TQQQ": min(1.0, ps)}
        if target != pos: apply_trade(target)
        continue
    if reg == 1:
        target = {"TQQQ": min(0.60, ps * 0.60)}
        if target != pos: apply_trade(target)
        continue
    if reg in (2, 3):
        target = {"TQQQ": min(1.0, ps)}
        if abs(pos.get("TQQQ", 0) - target.get("TQQQ", 0)) > 0.05: apply_trade(target)
        continue

    stop_hit = False
    for et in ["SOXL", "FNGU"]:
        if pos.get(et, 0) > 0.05 and et in assets:
            cp = assets[et][i]
            if cp > 0:
                trail[et] = max(trail.get(et, cp), cp)
                if cp / trail[et] - 1 <= -sp:
                    apply_trade({"TQQQ": min(1.0, ps)})
                    trail = {"TQQQ": assets.get("TQQQ", portfolio_history)[i]}
                    if et == "SOXL": soxl_stop = cp
                    stop_hit = True; break
    if stop_hit: continue

    if bool(do_arr[i]) or cur is None:
        rnk, sc, neg = get_best_asset(i, soxl_sl_active)
        best_e = rnk[0][0]
        if cur and cur != best_e:
            thr = 0.02 if best_e == "SOXL" else 0.05
            if sc.get(best_e, 0) - sc.get(cur, -99) < thr: best_e = cur
        alloc = min(1.0, max(0.0, ps * (0.5 if neg else 1.0)))
        target = {best_e: alloc}
        if target != pos:
            apply_trade(target)
            trail[best_e] = assets.get(best_e, portfolio_history)[i]

# ── 4. PERFORMANCE EVALUATION ─────────────────────────────────────────────
print("4/4: Berechne Performance-Kennzahlen...\n")

years = (N - 1) / 252.0
final_cagr = (portfolio_history[-1] / portfolio_history[0]) ** (1 / years) - 1
daily_returns = portfolio_history[1:] / portfolio_history[:-1] - 1
sharpe_ratio = (daily_returns.mean() - 0.02 / 252) / daily_returns.std() * np.sqrt(252)
drawdown = portfolio_history / np.maximum.accumulate(portfolio_history) - 1
max_drawdown = drawdown.min()

print("=" * 60)
print("     LEVERAGED ETF SYSTEM - NEUE BASELINE (11 / 48)")
print("=" * 60)
print(f" Testzeitraum:          {START_DATE} bis {END_DATE} ({years:.1f} Jahre)")
print(f" Lookback-Fenster:      Fast = 11 Tage (75%) | Slow = 48 Tage (25%)")
print("-" * 60)
print(f" CAGR (Jahresrendite):  {final_cagr * 100:.2f}%")
print(f" Sharpe Ratio:          {sharpe_ratio:.3f}")
print(f" Maximaler Drawdown:    {max_drawdown * 100:.2f}%")
print("=" * 60)
