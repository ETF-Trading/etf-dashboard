"""
Leveraged ETF Momentum Dashboard
==================================
Mobile-freundliches Streamlit-Dashboard fuer die taegliche Ueberwachung von:
  - Marktphase (Bull/Bear via SMA55)
  - ATR-Regime & DynStop fuer SOXL/FNGU
  - Hurst-Chop-Signal (Fenster=100, Schwelle=0,50)
  - Kreditmarkt-Stress-Signal (HYG/IEF, MA=50, Tiefe=2,5%)
  - Momentum-Scores aller 4 ETFs
  - Konkrete Stop-Kurse fuer SOXL/FNGU

Deployment: Streamlit Community Cloud (siehe Anleitung am Ende der Datei)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── SEITEN-KONFIGURATION (mobile-optimiert) ──────────────────────────────
st.set_page_config(
    page_title="ETF Momentum Dashboard",
    page_icon="📊",
    layout="centered",   # "centered" ist besser fuer schmale Handy-Screens als "wide"
    initial_sidebar_state="collapsed",
)

# ── PARAMETER ─────────────────────────────────────────────────────────────
SMA_PERIOD   = 55
ATR_PERIOD   = 14
DYN_BASE     = 0.022

DS_ATR_LO    = 1.8
DS_ATR_HI    = 3.0
DS_STOP_LO   = 0.21
DS_STOP_MID  = 0.14
DS_STOP_HI   = 0.09

LOCK_RECOV   = 0.08
DIP_TRIGGER  = 0.42
DIP_ALLOC    = 1.00

MTF_WINDOWS  = (10, 21, 63)
MTF_WEIGHTS  = (0.40, 0.40, 0.20)

HURST_WINDOW = 100
HURST_THR    = 0.50

CREDIT_MA    = 50
CREDIT_DEPTH = 0.025

UNIVERSE = ["TQQQ", "SOXL", "FNGU", "UPRO"]


# ── DATEN LADEN (gecached, damit nicht bei jedem Klick neu geladen wird) ──
@st.cache_data(ttl=3600)  # Cache fuer 1 Stunde
def load_data():
    end   = datetime.today()
    start = end - timedelta(days=400)

    def dl(ticker):
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
            return df if not df.empty else None
        except Exception:
            return None

    data = {}
    data["QQQ"] = dl("QQQ")
    data["VIX"] = dl("^VIX")
    data["HYG"] = dl("HYG")
    data["IEF"] = dl("IEF")
    for t in UNIVERSE:
        data[t] = dl(t)
    return data


def hurst_rs(x):
    n = len(x)
    if n < 20:
        return 0.5
    lags = np.unique(np.linspace(8, n // 2, 6).astype(int))
    rs_pts = []
    for lag in lags:
        if lag < 2:
            continue
        nseg = n // lag
        if nseg < 1:
            continue
        rs_seg = []
        for k in range(nseg):
            seg = x[k*lag:(k+1)*lag]
            mean = seg.mean()
            dev = np.cumsum(seg - mean)
            R = dev.max() - dev.min()
            S = seg.std()
            if S > 0:
                rs_seg.append(R / S)
        if rs_seg:
            rs_pts.append((lag, np.mean(rs_seg)))
    if len(rs_pts) < 2:
        return 0.5
    lx = np.log([p[0] for p in rs_pts])
    ly = np.log([p[1] for p in rs_pts])
    slope = np.polyfit(lx, ly, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def calc_stop(series, last_entry, stop_pct, lock_recov):
    since = series.loc[last_entry:] if last_entry in series.index else series.tail(252)
    peak = float(since.max())
    peak_date = since.idxmax()
    cur = float(series.iloc[-1])
    stop = peak * (1 - stop_pct)
    dist = (cur - stop) / cur
    return {
        "cur": cur, "peak": peak, "peak_date": peak_date,
        "stop": stop, "dist": dist, "stopped": cur < stop,
        "reentry": stop * (1 + lock_recov),
    }


# ── HAUPTLOGIK ────────────────────────────────────────────────────────────
def compute_signal():
    data = load_data()
    qqq_df = data["QQQ"]
    if qqq_df is None:
        return None

    qqq = qqq_df["Close"].squeeze()
    qhi = qqq_df["High"].squeeze()
    qlo = qqq_df["Low"].squeeze()
    vix_df = data["VIX"]
    vix = vix_df["Close"].squeeze() if vix_df is not None else pd.Series(15.0, index=qqq.index)

    etf_close = {}
    for t in UNIVERSE:
        if data[t] is not None:
            etf_close[t] = data[t]["Close"].squeeze()

    sma = qqq.rolling(SMA_PERIOD).mean()
    pc = qqq.shift(1)
    tr = pd.concat([qhi - qlo, (qhi - pc).abs(), (qlo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
    atr_pct = (atr / qqq * 100).dropna()

    date = qqq.index[-1]
    qqq_now = float(qqq.iloc[-1])
    sma_now = float(sma.iloc[-1])
    atr_now = float(atr_pct.iloc[-1])
    vix_now = float(vix.reindex(qqq.index).ffill().iloc[-1])
    px_bull = qqq_now >= sma_now

    if atr_now < DS_ATR_LO:
        stop_pct, regime = DS_STOP_LO, "RUHIG"
    elif atr_now > DS_ATR_HI:
        stop_pct, regime = DS_STOP_HI, "VOLATIL"
    else:
        stop_pct, regime = DS_STOP_MID, "NORMAL"

    dyn_pos = min(1.0, DYN_BASE / (atr_now / 100.0)) if atr_now > 0 else 1.0

    # Hurst
    log_ret = np.log(qqq / qqq.shift(1)).dropna().values
    hurst_now = hurst_rs(log_ret[-HURST_WINDOW:]) if len(log_ret) >= HURST_WINDOW else 0.5
    hurst_chop = hurst_now < HURST_THR

    # Credit
    credit_available = data["HYG"] is not None and data["IEF"] is not None
    ratio_now = ma_now = credit_dist_pct = None
    credit_stress = False
    if credit_available:
        hyg = data["HYG"]["Close"].squeeze()
        ief = data["IEF"]["Close"].squeeze()
        ratio = (hyg / ief).reindex(qqq.index).ffill()
        ratio_ma = ratio.rolling(CREDIT_MA).mean()
        ratio_now = float(ratio.iloc[-1])
        ma_now = float(ratio_ma.iloc[-1])
        credit_stress = ratio_now < ma_now * (1 - CREDIT_DEPTH)
        threshold = ma_now * (1 - CREDIT_DEPTH)
        credit_dist_pct = (ratio_now / threshold - 1) * 100 if threshold > 0 else 0

    is_bull = px_bull and not credit_stress

    # Momentum
    ww_total = sum(MTF_WEIGHTS)
    scores = {}
    for name, series in etf_close.items():
        s = 0.0
        for w, lb in zip(MTF_WEIGHTS, MTF_WINDOWS):
            if len(series) > lb:
                p0 = float(series.iloc[-1 - lb])
                p1 = float(series.iloc[-1])
                if p0 > 0:
                    s += (w / ww_total) * (p1 / p0 - 1)
        scores[name] = s
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0] if ranked else "TQQQ"

    # Stop-Kurse
    above = qqq >= sma
    cross = above & ~above.shift(1).fillna(False)
    bull_entries = cross[cross].index
    last_entry = bull_entries[-1] if len(bull_entries) > 0 else qqq.index[0]

    stops = {}
    for name in ["SOXL", "FNGU"]:
        if name in etf_close:
            stops[name] = calc_stop(etf_close[name], last_entry, stop_pct, LOCK_RECOV)

    return dict(
        date=date, qqq_now=qqq_now, sma_now=sma_now, atr_now=atr_now, vix_now=vix_now,
        px_bull=px_bull, is_bull=is_bull, stop_pct=stop_pct, regime=regime, dyn_pos=dyn_pos,
        hurst_now=hurst_now, hurst_chop=hurst_chop,
        credit_available=credit_available, ratio_now=ratio_now, ma_now=ma_now,
        credit_dist_pct=credit_dist_pct, credit_stress=credit_stress,
        scores=scores, ranked=ranked, best=best, etf_close=etf_close,
        last_entry=last_entry, stops=stops,
    )


# ── UI ─────────────────────────────────────────────────────────────────────

st.title("📊 ETF Momentum Dashboard")

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🔄 Aktualisieren"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Lade Marktdaten..."):
    sig = compute_signal()

if sig is None:
    st.error("⚠️ Daten konnten nicht geladen werden. Bitte später erneut versuchen.")
    st.stop()

st.caption(f"Stand: {sig['date'].strftime('%A, %d.%m.%Y')}  ·  "
           f"Letztes Update der Daten: {datetime.now().strftime('%H:%M')}")

st.divider()

# ── HAUPT-STATUS (groß, auf einen Blick) ─────────────────────────────────
if sig["is_bull"] and not sig["hurst_chop"]:
    st.success(f"### 🟢 BULL — Aktiv investiert\n**{sig['best']}** halten/kaufen "
               f"({sig['dyn_pos']:.0%} Portfolio)")
elif sig["is_bull"] and sig["hurst_chop"]:
    st.warning(f"### 🟡 BULL + CHOP — Defensiv investiert\n"
               f"**QQQ (1×, unleveraged)** statt Hebel-ETF "
               f"({sig['dyn_pos']:.0%} Portfolio)")
else:
    reasons = []
    if not sig["px_bull"]:
        reasons.append("Preis unter SMA55")
    if sig["credit_stress"]:
        reasons.append("Kreditmarkt-Stress")
    st.error(f"### 🔴 BEAR — Cash halten\nGrund: {', '.join(reasons)}")

st.divider()

# ── KENNZAHLEN-GRID ───────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.metric("QQQ", f"${sig['qqq_now']:.2f}",
              f"{(sig['qqq_now']/sig['sma_now']-1)*100:+.1f}% vs SMA55")
with c2:
    st.metric("VIX", f"{sig['vix_now']:.1f}")

c3, c4 = st.columns(2)
with c3:
    st.metric("ATR-Regime", sig["regime"], f"ATR {sig['atr_now']:.2f}%")
with c4:
    st.metric("DynStop", f"{int(sig['stop_pct']*100)}%", f"DynPos {sig['dyn_pos']:.0%}")

st.divider()

# ── HURST-CHOP ─────────────────────────────────────────────────────────────
st.subheader("🌊 Hurst-Chop-Signal")
hc1, hc2 = st.columns(2)
with hc1:
    st.metric("Hurst-Exponent", f"{sig['hurst_now']:.3f}",
              f"Schwelle {HURST_THR}")
with hc2:
    if sig["hurst_chop"]:
        st.markdown("**Status:** 🟡 CHOP AKTIV")
        st.caption("→ Wechsel zu unleveraged QQQ")
    else:
        st.markdown("**Status:** 🟢 Trend intakt")
        st.caption("→ normale Momentum-Rotation")

st.divider()

# ── KREDITMARKT-SIGNAL ──────────────────────────────────────────────────────
st.subheader("🏦 Kreditmarkt-Signal (HYG/IEF)")
if sig["credit_available"]:
    cc1, cc2 = st.columns(2)
    with cc1:
        st.metric("HYG/IEF Ratio", f"{sig['ratio_now']:.4f}",
                  f"{sig['credit_dist_pct']:+.2f}% vs. Schwelle")
    with cc2:
        if sig["credit_stress"]:
            st.markdown("**Status:** 🔴 STRESS AKTIV")
            st.caption("→ Bull-Wiedereinstieg gesperrt")
        else:
            st.markdown("**Status:** 🟢 Kein Stress")
            st.caption("→ kein Veto")
    st.caption(f"{CREDIT_MA}-Tage-MA: {sig['ma_now']:.4f}  ·  "
               f"Schwelle: {sig['ma_now']*(1-CREDIT_DEPTH):.4f}")
else:
    st.info("HYG/IEF nicht verfügbar — Signal neutral")

st.divider()

# ── MOMENTUM-SCORES ────────────────────────────────────────────────────────
st.subheader("📈 Momentum-Scores")
rows = []
for rank, (name, score) in enumerate(sig["ranked"], 1):
    if name not in sig["etf_close"]:
        continue
    s = sig["etf_close"][name]
    p = float(s.iloc[-1])
    d10 = (p / float(s.iloc[-11]) - 1) * 100 if len(s) > 10 else 0
    d21 = (p / float(s.iloc[-22]) - 1) * 100 if len(s) > 21 else 0
    d63 = (p / float(s.iloc[-64]) - 1) * 100 if len(s) > 63 else 0
    marker = "👑" if name == sig["best"] else ""
    rows.append({
        "Rang": rank, "ETF": f"{marker} {name}", "Score": f"{score:+.4f}",
        "Kurs": f"${p:.2f}", "10d": f"{d10:+.1f}%",
        "21d": f"{d21:+.1f}%", "63d": f"{d63:+.1f}%",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()

# ── TRAILING STOPS ──────────────────────────────────────────────────────────
st.subheader("🛑 Trailing Stops (SOXL / FNGU)")
st.caption(f"Peak-Berechnung seit Bull-Entry: {sig['last_entry'].strftime('%d.%m.%Y')}")

for name, info in sig["stops"].items():
    with st.container(border=True):
        st.markdown(f"**{name}**")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Peak", f"${info['peak']:.2f}")
        with s2:
            st.metric("Aktuell", f"${info['cur']:.2f}")
        with s3:
            st.metric("Stop", f"${info['stop']:.2f}")
        if info["stopped"]:
            st.error(f"⚠️ UNTER STOP! Re-Entry erst ab ${info['reentry']:.2f}")
        else:
            st.success(f"✅ Puffer: {info['dist']*100:+.1f}%")

st.divider()
st.caption(
    f"Parameter: SMA{SMA_PERIOD} · DynStop {int(DS_STOP_LO*100)}/{int(DS_STOP_MID*100)}/"
    f"{int(DS_STOP_HI*100)}% @ ATR {DS_ATR_LO}/{DS_ATR_HI}% · "
    f"Dip {int(DIP_TRIGGER*100)}%/{int(DIP_ALLOC*100)}% · "
    f"Hurst(w{HURST_WINDOW},t{HURST_THR}) · Credit(ma{CREDIT_MA},d{CREDIT_DEPTH*100:.1f}%)"
)
st.caption("⚠️ Keine Anlageberatung. Nur für private Nutzung.")
