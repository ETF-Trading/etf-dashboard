"""
Leveraged ETF Momentum Dashboard
Finale Baseline: SMA55 | DynStop | DipAlloc100% | DynPos 0.022
               Hurst-Ensemble w100/t0.50 AND w30/t0.46
               Credit-Veto HYG/IEF ma50/d2.5%
               atr_lo=2.5% | atr_hi=5.5%
               Momentum-Rotation: 10/21/63 mit (0.50, 0.30, 0.20)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ETF Momentum Dashboard",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── PARAMETER ──────────────────────────────────────────────────────────────
SMA_PERIOD   = 55
ATR_PERIOD   = 14
ATR_LO       = 2.5
ATR_HI       = 5.5
DYN_BASE     = 0.022

DS_ATR_LO    = 1.8
DS_ATR_HI    = 3.0
DS_STOP_LO   = 0.21
DS_STOP_MID  = 0.14
DS_STOP_HI   = 0.09

LOCK_RECOV   = 0.08
DIP_TRIGGER  = 0.42
DIP_ALLOC    = 1.00

# Neue Baseline Momentum-Gewichtung
MTF_WINDOWS  = (10, 21, 63)
MTF_WEIGHTS  = (0.50, 0.30, 0.20)

HURST_W1, HURST_T1 = 100, 0.50
HURST_W2, HURST_T2 = 30, 0.46

CREDIT_MA    = 50
CREDIT_DEPTH = 0.025

UNIVERSE = ["TQQQ", "SOXL", "FNGU", "UPRO"]


# ── DATEN (gecacht, 1h) ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    end   = datetime.today()
    start = end - timedelta(days=450)

    def dl(ticker):
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
            return df if not df.empty else None
        except:
            return None

    data = {}
    for key in ["QQQ", "^VIX", "HYG", "IEF"] + UNIVERSE:
        data[key] = dl(key)
    return data


def hurst_rs(x):
    n = len(x)
    if n < 20: return 0.5
    lags = np.unique(np.linspace(8, n // 2, 6).astype(int))
    rs_pts = []
    for lag in lags:
        if lag < 2: continue
        nseg = n // lag
        if nseg < 1: continue
        rs_seg = []
        for k in range(nseg):
            seg = x[k*lag:(k+1)*lag]
            mean = seg.mean(); dev = np.cumsum(seg - mean)
            R = dev.max() - dev.min(); S = seg.std()
            if S > 0: rs_seg.append(R / S)
        if rs_seg: rs_pts.append((lag, np.mean(rs_seg)))
    if len(rs_pts) < 2: return 0.5
    lx = np.log([p[0] for p in rs_pts])
    ly = np.log([p[1] for p in rs_pts])
    return float(np.clip(np.polyfit(lx, ly, 1)[0], 0.0, 1.0))


def compute_signal():
    data = load_data()
    qqq_df = data["QQQ"]
    if qqq_df is None:
        return None

    qqq = qqq_df["Close"].squeeze()
    qhi_raw = qqq_df["High"].squeeze() if "High" in qqq_df.columns else qqq
    qlo_raw = qqq_df["Low"].squeeze()  if "Low"  in qqq_df.columns else qqq
    qhi = qhi_raw.where(qhi_raw.notna(), qqq)
    qlo = qlo_raw.where(qlo_raw.notna(), qqq)

    vix_df = data["^VIX"]
    vix = vix_df["Close"].squeeze() if vix_df is not None else pd.Series(15.0, index=qqq.index)

    etf_close = {}
    for t in UNIVERSE:
        if data[t] is not None:
            etf_close[t] = data[t]["Close"].squeeze()

    # SMA & ATR
    sma = qqq.rolling(SMA_PERIOD).mean()
    pc  = qqq.shift(1)
    tr  = pd.concat([qhi - qlo, (qhi - pc).abs(), (qlo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
    atr_pct = (atr / qqq * 100).fillna(2.0)

    date    = qqq.index[-1]
    qqq_now = float(qqq.iloc[-1])
    sma_now = float(sma.iloc[-1])
    atr_now = float(atr_pct.iloc[-1])
    vix_now = float(vix.reindex(qqq.index).ffill().iloc[-1])
    px_bull = qqq_now >= sma_now

    # DynStop-Regime
    if atr_now < DS_ATR_LO:
        stop_pct, stop_regime = DS_STOP_LO, "RUHIG"
    elif atr_now > DS_ATR_HI:
        stop_pct, stop_regime = DS_STOP_HI, "VOLATIL"
    else:
        stop_pct, stop_regime = DS_STOP_MID, "NORMAL"

    # ATR-Hauptregime
    if atr_now > ATR_HI:
        atr_regime = "DEFENSIV"
    elif atr_now > ATR_LO:
        atr_regime = "ERHÖHT"
    else:
        atr_regime = "AGGRESSIV"

    dyn_pos = min(1.0, DYN_BASE / (atr_now / 100.0)) if atr_now > 0 else 1.0

    # Hurst-Ensemble
    log_ret = np.log(qqq / qqq.shift(1)).dropna().values
    h1 = hurst_rs(log_ret[-HURST_W1:]) if len(log_ret) >= HURST_W1 else 0.5
    h2 = hurst_rs(log_ret[-HURST_W2:]) if len(log_ret) >= HURST_W2 else 0.5
    hurst_chop = (h1 < HURST_T1) and (h2 < HURST_T2)

    # Kredit-Signal
    credit_ok = data["HYG"] is not None and data["IEF"] is not None
    ratio_now = ma_now = credit_dist = None
    credit_stress = False
    if credit_ok:
        hyg   = data["HYG"]["Close"].squeeze()
        ief   = data["IEF"]["Close"].squeeze()
        ratio = (hyg / ief).reindex(qqq.index).ffill()
        ratio_ma  = ratio.rolling(CREDIT_MA).mean()
        ratio_now = float(ratio.iloc[-1])
        ma_now    = float(ratio_ma.iloc[-1])
        threshold = ma_now * (1 - CREDIT_DEPTH)
        credit_stress = ratio_now < threshold
        credit_dist   = (ratio_now / threshold - 1) * 100 if threshold > 0 else 0

    is_bull = px_bull and not credit_stress

    # Momentum-Scores
    ww_total = sum(MTF_WEIGHTS)
    scores = {}
    for name, series in etf_close.items():
        s = 0.0
        for w, lb in zip(MTF_WEIGHTS, MTF_WINDOWS):
            if len(series) > lb:
                p0 = float(series.iloc[-1 - lb])
                p1 = float(series.iloc[-1])
                if p0 > 0: s += (w / ww_total) * (p1 / p0 - 1)
        scores[name] = s
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best   = ranked[0][0] if ranked else "TQQQ"

    # Trailing Stop
    above       = qqq >= sma
    cross       = above & ~above.shift(1).fillna(False)
    bull_entries = cross[cross].index
    last_entry   = bull_entries[-1] if len(bull_entries) > 0 else qqq.index[0]

    def calc_stop(name):
        if name not in etf_close: return None
        s     = etf_close[name]
        since = s.loc[last_entry:] if last_entry in s.index else s.tail(252)
        peak  = float(since.max()); peak_d = since.idxmax()
        cur   = float(s.iloc[-1]); stop = peak * (1 - stop_pct)
        return {"cur": cur, "peak": peak, "peak_date": peak_d,
                "stop": stop, "dist": (cur - stop) / cur,
                "stopped": cur < stop, "reentry": stop * (1 + LOCK_RECOV)}

    return dict(
        date=date, qqq_now=qqq_now, sma_now=sma_now,
        atr_now=atr_now, atr_regime=atr_regime, vix_now=vix_now,
        px_bull=px_bull, is_bull=is_bull,
        stop_pct=stop_pct, stop_regime=stop_regime, dyn_pos=dyn_pos,
        h1=h1, h2=h2, hurst_chop=hurst_chop,
        credit_ok=credit_ok, ratio_now=ratio_now, ma_now=ma_now,
        credit_dist=credit_dist, credit_stress=credit_stress,
        scores=scores, ranked=ranked, best=best, etf_close=etf_close,
        last_entry=last_entry,
        stops={"SOXL": calc_stop("SOXL"), "FNGU": calc_stop("FNGU")},
    )


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("📊 ETF Momentum Dashboard")

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🔄 Aktualisieren"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Lade Marktdaten …"):
    sig = compute_signal()

if sig is None:
    st.error("⚠️ Daten nicht verfügbar. Bitte später erneut versuchen.")
    st.stop()

st.caption(f"Stand: {sig['date'].strftime('%A, %d.%m.%Y')}  ·  "
           f"Aktualisiert: {datetime.now().strftime('%H:%M')}")

st.divider()

# ── HAUPT-STATUS ───────────────────────────────────────────────────────────
if sig["is_bull"] and not sig["hurst_chop"]:
    st.success(f"### 🟢 BULL — {sig['best']} kaufen\n"
               f"Allokation: **{sig['dyn_pos']:.0%}** des Portfolios")
elif sig["is_bull"] and sig["hurst_chop"]:
    st.warning(f"### 🟡 BULL + CHOP — QQQ (1×) kaufen\n"
               f"Hurst-Ensemble: Chop aktiv → kein Hebel-ETF  "
               f"({sig['dyn_pos']:.0%} Portfolio)")
else:
    reasons = []
    if not sig["px_bull"]: reasons.append(f"Preis unter SMA{SMA_PERIOD}")
    if sig["credit_stress"]: reasons.append("Kredit-Veto")
    st.error(f"### 🔴 BEAR — Cash halten\nGrund: {', '.join(reasons)}")

st.divider()

# ── KENNZAHLEN ─────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.metric("QQQ", f"${sig['qqq_now']:.2f}",
              f"{(sig['qqq_now']/sig['sma_now']-1)*100:+.1f}% vs SMA{SMA_PERIOD}")
with c2:
    st.metric("VIX", f"{sig['vix_now']:.1f}")

c3, c4 = st.columns(2)
with c3:
    st.metric("ATR-Regime", sig["atr_regime"], f"ATR {sig['atr_now']:.2f}%")
with c4:
    st.metric("DynStop", f"{int(sig['stop_pct']*100)}% ({sig['stop_regime']})",
              f"DynPos {sig['dyn_pos']:.0%}")

st.divider()

# ── HURST-ENSEMBLE ─────────────────────────────────────────────────────────
st.subheader("🌊 Hurst-Ensemble")
st.caption(f"AND-Logik: w{HURST_W1}/t{HURST_T1} UND w{HURST_W2}/t{HURST_T2}")

hc1, hc2, hc3 = st.columns(3)
with hc1:
    delta1 = f"< {HURST_T1} → Chop" if sig["h1"] < HURST_T1 else f"≥ {HURST_T1} → Trend"
    st.metric(f"H{HURST_W1} (lang)", f"{sig['h1']:.3f}", delta1)
with hc2:
    delta2 = f"< {HURST_T2} → Chop" if sig["h2"] < HURST_T2 else f"≥ {HURST_T2} → Trend"
    st.metric(f"H{HURST_W2} (kurz)", f"{sig['h2']:.3f}", delta2)
with hc3:
    if sig["hurst_chop"]:
        st.markdown("**Ensemble:**\n🟡 CHOP AKTIV")
    else:
        st.markdown("**Ensemble:**\n🟢 Trend intakt")

st.divider()

# ── KREDIT-SIGNAL ──────────────────────────────────────────────────────────
st.subheader("🏦 Kredit-Signal (HYG/IEF)")
st.caption(f"MA{CREDIT_MA} × {1-CREDIT_DEPTH:.3f} — Kreditstress = Bull-Veto")

if sig["credit_ok"]:
    cc1, cc2 = st.columns(2)
    with cc1:
        st.metric("HYG/IEF Ratio", f"{sig['ratio_now']:.4f}",
                  f"{sig['credit_dist']:+.2f}% vs. Schwelle")
    with cc2:
        if sig["credit_stress"]:
            st.markdown("**Status:** 🔴 STRESS — Bull-Veto aktiv")
        else:
            st.markdown("**Status:** 🟢 Kein Stress")
    st.caption(f"MA{CREDIT_MA}: {sig['ma_now']:.4f}  ·  "
               f"Schwelle: {sig['ma_now']*(1-CREDIT_DEPTH):.4f}")
else:
    st.info("HYG/IEF nicht verfügbar — Signal neutral")

st.divider()

# ── MOMENTUM-SCORES ────────────────────────────────────────────────────────
st.subheader("📈 Momentum-Scores")
rows = []
for rank, (name, score) in enumerate(sig["ranked"], 1):
    if name not in sig["etf_close"]: continue
    s   = sig["etf_close"][name]; p = float(s.iloc[-1])
    d10 = (p/float(s.iloc[-11])-1)*100 if len(s)>10 else 0
    d21 = (p/float(s.iloc[-22])-1)*100 if len(s)>21 else 0
    d63 = (p/float(s.iloc[-64])-1)*100 if len(s)>63 else 0
    marker = "👑" if name == sig["best"] else ""
    rows.append({"Rang": rank, "ETF": f"{marker} {name}",
                 "Score": f"{score:+.4f}", "Kurs": f"${p:.2f}",
                 "10d": f"{d10:+.1f}%", "21d": f"{d21:+.1f}%", "63d": f"{d63:+.1f}%"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()

# ── TRAILING STOPS ─────────────────────────────────────────────────────────
st.subheader("🛑 Trailing Stops (SOXL / FNGU)")
st.caption(f"Peak seit Bull-Entry: {sig['last_entry'].strftime('%d.%m.%Y')}")

for name, info in sig["stops"].items():
    if info is None: continue
    with st.container(border=True):
        st.markdown(f"**{name}**")
        s1, s2, s3 = st.columns(3)
        with s1: st.metric("Peak",    f"${info['peak']:.2f}")
        with s2: st.metric("Aktuell", f"${info['cur']:.2f}")
        with s3: st.metric("Stop",    f"${info['stop']:.2f}")
        if info["stopped"]:
            st.error(f"⚠️ UNTER STOP! Re-Entry erst ab ${info['reentry']:.2f}")
        else:
            st.success(f"✅ Puffer: {info['dist']*100:+.1f}%")

st.divider()
st.caption(
    f"SMA{SMA_PERIOD} · ATR-lo/hi {ATR_LO}/{ATR_HI}% · "
    f"DynStop {int(DS_STOP_LO*100)}/{int(DS_STOP_MID*100)}/{int(DS_STOP_HI*100)}% "
    f"@ {DS_ATR_LO}/{DS_ATR_HI}% · "
    f"Hurst w{HURST_W1}/t{HURST_T1} AND w{HURST_W2}/t{HURST_T2} · "
    f"Credit ma{CREDIT_MA}/d{CREDIT_DEPTH*100:.1f}% · "
    f"DynPos {DYN_BASE} · Dip {int(DIP_TRIGGER*100)}%/{int(DIP_ALLOC*100)}%"
)
st.caption("⚠️ Keine Anlageberatung. Nur für private Nutzung.")
