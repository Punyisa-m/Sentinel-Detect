"""
Sentinel-Detect | dashboard.py  ·  Apple Liquid Glass Edition
==============================================================
Run with:  streamlit run dashboard.py
"""

import sqlite3, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_generator import generate_realtime_event, engineer_features, init_db, DB_PATH
from model_engine   import (
    score_single_event, get_integrity_score, get_flagged_activities,
    get_scatter_data, get_audit_tasks, train_model, score_logs, MODEL_PATH,
)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel-Detect",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LIQUID GLASS CSS ─────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ╔══════════════════════════════════════╗
   ║  DESIGN TOKENS                       ║
   ╚══════════════════════════════════════╝ */
:root {
  /* Mesh background colours */
  --bg-base:        #f0f2ff;
  --bg-mesh-a:      #dce4ff;
  --bg-mesh-b:      #e8d5ff;
  --bg-mesh-c:      #d0f0ff;

  /* Glass surfaces */
  --glass-bg:       rgba(255,255,255,0.58);
  --glass-bg-deep:  rgba(255,255,255,0.72);
  --glass-border:   rgba(255,255,255,0.90);
  --glass-shadow:   0 8px 32px rgba(100,120,200,0.10),
                    0 2px  8px rgba(100,120,200,0.07);
  --glass-hover:    0 16px 48px rgba(100,120,200,0.16),
                    0  4px 16px rgba(100,120,200,0.10);
  --specular:       inset 0 1px 0 rgba(255,255,255,0.95),
                    inset 1px 0 0 rgba(255,255,255,0.60);

  /* Radius */
  --r-card: 32px;
  --r-btn:  16px;
  --r-tag:  100px;

  /* Typography */
  --ink-900: #0d0d1a;
  --ink-700: #1e2240;
  --ink-500: #4a5080;
  --ink-300: #9298b8;
  --ink-100: #c8ccdf;

  /* Brand accents */
  --blue:    #007AFF;
  --blue-lt: rgba(0,122,255,0.12);
  --green:   #34C759;
  --green-lt:rgba(52,199,89,0.12);
  --amber:   #FF9500;
  --amber-lt:rgba(255,149,0,0.12);
  --red:     #FF3B30;
  --red-lt:  rgba(255,59,48,0.12);
  --violet:  #AF52DE;
  --violet-lt:rgba(175,82,222,0.12);

  /* Neumorphic (light surface) */
  --neu-shadow: 4px 4px 12px rgba(160,180,220,0.35),
               -3px -3px 8px rgba(255,255,255,0.90);
  --neu-inset:  inset 3px 3px 8px rgba(160,180,220,0.30),
                inset -2px -2px 6px rgba(255,255,255,0.85);
}

/* ╔══════════════════════════════════════╗
   ║  GLOBAL RESET & BACKGROUND           ║
   ╚══════════════════════════════════════╝ */
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
  font-family: 'Inter', sans-serif !important;
  color: var(--ink-700) !important;
}

/* Animated mesh-gradient background */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 60% at 10% 10%,  var(--bg-mesh-a) 0%, transparent 60%),
    radial-gradient(ellipse 60% 80% at 90% 5%,   var(--bg-mesh-b) 0%, transparent 55%),
    radial-gradient(ellipse 70% 50% at 50% 95%,  var(--bg-mesh-c) 0%, transparent 60%),
    radial-gradient(ellipse 50% 60% at 80% 60%,  var(--bg-mesh-a) 0%, transparent 50%),
    var(--bg-base) !important;
  animation: meshShift 14s ease-in-out infinite alternate;
}

@keyframes meshShift {
  0%   { filter: hue-rotate(0deg); }
  100% { filter: hue-rotate(10deg); }
}

/* Noise grain overlay for depth */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
  opacity: 0.6;
}

[data-testid="stMainBlockContainer"] { position: relative; z-index: 1; padding-top: 0 !important; }

/* ╔══════════════════════════════════════╗
   ║  SIDEBAR – LIQUID GLASS              ║
   ╚══════════════════════════════════════╝ */
[data-testid="stSidebar"] {
  background: rgba(240,242,255,0.70) !important;
  backdrop-filter: blur(24px) saturate(1.8) !important;
  border-right: 1px solid var(--glass-border) !important;
  box-shadow: 2px 0 24px rgba(100,120,200,0.08) !important;
}

[data-testid="stSidebar"] * {
  font-family: 'Inter', sans-serif !important;
  color: var(--ink-500) !important;
}

[data-testid="stSidebar"] .stButton > button {
  background: var(--glass-bg-deep) !important;
  border: 1px solid rgba(255,255,255,0.85) !important;
  border-radius: var(--r-btn) !important;
  color: var(--ink-700) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  box-shadow: var(--neu-shadow) !important;
  transition: all 0.22s cubic-bezier(.34,1.56,.64,1) !important;
  width: 100% !important;
  padding: 0.55rem 1rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  box-shadow: var(--neu-inset) !important;
  transform: scale(0.97) !important;
  color: var(--blue) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
  box-shadow: var(--neu-inset) !important;
  transform: scale(0.95) !important;
}

/* ╔══════════════════════════════════════╗
   ║  TYPOGRAPHY                          ║
   ╚══════════════════════════════════════╝ */
h1, h2, h3, h4 {
  font-family: 'Inter', sans-serif !important;
  color: var(--ink-900) !important;
  letter-spacing: -0.03em !important;
  font-weight: 700 !important;
}
p, span, li, div, label, td, th {
  font-family: 'Inter', sans-serif !important;
  color: var(--ink-500) !important;
}

/* ╔══════════════════════════════════════╗
   ║  GLASS CARD  (base component)        ║
   ╚══════════════════════════════════════╝ */
.g-card {
  background: var(--glass-bg);
  backdrop-filter: blur(20px) saturate(1.6);
  -webkit-backdrop-filter: blur(20px) saturate(1.6);
  border: 1px solid var(--glass-border);
  border-radius: var(--r-card);
  box-shadow: var(--glass-shadow), var(--specular);
  padding: 24px 22px;
  position: relative;
  overflow: hidden;
  transition: box-shadow 0.3s ease, transform 0.25s cubic-bezier(.34,1.56,.64,1);
}
.g-card:hover {
  box-shadow: var(--glass-hover), var(--specular);
  transform: translateY(-2px);
}
/* Top specular sheen line */
.g-card::after {
  content: '';
  position: absolute;
  top: 0; left: 12px; right: 12px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,1), transparent);
  border-radius: 0 0 4px 4px;
}

/* ╔══════════════════════════════════════╗
   ║  KPI BENTO CARDS                     ║
   ╚══════════════════════════════════════╝ */
.kpi-wrap {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}

.kpi-g {
  background: var(--glass-bg);
  backdrop-filter: blur(20px) saturate(1.6);
  border: 1px solid var(--glass-border);
  border-radius: var(--r-card);
  box-shadow: var(--glass-shadow), var(--specular);
  padding: 22px 20px 18px;
  position: relative;
  overflow: hidden;
  transition: all 0.25s ease;
}
.kpi-g:hover { box-shadow: var(--glass-hover), var(--specular); transform: translateY(-3px); }
.kpi-g::after { content:''; position:absolute; top:0; left:10px; right:10px; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,1),transparent); }

.kpi-eyebrow {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-300) !important;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.kpi-num {
  font-size: 2.8rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.04em;
  margin-bottom: 8px;
}
.kpi-sub {
  font-size: 0.75rem;
  font-weight: 400;
  color: var(--ink-300) !important;
}

/* Accent blobs inside KPI cards */
.kpi-blob {
  position: absolute;
  bottom: -20px; right: -20px;
  width: 80px; height: 80px;
  border-radius: 50%;
  opacity: 0.12;
  filter: blur(16px);
}

/* ╔══════════════════════════════════════╗
   ║  PROGRESS RING                       ║
   ╚══════════════════════════════════════╝ */
.ring-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
}
.ring-svg { filter: drop-shadow(0 0 8px var(--ring-color, rgba(0,122,255,0.5))); }
.ring-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-400, var(--ink-300)) !important; }

/* ╔══════════════════════════════════════╗
   ║  SECTION HEADER                      ║
   ╚══════════════════════════════════════╝ */
.sec-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 28px 0 14px;
}
.sec-head-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--ink-900) !important;
  letter-spacing: -0.02em;
}
.sec-head-sub {
  font-size: 0.75rem;
  color: var(--ink-300) !important;
}

/* ╔══════════════════════════════════════╗
   ║  ALERT FEED                          ║
   ╚══════════════════════════════════════╝ */
.feed-wrap { max-height: 360px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: rgba(0,122,255,0.2) transparent; }
.feed-wrap::-webkit-scrollbar { width: 3px; }
.feed-wrap::-webkit-scrollbar-thumb { background: rgba(0,122,255,0.25); border-radius: 4px; }

.feed-item {
  background: var(--glass-bg-deep);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  padding: 12px 14px;
  margin-bottom: 8px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--glass-shadow);
}
.feed-item::before {
  content: '';
  position: absolute;
  top: 0; left: 0; bottom: 0;
  width: 3px;
  background: var(--feed-accent, var(--red));
  border-radius: 0 0 0 0;
}
.feed-item.warn::before { background: var(--amber); }

.feed-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 100px;
  font-size: 0.68rem;
  font-weight: 600;
  background: var(--red-lt);
  color: var(--red) !important;
  margin-right: 6px;
}
.feed-pill.warn { background: var(--amber-lt); color: var(--amber) !important; }
.feed-emp { font-weight: 700; font-size: 0.82rem; color: var(--ink-900) !important; }
.feed-reason { font-size: 0.74rem; color: var(--ink-300) !important; margin-top: 3px; line-height: 1.5; }
.feed-time { font-size: 0.68rem; color: var(--ink-100) !important; }

.feed-empty {
  text-align: center;
  padding: 36px 20px;
  color: var(--ink-300) !important;
}
.feed-empty-icon { font-size: 2rem; margin-bottom: 8px; opacity: 0.35; }

/* ╔══════════════════════════════════════╗
   ║  LIQUID WAVE ANIMATION               ║
   ╚══════════════════════════════════════╝ */
.wave-container {
  width: 100%;
  height: 110px;
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(0,122,255,0.06), rgba(175,82,222,0.06));
  margin: 12px 0;
}
.wave-svg { width: 200%; height: 100%; animation: waveScroll 6s linear infinite; }
.wave-svg.wave2 { animation: waveScroll2 9s linear infinite; opacity: 0.5; position: absolute; top: 0; left: 0; }
@keyframes waveScroll  { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
@keyframes waveScroll2 { 0% { transform: translateX(-50%); } 100% { transform: translateX(0); } }

/* ╔══════════════════════════════════════╗
   ║  DIVIDER                             ║
   ╚══════════════════════════════════════╝ */
.glass-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(160,180,220,0.35), transparent);
  margin: 24px 0;
}

/* ╔══════════════════════════════════════╗
   ║  STATUS BADGE                        ║
   ╚══════════════════════════════════════╝ */
.s-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 12px; border-radius: var(--r-tag);
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
}
.s-badge.green  { background: var(--green-lt);  color: var(--green)  !important; border: 1px solid rgba(52,199,89,0.25);  }
.s-badge.amber  { background: var(--amber-lt);  color: var(--amber)  !important; border: 1px solid rgba(255,149,0,0.25);  }
.s-badge.red    { background: var(--red-lt);    color: var(--red)    !important; border: 1px solid rgba(255,59,48,0.25);  }
.s-badge.blue   { background: var(--blue-lt);   color: var(--blue)   !important; border: 1px solid rgba(0,122,255,0.25); }
.s-badge.violet { background: var(--violet-lt); color: var(--violet) !important; border: 1px solid rgba(175,82,222,0.25);}
.live-dot { width:6px; height:6px; border-radius:50%; background:currentColor; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.75)} }

/* ╔══════════════════════════════════════╗
   ║  PAGE TITLE                          ║
   ╚══════════════════════════════════════╝ */
.page-hero {
  padding: 20px 0 28px;
  border-bottom: 1px solid rgba(160,180,220,0.2);
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}
.page-hero-text h1 {
  font-size: 1.55rem !important;
  font-weight: 800 !important;
  color: var(--ink-900) !important;
  letter-spacing: -0.04em !important;
  line-height: 1.1 !important;
  margin-bottom: 4px !important;
}
.page-hero-text p { font-size: 0.8rem !important; color: var(--ink-300) !important; }
.hero-icon {
  width: 52px; height: 52px; border-radius: 16px; flex-shrink: 0;
  background: linear-gradient(135deg, #007AFF, #AF52DE);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  box-shadow: 0 8px 24px rgba(0,122,255,0.25), inset 0 1px 0 rgba(255,255,255,0.35);
}

/* ╔══════════════════════════════════════╗
   ║  STREAMLIT COMPONENT OVERRIDES       ║
   ╚══════════════════════════════════════╝ */
/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 20px !important; overflow: hidden !important; border: 1px solid rgba(255,255,255,0.8) !important; }
.stDataFrameGlideDataEditor { background: var(--glass-bg) !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
  background: var(--glass-bg-deep) !important;
  border: 1px solid rgba(255,255,255,0.85) !important;
  border-radius: 14px !important;
  box-shadow: var(--neu-shadow) !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
  background: var(--glass-bg-deep) !important;
  border: 1px solid rgba(255,255,255,0.85) !important;
  border-radius: 14px !important;
  box-shadow: var(--neu-shadow) !important;
}

/* Info/success/warning boxes */
[data-testid="stInfo"]    { background: rgba(0,122,255,0.06)  !important; border: 1px solid rgba(0,122,255,0.2)   !important; border-radius: 18px !important; }
[data-testid="stSuccess"] { background: rgba(52,199,89,0.07)  !important; border: 1px solid rgba(52,199,89,0.25)  !important; border-radius: 18px !important; }
[data-testid="stWarning"] { background: rgba(255,149,0,0.07)  !important; border: 1px solid rgba(255,149,0,0.25)  !important; border-radius: 18px !important; }

/* Caption */
[data-testid="stCaptionContainer"] p { color: var(--ink-300) !important; font-size: 0.73rem !important; }

/* Remove default deco */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; backdrop-filter: blur(20px) !important; }

/* Main block padding */
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* ╔══════════════════════════════════════╗
   ║  FOOTER                              ║
   ╚══════════════════════════════════════╝ */
.page-footer {
  text-align: center;
  padding: 20px 0 8px;
  font-size: 0.7rem;
  color: var(--ink-100) !important;
  letter-spacing: 0.06em;
  border-top: 1px solid rgba(160,180,220,0.2);
  margin-top: 36px;
}
</style>
""", unsafe_allow_html=True)


# ── PLOTLY GLASS THEME ────────────────────────────────────────────────────────
PLOT_BG   = "rgba(255,255,255,0.0)"
GRID_CLR  = "rgba(160,180,220,0.18)"
FONT_CLR  = "#9298b8"
PLOT_BASE = dict(
    paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Inter, sans-serif", color=FONT_CLR, size=11),
    margin=dict(t=24, b=36, l=4, r=4),
    legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(255,255,255,0.9)",
                borderwidth=1, font=dict(size=10, color="#4a5080")),
    xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zeroline=False, tickfont=dict(size=10)),
)
IOS_BLUE   = "#007AFF"
IOS_GREEN  = "#34C759"
IOS_RED    = "#FF3B30"
IOS_AMBER  = "#FF9500"
IOS_VIOLET = "#AF52DE"
COLOR_MAP  = {"Normal": IOS_BLUE, "Anomaly": IOS_RED}


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("alert_feed",[]), ("event_count",0), ("model_ready", Path(MODEL_PATH).exists())]:
    if k not in st.session_state: st.session_state[k] = v


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 4px 22px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <div style="width:36px;height:36px;border-radius:11px;
                    background:linear-gradient(135deg,#007AFF,#AF52DE);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;box-shadow:0 4px 12px rgba(0,122,255,0.3);">🛡️</div>
        <div>
          <div style="font-family:Inter,sans-serif;font-size:1rem;font-weight:800;
                      color:#0d0d1a;letter-spacing:-0.03em;">Sentinel-Detect</div>
          <div style="font-size:0.68rem;color:#9298b8;letter-spacing:0.04em;text-transform:uppercase;margin-top:1px;">AI Threat Platform</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(160,180,220,0.4),transparent);margin-bottom:18px;"></div>
    <p style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#c8ccdf;margin-bottom:10px;">Controls</p>
    """, unsafe_allow_html=True)

    if st.button("⟳  Initialise Database", use_container_width=True):
        from data_generator import generate_normal_log, inject_anomalies, store_logs
        with st.spinner("Generating logs…"):
            init_db(DB_PATH)
            feat = engineer_features(inject_anomalies(generate_normal_log(500), 40))
            store_logs(feat, DB_PATH)
        st.success("540 records ready.")

    if st.button("◈  Train Detection Model", use_container_width=True):
        with st.spinner("Training Isolation Forest…"):
            train_model(DB_PATH); score_logs(DB_PATH, retrain=False)
        st.session_state.model_ready = True
        st.success("Model ready.")

    if st.button("▷  Simulate Live Event", use_container_width=True):
        if not st.session_state.model_ready:
            st.warning("Train the model first.")
        else:
            result = score_single_event(generate_realtime_event(), DB_PATH)
            st.session_state.event_count += 1
            if result["is_anomaly_pred"]:
                st.session_state.alert_feed.insert(0, {
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "emp": result["employee_id"], "action": result["action_type"],
                    "score": result["anomaly_score"],
                    "reason": result["anomaly_reason"] or "Multivariate deviation",
                    "sev": "high" if result["anomaly_score"] < -0.15 else "medium",
                })
                st.session_state.alert_feed = st.session_state.alert_feed[:50]

    st.markdown("""<div style="height:18px"></div>
    <p style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#c8ccdf;margin-bottom:10px;">Filters</p>""",
    unsafe_allow_html=True)

    dept_filter   = st.multiselect("Department",  ["All","Engineering","Finance","HR","Legal","Marketing","Operations"], default=["All"])
    action_filter = st.multiselect("Action Type", ["All","Access Doc","Submit Task","Financial Claim"], default=["All"])

    model_color = IOS_GREEN if st.session_state.model_ready else IOS_RED
    model_label = "Ready" if st.session_state.model_ready else "Offline"
    st.markdown(f"""
    <div style="margin-top:18px;background:rgba(255,255,255,0.55);
                border:1px solid rgba(255,255,255,0.9);border-radius:18px;
                padding:14px 16px;box-shadow:4px 4px 12px rgba(160,180,220,0.2),
                -2px -2px 6px rgba(255,255,255,0.85);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:0.72rem;color:#9298b8;">Events simulated</span>
        <span style="font-size:0.8rem;font-weight:700;color:#007AFF;">{st.session_state.event_count}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.72rem;color:#9298b8;">Model</span>
        <span style="font-size:0.72rem;font-weight:600;color:{model_color};">● {model_label}</span>
      </div>
    </div>""", unsafe_allow_html=True)


# ── DATA FETCH ────────────────────────────────────────────────────────────────
def apply_f(df):
    if "All" not in dept_filter   and dept_filter:   df = df[df["department"].isin(dept_filter)]
    if "All" not in action_filter and action_filter: df = df[df["action_type"].isin(action_filter)]
    return df

integrity_score = get_integrity_score(DB_PATH)
flagged_df = get_flagged_activities(DB_PATH)
audit_df   = get_audit_tasks(DB_PATH)
scatter_df = get_scatter_data(DB_PATH)

conn = sqlite3.connect(DB_PATH)
total_events = int(pd.read_sql("SELECT COUNT(*) AS c FROM activity_logs", conn)["c"].iloc[0])
conn.close()
open_tasks   = int(len(audit_df[audit_df["status"]=="OPEN"])) if not audit_df.empty else 0
anom_pct     = round(len(flagged_df)/max(total_events,1)*100, 1)

def score_style(s):
    if s >= 80: return IOS_GREEN,  "green",  "HEALTHY"
    if s >= 55: return IOS_AMBER,  "amber",  "MODERATE"
    return             IOS_RED,   "red",    "AT RISK"
sc_color, sc_cls, sc_lbl = score_style(integrity_score)


# ── PAGE HERO ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-hero">
  <div class="hero-icon">🛡️</div>
  <div class="page-hero-text">
    <h1>Security Audit Dashboard</h1>
    <p>AI-powered anomaly detection · Insider threat monitoring · Compliance assurance
       &nbsp;<span class="s-badge blue"><span class="live-dot"></span>Live</span></p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI BENTO ROW ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

def render_ring(value, max_val, color, size=72, stroke=7):
    r   = (size - stroke*2) / 2
    circ = 2 * math.pi * r
    dash = (value / max_val) * circ
    return f"""
    <svg width="{size}" height="{size}" class="ring-svg" style="--ring-color:{color}40;">
      <circle cx="{size/2}" cy="{size/2}" r="{r}"
              fill="none" stroke="rgba(160,180,220,0.18)"
              stroke-width="{stroke}" />
      <circle cx="{size/2}" cy="{size/2}" r="{r}"
              fill="none" stroke="{color}"
              stroke-width="{stroke}"
              stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {circ:.1f}"
              transform="rotate(-90 {size/2} {size/2})"
              style="filter:drop-shadow(0 0 5px {color}80);" />
      <text x="{size/2}" y="{size/2+5}" text-anchor="middle"
            font-family="Inter,sans-serif" font-size="14" font-weight="700"
            fill="{color}">{int(value)}</text>
    </svg>"""

with k1:
    ring = render_ring(integrity_score, 100, sc_color)
    st.markdown(f"""
    <div class="kpi-g">
      <div class="kpi-blob" style="background:{sc_color};"></div>
      <div class="kpi-eyebrow">🏢 Integrity Score</div>
      <div style="display:flex;align-items:center;gap:14px;">
        {ring}
        <div>
          <div class="kpi-num" style="color:{sc_color};">{integrity_score}</div>
          <span class="s-badge {sc_cls}">{sc_lbl}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with k2:
    ring2 = render_ring(min(len(flagged_df), 100), 100, IOS_RED)
    st.markdown(f"""
    <div class="kpi-g">
      <div class="kpi-blob" style="background:{IOS_RED};"></div>
      <div class="kpi-eyebrow">⚠ Anomalies</div>
      <div style="display:flex;align-items:center;gap:14px;">
        {ring2}
        <div>
          <div class="kpi-num" style="color:{IOS_RED};">{len(flagged_df)}</div>
          <div class="kpi-sub">{anom_pct}% of total</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with k3:
    ring3 = render_ring(min(open_tasks, 100), 100, IOS_AMBER)
    st.markdown(f"""
    <div class="kpi-g">
      <div class="kpi-blob" style="background:{IOS_AMBER};"></div>
      <div class="kpi-eyebrow">📋 Open Tasks</div>
      <div style="display:flex;align-items:center;gap:14px;">
        {ring3}
        <div>
          <div class="kpi-num" style="color:{IOS_AMBER};">{open_tasks}</div>
          <div class="kpi-sub">Pending review</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with k4:
    ring4 = render_ring(min(total_events/10, 100), 100, IOS_VIOLET)
    st.markdown(f"""
    <div class="kpi-g">
      <div class="kpi-blob" style="background:{IOS_VIOLET};"></div>
      <div class="kpi-eyebrow">◈ Events</div>
      <div style="display:flex;align-items:center;gap:14px;">
        {ring4}
        <div>
          <div class="kpi-num" style="color:{IOS_VIOLET};">{total_events}</div>
          <div class="kpi-sub">Total monitored</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)


# ── AI INSIGHTS CARD + ALERT FEED ────────────────────────────────────────────
row2a, row2b = st.columns([1.3, 1])

with row2a:
    st.markdown('<div class="sec-head"><span class="sec-head-title">AI Insights</span><span class="sec-head-sub">Liquid anomaly surface</span></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="g-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
        <div>
          <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#9298b8;">Detection Wave</div>
          <div style="font-size:1.5rem;font-weight:800;color:#0d0d1a;letter-spacing:-0.04em;margin-top:2px;">
            {len(flagged_df)} <span style="font-size:0.9rem;font-weight:500;color:#9298b8;">anomalies surfaced</span>
          </div>
        </div>
        <span class="s-badge {'red' if anom_pct > 10 else 'green'}">{anom_pct}% anomaly rate</span>
      </div>

      <!-- Liquid Wave -->
      <div class="wave-container">
        <svg class="wave-svg" viewBox="0 0 1440 110" preserveAspectRatio="none">
          <defs>
            <linearGradient id="wg1" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#007AFF;stop-opacity:0.35"/>
              <stop offset="100%" style="stop-color:#AF52DE;stop-opacity:0.08"/>
            </linearGradient>
          </defs>
          <path d="M0,55 C120,20 240,90 360,55 C480,20 600,90 720,55 C840,20 960,90 1080,55 C1200,20 1320,90 1440,55
                   L1440,110 L720,110 L0,110 Z" fill="url(#wg1)"/>
          <path d="M0,65 C120,35 240,95 360,65 C480,35 600,95 720,65 C840,35 960,95 1080,65 C1200,35 1320,95 1440,65
                   L1440,110 L720,110 L0,110 Z" fill="rgba(0,122,255,0.12)"/>
        </svg>
        <svg class="wave-svg wave2" viewBox="0 0 1440 110" preserveAspectRatio="none" style="position:absolute;top:0;left:0;">
          <path d="M0,70 C180,30 360,100 540,65 C720,30 900,100 1080,65 C1260,30 1350,90 1440,70
                   L1440,110 L0,110 Z" fill="rgba(175,82,222,0.10)"/>
        </svg>
      </div>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px;">
        <div style="background:rgba(0,122,255,0.06);border:1px solid rgba(0,122,255,0.15);border-radius:16px;padding:12px 14px;">
          <div style="font-size:0.68rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#007AFF;">Access</div>
          <div style="font-size:1.3rem;font-weight:800;color:#0d0d1a;letter-spacing:-0.03em;margin-top:3px;">
            {len(flagged_df[flagged_df['action_type']=='Access Doc']) if not flagged_df.empty else 0}
          </div>
        </div>
        <div style="background:rgba(52,199,89,0.06);border:1px solid rgba(52,199,89,0.15);border-radius:16px;padding:12px 14px;">
          <div style="font-size:0.68rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#34C759;">Tasks</div>
          <div style="font-size:1.3rem;font-weight:800;color:#0d0d1a;letter-spacing:-0.03em;margin-top:3px;">
            {len(flagged_df[flagged_df['action_type']=='Submit Task']) if not flagged_df.empty else 0}
          </div>
        </div>
        <div style="background:rgba(255,59,48,0.06);border:1px solid rgba(255,59,48,0.15);border-radius:16px;padding:12px 14px;">
          <div style="font-size:0.68rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#FF3B30;">Financial</div>
          <div style="font-size:1.3rem;font-weight:800;color:#0d0d1a;letter-spacing:-0.03em;margin-top:3px;">
            {len(flagged_df[flagged_df['action_type']=='Financial Claim']) if not flagged_df.empty else 0}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with row2b:
    st.markdown('<div class="sec-head"><span class="sec-head-title">Real-Time Alerts</span><span class="sec-head-sub">Live event stream</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-card" style="height:298px;">', unsafe_allow_html=True)

    if not st.session_state.alert_feed:
        st.markdown("""
        <div class="feed-empty">
          <div class="feed-empty-icon">🔍</div>
          <div style="font-size:0.82rem;color:#9298b8;">No alerts yet</div>
          <div style="font-size:0.72rem;color:#c8ccdf;margin-top:4px;">Simulate events via the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        html = '<div class="feed-wrap">'
        for a in st.session_state.alert_feed:
            css  = "feed-item warn" if a["sev"]=="medium" else "feed-item"
            pill = "feed-pill warn" if a["sev"]=="medium" else "feed-pill"
            html += f"""
            <div class="{css}">
              <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <span class="{pill}">{a['score']:.4f}</span>
                <span class="feed-emp">{a['emp']}</span>
                <span style="color:#c8ccdf;font-size:0.7rem;margin-left:auto;">{a['t']}</span>
              </div>
              <div class="feed-reason">{a['reason']}</div>
            </div>"""
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)


# ── GAUGE + DEPT CHART ────────────────────────────────────────────────────────
row3a, row3b = st.columns([1, 1.6])

with row3a:
    st.markdown('<div class="sec-head"><span class="sec-head-title">Health Gauge</span></div>', unsafe_allow_html=True)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=integrity_score,
        delta={"reference": 85, "increasing": {"color": IOS_GREEN}, "decreasing": {"color": IOS_RED}},
        number={"font": {"color": sc_color, "family": "Inter, sans-serif", "size": 40}, "suffix": ""},
        gauge={
            "axis": {"range": [0,100], "tickcolor": "#c8ccdf", "tickwidth":1,
                     "tickfont":{"color":"#9298b8","size":9,"family":"Inter"}},
            "bar":  {"color": sc_color, "thickness": 0.20},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range":[0,55],  "color":"rgba(255,59,48,0.07)"},
                {"range":[55,80], "color":"rgba(255,149,0,0.07)"},
                {"range":[80,100],"color":"rgba(52,199,89,0.07)"},
            ],
            "threshold": {"line":{"color":IOS_BLUE,"width":2}, "thickness":0.65, "value":85},
        },
        title={"text":"Company Health", "font":{"color":"#9298b8","size":11,"family":"Inter"}},
    ))
    fig_g.update_layout(**{**PLOT_BASE, "height":240, "margin":dict(t=28,b=8,l=20,r=20)})
    st.plotly_chart(fig_g, use_container_width=True)

with row3b:
    st.markdown('<div class="sec-head"><span class="sec-head-title">Anomaly Rate by Department</span></div>', unsafe_allow_html=True)
    if not scatter_df.empty:
        sdf = apply_f(scatter_df).copy()
        dept_s = sdf.groupby("department").agg(
            total=("is_anomaly_pred","count"), anomalies=("is_anomaly_pred","sum")
        ).reset_index()
        dept_s["rate"] = (dept_s["anomalies"]/dept_s["total"]*100).round(1)

        fig_dept = px.bar(dept_s.sort_values("rate"),
                          x="rate", y="department", orientation="h",
                          color="rate", text="rate",
                          color_continuous_scale=[IOS_GREEN, IOS_AMBER, IOS_RED])
        fig_dept.update_traces(texttemplate="%{text}%", textposition="outside",
                               marker_line_width=0, marker_cornerradius=8)
        fig_dept.update_layout(**{**PLOT_BASE, "height":240,
                                  "coloraxis_showscale":False,
                                  "margin":dict(t=8,b=24,l=4,r=40)})
        st.plotly_chart(fig_dept, use_container_width=True)

st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)


# ── SCATTER + HISTOGRAM ───────────────────────────────────────────────────────
st.markdown('<div class="sec-head"><span class="sec-head-title">Anomaly Visualisations</span><span class="sec-head-sub">Feature space analysis</span></div>', unsafe_allow_html=True)

if scatter_df.empty:
    st.info("Score the logs first to see visualisations.")
else:
    sdf = apply_f(scatter_df).copy()
    sdf["Status"] = sdf["is_anomaly_pred"].map({0:"Normal",1:"Anomaly"})
    v1, v2 = st.columns(2)

    with v1:
        fig1 = px.scatter(sdf, x="duration_minutes", y="security_level",
                          color="Status", color_discrete_map=COLOR_MAP,
                          hover_data=["action_type","department","anomaly_score"],
                          opacity=0.70, title="Duration vs Security Level")
        fig1.update_layout(**{**PLOT_BASE, "height":320})
        fig1.update_yaxes(tickvals=[1,2,3,4],
                          ticktext=["Public","Internal","Confidential","Restricted"])
        fig1.update_traces(marker=dict(size=7, line=dict(width=0)))
        st.plotly_chart(fig1, use_container_width=True)

    with v2:
        fig3 = px.histogram(sdf, x="anomaly_score", color="Status", nbins=48,
                            barmode="overlay", opacity=0.75,
                            color_discrete_map=COLOR_MAP,
                            title="Score Distribution")
        fig3.add_vline(x=-0.10, line_dash="dot", line_color=IOS_VIOLET,
                       line_width=1.5,
                       annotation_text="Threshold",
                       annotation_font_color=IOS_VIOLET,
                       annotation_font_size=10)
        fig3.update_layout(**{**PLOT_BASE, "height":320})
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    fin = sdf[sdf["transaction_amount"]>0].copy()
    if not fin.empty:
        v3, v4 = st.columns(2)
        with v3:
            fig2 = px.scatter(fin, x="transaction_amount", y="freq_score",
                              color="Status", size="transaction_amount", size_max=16,
                              color_discrete_map=COLOR_MAP, opacity=0.75,
                              hover_data=["action_type","department","anomaly_score"],
                              title="Transaction Amount vs Frequency")
            fig2.update_layout(**{**PLOT_BASE,"height":300})
            fig2.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig2, use_container_width=True)

        with v4:
            if not scatter_df.empty and "hour_of_day" in scatter_df.columns:
                hourly = scatter_df.dropna(subset=["hour_of_day"]).groupby(["hour_of_day","is_anomaly_pred"]).size().reset_index(name="count")
                hourly["type"] = hourly["is_anomaly_pred"].map({0:"Normal",1:"Anomaly"})
                fig5 = px.line(hourly, x="hour_of_day", y="count", color="type",
                               color_discrete_map=COLOR_MAP,
                               title="Events by Hour of Day",
                               markers=True)
                fig5.update_traces(line=dict(width=2.5))
                fig5.update_layout(**{**PLOT_BASE,"height":300})
                fig5.update_xaxes(title="Hour", tickvals=list(range(0,24,3)))
                st.plotly_chart(fig5, use_container_width=True)

st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)


# ── FLAGGED ACTIVITIES ────────────────────────────────────────────────────────
st.markdown('<div class="sec-head"><span class="sec-head-title">Flagged Activities</span><span class="sec-head-sub">XAI-annotated anomaly log</span></div>', unsafe_allow_html=True)

if flagged_df.empty:
    st.info("No flagged activities. Train the model and score logs first.")
else:
    disp = apply_f(flagged_df).copy()
    disp["anomaly_score"] = disp["anomaly_score"].round(4)
    disp["transaction_amount"] = disp["transaction_amount"].apply(
        lambda x: f"${x:,.2f}" if x and x>0 else "—")
    disp = disp.rename(columns={
        "employee_id":"Employee","department":"Dept","action_type":"Action",
        "timestamp":"Timestamp","duration_minutes":"Duration (min)",
        "security_level":"Sec Level","transaction_amount":"Amount",
        "anomaly_score":"Score","anomaly_reason":"🤖 XAI Reason",
    })
    st.dataframe(
        disp.drop(columns=["id"],errors="ignore"),
        use_container_width=True, height=280,
        column_config={
            "Score":     st.column_config.NumberColumn(format="%.4f"),
            "Sec Level": st.column_config.NumberColumn(help="1=Public 2=Internal 3=Confidential 4=Restricted"),
        },
    )
    st.caption(f"{len(disp)} flagged · {total_events} total monitored")

st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)


# ── AUDIT TASKS ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head"><span class="sec-head-title">Audit Tasks</span><span class="sec-head-sub">Orchestra-Agent hand-off queue</span></div>', unsafe_allow_html=True)

if audit_df.empty:
    st.info("No audit tasks yet.")
else:
    sf = st.selectbox("Filter by Status", ["ALL","OPEN","CLOSED"])
    adf = audit_df[audit_df["status"]==sf].copy() if sf!="ALL" else audit_df.copy()
    adf["anomaly_score"] = adf["anomaly_score"].round(4)
    st.dataframe(
        adf[["id","employee_id","department","action_type","timestamp",
             "anomaly_score","reason","status","created_at"]].rename(columns={
            "id":"Task #","employee_id":"Employee","department":"Dept",
            "action_type":"Action","timestamp":"Event Time",
            "anomaly_score":"Score","reason":"XAI Reason",
            "status":"Status","created_at":"Created At",
        }),
        use_container_width=True, height=240,
    )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-footer">
  SENTINEL-DETECT v1.0 &nbsp;·&nbsp; Isolation Forest · scikit-learn
  &nbsp;·&nbsp; Threshold −0.10 &nbsp;·&nbsp; Refresh to update all metrics
</div>
""", unsafe_allow_html=True)