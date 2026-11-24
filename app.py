"""Always-On Multi-Sensor Oscilloscope, Animated 3-Tier Parallel Orchestra & Bilingual HITL Control Tower UI."""

from __future__ import annotations

import hashlib
import html
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.graph import apply_human_decision, stream_maintenance_workflow
from src.i18n import get_scenario_label, localize_zone, t
from src.llm_config import (
    PROVIDER_CATALOG,
    detect_key_provider_mismatch,
    fetch_live_models,
    resolve_api_key,
)
from src.seed_data import (
    GMP_VIBRATION_LIMIT_RMS,
    REPORTS_DIR,
    ensure_seed_data,
    reset_demo_environment,
)
from src.state import DEMO_SCENARIOS, MaintenanceState, create_custom_incident_state, create_initial_state
from src.telemetry import SimulatedPlantTelemetryAdapter
from src.tools import (
    build_incident_certificate_pdf,
    compute_monthly_kpis,
    extract_pdf_page_text,
    find_document_by_code_or_name,
    generate_monthly_excel_report_impl,
    get_capa_dataframe,
    get_inventory_dataframe,
    get_monthly_incidents_dataframe,
    get_purchase_requisitions_dataframe,
    list_controlled_documents,
    render_pdf_page_png,
)

# ---------------------------------------------------------------------------
# Page Configuration & Bootstrapping
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PharmaGMP Control Tower | 3-Tier Multi-Agent Orchestra",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "db_booted" not in st.session_state:
    ensure_seed_data()
    st.session_state["db_booted"] = True

if "telemetry_adapter" not in st.session_state:
    st.session_state["telemetry_adapter"] = SimulatedPlantTelemetryAdapter(auto_anomaly_enabled=True)

if "active_view_idx" not in st.session_state:
    st.session_state["active_view_idx"] = 0

BAY_SCENARIO_GROUPS: Dict[str, List[str]] = {
    "TP-01": ["scenario_1", "scenario_tp01_punch", "scenario_tp01_roller"],
    "RP-04": ["scenario_2", "scenario_rp04_impeller", "scenario_rp04_softfoot"],
    "PK-02": ["scenario_3", "scenario_pk02_servo", "scenario_pk02_sealing"],
}


# ---------------------------------------------------------------------------
# Projection-Ready Industrial SCADA CSS & Native SVG Animations
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
header[data-testid="stHeader"] {
    background: rgba(11, 17, 32, 0.85);
}
.block-container {
    padding-top: 3.35rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
.ct-banner {
    background: linear-gradient(125deg, #090F1E 0%, #0F172A 55%, #0C4A6E 100%);
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.4);
}
.ct-title { font-size: 1.46rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; }
.ct-subtitle { font-size: 0.85rem; color: #94A3B8; margin-top: 0.15rem; }
.badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-approved { background: #064E3B; color: #34D399; border: 1px solid #10B981; }
.badge-signed { background: #14532D; color: #86EFAC; border: 1px solid #22C55E; }
.badge-rejected { background: #7F1D1D; color: #FCA5A5; border: 1px solid #EF4444; }
.badge-escalated { background: #78350F; color: #FCD34D; border: 1px solid #F59E0B; }
.badge-awaiting { background: #451A03; color: #FBBF24; border: 1px solid #F59E0B; }
.badge-pending { background: #1E3A8A; color: #93C5FD; border: 1px solid #3B82F6; }
.badge-standby { background: #1E293B; color: #94A3B8; border: 1px solid #475569; }
.bay-card {
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.78rem 0.9rem 0.68rem 0.9rem;
    margin-bottom: 0.35rem;
}
.bay-alarm { border-color: #F59E0B; box-shadow: 0 0 16px rgba(245, 158, 11, 0.22); }
.bay-rejected { border-color: #EF4444; box-shadow: 0 0 18px rgba(239, 68, 68, 0.30); }
.bay-ok { border-color: #10B981; box-shadow: 0 0 14px rgba(16, 185, 129, 0.20); }
.bay-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.32rem; }
.bay-suite { font-size: 0.70rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; }
.bay-name { font-size: 0.96rem; font-weight: 800; color: #F8FAFC; }
.vib-track {
    position: relative;
    height: 9px;
    background: #1E293B;
    border-radius: 999px;
    overflow: hidden;
    margin: 0.38rem 0 0.22rem 0;
}
.vib-limit-tick { position: absolute; left: 20.8%; top: 0; bottom: 0; width: 2px; background: #F8FAFC; z-index: 3; }
.hitl-panel {
    background: linear-gradient(135deg, #271E0B 0%, #0F172A 100%);
    border: 2px solid #F59E0B;
    border-radius: 14px;
    padding: 0.95rem 1.25rem;
    margin: 0.55rem 0 0.75rem 0;
    box-shadow: 0 0 28px rgba(245, 158, 11, 0.28);
}
.exec-close-banner {
    background: linear-gradient(135deg, #052E24 0%, #0F172A 65%, #0C4A6E 100%);
    border: 2px solid #10B981;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin: 0.65rem 0 0.75rem 0;
    box-shadow: 0 0 26px rgba(16, 185, 129, 0.22);
}
.exec-hold-banner {
    background: linear-gradient(135deg, #450A0A 0%, #0F172A 75%);
    border: 2px solid #EF4444;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin: 0.65rem 0 0.75rem 0;
}
.svg-panel {
    background: #0B1222;
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 0.55rem 0.65rem;
    margin-bottom: 0.7rem;
}
.spotlight-box {
    background: #0F172A;
    border-left: 4px solid #38BDF8;
    border-top: 1px solid #1E293B;
    border-right: 1px solid #1E293B;
    border-bottom: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.75rem;
    color: #E2E8F0;
    font-size: 0.87rem;
}
.trace-scroll {
    max-height: 410px;
    overflow-y: auto;
    padding-right: 0.2rem;
}
.trace-card {
    background: #0F172A;
    border: 1px solid #334155;
    border-left: 4px solid #38BDF8;
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #E2E8F0;
}
.trace-card-orch { border-left-color: #F59E0B; }
.trace-card-tech { border-left-color: #A855F7; }
.trace-card-qa { border-left-color: #10B981; }
.trace-card-human { border-left-color: #FBBF24; background: #19160E; }
.trace-head { display: flex; justify-content: space-between; align-items: center; gap: 0.4rem; margin-bottom: 0.3rem; }
.trace-actor { font-weight: 800; color: #F8FAFC; font-size: 0.86rem; }
.trace-tier { font-size: 0.70rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; }
.trace-line { margin: 0.16rem 0; line-height: 1.38; color: #CBD5E1; }
.trace-label { color: #94A3B8; font-weight: 700; }
.audit-table-wrap {
    max-height: 410px;
    overflow-y: auto;
    border: 1px solid #334155;
    border-radius: 10px;
    background: #0F172A;
}
table.ebmr-table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
table.ebmr-table th {
    position: sticky;
    top: 0;
    background: #1E293B;
    color: #E2E8F0;
    text-align: left;
    padding: 0.5rem 0.55rem;
    border-bottom: 2px solid #334155;
    font-weight: 700;
}
table.ebmr-table td {
    padding: 0.45rem 0.55rem;
    border-bottom: 1px solid #1E293B;
    color: #E2E8F0;
    vertical-align: top;
}
table.ebmr-table tr:nth-child(even) { background: #111827; }
table.sensor-strip { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.25rem; background: #0B1222; border-radius: 8px; overflow: hidden; }
table.sensor-strip th { background: #172540; color: #93C5FD; padding: 0.38rem 0.5rem; text-align: left; border-bottom: 1px solid #1E3A5F; }
table.sensor-strip td { padding: 0.36rem 0.5rem; color: #E2E8F0; border-bottom: 1px solid #1E293B; }
.doc-card {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.35rem;
}
.doc-card-active { border-color: #38BDF8; background: #0C1A32; }
.kpi-card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 0.82rem 0.95rem; }
.kpi-label { font-size: 0.74rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; }
.kpi-val { font-size: 1.48rem; font-weight: 800; color: #F8FAFC; margin-top: 0.14rem; }
.kpi-sub { font-size: 0.75rem; color: #38BDF8; margin-top: 0.14rem; }

@keyframes flowDash {
    from { stroke-dashoffset: 36; }
    to { stroke-dashoffset: 0; }
}
@keyframes pulseActive {
    0%, 100% { filter: drop-shadow(0 0 4px rgba(56, 189, 248, 0.45)); }
    50% { filter: drop-shadow(0 0 14px rgba(56, 189, 248, 0.95)); }
}
.flow-active { stroke-dasharray: 8 6; animation: flowDash 0.85s linear infinite; }
.node-active { animation: pulseActive 1.3s ease-in-out infinite; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Helpers & Rendering Primitives
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _cached_live_models(provider: str, key_fingerprint: str, api_key: Optional[str]) -> Tuple[List[str], Optional[str]]:
    _ = key_fingerprint
    return fetch_live_models(provider, api_key)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_pdf_page_png(pdf_path: str, page_number: int, mtime: float) -> bytes:
    _ = mtime
    return render_pdf_page_png(pdf_path, page_number=page_number, scale=1.75)


def _strip_html_indent(raw: str) -> str:
    return "\n".join(line.lstrip() for line in raw.splitlines() if line.strip())


def _gate_badge_html(gate: str) -> str:
    g = (gate or "PENDING").upper()
    cls = {
        "APPROVED": "badge-approved",
        "SIGNED_OFF": "badge-signed",
        "REJECTED": "badge-rejected",
        "HUMAN_REJECTED": "badge-rejected",
        "ESCALATED": "badge-escalated",
        "AWAITING_HUMAN": "badge-awaiting",
        "PENDING": "badge-pending",
    }.get(g, "badge-standby")
    return f'<span class="badge {cls}">{html.escape(g)}</span>'


def _equipment_svg(machine_id: str, accent: str) -> str:
    if machine_id == "TP-01":
        return _strip_html_indent(
            f"""
            <svg viewBox="0 0 120 68" width="96" height="54" xmlns="http://www.w3.org/2000/svg">
              <rect x="8" y="54" width="104" height="8" rx="3" fill="#1E293B" stroke="#475569"/>
              <rect x="26" y="14" width="68" height="40" rx="5" fill="#0F172A" stroke="{accent}" stroke-width="2"/>
              <ellipse cx="60" cy="24" rx="22" ry="6" fill="#1E293B" stroke="{accent}" stroke-width="1.5"/>
              <rect x="42" y="24" width="36" height="18" fill="#1E293B" stroke="#64748B"/>
              <line x1="49" y1="24" x2="49" y2="42" stroke="{accent}" stroke-width="2"/>
              <line x1="60" y1="24" x2="60" y2="42" stroke="{accent}" stroke-width="2"/>
              <line x1="71" y1="24" x2="71" y2="42" stroke="{accent}" stroke-width="2"/>
              <circle cx="96" cy="22" r="5" fill="{accent}"/>
            </svg>
            """
        )
    if machine_id == "RP-04":
        return _strip_html_indent(
            f"""
            <svg viewBox="0 0 120 68" width="96" height="54" xmlns="http://www.w3.org/2000/svg">
              <rect x="10" y="54" width="100" height="8" rx="3" fill="#1E293B" stroke="#475569"/>
              <circle cx="38" cy="34" r="20" fill="#0F172A" stroke="{accent}" stroke-width="2.2"/>
              <circle cx="38" cy="34" r="8" fill="#1E293B" stroke="{accent}" stroke-width="1.5"/>
              <rect x="33" y="8" width="10" height="10" fill="#1E293B" stroke="{accent}"/>
              <rect x="58" y="29" width="14" height="10" fill="#334155" stroke="{accent}" stroke-width="1.5"/>
              <rect x="72" y="20" width="34" height="28" rx="4" fill="#0F172A" stroke="#64748B" stroke-width="1.8"/>
              <line x1="78" y1="26" x2="100" y2="26" stroke="#475569" stroke-width="2"/>
              <line x1="78" y1="34" x2="100" y2="34" stroke="#475569" stroke-width="2"/>
              <line x1="78" y1="42" x2="100" y2="42" stroke="#475569" stroke-width="2"/>
            </svg>
            """
        )
    return _strip_html_indent(
        f"""
        <svg viewBox="0 0 120 68" width="96" height="54" xmlns="http://www.w3.org/2000/svg">
          <rect x="8" y="54" width="104" height="8" rx="3" fill="#1E293B" stroke="#475569"/>
          <circle cx="32" cy="34" r="16" fill="#0F172A" stroke="{accent}" stroke-width="2"/>
          <circle cx="88" cy="34" r="12" fill="#0F172A" stroke="{accent}" stroke-width="2"/>
          <line x1="32" y1="18" x2="88" y2="22" stroke="{accent}" stroke-width="3" stroke-dasharray="5,3"/>
          <line x1="32" y1="50" x2="88" y2="46" stroke="{accent}" stroke-width="3" stroke-dasharray="5,3"/>
          <circle cx="32" cy="34" r="4" fill="{accent}"/>
          <circle cx="88" cy="34" r="4" fill="{accent}"/>
        </svg>
        """
    )


def _render_factory_bay_card(
    sc: Dict[str, Any],
    active_state: Optional[MaintenanceState],
    lang: str = "en",
    live_sample: Optional[Dict[str, Any]] = None,
    live_rms_override: Optional[float] = None,
) -> str:
    mid = sc["machine_id"]
    baseline = float(sc["baseline_rms"])
    rpm = int(sc["nominal_rpm"])
    rate = float(sc["downtime_rate_per_min"])

    is_target = bool(active_state and active_state.get("machine_id") == mid)
    if is_target and active_state is not None:
        maint = active_state.get("maintenance_report", {})
        inv = active_state.get("inventory_report", {})
        if active_state.get("last_gate_decision") == "HUMAN_REJECTED":
            current_rms = float(active_state.get("alarm_vibration_rms", sc["alarm_vibration_rms"]))
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_human_hold"), "badge-rejected", "bay-card bay-rejected", "#EF4444"
        elif active_state.get("awaiting_human_approval"):
            current_rms = float(
                live_rms_override
                or maint.get("post_repair_vibration_rms")
                or active_state.get("alarm_vibration_rms", sc["alarm_vibration_rms"])
            )
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_awaiting_qa"), "badge-awaiting", "bay-card bay-alarm", "#F59E0B"
        elif active_state.get("protocol_closed") and not active_state.get("escalated"):
            current_rms = float(maint.get("post_repair_vibration_rms", baseline))
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_cleared"), "badge-signed", "bay-card bay-ok", "#10B981"
        elif active_state.get("last_gate_decision") == "REJECTED":
            current_rms = float(live_rms_override or maint.get("post_repair_vibration_rms", sc["alarm_vibration_rms"]))
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_escalated"), "badge-rejected", "bay-card bay-rejected", "#EF4444"
        elif live_rms_override is not None:
            current_rms = float(live_rms_override)
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_orch_active"), "badge-pending", "bay-card bay-alarm", "#38BDF8"
        else:
            current_rms = float(active_state.get("alarm_vibration_rms", sc["alarm_vibration_rms"]))
            status_lbl = t(lang, "status_alt_part") if inv.get("used_alternative") else t(lang, "status_crit_trip")
            badge_cls, bay_cls, accent = "badge-escalated", "bay-card bay-alarm", "#F59E0B"
    else:
        current_rms = float(live_sample["rms"]) if live_sample else baseline
        if current_rms > 4.5:
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_trip_breach"), "badge-rejected", "bay-card bay-rejected", "#EF4444"
        elif current_rms > 2.5:
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_zone_c"), "badge-escalated", "bay-card bay-alarm", "#F59E0B"
        else:
            status_lbl, badge_cls, bay_cls, accent = t(lang, "status_nominal"), "badge-approved", "bay-card", "#38BDF8"

    pct = max(5.0, min(100.0, (current_rms / 12.0) * 100.0))
    bar_color = "#10B981" if current_rms <= GMP_VIBRATION_LIMIT_RMS else ("#EF4444" if current_rms >= 4.5 else "#F59E0B")
    kurt_val = float(live_sample.get("kurtosis", 3.0)) if live_sample else 3.0
    temp_val = float(live_sample.get("temp_c", 46.0)) if live_sample else 46.0
    svg_icon = _equipment_svg(mid, accent)

    return _strip_html_indent(
        f"""
        <div class="{bay_cls}">
          <div class="bay-head">
            <div>
              <div class="bay-suite">{html.escape(sc['area'])}</div>
              <div class="bay-name">{html.escape(sc['machine_name'])}</div>
            </div>
            <span class="badge {badge_cls}">{html.escape(status_lbl)}</span>
          </div>
          <div style="display:flex;align-items:center;gap:0.65rem;">
            {svg_icon}
            <div style="flex:1;font-size:0.79rem;color:#CBD5E1;line-height:1.40;">
              <div><b>{html.escape(t(lang, 'bay_live_sensor'))}:</b> <span style="color:{bar_color};font-weight:800;">{current_rms:.2f} mm/s</span> (Kurt {kurt_val:.1f} | {temp_val:.1f}&deg;C)</div>
              <div><b>{html.escape(t(lang, 'bay_speed'))}:</b> {rpm} RPM &bull; <b>{html.escape(t(lang, 'bay_limit'))}:</b> &le; {GMP_VIBRATION_LIMIT_RMS} mm/s</div>
              <div><b>{html.escape(t(lang, 'bay_rate'))}:</b> ${rate:.0f}/min &bull; <b>{html.escape(t(lang, 'bay_part'))}:</b> <code>{html.escape(sc['expected_part_code'])}</code></div>
            </div>
          </div>
          <div class="vib-track">
            <div class="vib-limit-tick" title="GMP Limit 2.5 mm/s RMS"></div>
            <div style="width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:999px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#64748B;">
            <span>0.0</span><span>GMP Limit: 2.5 | Trip: 4.5</span><span>12.0</span>
          </div>
        </div>
        """
    )


def _render_hierarchy_orchestra_svg(
    active_node: str = "standby",
    state: Optional[MaintenanceState] = None,
    lang: str = "en",
) -> str:
    """Render Animated 3-Tier Technician-Engineer-QA Hierarchy & Command Flow SVG."""
    st_dict = state or {}
    last_gate = str(st_dict.get("last_gate_decision", "STANDBY"))
    awaiting_human = bool(st_dict.get("awaiting_human_approval", False))
    diag_done = bool(st_dict.get("diagnostic_done"))
    pre_done = bool(st_dict.get("precheck_done"))
    diag_ok = bool(st_dict.get("diagnostic_approved"))
    stock_ok = bool(st_dict.get("stock_approved"))
    maint_ok = bool(st_dict.get("maintenance_approved"))
    clear_ok = bool(st_dict.get("clearance_approved"))
    analyt_ok = bool(st_dict.get("analyst_approved"))
    replan_done = bool(st_dict.get("engineering_replan_done"))
    escalated_up = bool(st_dict.get("escalation_to_engineer")) or replan_done
    closed_ok = bool(st_dict.get("protocol_closed")) and not bool(st_dict.get("escalated"))

    def style_of(nid: str, done: bool = False) -> Tuple[str, str, str]:
        if active_node == nid:
            cls = "node-active"
            if last_gate in ("REJECTED", "HUMAN_REJECTED"):
                return "#450A0A", "#EF4444", cls
            return "#0C4A6E", "#38BDF8", cls
        if done:
            return "#052E24", "#10B981", ""
        return "#111C33", "#334155", ""

    orch_f, orch_s, orch_cls = style_of("orchestrator", closed_ok)
    if awaiting_human:
        orch_f, orch_s, orch_cls = "#451A03", "#F59E0B", "node-active"
    elif last_gate in ("REJECTED", "HUMAN_REJECTED") and active_node == "orchestrator":
        orch_f, orch_s, orch_cls = "#450A0A", "#EF4444", "node-active"

    d_f, d_s, d_cls = style_of("diagnostic_engineer", diag_ok or diag_done)
    p_f, p_s, p_cls = style_of("stock_precheck", pre_done)
    s_f, s_s, s_cls = style_of("stock_agent", stock_ok)
    a_f, a_s, a_cls = style_of("analyst_capa", analyt_ok)
    t_f, t_s, t_cls = style_of("field_technician", maint_ok)
    c_f, c_s, c_cls = style_of("qa_clearance", clear_ok)

    sensor_f, sensor_s = ("#0C4A6E", "#38BDF8") if state else ("#0D1B2E", "#22D3EE")
    if closed_ok:
        orch_sub = "SIGNED OFF &amp; eBMR ARCHIVED" if lang != "tr" else "İMZALANDI &amp; eBMR ARŞİVLENDİ"
    elif awaiting_human:
        gt = st_dict.get("pending_approval", {}).get("gate_type", "AUTH")
        orch_sub = f"PAUSED: HUMAN QA PERMISSION ({gt})" if lang != "tr" else f"BEKLİYOR: İNSAN QA İZNİ ({gt})"
    elif state:
        cmd_short = str(st_dict.get("current_instruction") or "")[:64]
        orch_sub = f"Gate: {last_gate} (#{st_dict.get('iteration_count', 0)}) | {cmd_short}"
    else:
        orch_sub = t(lang, "svg_orch_standby")

    esc_col = "#EF4444" if escalated_up else "#475569"
    esc_cls = "flow-active" if escalated_up else ""
    esc_w = "2.8" if escalated_up else "1.4"
    esc_op = "1.0" if escalated_up else "0.45"

    par_active = active_node in ("diagnostic_engineer", "stock_precheck")
    par_col = "#38BDF8" if par_active else ("#10B981" if (diag_done and pre_done) else "#475569")
    par_cls = "flow-active" if par_active else ""

    stock_flow_cls = "flow-active" if active_node == "stock_agent" else ""
    tech_flow_cls = "flow-active" if active_node == "field_technician" else ""
    clr_flow_cls = "flow-active" if active_node == "qa_clearance" else ""
    anl_flow_cls = "flow-active" if active_node == "analyst_capa" else ""

    return _strip_html_indent(
        f"""
        <div class="svg-panel">
        <svg viewBox="0 0 1180 468" width="100%" style="max-height:385px;display:block;" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="a-cyan" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#38BDF8"/></marker>
            <marker id="a-green" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#10B981"/></marker>
            <marker id="a-slate" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#475569"/></marker>
            <marker id="a-red" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#EF4444"/></marker>
          </defs>

          <rect x="8" y="8" width="1164" height="118" rx="10" fill="#0F172A" stroke="#1E293B"/>
          <text x="22" y="25" fill="#38BDF8" font-size="11" font-weight="800" font-family="sans-serif">{html.escape(t(lang, 'svg_tier3'))}</text>

          <rect x="8" y="136" width="1164" height="162" rx="10" fill="#0D1628" stroke="#1E293B"/>
          <text x="22" y="153" fill="#38BDF8" font-size="11" font-weight="800" font-family="sans-serif">{html.escape(t(lang, 'svg_tier2'))}</text>

          <rect x="8" y="308" width="1164" height="150" rx="10" fill="#0F172A" stroke="#1E293B"/>
          <text x="22" y="325" fill="#38BDF8" font-size="11" font-weight="800" font-family="sans-serif">{html.escape(t(lang, 'svg_tier1'))}</text>

          <rect x="26" y="38" width="210" height="72" rx="10" fill="{sensor_f}" stroke="{sensor_s}" stroke-width="2.2"/>
          <text x="131" y="65" text-anchor="middle" fill="#F8FAFC" font-size="13.5" font-weight="700" font-family="sans-serif">{html.escape(t(lang, 'svg_sensor_title'))}</text>
          <text x="131" y="87" text-anchor="middle" fill="#94A3B8" font-size="11" font-family="sans-serif">{html.escape(str(st_dict.get('machine_id', 'TP-01 / RP-04 / PK-02')))} (&gt;2.5 Watch)</text>

          <g class="{orch_cls}">
            <rect x="305" y="34" width="470" height="78" rx="12" fill="{orch_f}" stroke="{orch_s}" stroke-width="2.6"/>
            <text x="540" y="61" text-anchor="middle" fill="#F8FAFC" font-size="15" font-weight="800" font-family="sans-serif">{html.escape(t(lang, 'svg_orch_title'))}</text>
            <text x="540" y="85" text-anchor="middle" fill="#FDE68A" font-size="11" font-weight="700" font-family="sans-serif">{html.escape(orch_sub)}</text>
          </g>

          <g class="{c_cls}">
            <rect x="845" y="38" width="182" height="72" rx="10" fill="{c_f}" stroke="{c_s}" stroke-width="2.2"/>
            <text x="936" y="65" text-anchor="middle" fill="#F8FAFC" font-size="13" font-weight="700" font-family="sans-serif">{html.escape(t(lang, 'svg_clr_title'))}</text>
            <text x="936" y="86" text-anchor="middle" fill="#CBD5E1" font-size="11" font-family="sans-serif">{html.escape(t(lang, 'svg_clr_sub'))}</text>
          </g>

          <rect x="1050" y="38" width="108" height="72" rx="10" fill="{'#052E24' if closed_ok else '#111C33'}" stroke="{'#22C55E' if closed_ok else '#334155'}" stroke-width="2.2"/>
          <text x="1104" y="66" text-anchor="middle" fill="#F8FAFC" font-size="12.5" font-weight="800" font-family="sans-serif">eBMR</text>
          <text x="1104" y="86" text-anchor="middle" fill="#94A3B8" font-size="11" font-family="sans-serif">{'SIGNED OFF' if closed_ok else 'Pending'}</text>

          <line x1="236" y1="74" x2="301" y2="74" class="{ 'flow-active' if state else '' }" stroke="#38BDF8" stroke-width="2.4" marker-end="url(#a-cyan)"/>
          <line x1="775" y1="74" x2="841" y2="74" class="{clr_flow_cls}" stroke="{'#10B981' if clear_ok else ('#38BDF8' if active_node == 'qa_clearance' else '#475569')}" stroke-width="2.4" marker-end="url(#{'a-green' if clear_ok else 'a-cyan'})"/>
          <line x1="1027" y1="74" x2="1046" y2="74" stroke="{'#10B981' if closed_ok else '#475569'}" stroke-width="2.2" marker-end="url(#{'a-green' if closed_ok else 'a-slate'})"/>

          <line x1="540" y1="112" x2="540" y2="145" class="{par_cls}" stroke="{par_col}" stroke-width="2.4"/>
          <line x1="160" y1="145" x2="1010" y2="145" stroke="#334155" stroke-width="2"/>

          <line x1="160" y1="145" x2="160" y2="182" class="{par_cls}" stroke="{par_col}" stroke-width="2.4" marker-end="url(#{'a-green' if diag_ok else 'a-cyan'})"/>
          <g class="{d_cls}">
            <rect x="30" y="186" width="260" height="98" rx="11" fill="{d_f}" stroke="{d_s}" stroke-width="2.4"/>
            <text x="160" y="214" text-anchor="middle" fill="#F8FAFC" font-size="13.5" font-weight="700" font-family="sans-serif">{html.escape(t(lang, 'svg_eng_title'))}</text>
            <text x="160" y="235" text-anchor="middle" fill="#CBD5E1" font-size="11" font-family="sans-serif">{html.escape(t(lang, 'svg_eng_sub1'))}</text>
            <text x="160" y="255" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-family="sans-serif">{html.escape(t(lang, 'svg_eng_sub2'))}</text>
          </g>

          <line x1="440" y1="145" x2="440" y2="182" class="{par_cls}" stroke="{par_col}" stroke-width="2.4" marker-end="url(#{'a-green' if pre_done else 'a-cyan'})"/>
          <g class="{p_cls}">
            <rect x="315" y="186" width="250" height="98" rx="11" fill="{p_f}" stroke="{p_s}" stroke-width="2.4"/>
            <text x="440" y="214" text-anchor="middle" fill="#F8FAFC" font-size="13.5" font-weight="700" font-family="sans-serif">{html.escape(t(lang, 'svg_pre_title'))}</text>
            <text x="440" y="235" text-anchor="middle" fill="#CBD5E1" font-size="11" font-family="sans-serif">{html.escape(t(lang, 'svg_pre_sub1'))}</text>
            <text x="440" y="255" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-family="sans-serif">{html.escape(t(lang, 'svg_pre_sub2'))}</text>
          </g>

          <line x1="725" y1="145" x2="725" y2="182" class="{stock_flow_cls}" stroke="{'#10B981' if stock_ok else ('#38BDF8' if active_node == 'stock_agent' else '#475569')}" stroke-width="2.4" marker-end="url(#{'a-green' if stock_ok else ('a-cyan' if active_node == 'stock_agent' else 'a-slate')})"/>
          <g class="{s_cls}">
            <rect x="595" y="186" width="260" height="98" rx="11" fill="{s_f}" stroke="{s_s}" stroke-width="2.4"/>
            <text x="725" y="214" text-anchor="middle" fill="#F8FAFC" font-size="13.5" font-weight="700" font-family="sans-serif">{html.escape(t(lang, 'svg_stk_title'))}</text>
            <text x="725" y="235" text-anchor="middle" fill="#CBD5E1" font-size="11" font-family="sans-serif">{html.escape(t(lang, 'svg_stk_sub1'))}</text>
            <text x="725" y="255" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-family="sans-serif">{html.escape(t(lang, 'svg_stk_sub2'))}</text>
          </g>

          <line x1="1010" y1="145" x2="1010" y2="182" class="{anl_flow_cls}" stroke="{'#10B981' if analyt_ok else ('#38BDF8' if active_node == 'analyst_capa' else '#475569')}" stroke-width="2.4" marker-end="url(#{'a-green' if analyt_ok else ('a-cyan' if active_node == 'analyst_capa' else 'a-slate')})"/>
          <g class="{a_cls}">
            <rect x="880" y="186" width="260" height="98" rx="11" fill="{a_f}" stroke="{a_s}" stroke-width="2.4"/>
            <text x="1010" y="214" text-anchor="middle" fill="#F8FAFC" font-size="13.5" font-weight="700" font-family="sans-serif">{html.escape(t(lang, 'svg_anl_title'))}</text>
            <text x="1010" y="235" text-anchor="middle" fill="#CBD5E1" font-size="11" font-family="sans-serif">{html.escape(t(lang, 'svg_anl_sub1'))}</text>
            <text x="1010" y="255" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-family="sans-serif">{html.escape(t(lang, 'svg_anl_sub2'))}</text>
          </g>

          <rect x="232" y="156" width="136" height="22" rx="11" fill="#0C4A6E" stroke="#38BDF8" stroke-width="1.2"/>
          <text x="300" y="171" text-anchor="middle" fill="#E0F2FE" font-size="10" font-weight="700" font-family="sans-serif">1. PARALLEL SEND()</text>

          <line x1="725" y1="284" x2="725" y2="338" class="{tech_flow_cls}" stroke="{'#10B981' if maint_ok else ('#38BDF8' if active_node == 'field_technician' else '#475569')}" stroke-width="2.4" marker-end="url(#{'a-green' if maint_ok else ('a-cyan' if active_node == 'field_technician' else 'a-slate')})"/>
          <g class="{t_cls}">
            <rect x="445" y="342" width="560" height="94" rx="12" fill="{t_f}" stroke="{t_s}" stroke-width="2.6"/>
            <text x="725" y="371" text-anchor="middle" fill="#F8FAFC" font-size="14.5" font-weight="800" font-family="sans-serif">{html.escape(t(lang, 'svg_tech_title'))}</text>
            <text x="725" y="393" text-anchor="middle" fill="#CBD5E1" font-size="11.5" font-family="sans-serif">{html.escape(t(lang, 'svg_tech_sub1'))}</text>
            <text x="725" y="414" text-anchor="middle" fill="#94A3B8" font-size="11" font-family="sans-serif">{html.escape(t(lang, 'svg_tech_sub2'))}</text>
          </g>

          <path d="M 445 388 C 160 388, 160 335, 160 288" class="{esc_cls}" fill="none" stroke="{esc_col}" stroke-width="{esc_w}" stroke-dasharray="6,4" opacity="{esc_op}" marker-end="url(#{'a-red' if escalated_up else 'a-slate'})"/>
          <text x="275" y="370" text-anchor="middle" fill="{'#FCA5A5' if escalated_up else '#64748B'}" font-size="11" font-weight="700" font-family="sans-serif">
            {html.escape(t(lang, 'svg_esc_label'))}
          </text>
        </svg>
        </div>
        """
    )


def _provenance_badge_html(source: str, lang: str = "en") -> str:
    src = str(source or "OFFLINE_DEMO").upper()
    if src == "LIVE_LLM":
        return (
            f'<span style="background:#1E3A8A;color:#BFDBFE;border:1px solid #3B82F6;'
            f'border-radius:999px;padding:0.12rem 0.48rem;font-size:0.68rem;font-weight:800;">'
            f'🧠 {html.escape(t(lang, "badge_live_llm"))}</span>'
        )
    if src == "LLM_GUARDRAILED":
        return (
            f'<span style="background:#78350F;color:#FDE68A;border:1px solid #F59E0B;'
            f'border-radius:999px;padding:0.12rem 0.48rem;font-size:0.68rem;font-weight:800;">'
            f'🛡️ {html.escape(t(lang, "badge_guardrailed"))}</span>'
        )
    if src == "HUMAN_QA":
        return (
            f'<span style="background:#713F12;color:#FEF08A;border:1px solid #EAB308;'
            f'border-radius:999px;padding:0.12rem 0.48rem;font-size:0.68rem;font-weight:800;">'
            f'👤 {html.escape(t(lang, "badge_human_qa"))}</span>'
        )
    return (
        f'<span style="background:#1F2937;color:#CBD5E1;border:1px solid #475569;'
        f'border-radius:999px;padding:0.12rem 0.48rem;font-size:0.68rem;font-weight:800;">'
        f'🧪 {html.escape(t(lang, "badge_offline_demo"))}</span>'
    )


def _render_agent_traces_html(traces: List[Dict[str, Any]], lang: str = "en") -> str:
    if not traces:
        return (
            '<div class="spotlight-box" style="text-align:center;color:#94A3B8;">'
            f"{t(lang, 'traces_empty')}"
            "</div>"
        )
    cards: List[str] = []
    step_word = "Adım" if lang == "tr" else "Step"
    for idx, tr in enumerate(reversed(traces), start=1):
        step_num = len(traces) - idx + 1
        node = str(tr.get("node", ""))
        cls = "trace-card"
        if node == "orchestrator":
            cls += " trace-card-orch"
        elif node == "field_technician":
            cls += " trace-card-tech"
        elif node == "qa_clearance":
            cls += " trace-card-qa"
        elif node == "human_qa":
            cls += " trace-card-human"

        acts = tr.get("actions_taken") or []
        acts_str = " | ".join(html.escape(str(a)) for a in acts) if acts else "State inspection"
        prov_badge = _provenance_badge_html(str(tr.get("decision_source", "OFFLINE_DEMO")), lang=lang)
        raw_payload = tr.get("llm_raw_payload")
        raw_details_html = ""
        if raw_payload:
            import json as _json

            pretty_json = html.escape(_json.dumps(raw_payload, indent=2, ensure_ascii=False))
            raw_details_html = (
                f'<details style="margin-top:0.35rem;background:#0B1222;border:1px solid #1E293B;border-radius:6px;padding:0.3rem 0.55rem;">'
                f'<summary style="cursor:pointer;font-size:0.74rem;color:#38BDF8;font-weight:700;">🔍 {html.escape(t(lang, "inspect_raw_llm"))}</summary>'
                f'<pre style="margin:0.35rem 0 0 0;white-space:pre-wrap;word-break:break-word;font-size:0.72rem;color:#93C5FD;max-height:190px;overflow-y:auto;">{pretty_json}</pre>'
                f'</details>'
            )

        cards.append(
            _strip_html_indent(
                f"""
                <div class="{cls}">
                  <div class="trace-head">
                    <div>
                      <span class="trace-actor">{step_word} {step_num}: {html.escape(str(tr.get('agent_name', node)))}</span>
                      <span class="trace-tier"> &bull; {html.escape(str(tr.get('tier', '')))}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.4rem;">
                      {prov_badge}
                      <span style="color:#64748B;font-size:0.71rem;"><code>{html.escape(str(tr.get('timestamp_utc', ''))[-9:])}</code></span>
                    </div>
                  </div>
                  <div class="trace-line"><span class="trace-label">{html.escape(t(lang, 'trace_cmd'))}</span> {html.escape(str(tr.get('command_received', '')))}</div>
                  <div class="trace-line"><span class="trace-label">{html.escape(t(lang, 'trace_think'))}</span> {html.escape(str(tr.get('thinking', '')))}</div>
                  <div class="trace-line"><span class="trace-label">{html.escape(t(lang, 'trace_act'))}</span> <code>{acts_str}</code></div>
                  <div class="trace-line"><span class="trace-label">{html.escape(t(lang, 'trace_verdict'))}</span> {html.escape(str(tr.get('output_verdict', '')))}</div>
                  <div class="trace-line"><span class="trace-label">{html.escape(t(lang, 'trace_next'))}</span> <code>{html.escape(str(tr.get('delegated_to', '')))}</code></div>
                  {raw_details_html}
                </div>
                """
            )
        )
    return f'<div class="trace-scroll">{"".join(cards)}</div>'


def _render_audit_table_html(audit_rows: List[Dict[str, Any]], lang: str = "en") -> str:
    if not audit_rows:
        return (
            '<div class="spotlight-box" style="text-align:center;color:#94A3B8;">'
            f"{t(lang, 'audit_empty')}"
            "</div>"
        )
    rows_html = []
    for row in audit_rows:
        ts = html.escape(str(row.get("timestamp_utc", ""))[-9:])
        step = html.escape(str(row.get("step_number", "")))
        agent = html.escape(str(row.get("agent_name", "")))
        action = html.escape(str(row.get("action_summary", "")))
        notes = html.escape(str(row.get("audit_notes", "")))
        rec_hash = html.escape(str(row.get("record_hash", "")))
        badge = _gate_badge_html(str(row.get("gate_decision", "PENDING")))
        rows_html.append(
            f"<tr>"
            f"<td><b>#{step}</b></td>"
            f"<td><code>{ts}</code></td>"
            f"<td><b>{agent}</b></td>"
            f"<td>{action}</td>"
            f"<td>{badge}</td>"
            f"<td>{notes}<br/><span style='color:#64748B;font-size:0.70rem;'>SHA256: {rec_hash}...</span></td>"
            f"</tr>"
        )

    return _strip_html_indent(
        f"""
        <div class="audit-table-wrap">
        <table class="ebmr-table">
        <thead><tr>
        <th style="width:52px;">{html.escape(t(lang, 'col_step'))}</th>
        <th style="width:82px;">{html.escape(t(lang, 'col_utc'))}</th>
        <th style="width:155px;">{html.escape(t(lang, 'col_actor'))}</th>
        <th style="width:205px;">{html.escape(t(lang, 'col_action'))}</th>
        <th style="width:115px;">{html.escape(t(lang, 'col_gate'))}</th>
        <th>{html.escape(t(lang, 'col_notes'))}</th>
        </tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
        </table></div>
        """
    )


def _get_latest_excel_bytes() -> Tuple[Optional[bytes], str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        res = generate_monthly_excel_report_impl()
        p = Path(res["excel_path"])
        return p.read_bytes(), p.name
    latest = files[0]
    return latest.read_bytes(), latest.name


def _render_executive_closure_report(active_state: MaintenanceState, lang: str) -> None:
    """Render the prominent top-of-page smartboard Executive Final Incident Closure Report."""
    diag_c = active_state.get("diagnostic_report", {}) or {}
    inv_c = active_state.get("inventory_report", {}) or {}
    maint_c = active_state.get("maintenance_report", {}) or {}
    clr_c = active_state.get("clearance_report", {}) or {}
    an_c = active_state.get("analytics_report", {}) or {}
    capa_c = active_state.get("capa_report", {}) or {}
    pr_c = active_state.get("purchase_requisition", {}) or {}
    pre_c = active_state.get("precheck_report", {}) or {}
    replan_c = active_state.get("engineering_replan", {}) or {}
    hdec = active_state.get("human_decisions", {}) or {}

    gate_fin = str(active_state.get("last_gate_decision", "SIGNED_OFF"))
    is_hold = gate_fin in ("HUMAN_REJECTED", "ESCALATED", "REJECTED")
    banner_cls = "exec-hold-banner" if is_hold else "exec-close-banner"
    banner_title = t(lang, "exec_hold_title") if is_hold else t(lang, "exec_close_title")

    pre_rms = float(active_state.get("alarm_vibration_rms", 0.0) or 0.0)
    post_rms = float(maint_c.get("post_repair_vibration_rms", 0.0) or 0.0)
    att_n = int(maint_c.get("attempt_count", 0) or 0)
    drop_pct = ((pre_rms - post_rms) / pre_rms * 100.0) if (pre_rms > 0 and post_rms > 0) else 0.0
    sop_code = str(diag_c.get("sop_code") or "SOP-MNT-001")
    sop_page = int(diag_c.get("page_number") or 1)
    line_status = str(clr_c.get("line_status_after_clearance", "HELD"))
    tail_hash = ((active_state.get("audit_trail") or [{}])[-1]).get("record_hash", "N/A")[:16]

    part_code = str(inv_c.get("reserved_part_code") or diag_c.get("required_part_code") or "Held")
    total_cost = float(an_c.get("total_cost_usd", 0.0) or 0.0)
    part_cost = float(an_c.get("part_cost_usd", inv_c.get("unit_cost_usd", 0.0)) or 0.0)
    labor_cost = float(an_c.get("labor_cost_usd", 0.0) or 0.0)
    loss_cost = float(an_c.get("production_loss_usd", 0.0) or 0.0)
    downtime_m = int(an_c.get("downtime_minutes", 0) or 0)
    mttr_m = int(an_c.get("mttr_minutes", downtime_m) or 0)
    severity = str(an_c.get("damage_severity", "N/A"))

    with st.container():
        st.markdown(
            _strip_html_indent(
                f"""
                <div class="{banner_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.6rem;">
                    <div>
                      <div style="font-size:1.18rem;font-weight:800;color:#F8FAFC;">
                        {html.escape(banner_title)} &mdash;
                        <code>{html.escape(active_state['ticket_id'])}</code> ({html.escape(active_state['machine_id'])})
                      </div>
                      <div style="color:#94A3B8;font-size:0.82rem;margin-top:0.15rem;">
                        {html.escape(t(lang, 'exec_report_subtitle'))} &bull; SHA-256: <code>{html.escape(tail_hash)}</code>
                      </div>
                    </div>
                    {_gate_badge_html(gate_fin)}
                  </div>
                  <div style="color:#E2E8F0;font-size:0.88rem;margin-top:0.45rem;">
                    <b>Asset:</b> {html.escape(str(active_state.get('machine_name', active_state['machine_id'])))} &bull;
                    <b>Failure Mode:</b> {html.escape(str(diag_c.get('failure_mode', active_state.get('alarm_type', 'N/A'))))} &bull;
                    <b>SOP:</b> <code>{html.escape(sop_code)} p.{sop_page}</code> &bull;
                    <b>Line:</b> {html.escape(line_status)}
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # 4 High-Contrast HTML Executive KPI Cards
        is_no_part = bool(diag_c.get("no_part_required") or part_code.upper() == "NONE")
        if post_rms > 0:
            vib_main = f"{pre_rms:.1f} &rarr; {post_rms:.2f} mm/s"
            vib_sub = f"-{drop_pct:.0f}% {html.escape(t(lang, 'exec_reduction_lbl'))} (&le; {GMP_VIBRATION_LIMIT_RMS} GMP)"
        elif is_no_part and gate_fin == "SIGNED_OFF":
            vib_main = f"{pre_rms:.2f} mm/s (OK)"
            vib_sub = f"No Part / Calibration (&le; {GMP_VIBRATION_LIMIT_RMS} GMP)"
        else:
            vib_main = f"{pre_rms:.1f} mm/s (HELD)"
            vib_sub = "Stopped by Human QA"
        att_lbl = (
            ("Revizyon (2 Deneme)" if lang == "tr" else "Gate 3 Realign (2x)")
            if att_n > 1
            else (
                ("İlk Seferde Çözüm" if lang == "tr" else "First-Time-Fix")
                if att_n == 1
                else (
                    ("Parça Değişimi Yok" if lang == "tr" else "Short Route (No Swap)")
                    if (is_no_part and gate_fin == "SIGNED_OFF")
                    else ("Durduruldu" if lang == "tr" else "Work Held")
                )
            )
        )
        part_sub_bits = []
        if inv_c.get("used_alternative"):
            part_sub_bits.append("Alt GMP" if lang == "tr" else "Qualified Alt")
        if pr_c.get("created"):
            part_sub_bits.append(f"PO: {pr_c['pr_id']}")
        if capa_c.get("capa_triggered"):
            part_sub_bits.append(f"CAPA: {capa_c['capa_id']}")
        if not part_sub_bits:
            part_sub_bits.append("No Hardware Swap" if is_no_part else f"Bin: {inv_c.get('bin_location', 'N/A')}")

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            f'<div class="kpi-card" style="border-top:3px solid #10B981;">'
            f'<div class="kpi-label">{html.escape(t(lang, "exec_kpi_vib"))}</div>'
            f'<div class="kpi-val">{vib_main}</div>'
            f'<div class="kpi-sub" style="color:#34D399;">{vib_sub}</div></div>',
            unsafe_allow_html=True,
        )
        k2.markdown(
            f'<div class="kpi-card" style="border-top:3px solid #F59E0B;">'
            f'<div class="kpi-label">{html.escape(t(lang, "exec_kpi_cost"))}</div>'
            f'<div class="kpi-val">${total_cost:,.2f}</div>'
            f'<div class="kpi-sub">Severity: {html.escape(severity)} | MTTR: {mttr_m}m</div></div>',
            unsafe_allow_html=True,
        )
        k3.markdown(
            f'<div class="kpi-card" style="border-top:3px solid #38BDF8;">'
            f'<div class="kpi-label">{html.escape(t(lang, "exec_kpi_dt"))} / {html.escape(t(lang, "exec_kpi_att"))}</div>'
            f'<div class="kpi-val">{downtime_m} min / {att_n}x</div>'
            f'<div class="kpi-sub">{html.escape(att_lbl)}</div></div>',
            unsafe_allow_html=True,
        )
        k4.markdown(
            f'<div class="kpi-card" style="border-top:3px solid #A855F7;">'
            f'<div class="kpi-label">{html.escape(t(lang, "exec_kpi_part"))}</div>'
            f'<div class="kpi-val">{html.escape(part_code)}</div>'
            f'<div class="kpi-sub">{html.escape(" | ".join(part_sub_bits))}</div></div>',
            unsafe_allow_html=True,
        )

        # Two-column Executive Visual Chart + Financial Breakdown (left) & RCA/Governance Proof (right)
        col_l, col_r = st.columns([1.06, 1.0])
        with col_l:
            st.markdown(f"##### 📊 {t(lang, 'exec_sec_vib')}")
            x_lbls = ["Baseline", f"Reported ({pre_rms:.1f})" if is_no_part else f"Alarm ({pre_rms:.1f})"]
            y_vals = [1.20, pre_rms]
            bar_cols = ["#38BDF8", "#10B981" if (is_no_part and pre_rms <= GMP_VIBRATION_LIMIT_RMS) else "#EF4444"]
            attempts_hist = list(maint_c.get("attempts_history") or active_state.get("maintenance_attempts") or [])
            for att in attempts_hist[:3]:
                a_no = att.get("attempt_number", 1)
                a_rms = float(att.get("post_repair_vibration_rms", post_rms) or 0.0)
                x_lbls.append(f"Attempt #{a_no}")
                y_vals.append(a_rms)
                bar_cols.append("#10B981" if a_rms <= GMP_VIBRATION_LIMIT_RMS else "#F59E0B")
            if not attempts_hist and post_rms > 0:
                x_lbls.append("Final Post-Repair")
                y_vals.append(post_rms)
                bar_cols.append("#10B981" if post_rms <= GMP_VIBRATION_LIMIT_RMS else "#EF4444")

            fig_rec = go.Figure(
                data=[
                    go.Bar(
                        x=x_lbls,
                        y=y_vals,
                        marker_color=bar_cols,
                        text=[f"{v:.2f} mm/s" for v in y_vals],
                        textposition="outside",
                    )
                ]
            )
            fig_rec.add_hline(
                y=GMP_VIBRATION_LIMIT_RMS,
                line_dash="dash",
                line_color="#EF4444",
                line_width=2.0,
                annotation_text=f"GMP Limit ({GMP_VIBRATION_LIMIT_RMS} mm/s)",
            )
            fig_rec.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1222",
                plot_bgcolor="#0F172A",
                height=245,
                margin=dict(t=26, b=24, l=38, r=16),
                yaxis=dict(title="mm/s RMS", range=[0, max(y_vals + [4.0]) * 1.22]),
                showlegend=False,
            )
            st.plotly_chart(fig_rec, width="stretch", key="closure_vib_recovery_chart")

            st.markdown(f"##### 💵 {t(lang, 'exec_sec_fin')}")
            pct_p = (part_cost / total_cost * 100.0) if total_cost > 0 else 0.0
            pct_l = (labor_cost / total_cost * 100.0) if total_cost > 0 else 0.0
            pct_d = (loss_cost / total_cost * 100.0) if total_cost > 0 else 0.0
            st.markdown(
                _strip_html_indent(
                    f"""
                    <div class="audit-table-wrap" style="max-height:none;">
                    <table class="ebmr-table">
                      <thead><tr><th>Cost Item</th><th>Basis</th><th>Share</th><th>USD</th></tr></thead>
                      <tbody>
                        <tr><td>Spare Part</td><td>1x <code>{html.escape(part_code)}</code></td><td>{pct_p:.1f}%</td><td><b>${part_cost:,.2f}</b></td></tr>
                        <tr><td>Technician Labor</td><td>{downtime_m} min @ $95/hr</td><td>{pct_l:.1f}%</td><td><b>${labor_cost:,.2f}</b></td></tr>
                        <tr><td>Line Stoppage Loss</td><td>{html.escape(active_state['machine_id'])} idle rate</td><td>{pct_d:.1f}%</td><td><b>${loss_cost:,.2f}</b></td></tr>
                        <tr style="background:#172540;"><td><b>TOTAL IMPACT</b></td><td><b>{html.escape(severity)} | MTTR {mttr_m}m</b></td><td><b>100.0%</b></td><td><b>${total_cost:,.2f}</b></td></tr>
                      </tbody>
                    </table>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown(f"##### 🔍 {t(lang, 'exec_sec_rca')}")
            g2a_st = hdec.get("sql_reservation", {}).get("decision", "AUTO_APPROVED" if not active_state.get("hitl_enabled", True) else "PENDING")
            g3a_st = hdec.get("repair_attempt_1", {}).get("decision", "AUTO_APPROVED" if not active_state.get("hitl_enabled", True) else "PENDING")
            replan_html = (
                f"<br><b>Engineering Replan:</b> <code>{html.escape(str(replan_c.get('sop_ref')))}</code> &mdash; "
                f"{html.escape(str(replan_c.get('corrective_procedure', '')))} ({html.escape(str(replan_c.get('alignment_tolerance', '')))})"
                if replan_c
                else ""
            )
            capa_html = (
                f"<br><b>5-Why CAPA:</b> <code>{html.escape(str(capa_c.get('capa_id')))}</code> &mdash; {html.escape(str(capa_c.get('preventive_action', '')))}"
                if capa_c.get("capa_triggered")
                else ("Tatmin edici (30 gün tekrar eşiği altında)" if lang == "tr" else "Below 30-day repeat CAPA threshold")
            )
            st.markdown(
                _strip_html_indent(
                    f"""
                    <div class="spotlight-box" style="margin-bottom:0.65rem;line-height:1.52;">
                      <b>Root Cause:</b> {html.escape(str(diag_c.get('root_cause', 'Verified via Hybrid SOP RAG + telemetry.')))}<br>
                      <b>SOP Prescription:</b> <code>{html.escape(sop_code)} p.{sop_page}</code> &bull;
                      <b>Req &rarr; Issued:</b> <code>{html.escape(str(diag_c.get('required_part_code', part_code)))}</code> &rarr; <code>{html.escape(part_code)}</code>
                      ({html.escape(str(inv_c.get('bin_location', 'N/A')))}){replan_html}<br>
                      <b>Batch Pre-Check:</b> Lot <code>{html.escape(str(pre_c.get('active_batch_id', 'N/A')))}</code>
                      (Contamination: <b>{html.escape(str(pre_c.get('contamination_risk', 'LOW')))}</b>)<br>
                      <b>CAPA Status:</b> {capa_html if capa_c.get('capa_triggered') else html.escape(capa_html)}
                    </div>
                    <div class="audit-table-wrap" style="max-height:none;">
                    <table class="ebmr-table">
                      <thead><tr><th>{html.escape(t(lang, 'exec_gate_matrix_hdr'))}</th><th>Authority</th><th>State</th></tr></thead>
                      <tbody>
                        <tr><td>Gate 1: Diagnostic & SOP RAG</td><td>Tier 3 Conductor</td><td>{_gate_badge_html('APPROVED' if active_state.get('diagnostic_approved') else gate_fin)}</td></tr>
                        <tr><td>Gate 2A: SQL Warehouse Mutation</td><td>Human QA Lead</td><td>{_gate_badge_html(str(g2a_st))}</td></tr>
                        <tr><td>Gate 3A: Maintenance Cost Budget</td><td>Human QA Lead</td><td>{_gate_badge_html(str(g3a_st))}</td></tr>
                        <tr><td>Gate 3B: Post-Repair Vibration (&le; 2.5)</td><td>Tier 3 Conductor</td><td>{_gate_badge_html('APPROVED' if post_rms > 0 and post_rms <= GMP_VIBRATION_LIMIT_RMS else gate_fin)}</td></tr>
                        <tr><td>Gate 4: QA Line Clearance & Archive</td><td>QA + Conductor</td><td>{_gate_badge_html(gate_fin)}</td></tr>
                      </tbody>
                    </table>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        # Plain-Language Tier 1 -> Tier 3 Orchestra Storyboard
        if lang == "tr":
            story_items = [
                f"<li><b>1. Telemetri & Paralel Teşhis (Kademe 2):</b> Kenar sensörü <b>{pre_rms:.1f} mm/s</b> ile alarm verdi; Teşhis Mühendisi <code>{html.escape(sop_code)} s.{sop_page}</code> referansıyla kök nedeni bulurken Parti Ön Kontrol eşzamanlı <code>{html.escape(str(pre_c.get('active_batch_id', 'LOT')))}</code> partisini QA karantinasına aldı.</li>",
                f"<li><b>2. Stok & İnsan SQL İzni (Kademe 2 + İnsan QA):</b> İstenen <code>{html.escape(str(diag_c.get('required_part_code', 'N/A')))}</code> için depo SQL düşümü öncesinde İnsan QA onayı alındı; ayrılan parça <code>{html.escape(part_code)}</code>"
                + (" <i>(stok=0 sonrası GMP onaylı muadil)</i>" if inv_c.get("used_alternative") else "")
                + (f"; otomatik <code>{html.escape(str(pr_c['pr_id']))}</code> açıldı.</li>" if pr_c.get("created") else ".</li>"),
                f"<li><b>3. Saha Müdahalesi & Revizyon Döngüsü (Kademe 1 &rarr; Kademe 3):</b> Teknisyen <b>{att_n}</b> müdahale gerçekleştirdi; nihai titreşim <b>{post_rms:.2f} mm/s</b> (&le; 2.5 mm/s ISO 20816-3 limiti içinde) olarak doğrulandı.</li>"
                if att_n
                else "<li><b>3. İnsan QA Durdurma Kararı:</b> Fiziksel onarım İnsan QA bütçe kapısında durduruldu; hiçbir saha müdahalesi yapılmadı.</li>",
                f"<li><b>4. Hat Serbest Bırakma & Finansal Mühür (Kademe 3):</b> QA Hat Onayı sonucu <b>{html.escape(line_status)}</b> olarak kapatıldı; <b>${total_cost:,.2f}</b> toplam etki 4 sayfalı Excel'e ve SHA-256 defterine mühürlendi.</li>",
            ]
        else:
            story_items = [
                f"<li><b>1. Edge Trip & Parallel Intake (Tier 2):</b> Continuous sensor tripped at <b>{pre_rms:.1f} mm/s RMS</b>; Diagnostic Engineer pinned root cause to <code>{html.escape(sop_code)} p.{sop_page}</code> while Batch Pre-Check quarantined <code>{html.escape(str(pre_c.get('active_batch_id', 'LOT')))}</code> in parallel.</li>",
                f"<li><b>2. Inventory & Human SQL Approval (Tier 2 + Human QA):</b> Before modifying SQL stock for <code>{html.escape(str(diag_c.get('required_part_code', 'N/A')))}</code>, Human QA reviewed & approved the mutation; issued <code>{html.escape(part_code)}</code>"
                + (" <i>(GMP-qualified alt after stockout)</i>" if inv_c.get("used_alternative") else "")
                + (f" and drafted Auto-PO <code>{html.escape(str(pr_c['pr_id']))}</code>.</li>" if pr_c.get("created") else ".</li>"),
                f"<li><b>3. Field Execution & Gate 3B Verification (Tier 1 &harr; Tier 2/3):</b> Field Technician completed <b>{att_n}</b> repair attempt(s); verified final RMS <b>{post_rms:.2f} mm/s</b> (&le; 2.5 mm/s ISO 20816-3 release limit).</li>"
                if att_n
                else "<li><b>3. Governance Hold:</b> Maintenance work halted at Human QA authorization gate before physical tool entry.</li>",
                f"<li><b>4. QA Line Release & Financial Archival (Tier 3):</b> QA Line Clearance set asset to <b>{html.escape(line_status)}</b>; Analyst sealed <b>${total_cost:,.2f}</b> impact into SQLite, the 4-sheet Excel report, and SHA-256 tail <code>{html.escape(tail_hash)}</code>.</li>",
            ]

        st.markdown(
            _strip_html_indent(
                f"""
                <div class="spotlight-box" style="margin-top:0.45rem;margin-bottom:0.55rem;">
                  <div style="font-weight:800;color:#F8FAFC;margin-bottom:0.35rem;">🎼 {html.escape(t(lang, 'exec_sec_story'))}</div>
                  <ul style="margin:0;padding-left:1.25rem;color:#E2E8F0;font-size:0.84rem;line-height:1.5;">
                    {''.join(story_items)}
                  </ul>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # Executive Dynamic Route & Provenance Proof Strip
        traces_all = list(active_state.get("agent_traces", []) or [])
        route_nodes = [str(tr.get("node", "")) for tr in traces_all] + [f"FINISH ({gate_fin})"]
        live_cnt = sum(1 for tr in traces_all if tr.get("decision_source") == "LIVE_LLM")
        guard_cnt = sum(1 for tr in traces_all if tr.get("decision_source") == "LLM_GUARDRAILED")
        human_cnt = sum(1 for tr in traces_all if tr.get("decision_source") == "HUMAN_QA")
        off_cnt = sum(1 for tr in traces_all if tr.get("decision_source") in (None, "OFFLINE_DEMO"))
        route_str = " &rarr; ".join(html.escape(n) for n in route_nodes)
        st.markdown(
            f'<div class="spotlight-box" style="margin-top:0.2rem;margin-bottom:0.55rem;padding:0.48rem 0.85rem;">'
            f'<b>🧭 {html.escape(t(lang, "exec_route_proof_lbl"))}:</b> <code>{route_str}</code><br>'
            f'<span style="font-size:0.78rem;color:#93C5FD;">'
            f'Live LLM Steps: <b>{live_cnt}</b> &bull; GMP Safety Interlocks: <b>{guard_cnt}</b> &bull; '
            f'Human QA Gates: <b>{human_cnt}</b> &bull; Offline Steps: <b>{off_cnt}</b>'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # 3-Button Executive Action Bar: 2-Page PDF + 4-Sheet Excel + Cited SOP Deep-Link
        dl_c1, dl_c2, dl_c3 = st.columns(3)
        cert_bytes = build_incident_certificate_pdf(active_state)
        dl_c1.download_button(
            t(lang, "exec_dl_pdf"),
            data=cert_bytes,
            file_name=f"Incident_Certificate_{active_state['ticket_id']}.pdf",
            mime="application/pdf",
            width="stretch",
            type="primary",
            key="dl_cert_pdf_closure",
        )
        xl_bytes_c, xl_name_c = _get_latest_excel_bytes()
        if xl_bytes_c:
            dl_c2.download_button(
                t(lang, "exec_dl_xlsx"),
                data=xl_bytes_c,
                file_name=xl_name_c,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                key="dl_monthly_xlsx_closure",
            )
        if dl_c3.button(f"{t(lang, 'exec_btn_sop')} ({sop_code} p.{sop_page})", width="stretch", key="closure_open_sop_btn"):
            st.session_state["inline_doc_code"] = sop_code
            st.session_state["inline_doc_page"] = sop_page
            st.session_state["active_view_idx"] = 1
            st.rerun()


def _run_and_stream_into_placeholders(
    state_to_run: MaintenanceState,
    selected_bay_scenarios: Dict[str, str],
    bay_placeholders: Dict[str, Any],
    svg_placeholder: Any,
    spotlight_placeholder: Any,
    traces_placeholder: Any,
    audit_placeholder: Any,
    header_placeholder: Any,
    lang: str = "en",
    watch_pacing: float = 0.0,
) -> MaintenanceState:
    final_state = state_to_run
    adapter: SimulatedPlantTelemetryAdapter = st.session_state["telemetry_adapter"]
    latest_map = adapter.get_latest_all()

    for node_name, _, merged_state, ctx in stream_maintenance_workflow(state_to_run):
        final_state = merged_state
        _render_top_banner(header_placeholder, final_state, lang=lang)
        for mid, ph in bay_placeholders.items():
            sc_item = DEMO_SCENARIOS[selected_bay_scenarios[mid]]
            override_rms = ctx.latest_vibration_rms if mid == final_state["machine_id"] else None
            ph.markdown(
                _render_factory_bay_card(
                    sc_item,
                    final_state,
                    lang=lang,
                    live_sample=latest_map.get(mid),
                    live_rms_override=override_rms,
                ),
                unsafe_allow_html=True,
            )

        svg_placeholder.markdown(_render_hierarchy_orchestra_svg(node_name, final_state, lang=lang), unsafe_allow_html=True)
        ev_list = final_state.get("agent_events", [])
        if ev_list:
            last_ev = ev_list[-1]
            spotlight_placeholder.markdown(
                f'<div class="spotlight-box"><b>{html.escape(t(lang, "live_step_lbl"))} ({html.escape(str(last_ev.get("node", "")).upper())}):</b> '
                f'{_gate_badge_html(str(last_ev.get("gate_decision", "PENDING")))} &nbsp; '
                f'{html.escape(str(last_ev.get("summary", "")))}</div>',
                unsafe_allow_html=True,
            )
        traces_placeholder.markdown(_render_agent_traces_html(final_state.get("agent_traces", []), lang=lang), unsafe_allow_html=True)
        audit_placeholder.markdown(_render_audit_table_html(final_state.get("audit_trail", []), lang=lang), unsafe_allow_html=True)

        if watch_pacing > 0.0 and not final_state.get("awaiting_human_approval"):
            time.sleep(min(1.2, float(watch_pacing)))

    if final_state.get("protocol_closed") and not final_state.get("escalated"):
        post_rms = float(final_state.get("maintenance_report", {}).get("post_repair_vibration_rms", 1.4))
        adapter.mark_repaired(final_state["machine_id"], post_rms)

    st.session_state["last_state"] = final_state
    return final_state


# ---------------------------------------------------------------------------
# Sidebar: Language Switcher & Global System Config Only (1 Function = 1 Button)
# ---------------------------------------------------------------------------

with st.sidebar:
    lang_choice = st.radio(
        "🌐 Language / Dil",
        options=["🇬🇧 English", "🇹🇷 Türkçe"],
        index=0,
        horizontal=True,
        key="lang_radio",
    )
    lang = "tr" if "Türkçe" in lang_choice else "en"

    active_state_peek: Optional[MaintenanceState] = st.session_state.get("last_state")
    if active_state_peek and active_state_peek.get("awaiting_human_approval"):
        st.warning(t(lang, "hitl_sidebar_alert"))

    st.markdown(t(lang, "sys_config"))

    has_server_key = bool(resolve_api_key("groq")[0] or resolve_api_key("gemini")[0])
    default_exec_idx = 0 if has_server_key else 1

    execution_mode = st.radio(
        t(lang, "exec_engine"),
        options=["live", "offline_demo"],
        format_func=lambda m: t(lang, "mode_live") if m == "live" else t(lang, "mode_offline"),
        index=default_exec_idx,
        key="exec_mode_radio",
        help=t(lang, "mode_help"),
    )

    provider_keys = list(PROVIDER_CATALOG.keys())
    selected_provider = st.selectbox(
        t(lang, "provider_label"),
        options=provider_keys,
        format_func=lambda k: PROVIDER_CATALOG[k]["label"],
        index=0,
        key="provider_select",
    )
    p_cfg = PROVIDER_CATALOG[selected_provider]

    sidebar_api_key = ""
    if p_cfg["requires_key"]:
        sidebar_api_key = st.text_input(
            f"{selected_provider.upper()} API Key",
            type="password",
            placeholder=t(lang, "api_key_placeholder"),
            key="api_key_input",
        )

    resolved_key, key_source = resolve_api_key(selected_provider, sidebar_api_key)
    mismatch_warn = detect_key_provider_mismatch(selected_provider, resolved_key)
    if mismatch_warn:
        st.error(f"⚠️ Key Mismatch: {mismatch_warn}")

    key_fp = hashlib.sha256((resolved_key or "none").encode("utf-8")).hexdigest()[:12]
    if resolved_key and not mismatch_warn:
        live_models, model_err = _cached_live_models(selected_provider, key_fp, resolved_key)
    else:
        live_models, model_err = list(p_cfg["models"]), None

    col_m1, col_m2 = st.columns([3.2, 1.0])
    with col_m2:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄", help=t(lang, "refresh_models_help"), key="refresh_models_btn"):
            _cached_live_models.clear()
            st.rerun()
    with col_m1:
        selected_model = st.selectbox(t(lang, "model_label"), options=live_models, index=0, key="model_select")

    custom_model_id = st.text_input(
        t(lang, "custom_model"),
        value="",
        placeholder=t(lang, "custom_model_ph"),
        key="custom_model_input",
    ).strip()
    if custom_model_id:
        selected_model = custom_model_id

    if execution_mode == "offline_demo":
        st.info(t(lang, "offline_badge_info"))
        if sidebar_api_key.strip() and not mismatch_warn:
            st.success(t(lang, "offline_byok_ready"))
    elif not p_cfg["requires_key"]:
        st.success(f"🟢 Ollama: {resolved_key}")
    elif resolved_key and not mismatch_warn:
        sync_tag = f"{len(live_models)} live models" if not model_err else "Catalog fallback"
        st.success(t(lang, "key_ready", src=key_source, sync=sync_tag))
    elif not resolved_key:
        st.warning(t(lang, "key_missing"))

    st.divider()
    st.markdown(t(lang, "hitl_header"))
    hitl_enabled = st.toggle(
        t(lang, "hitl_toggle"),
        value=True,
        key="hitl_toggle",
        help=t(lang, "hitl_toggle_help"),
    )
    qa_reviewer = st.text_input(t(lang, "qa_reviewer"), value="QA Lead" if lang == "en" else "QA Lideri", key="qa_reviewer_name")

    st.markdown(t(lang, "pacing_header"))
    watch_pacing = st.slider(
        t(lang, "watch_pacing"),
        min_value=0.0,
        max_value=1.2,
        value=0.2,
        step=0.05,
        key="watch_pacing_slider",
        help=t(lang, "watch_pacing_help"),
    )
    pacing_delay = st.slider(
        t(lang, "api_pacing"),
        min_value=0.0,
        max_value=3.0,
        value=float(p_cfg["default_pacing_sec"]),
        step=0.1,
        key="pacing_slider",
    )

    st.divider()
    if st.button(t(lang, "reset_btn"), width="stretch", key="sb_reset_db"):
        reset_demo_environment()
        st.session_state["telemetry_adapter"].reset()
        for k in ["last_state", "auto_trigger_scenario", "auto_trigger_source", "resume_state_to_stream"]:
            st.session_state.pop(k, None)
        st.success(t(lang, "reset_ok"))
        st.rerun()


# ---------------------------------------------------------------------------
# Top Command Banner, 3-View Navigation & Unmissable Global HITL QA Gate Card
# ---------------------------------------------------------------------------

active_state: Optional[MaintenanceState] = st.session_state.get("last_state")
header_placeholder = st.empty()


def _render_top_banner(
    target: Any,
    state_obj: Optional[MaintenanceState],
    lang: str = "en",
    status_override: Optional[str] = None,
) -> None:
    if state_obj:
        tkt = state_obj["ticket_id"]
        machine = state_obj["machine_name"]
        alarm = f"{state_obj['alarm_vibration_rms']} mm/s RMS"
        mode_lbl = t(lang, "mode_live_badge") if state_obj["execution_mode"] == "live" else t(lang, "mode_offline_badge")
        model_lbl = f"{state_obj['llm_provider']}/{state_obj['llm_model']}"
        if status_override:
            badge = _gate_badge_html(status_override)
        elif state_obj.get("error_message"):
            badge = _gate_badge_html("ESCALATED")
        elif state_obj.get("awaiting_human_approval"):
            badge = _gate_badge_html("AWAITING_HUMAN")
        elif state_obj.get("protocol_closed"):
            badge = _gate_badge_html(state_obj.get("last_gate_decision", "SIGNED_OFF"))
        else:
            badge = _gate_badge_html("PENDING")
    else:
        tkt = "MONITORING"
        machine = t(lang, "scope_monitoring")
        alarm = t(lang, "limit_banner", limit=GMP_VIBRATION_LIMIT_RMS)
        mode_lbl = t(lang, "mode_live_badge") if execution_mode == "live" else t(lang, "mode_offline_badge")
        model_lbl = f"{selected_provider}/{selected_model}"
        badge = f'<span class="badge badge-approved">{html.escape(t(lang, "streaming_live"))}</span>'

    hitl_lbl = t(lang, "enforced") if hitl_enabled else t(lang, "autonomous")

    target.markdown(
        _strip_html_indent(
            f"""
            <div class="ct-banner">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.8rem;">
                <div>
                  <div class="ct-title">{html.escape(t(lang, 'banner_title'))}</div>
                  <div class="ct-subtitle">{html.escape(t(lang, 'banner_subtitle'))}</div>
                </div>
                <div>{badge}</div>
              </div>
              <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.6rem;font-size:0.83rem;color:#E2E8F0;">
                <span><b>{html.escape(t(lang, 'ticket'))}:</b> <code>{html.escape(tkt)}</code></span>
                <span><b>{html.escape(t(lang, 'scope'))}:</b> {html.escape(machine)}</span>
                <span><b>{html.escape(t(lang, 'telemetry'))}:</b> <span style="color:#38BDF8;font-weight:700;">{html.escape(alarm)}</span></span>
                <span><b>{html.escape(t(lang, 'engine'))}:</b> {html.escape(mode_lbl)} (<code>{html.escape(model_lbl)}</code>)</span>
                <span><b>{html.escape(t(lang, 'hitl_qa'))}:</b> {html.escape(hitl_lbl)}</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


_render_top_banner(header_placeholder, active_state, lang=lang)

VIEW_LABELS = [t(lang, "view_0"), t(lang, "view_1"), t(lang, "view_2")]
nav_cols = st.columns(3)
for idx, lbl in enumerate(VIEW_LABELS):
    with nav_cols[idx]:
        is_active = int(st.session_state["active_view_idx"]) == idx
        if st.button(
            lbl,
            width="stretch",
            type="primary" if is_active else "secondary",
            key=f"nav_btn_{idx}",
        ):
            st.session_state["active_view_idx"] = idx
            st.rerun()
active_view_idx = int(st.session_state["active_view_idx"])

# Global Unmissable HITL Authorization Card directly below top navigation (visible on ALL views without scrolling!)
if active_state and active_state.get("awaiting_human_approval"):
    pending = active_state.get("pending_approval", {})
    gate_type = str(pending.get("gate_type", "AUTHORIZATION"))
    approval_id = str(pending.get("approval_id", "gate_auth"))

    with st.container():
        st.markdown(
            _strip_html_indent(
                f"""
                <div class="hitl-panel">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.6rem;">
                    <div style="font-size:1.08rem;font-weight:800;color:#FDE68A;">
                      {html.escape(t(lang, 'hitl_banner_title', title=str(pending.get('title', gate_type))))}
                    </div>
                    {_gate_badge_html('AWAITING_HUMAN')}
                  </div>
                  <div style="color:#E2E8F0;font-size:0.89rem;margin-top:0.35rem;">
                    {html.escape(str(pending.get('summary', '')))}
                  </div>
                  <div style="color:#FBBF24;font-size:0.79rem;margin-top:0.25rem;font-weight:600;">
                    {html.escape(t(lang, 'hitl_top_reminder'))}
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        if gate_type == "SQL_MUTATION":
            if pending.get("is_alternative"):
                st.warning(t(lang, "hitl_alt_warning", req=pending.get("requested_part_code"), alt=pending.get("part_code")))
            pr_info = pending.get("purchase_requisition", {})
            if pr_info.get("created"):
                st.info(
                    t(
                        lang,
                        "hitl_auto_po_info",
                        pr_id=pr_info["pr_id"],
                        qty=pr_info["order_qty"],
                        part=pr_info["part_code"],
                        cost=f"{pr_info['est_order_cost_usd']:,.2f}",
                    )
                )
            st.code(pending.get("sql_preview", ""), language="sql")
        else:
            if gate_type == "REWORK_COST" and pending.get("engineering_replan"):
                rep = pending["engineering_replan"]
                st.info(
                    t(
                        lang,
                        "hitl_replan_info",
                        sop_ref=rep.get("sop_ref"),
                        proc=rep.get("corrective_procedure"),
                        tol=rep.get("alignment_tolerance"),
                        tq=rep.get("torque_spec_nm"),
                    )
                )
            cb = pending.get("cost_breakdown", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t(lang, "hitl_part_cost"), f"${cb.get('part_cost_usd', 0):,.2f}")
            c2.metric(t(lang, "hitl_labor_cost"), f"${cb.get('est_labor_cost_usd', 0):,.2f}", f"{cb.get('est_labor_hours', 1.5)} hrs")
            c3.metric(t(lang, "hitl_downtime_loss"), f"${cb.get('est_production_loss_usd', 0):,.2f}", f"{cb.get('est_downtime_min', 60)} min")
            if gate_type == "REWORK_COST":
                c4.metric(
                    t(lang, "hitl_cum_total"),
                    f"${cb.get('est_total_cost_usd', 0):,.2f}",
                    f"+${cb.get('incremental_rework_cost_usd', 0):,.2f} rework",
                    delta_color="inverse",
                )
            else:
                c4.metric(t(lang, "hitl_total_budget"), f"${cb.get('est_total_cost_usd', 0):,.2f}")

        qa_comment = st.text_input(
            t(lang, "hitl_comment_label"),
            value="",
            placeholder=t(lang, "hitl_comment_ph"),
            key=f"hitl_comment_{approval_id}",
        )
        b_app, b_den = st.columns(2)
        approve_clicked = b_app.button(t(lang, "hitl_approve_btn"), width="stretch", type="primary", key="hitl_approve_btn")
        deny_clicked = b_den.button(t(lang, "hitl_deny_btn"), width="stretch", key="hitl_deny_btn")

        if approve_clicked:
            resumed_state = apply_human_decision(
                active_state,
                approval_id=approval_id,
                decision="APPROVED",
                reviewer=qa_reviewer,
                comment=qa_comment,
            )
            resumed_state["language"] = lang
            st.session_state["last_state"] = resumed_state
            st.session_state["resume_state_to_stream"] = resumed_state
            st.session_state["active_view_idx"] = 0
            st.rerun()
        elif deny_clicked:
            held_state = apply_human_decision(
                active_state,
                approval_id=approval_id,
                decision="REJECTED",
                reviewer=qa_reviewer,
                comment=qa_comment or ("İnsan QA tarafından reddedildi." if lang == "tr" else "Denied by Human QA Reviewer."),
            )
            st.session_state["last_state"] = held_state
            st.rerun()


# ---------------------------------------------------------------------------
# View 1: Always-On Oscilloscope, 9-Fault Bays, Animated Orchestra & Traces
# ---------------------------------------------------------------------------

if active_view_idx == 0:
    adapter: SimulatedPlantTelemetryAdapter = st.session_state["telemetry_adapter"]
    has_open_ticket = bool(active_state and not active_state.get("protocol_closed"))

    if active_state and active_state.get("protocol_closed") and not active_state.get("error_message"):
        _render_executive_closure_report(active_state, lang)

    mon_c1, mon_c2, mon_c3, mon_c4, mon_c5 = st.columns([1.75, 0.85, 0.95, 0.95, 0.95])
    with mon_c1:
        st.markdown(t(lang, "osc_title"))
    with mon_c2:
        stream_active = st.toggle(t(lang, "toggle_live_stream"), value=True, key="stream_active_toggle")
    with mon_c3:
        auto_drift = st.toggle(
            t(lang, "toggle_auto_drift"),
            value=True,
            key="auto_anomaly_toggle",
            help=t(lang, "toggle_auto_drift_help"),
        )
        adapter.set_auto_anomaly(bool(auto_drift and not has_open_ticket))
    with mon_c4:
        progressive_ramp = st.toggle(t(lang, "toggle_progressive"), value=False, key="progressive_ramp_toggle")
    with mon_c5:
        high_turb = st.toggle(t(lang, "toggle_high_variance"), value=False, key="turbulence_toggle")
        adapter.set_turbulence_mode("high" if high_turb else "normal")

    if auto_drift and not has_open_ticket:
        st.caption(t(lang, "auto_drift_active_caption"))

    paused_global = bool(active_state and active_state.get("awaiting_human_approval"))
    run_interval = 1.5 if (stream_active and not paused_global) else None

    @st.fragment(run_every=run_interval)
    def _live_oscilloscope_fragment() -> None:
        cur_adapter: SimulatedPlantTelemetryAdapter = st.session_state["telemetry_adapter"]
        cur_state = st.session_state.get("last_state")
        paused_for_hitl = bool(cur_state and cur_state.get("awaiting_human_approval"))

        tripped_sc: Optional[str] = None
        if st.session_state.get("stream_active_toggle", True) and not paused_for_hitl:
            tripped_sc = cur_adapter.step_tick()

        fig = go.Figure()
        palette = {"TP-01": "#38BDF8", "RP-04": "#A855F7", "PK-02": "#10B981"}
        for mid, col_hex in palette.items():
            hist = cur_adapter.get_history(mid)
            fig.add_trace(
                go.Scatter(
                    x=[h["timestamp"] for h in hist],
                    y=[h["rms"] for h in hist],
                    mode="lines+markers",
                    name=f"{mid} ({hist[-1]['rms']:.2f} mm/s)" if hist else mid,
                    line=dict(color=col_hex, width=2.4),
                    marker=dict(size=4),
                )
            )
        fig.add_hline(y=2.5, line_dash="dash", line_color="#F59E0B", line_width=1.8, annotation_text="Warning 2.5 mm/s")
        fig.add_hline(y=4.5, line_dash="dash", line_color="#EF4444", line_width=2.2, annotation_text="Auto-Trip 4.5 mm/s (Zone D)")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1222",
            plot_bgcolor="#0F172A",
            height=210,
            margin=dict(t=12, b=22, l=45, r=20),
            yaxis=dict(title="mm/s RMS", range=[0, 12.2]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, width="stretch", key="live_oscilloscope_chart")

        snap = cur_adapter.get_latest_all()
        s_rows = []
        for mid in ("TP-01", "RP-04", "PK-02"):
            r = snap.get(mid, {})
            rms_v = float(r.get("rms", 1.2))
            z_badge = _gate_badge_html("APPROVED" if rms_v <= 2.5 else ("REJECTED" if rms_v > 4.5 else "ESCALATED"))
            zone_lbl = localize_zone(str(r.get("zone", "Zone A (Good)")), lang)
            s_rows.append(
                f"<tr>"
                f"<td><b>{mid}</b></td>"
                f"<td><b>{rms_v:.2f} mm/s</b></td>"
                f"<td>{r.get('radial_x', 0):.2f} / {r.get('radial_y', 0):.2f} / {r.get('axial_z', 0):.2f}</td>"
                f"<td>{r.get('peak_g', 0):.2f} g</td>"
                f"<td>{r.get('kurtosis', 3.0):.2f}</td>"
                f"<td>{r.get('freq_hz', 0):.1f} Hz</td>"
                f"<td>{r.get('temp_c', 0):.1f} &deg;C</td>"
                f"<td>{r.get('motor_amps', 0):.1f} A</td>"
                f"<td>{html.escape(zone_lbl)} {z_badge}</td>"
                f"</tr>"
            )
        st.markdown(
            _strip_html_indent(
                f"""
                <table class="sensor-strip">
                <thead><tr>
                <th>{html.escape(t(lang, 'col_asset'))}</th>
                <th>{html.escape(t(lang, 'col_rms'))}</th>
                <th>{html.escape(t(lang, 'col_triaxial'))}</th>
                <th>{html.escape(t(lang, 'col_peak_g'))}</th>
                <th>{html.escape(t(lang, 'col_kurtosis'))}</th>
                <th>{html.escape(t(lang, 'col_freq'))}</th>
                <th>{html.escape(t(lang, 'col_temp'))}</th>
                <th>{html.escape(t(lang, 'col_amps'))}</th>
                <th>{html.escape(t(lang, 'col_zone'))}</th>
                </tr></thead>
                <tbody>{''.join(s_rows)}</tbody>
                </table>
                """
            ),
            unsafe_allow_html=True,
        )

        if tripped_sc:
            st.session_state["auto_trigger_scenario"] = tripped_sc
            st.session_state["auto_trigger_source"] = "AUTONOMOUS_DRIFT_DETECTION"
            st.rerun()

    _live_oscilloscope_fragment()

    # 3 Equipment Bay Cards with 3 Real Faults Per Machine (1 Button = 1 Function)
    bay_cols = st.columns(3)
    bay_placeholders: Dict[str, Any] = {}
    selected_bay_scenarios: Dict[str, str] = {}
    latest_map = adapter.get_latest_all()
    triggered_scenario: Optional[str] = st.session_state.pop("auto_trigger_scenario", None)
    trigger_src = st.session_state.pop("auto_trigger_source", "EDGE_SENSOR_AUTO_TRIP")
    pending_resume: Optional[MaintenanceState] = st.session_state.pop("resume_state_to_stream", None)

    for idx, mid in enumerate(["TP-01", "RP-04", "PK-02"]):
        sc_options = BAY_SCENARIO_GROUPS[mid]
        with bay_cols[idx]:
            chosen_sc_id = st.selectbox(
                t(lang, "bay_fault_select", mid=mid),
                options=sc_options,
                format_func=lambda sid: get_scenario_label(sid, lang, DEMO_SCENARIOS[sid]["label"]),
                index=0,
                key=f"bay_fault_select_{mid}",
            )
            selected_bay_scenarios[mid] = chosen_sc_id
            sc = DEMO_SCENARIOS[chosen_sc_id]
            bay_placeholders[mid] = st.empty()
            bay_placeholders[mid].markdown(
                _render_factory_bay_card(sc, active_state, lang=lang, live_sample=latest_map.get(mid)),
                unsafe_allow_html=True,
            )
            if st.button(t(lang, "bay_inject_btn", mid=mid), width="stretch", key=f"bay_inject_btn_{mid}"):
                immediate_trip = adapter.inject_fault(chosen_sc_id, progressive=progressive_ramp)
                if immediate_trip:
                    triggered_scenario = immediate_trip
                    trigger_src = "EDGE_SENSOR_TRIP_INJECTED"

    with st.expander(t(lang, "custom_box_title"), expanded=False):
        st.caption(t(lang, "custom_box_desc"))
        cc1, cc2 = st.columns([1.0, 1.0])
        with cc1:
            custom_mid = st.selectbox(
                t(lang, "custom_asset_lbl"),
                options=["TP-01", "RP-04", "PK-02"],
                index=0,
                key="custom_asset_select",
            )
        with cc2:
            custom_rms = st.number_input(
                t(lang, "custom_rms_lbl"),
                min_value=0.4,
                max_value=14.0,
                value=1.9,
                step=0.1,
                key="custom_rms_input",
            )
        custom_sym = st.text_area(
            t(lang, "custom_symptom_lbl"),
            value=t(lang, "custom_symptom_default"),
            height=68,
            key="custom_symptom_input",
        )
        custom_dispatch_clicked = st.button(
            t(lang, "custom_dispatch_btn"),
            width="stretch",
            key="custom_incident_dispatch_btn",
        )

    svg_placeholder = st.empty()
    spotlight_placeholder = st.empty()

    col_trace, col_audit = st.columns([1.08, 1.0])
    with col_trace:
        st.markdown(t(lang, "traces_header"))
        traces_placeholder = st.empty()
    with col_audit:
        st.markdown(t(lang, "audit_header"))
        audit_placeholder = st.empty()

    if pending_resume is not None:
        active_state = _run_and_stream_into_placeholders(
            state_to_run=pending_resume,
            selected_bay_scenarios=selected_bay_scenarios,
            bay_placeholders=bay_placeholders,
            svg_placeholder=svg_placeholder,
            spotlight_placeholder=spotlight_placeholder,
            traces_placeholder=traces_placeholder,
            audit_placeholder=audit_placeholder,
            header_placeholder=header_placeholder,
            lang=lang,
            watch_pacing=watch_pacing,
        )
        st.rerun()
    elif custom_dispatch_clicked:
        if execution_mode == "live" and p_cfg["requires_key"] and (not resolved_key or mismatch_warn):
            svg_placeholder.markdown(_render_hierarchy_orchestra_svg("standby", active_state, lang=lang), unsafe_allow_html=True)
            st.error(
                mismatch_warn
                or t(lang, "key_missing_run_err", mid=custom_mid, prov=selected_provider.upper())
            )
        else:
            init_custom = create_custom_incident_state(
                machine_id=custom_mid,
                alarm_vibration_rms=float(custom_rms),
                symptom_summary=str(custom_sym),
                execution_mode=execution_mode,
                llm_provider=selected_provider,
                llm_model=selected_model,
                api_key=resolved_key,
                pacing_delay_sec=pacing_delay,
                hitl_enabled=hitl_enabled,
                language=lang,
            )
            _render_top_banner(header_placeholder, init_custom, lang=lang, status_override="PENDING")
            svg_placeholder.markdown(_render_hierarchy_orchestra_svg("orchestrator", init_custom, lang=lang), unsafe_allow_html=True)
            active_state = _run_and_stream_into_placeholders(
                state_to_run=init_custom,
                selected_bay_scenarios=selected_bay_scenarios,
                bay_placeholders=bay_placeholders,
                svg_placeholder=svg_placeholder,
                spotlight_placeholder=spotlight_placeholder,
                traces_placeholder=traces_placeholder,
                audit_placeholder=audit_placeholder,
                header_placeholder=header_placeholder,
                lang=lang,
                watch_pacing=watch_pacing,
            )
            st.rerun()
    elif triggered_scenario:
        if execution_mode == "live" and p_cfg["requires_key"] and (not resolved_key or mismatch_warn):
            svg_placeholder.markdown(_render_hierarchy_orchestra_svg("standby", active_state, lang=lang), unsafe_allow_html=True)
            st.error(
                mismatch_warn
                or t(lang, "key_missing_run_err", mid=DEMO_SCENARIOS[triggered_scenario]["machine_id"], prov=selected_provider.upper())
            )
        else:
            init_state = create_initial_state(
                scenario_id=triggered_scenario,
                execution_mode=execution_mode,
                llm_provider=selected_provider,
                llm_model=selected_model,
                api_key=resolved_key,
                pacing_delay_sec=pacing_delay,
                hitl_enabled=hitl_enabled,
                trigger_source=trigger_src,
                language=lang,
            )
            _render_top_banner(header_placeholder, init_state, lang=lang, status_override="PENDING")
            svg_placeholder.markdown(_render_hierarchy_orchestra_svg("orchestrator", init_state, lang=lang), unsafe_allow_html=True)
            active_state = _run_and_stream_into_placeholders(
                state_to_run=init_state,
                selected_bay_scenarios=selected_bay_scenarios,
                bay_placeholders=bay_placeholders,
                svg_placeholder=svg_placeholder,
                spotlight_placeholder=spotlight_placeholder,
                traces_placeholder=traces_placeholder,
                audit_placeholder=audit_placeholder,
                header_placeholder=header_placeholder,
                lang=lang,
                watch_pacing=watch_pacing,
            )
            st.rerun()
    else:
        svg_placeholder.markdown(_render_hierarchy_orchestra_svg("standby", active_state, lang=lang), unsafe_allow_html=True)
        if active_state and active_state.get("agent_events"):
            last_ev = active_state["agent_events"][-1]
            spotlight_placeholder.markdown(
                f'<div class="spotlight-box"><b>{html.escape(t(lang, "latest_gate_lbl"))}</b> '
                f'{_gate_badge_html(str(last_ev.get("gate_decision", "PENDING")))} &nbsp; '
                f'{html.escape(str(last_ev.get("summary", "")))}</div>',
                unsafe_allow_html=True,
            )

    if active_state and active_state.get("error_message"):
        st.error(active_state["error_message"])

    traces_placeholder.markdown(
        _render_agent_traces_html(active_state.get("agent_traces", []) if active_state else [], lang=lang),
        unsafe_allow_html=True,
    )
    audit_placeholder.markdown(
        _render_audit_table_html(active_state.get("audit_trail", []) if active_state else [], lang=lang),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# View 2: SOPs & Machine Operation Manuals (24-Page Clean 2-Col PDF Reader)
# ---------------------------------------------------------------------------

if active_view_idx == 1:
    st.markdown(t(lang, "view2_header"))
    docs = list_controlled_documents()
    if "inline_doc_code" not in st.session_state and docs:
        st.session_state["inline_doc_code"] = docs[0]["doc_code"]
        st.session_state["inline_doc_page"] = 1

    col_list, col_viewer = st.columns([1.0, 1.55])

    with col_list:
        st.markdown(t(lang, "view2_lib"))
        for doc in docs:
            is_sel = st.session_state.get("inline_doc_code") == doc["doc_code"]
            card_cls = "doc-card doc-card-active" if is_sel else "doc-card"
            st.markdown(
                _strip_html_indent(
                    f"""
                    <div class="{card_cls}">
                      <span class="badge badge-pending">{html.escape(doc['badge'])}</span>
                      <span style="color:#94A3B8;font-size:0.78rem;margin-left:0.4rem;">{doc['pages']}p</span>
                      <div style="font-size:0.96rem;font-weight:800;color:#F8FAFC;margin-top:0.35rem;">
                        {html.escape(doc['doc_code'])} — {html.escape(doc['title'])}
                      </div>
                      <div style="font-size:0.81rem;color:#CBD5E1;margin-top:0.25rem;">
                        {html.escape(doc['description'])}
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            if st.button(
                t(lang, "view2_viewing") if is_sel else t(lang, "view2_open"),
                width="stretch",
                type="primary" if is_sel else "secondary",
                key=f"lib_open_{doc['doc_code']}",
            ):
                st.session_state["inline_doc_code"] = doc["doc_code"]
                st.session_state["inline_doc_page"] = 1
                st.rerun()

    with col_viewer:
        sel_doc = find_document_by_code_or_name(st.session_state.get("inline_doc_code", "GUIDE-OPS-000"))
        if sel_doc:
            h1, h2 = st.columns([2.5, 1.0])
            with h1:
                st.markdown(f"#### {sel_doc['doc_code']} — {sel_doc['title']}")
            with h2:
                st.download_button(
                    t(lang, "view2_dl"),
                    data=Path(sel_doc["path"]).read_bytes(),
                    file_name=sel_doc["filename"],
                    mime="application/pdf",
                    width="stretch",
                    key=f"dl_pdf_{sel_doc['doc_code']}",
                )

            p_count = int(sel_doc["pages"])
            cur_p = max(1, min(int(st.session_state.get("inline_doc_page", 1)), p_count))
            chosen_p = st.radio(
                t(lang, "view2_page"),
                options=list(range(1, p_count + 1)),
                index=cur_p - 1,
                format_func=lambda p: t(lang, "view2_page_fmt", p=p),
                horizontal=True,
                key=f"pdf_page_radio_{sel_doc['doc_code']}",
            )
            st.session_state["inline_doc_page"] = int(chosen_p)

            mtime = Path(sel_doc["path"]).stat().st_mtime
            png_bytes = _cached_pdf_page_png(sel_doc["path"], int(chosen_p), mtime)
            st.image(png_bytes, caption=f"{sel_doc['filename']} — {chosen_p} / {p_count}", width="stretch")

            with st.expander(t(lang, "view2_text_exp"), expanded=False):
                st.text(extract_pdf_page_text(sel_doc["path"], int(chosen_p)))


# ---------------------------------------------------------------------------
# View 3: Technical Evidence, Line Clearance, CAPA, KPIs & 4-Sheet Excel
# ---------------------------------------------------------------------------

if active_view_idx == 2:
    st.markdown(t(lang, "view3_header"))
    t_rag, t_sql, t_vib, t_capa, t_kpi = st.tabs(
        [
            t(lang, "tab_rag"),
            t(lang, "tab_sql"),
            t(lang, "tab_vib"),
            t(lang, "tab_capa"),
            t(lang, "tab_kpi"),
        ]
    )

    with t_rag:
        diag = active_state.get("diagnostic_report", {}) if active_state else {}
        replan = active_state.get("engineering_replan", {}) if active_state else {}
        rag_chunks = (active_state.get("rag_chunks") if active_state else None) or diag.get("rag_evidence", [])

        c_d1, c_d2 = st.columns([1.0, 1.35])
        with c_d1:
            st.markdown("#### Tier 2 Reliability Engineer Diagnosis" if lang == "en" else "#### 2. Kademe Güvenilirlik Mühendisi Teşhisi")
            if diag:
                st.success(f"**SOP:** `{diag.get('sop_code')}` (p.{diag.get('page_number')})")
                st.markdown(f"**Failure Mode / Arıza:** {diag.get('failure_mode')}")
                st.markdown(f"**Root Cause / Kök Neden:** {diag.get('root_cause')}")
                st.markdown(f"**Required Part / Parça:** `{diag.get('required_part_code')}`")
                if diag.get("required_tools"):
                    st.markdown("**Tooling:** " + ", ".join(f"`{t_item}`" for t_item in diag["required_tools"]))
                if replan:
                    st.warning(
                        f"**Escalation Realignment Spec (`{replan.get('sop_ref')}`):** "
                        f"{replan.get('corrective_procedure')} ({replan.get('alignment_tolerance')})"
                    )
                if st.button(t(lang, "jump_sop_btn"), width="stretch", type="primary", key="jump_cited_pdf_btn"):
                    st.session_state["inline_doc_code"] = str(diag.get("sop_code", "SOP-MNT-001"))
                    st.session_state["inline_doc_page"] = int(diag.get("page_number", 1))
                    st.session_state["active_view_idx"] = 1
                    st.rerun()
            else:
                st.info("Inject or wait for autonomous drift on the Factory Floor to view SOP citations." if lang == "en" else "SOP atıflarını görmek için Fabrika Zemininde arıza tetikleyin veya otonom sapma bekleyin.")

        with c_d2:
            st.markdown("#### Hybrid TF-IDF + BM25 RAG Snippets" if lang == "en" else "#### Hibrit TF-IDF + BM25 RAG Parçaları")
            if rag_chunks:
                for idx, chunk in enumerate(rag_chunks):
                    with st.expander(
                        f"Rank #{chunk.get('rank')} | {chunk.get('sop_code')} (p.{chunk.get('page_number')}) — Score: {chunk.get('score')}",
                        expanded=(idx == 0),
                    ):
                        st.caption(f"BM25: {chunk.get('bm25_score')} | TF-IDF: {chunk.get('tfidf_score')} | {chunk.get('filename')}")
                        st.text(chunk.get("excerpt", ""))
            else:
                st.info("Retrieved PDF chunks with exact page numbers will appear here." if lang == "en" else "Sayfa numaralı PDF parçaları burada listelenecek.")

    with t_sql:
        inv_rep = active_state.get("inventory_report", {}) if active_state else {}
        pre_rep = active_state.get("precheck_report", {}) if active_state else {}
        sql_queries = active_state.get("sql_queries", []) if active_state else []

        if pre_rep:
            st.info(
                f"**Batch Risk ({pre_rep.get('active_batch_id')}):** "
                f"{pre_rep.get('contamination_risk')} | {pre_rep.get('qa_batch_hold_action')}"
            )

        c_s1, c_s2 = st.columns([1.05, 1.25])
        with c_s1:
            st.markdown("#### Material Requisition & Stock Gate" if lang == "en" else "#### Malzeme Talebi & Stok Kapısı")
            if inv_rep:
                if inv_rep.get("awaiting_sql_approval"):
                    st.warning(f"🔐 `{inv_rep.get('proposed_part_code')}` awaiting Human QA SQL authorization.")
                elif inv_rep.get("used_alternative"):
                    st.warning(
                        f"⚠️ Primary `{inv_rep.get('requested_part_code')}` Stockout (qty=0) -> "
                        f"Reserved Alt **`{inv_rep.get('reserved_part_code')}`**"
                    )
                elif inv_rep.get("reserved"):
                    st.success(
                        f"✅ Reserved **`{inv_rep.get('reserved_part_code')}`** from `{inv_rep.get('bin_location')}` "
                        f"(${inv_rep.get('unit_cost_usd', 0):,.2f}) | Rem: `{inv_rep.get('remaining_stock_qty')}`"
                    )
                else:
                    st.error("Stockout escalated to procurement.")
            else:
                st.info("Trigger a fault to inspect executed SQLite queries." if lang == "en" else "SQLite sorgularını görmek için bir olay başlatın.")

            st.markdown("#### Executed SQLite Statements" if lang == "en" else "#### Çalıştırılan SQLite Sorguları")
            if sql_queries:
                for q in sql_queries:
                    st.code(q.get("sql", ""), language="sql")
                    st.caption(f"Rows: {q.get('row_count')} @ {q.get('timestamp_utc')}")
            else:
                st.caption("No live SQL queries executed yet.")

        with c_s2:
            st.markdown("#### `spare_parts_inventory` (15 Parts)")
            st.dataframe(get_inventory_dataframe(), width="stretch", hide_index=True)
            st.markdown("#### `purchase_requisitions`")
            df_prs = get_purchase_requisitions_dataframe()
            if df_prs.empty:
                st.caption("No purchase requisitions drafted yet.")
            else:
                st.dataframe(df_prs, width="stretch", hide_index=True)

    with t_vib:
        maint_rep = active_state.get("maintenance_report", {}) if active_state else {}
        clear_rep = active_state.get("clearance_report", {}) if active_state else {}
        telemetry_list = active_state.get("telemetry_readings", []) if active_state else []

        if active_state and (telemetry_list or maint_rep):
            sc_meta = DEMO_SCENARIOS.get(active_state["scenario_id"], {})
            labels: List[str] = ["Baseline", "Alarm Trip"]
            values: List[float] = [
                float(sc_meta.get("baseline_rms", 1.2)),
                float(active_state["alarm_vibration_rms"]),
            ]
            colors: List[str] = ["#38BDF8", "#F59E0B"]
            post_readings = [t_r for t_r in telemetry_list if str(t_r.get("phase", "")).startswith("post_repair")]
            if post_readings:
                for r in post_readings:
                    rms = float(r.get("post_repair_vibration_rms", 1.5))
                    labels.append(f"Attempt #{r.get('attempt_number', 1)}")
                    values.append(rms)
                    colors.append("#10B981" if rms <= GMP_VIBRATION_LIMIT_RMS else "#EF4444")
            elif maint_rep:
                rms = float(maint_rep.get("post_repair_vibration_rms", 1.5))
                labels.append("Post-Repair")
                values.append(rms)
                colors.append("#10B981" if rms <= GMP_VIBRATION_LIMIT_RMS else "#EF4444")

            fig_vib = go.Figure()
            fig_vib.add_trace(
                go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2f} mm/s" for v in values], textposition="outside")
            )
            fig_vib.add_hline(
                y=GMP_VIBRATION_LIMIT_RMS,
                line_dash="dash",
                line_color="#EF4444",
                line_width=3,
                annotation_text=f"ISO 20816-3 / cGMP Limit ({GMP_VIBRATION_LIMIT_RMS} mm/s RMS)",
                annotation_position="top right",
            )
            fig_vib.update_layout(
                title=f"Vibration Telemetry Progression — {active_state['machine_name']}",
                yaxis_title="mm/s RMS",
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                height=350,
                margin=dict(t=55, b=35, l=50, r=30),
            )
            st.plotly_chart(fig_vib, width="stretch")

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("#### Tier 1 Technician Work Log" if lang == "en" else "#### 1. Kademe Teknisyen İşlem Kaydı")
                if maint_rep.get("attempts_history"):
                    st.dataframe(pd.DataFrame(maint_rep["attempts_history"]), width="stretch", hide_index=True)
            with col_v2:
                st.markdown("#### Tier 3 QA Line Clearance Checklist" if lang == "en" else "#### 3. Kademe QA Hat Onay Listesi")
                if clear_rep and clear_rep.get("checklist"):
                    st.success(f"**{clear_rep.get('line_status_after_clearance')}** @ {clear_rep.get('final_vibration_rms')} mm/s RMS")
                    st.dataframe(pd.DataFrame(clear_rep["checklist"]), width="stretch", hide_index=True)
                else:
                    st.caption("Pending post-repair QA Line Clearance.")
        else:
            st.info("Run any fault scenario to inspect vibration progression and QA Line Clearance." if lang == "en" else "Titreşim ilerleyişini görmek için bir senaryo çalıştırın.")

    with t_capa:
        st.markdown("#### `capa_records` (5-Why CAPA Register)")
        capa_now = active_state.get("capa_report", {}) if active_state else {}
        if capa_now and capa_now.get("capa_triggered"):
            st.warning(
                f"🛡️ **Auto-CAPA:** `{capa_now['capa_id']}` "
                f"(30d Repeat Count `{capa_now['machine_id']}`: **{capa_now['repeat_count_30d']}**)"
            )
            st.markdown(f"**5-Why:** {capa_now.get('five_why_root_cause')}")
            st.markdown(f"**CAPA:** {capa_now.get('preventive_action')}")
        st.dataframe(get_capa_dataframe(), width="stretch", hide_index=True)

    with t_kpi:
        df_month = get_monthly_incidents_dataframe()
        kpis = compute_monthly_kpis(df_month)

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{html.escape(t(lang, "kpi_dt"))}</div>'
            f'<div class="kpi-val">{kpis["total_downtime_min"]} min</div>'
            f'<div class="kpi-sub">{kpis["total_incidents"]} incidents</div></div>',
            unsafe_allow_html=True,
        )
        k2.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{html.escape(t(lang, "kpi_mttr"))}</div>'
            f'<div class="kpi-val">{kpis["avg_mttr_min"]:.1f} min</div>'
            f'<div class="kpi-sub">Mean Time To Repair</div></div>',
            unsafe_allow_html=True,
        )
        k3.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{html.escape(t(lang, "kpi_cost"))}</div>'
            f'<div class="kpi-val">${kpis["total_cost_usd"]:,.0f}</div>'
            f'<div class="kpi-sub">Parts + Labor ($95/hr) + Loss</div></div>',
            unsafe_allow_html=True,
        )
        k4.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{html.escape(t(lang, "kpi_ftf"))}</div>'
            f'<div class="kpi-val">{kpis["first_time_fix_rate_pct"]:.1f}%</div>'
            f'<div class="kpi-sub">Target &ge; 85.0% GMP QA</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        c_ch1, c_ch2 = st.columns(2)
        if not df_month.empty:
            with c_ch1:
                by_machine = df_month.groupby("machine_id", as_index=False)[
                    ["part_cost_usd", "labor_cost_usd", "production_loss_usd"]
                ].sum()
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Bar(name="Spare Parts ($)", x=by_machine["machine_id"], y=by_machine["part_cost_usd"], marker_color="#38BDF8"))
                fig_cost.add_trace(go.Bar(name="Labor ($)", x=by_machine["machine_id"], y=by_machine["labor_cost_usd"], marker_color="#A855F7"))
                fig_cost.add_trace(go.Bar(name="Production Loss ($)", x=by_machine["machine_id"], y=by_machine["production_loss_usd"], marker_color="#F59E0B"))
                fig_cost.update_layout(
                    barmode="stack",
                    title="Cost Breakdown by Machine ($ USD)",
                    template="plotly_dark",
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0F172A",
                    height=320,
                    margin=dict(t=50, b=35, l=45, r=20),
                )
                st.plotly_chart(fig_cost, width="stretch")

            with c_ch2:
                fig_trend = go.Figure()
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_month["ticket_id"],
                        y=df_month["total_cost_usd"],
                        mode="lines+markers",
                        name="Total Cost ($)",
                        line=dict(color="#10B981", width=3),
                        marker=dict(size=8),
                    )
                )
                fig_trend.update_layout(
                    title="Incident Financial Impact Trend ($ USD)",
                    template="plotly_dark",
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0F172A",
                    height=320,
                    margin=dict(t=50, b=35, l=45, r=20),
                )
                st.plotly_chart(fig_trend, width="stretch")

        excel_bytes, excel_name = _get_latest_excel_bytes()
        dl_col1, dl_col2 = st.columns([1.2, 2.8])
        with dl_col1:
            if excel_bytes:
                st.download_button(
                    label=t(lang, "dl_xlsx_view3"),
                    data=excel_bytes,
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                    type="primary",
                    key="dl_monthly_xlsx",
                )
        with dl_col2:
            st.caption(f"4-Sheet Workbook Ready: `{excel_name}` (`KPI Summary`, `Incident Log`, `CAPA & Reorders`, `GMP Audit Trail`)")

        st.markdown("#### `incident_analytics`")
        st.dataframe(df_month, width="stretch", hide_index=True)
