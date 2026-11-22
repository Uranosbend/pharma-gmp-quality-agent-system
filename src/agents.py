"""3-Tier Hierarchical LangGraph Orchestra: Conductor, Tier-2 Engineers/Specialists, Tier-1 Technician, Tier-3 QA."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.llm_config import (
    extract_json_object,
    fetch_live_models,
    get_chat_model,
    invoke_structured_decision,
    invoke_with_retry,
    is_model_not_found_error,
    resolve_fallback_model,
)
from src.seed_data import GMP_VIBRATION_LIMIT_RMS
from src.state import DEMO_SCENARIOS, MaintenanceState, OrchestratorDecision
from src.tools import (
    append_gmp_audit_log,
    assess_batch_risk,
    assess_batch_risk_impl,
    check_alternative_parts,
    check_alternative_parts_impl,
    create_capa_if_repeat_failure,
    create_capa_if_repeat_failure_impl,
    create_purchase_requisition,
    create_purchase_requisition_impl,
    estimate_repair_cost,
    execute_part_replacement,
    execute_part_replacement_impl,
    generate_monthly_excel_report,
    generate_monthly_excel_report_impl,
    get_active_run_context,
    log_incident_analytics,
    log_incident_analytics_impl,
    query_inventory_sql,
    query_inventory_sql_impl,
    read_sensor_telemetry,
    read_sensor_telemetry_impl,
    reserve_spare_part,
    reserve_spare_part_impl,
    run_post_repair_vibration_test,
    run_post_repair_vibration_test_impl,
    run_qa_line_clearance,
    run_qa_line_clearance_impl,
    search_sop_manuals,
    search_sop_manuals_impl,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _next_step_num(state: MaintenanceState) -> int:
    return len(state.get("audit_trail", [])) + 1


def _lang_prompt_hint(state: MaintenanceState) -> str:
    """Instruct Live LLMs to answer human-readable narrative values in Turkish when UI language='tr'."""
    if str(state.get("language", "en")).lower() == "tr":
        return (
            "\nIMPORTANT: Write all human-readable narrative values "
            "(failure_mode, root_cause, instruction, audit_notes, corrective_procedure) "
            "in clear, natural technical Turkish (Türkçe), while keeping exact JSON keys, "
            "SOP codes (SOP-MNT-001), and part codes (BRG-4402) unchanged."
        )
    return ""


_TR_AGENT_NAMES: Dict[str, Tuple[str, str]] = {
    "GMP Quality Conductor (Orchestrator)": ("Kademe 3 Yönetici Orkestratör", "GMP Kalite Şefi (Orkestratör)"),
    "Reliability Engineer (Escalation Mode)": ("Kademe 2 Güvenilirlik Mühendisliği", "Güvenilirlik Mühendisi (Eskalasyon Modu)"),
    "Reliability & Diagnostic Engineer": ("Kademe 2 Güvenilirlik Mühendisliği", "Güvenilirlik & Arıza Teşhis Mühendisi"),
    "Batch & Stock Pre-Check": ("Kademe 2 Operasyon & Parti Kalite", "Parti & Stok Ön Kontrol Ajanı"),
    "Stock & Procurement Agent": ("Kademe 2 Envanter & Satınalma", "Stok & Otomatik Satınalma Ajanı"),
    "Field Maintenance Technician": ("Kademe 1 Saha İcrası", "Saha Bakım Teknisyeni"),
    "QA Line Clearance Officer": ("Kademe 3 Kalite Güvence", "QA Hat Serbest Bırakma Uzmanı"),
    "Quality, Cost & CAPA Analyst": ("Kademe 2/3 Finans & Kalite Analitiği", "Kalite, Maliyet & CAPA Analisti"),
}


def _make_trace(
    node: str,
    tier: str,
    agent_name: str,
    command_received: str,
    thinking: str,
    actions_taken: List[str],
    output_verdict: str,
    delegated_to: str,
    timestamp_utc: Optional[str] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """Build structured watcher trace capturing incoming command, reasoning, actions, and handoff."""
    out_tier, out_name = (
        _TR_AGENT_NAMES.get(agent_name, (tier, agent_name))
        if str(lang).lower() == "tr"
        else (tier, agent_name)
    )
    return {
        "timestamp_utc": timestamp_utc or _utc_now(),
        "node": node,
        "tier": out_tier,
        "agent_name": out_name,
        "command_received": command_received,
        "thinking": thinking,
        "actions_taken": list(actions_taken),
        "output_verdict": output_verdict,
        "delegated_to": delegated_to,
    }


# ---------------------------------------------------------------------------
# Generic Autonomous LLM Tool-Calling Loop + 404 Model Auto-Fallback
# ---------------------------------------------------------------------------


def _run_specialist_tool_loop(
    llm: Any,
    system_prompt: str,
    task_prompt: str,
    tools: List[Any],
    pacing_delay: float,
    max_rounds: int = 5,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute a LangChain bind_tools ReAct loop with retry protection."""
    tool_map: Dict[str, Callable[..., Any]] = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_prompt),
    ]
    executed_tools: List[Dict[str, Any]] = []

    for _ in range(max_rounds):
        ai_msg = invoke_with_retry(llm_with_tools.invoke, messages, pacing_delay=pacing_delay)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            raw_content = getattr(ai_msg, "content", "")
            if isinstance(raw_content, list):
                raw_content = " ".join(
                    str(part.get("text", part) if isinstance(part, dict) else part)
                    for part in raw_content
                )
            return str(raw_content), executed_tools

        for tc in tool_calls:
            t_name = tc.get("name", "")
            t_args = tc.get("args", {}) or {}
            t_id = tc.get("id", t_name)
            tool_obj = tool_map.get(t_name)
            if tool_obj is None:
                obs_str = json.dumps({"error": f"Unknown tool '{t_name}'"})
            else:
                try:
                    obs_str = str(tool_obj.invoke(t_args))
                except Exception as exc:
                    obs_str = json.dumps({"error": f"Tool execution error: {exc}"})

            executed_tools.append(
                {
                    "tool": t_name,
                    "args": t_args,
                    "output_preview": obs_str[:320],
                    "timestamp_utc": _utc_now(),
                }
            )
            messages.append(ToolMessage(content=obs_str, tool_call_id=t_id))

    final_msg = invoke_with_retry(
        llm.invoke,
        messages + [HumanMessage(content="Provide your final JSON summary now based strictly on tool outputs.")],
        pacing_delay=pacing_delay,
    )
    raw_final = getattr(final_msg, "content", "")
    if isinstance(raw_final, list):
        raw_final = " ".join(
            str(part.get("text", part) if isinstance(part, dict) else part) for part in raw_final
        )
    return str(raw_final), executed_tools


def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        data = extract_json_object(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _invoke_live_with_model_fallback(
    state: MaintenanceState,
    call_fn: Callable[[str], Any],
) -> Tuple[Any, str, Optional[str]]:
    """Invoke a live LLM action with automatic 404 model_not_found recovery."""
    provider = state["llm_provider"]
    current_model = state["llm_model"]
    api_key = state.get("api_key")
    tried: List[str] = []

    for attempt in range(3):
        try:
            result = call_fn(current_model)
            note = (
                f"Auto-switched model from '{state['llm_model']}' to accessible '{current_model}' after 404."
                if current_model != state["llm_model"]
                else None
            )
            return result, current_model, note
        except Exception as exc:
            tried.append(current_model)
            if not is_model_not_found_error(exc) or attempt >= 2:
                if is_model_not_found_error(exc):
                    avail, _ = fetch_live_models(provider, api_key)
                    preview = ", ".join(avail[:5]) if avail else "none discovered"
                    raise RuntimeError(
                        f"Model '{current_model}' not found/accessible on {provider}. "
                        f"Available models detected: [{preview}]. Original error: {exc}"
                    ) from exc
                raise
            nxt = resolve_fallback_model(provider, current_model, api_key=api_key, tried_models=tried)
            if not nxt:
                raise
            current_model = nxt

    raise RuntimeError("Exhausted live model fallback attempts.")


# ---------------------------------------------------------------------------
# Tier 3 Conductor: GMP Quality Orchestrator Node
# ---------------------------------------------------------------------------


def _evaluate_orchestra_state(state: MaintenanceState) -> OrchestratorDecision:
    """Deterministic GMP-QA-010 quality gate & 3-tier hierarchy state machine."""
    if state.get("error_message"):
        return OrchestratorDecision(
            next_agent="FINISH",
            instruction="Workflow terminated due to execution error.",
            gate_decision="ESCALATED",
            reviewed_stage="closure",
            audit_notes=str(state["error_message"]),
        )

    diag = state.get("diagnostic_report", {})
    inv = state.get("inventory_report", {})
    maint = state.get("maintenance_report", {})
    clear = state.get("clearance_report", {})
    analyt = state.get("analytics_report", {})
    human_decisions = state.get("human_decisions", {})

    # Phase 0: Edge Alarm -> Parallel Fan-Out (Diagnostic Engineer + Batch/Stock Pre-Check)
    if not state.get("diagnostic_done") or not state.get("precheck_done"):
        return OrchestratorDecision(
            next_agent="parallel_intake",
            instruction=(
                f"Edge vibration trip on {state['machine_id']} ({state['alarm_vibration_rms']} mm/s RMS). "
                "Launch parallel orchestra: Reliability Diagnostic Engineer (SOP RAG) + Batch/Stock Pre-Check."
            ),
            gate_decision="PENDING",
            reviewed_stage="intake",
            audit_notes=(
                f"Sensor alarm ({state['alarm_vibration_rms']} mm/s > {GMP_VIBRATION_LIMIT_RMS} mm/s). "
                "Conducting parallel Diagnostic + Batch-Risk/Inventory fan-out."
            ),
        )

    # Phase 1: Gate 1 Verification -> Stock & Auto-Procurement Agent
    if not state.get("diagnostic_approved"):
        has_sop = bool(diag.get("sop_code")) and bool(diag.get("page_number"))
        has_root = bool(diag.get("root_cause")) and bool(diag.get("required_part_code"))
        if not (has_sop and has_root):
            return OrchestratorDecision(
                next_agent="diagnostic_engineer",
                instruction="REJECTED: Diagnostic Report missing valid SOP citation or spare part code. Re-run RAG.",
                gate_decision="REJECTED",
                reviewed_stage="diagnostic",
                audit_notes="Gate 1 REJECTED per GMP-QA-010 Sec 2: Incomplete SOP citation or root cause.",
            )
        part_code = diag["required_part_code"]
        pre = state.get("precheck_report", {})
        return OrchestratorDecision(
            next_agent="stock_agent",
            instruction=(
                f"Verify SQLite stock for {part_code}, draft Purchase Requisition if <= min_reorder_qty, "
                "and propose reservation for Human QA approval."
            ),
            gate_decision="APPROVED",
            reviewed_stage="diagnostic",
            audit_notes=(
                f"Gate 1 APPROVED: {diag['sop_code']} p.{diag['page_number']}; Batch {pre.get('active_batch_id', 'LOT')} "
                f"risk assessed; dispatching Stock Agent for {part_code}."
            ),
        )

    # Phase 2: Stock & HITL SQL Permission -> Repair Cost Auth -> Tier 1 Technician
    if not state.get("stock_approved"):
        if inv.get("escalate_procurement"):
            return OrchestratorDecision(
                next_agent="FINISH",
                instruction="ESCALATED: Zero stock for OEM and compatible alternative spare parts.",
                gate_decision="ESCALATED",
                reviewed_stage="stock",
                audit_notes="Gate 2 ESCALATED to Procurement: both OEM and alternative stock = 0.",
            )
        if not inv:
            return OrchestratorDecision(
                next_agent="stock_agent",
                instruction="Check inventory and prepare reservation.",
                gate_decision="PENDING",
                reviewed_stage="stock",
                audit_notes="Dispatching Stock Agent.",
            )
        if inv.get("awaiting_sql_approval"):
            if human_decisions.get("sql_reservation", {}).get("decision") == "APPROVED":
                target_part = inv.get("proposed_part_code", "BRG-4402")
                return OrchestratorDecision(
                    next_agent="stock_agent",
                    instruction=f"Human QA approved SQL write. Execute reserve_spare_part for {target_part} now.",
                    gate_decision="APPROVED",
                    reviewed_stage="stock",
                    audit_notes=f"SQL reservation approved by Human QA. Executing atomic stock decrement for {target_part}.",
                )
            return OrchestratorDecision(
                next_agent="FINISH",
                instruction="Awaiting Human QA Supervisor permission for SQLite stock update.",
                gate_decision="AWAITING_HUMAN",
                reviewed_stage="stock",
                audit_notes=(
                    f"Paused at HITL Gate (SQL_MUTATION): Proposed {inv.get('proposed_part_code')} "
                    f"(${inv.get('unit_cost_usd', 0):,.2f})."
                ),
            )
        if not inv.get("reserved"):
            return OrchestratorDecision(
                next_agent="stock_agent",
                instruction="Reservation incomplete; verify SQLite and reserve part.",
                gate_decision="REJECTED",
                reviewed_stage="stock",
                audit_notes="Gate 2 REJECTED: Material Requisition missing confirmed SQLite reservation.",
            )

        part_issued = inv.get("reserved_part_code")
        alt_note = (
            f"Approved qualified alternative {part_issued} after OEM {inv.get('requested_part_code')} stockout."
            if inv.get("used_alternative")
            else f"Reserved OEM part {part_issued} from {inv.get('bin_location')}."
        )
        return OrchestratorDecision(
            next_agent="field_technician",
            instruction=(
                f"Execute LOTO and replace {part_issued} on {state['machine_id']} per "
                f"{diag.get('sop_code', 'SOP-MNT-001')} p.{diag.get('page_number', 2)} and test vibration <= 2.5 mm/s."
            ),
            gate_decision="APPROVED",
            reviewed_stage="stock",
            audit_notes=f"Gate 2 APPROVED: {alt_note} Remaining stock: {inv.get('remaining_stock_qty')}.",
        )

    # Phase 3: Tier 1 Technician Execution & Tier 2 Reliability Engineer Escalation Hierarchy
    if not state.get("maintenance_approved"):
        if not maint:
            return OrchestratorDecision(
                next_agent="field_technician",
                instruction=f"Dispatching Tier 1 Field Technician for {state['machine_id']} repair.",
                gate_decision="APPROVED",
                reviewed_stage="stock",
                audit_notes="Repair budget authorized. Dispatching Field Technician Attempt #1.",
            )

        rms = maint.get("post_repair_vibration_rms")
        if rms is None or float(rms) > GMP_VIBRATION_LIMIT_RMS:
            # Upward escalation: Technician -> Reliability Engineer -> Rework Auth -> Technician Attempt #2
            if not state.get("engineering_replan_done"):
                return OrchestratorDecision(
                    next_agent="diagnostic_engineer",
                    instruction=(
                        f"ESCALATED FROM TIER 1 TECHNICIAN: Post-repair vibration is {rms} mm/s RMS (> 2.5 mm/s limit). "
                        "Inspect axial vibration pattern in SOP-MNT-002 p.3 and prescribe corrective engineering alignment plan."
                    ),
                    gate_decision="REJECTED",
                    reviewed_stage="maintenance",
                    audit_notes=(
                        f"Gate 3 REJECTED: Measured {rms} mm/s RMS (> {GMP_VIBRATION_LIMIT_RMS} mm/s). "
                        "Escalating UP from Tier 1 Technician to Tier 2 Reliability Engineer."
                    ),
                )
            replan = state.get("engineering_replan", {})
            return OrchestratorDecision(
                next_agent="field_technician",
                instruction=(
                    f"Execute Reliability Engineer's corrective plan: {replan.get('corrective_procedure', 'Cold dual-beam laser shaft realignment <= 0.05 mm')} "
                    f"({replan.get('sop_ref', 'SOP-MNT-002 p.3')}) and re-verify <= 2.5 mm/s RMS."
                ),
                gate_decision="APPROVED",
                reviewed_stage="engineering_replan",
                audit_notes=(
                    f"Tier 2 Engineer prescribed realignment ({replan.get('alignment_tolerance', '<= 0.05 mm')}). "
                    "Routing back down to Tier 1 Technician for Attempt #2."
                ),
            )

        return OrchestratorDecision(
            next_agent="qa_clearance",
            instruction=(
                f"Post-repair vibration {rms} mm/s RMS passed (<= 2.5 mm/s). "
                "Execute Tier 3 QA Line Clearance (LOTO removal, tool count, wipe/CIP, stability)."
            ),
            gate_decision="APPROVED",
            reviewed_stage="maintenance",
            audit_notes=f"Gate 3 APPROVED: {rms} mm/s RMS <= {GMP_VIBRATION_LIMIT_RMS} mm/s after {maint.get('attempt_count', 1)} attempt(s).",
        )

    # Phase 4: Tier 3 QA Line Clearance -> Analyst & CAPA
    if not state.get("clearance_approved"):
        if clear and clear.get("clearance_approved"):
            return OrchestratorDecision(
                next_agent="analyst_capa",
                instruction=(
                    f"QA Line Clearance passed for {state['machine_id']}. Compute MTTR & financial impact, "
                    "check repeat-failure CAPA threshold, and export 4-sheet Monthly Report (.xlsx)."
                ),
                gate_decision="APPROVED",
                reviewed_stage="clearance",
                audit_notes=f"Line Clearance APPROVED: {state['machine_id']} released for production; dispatching Analyst/CAPA.",
            )
        return OrchestratorDecision(
            next_agent="qa_clearance",
            instruction="Complete post-repair QA Line Clearance checklist.",
            gate_decision="PENDING",
            reviewed_stage="clearance",
            audit_notes="Dispatching QA Line Clearance Officer.",
        )

    # Phase 5: Analytics, Auto-CAPA & Final Conductor Sign-Off
    if not state.get("analyst_approved"):
        if analyt and analyt.get("logged_to_db") and analyt.get("excel_report_path"):
            capa = state.get("capa_report", {})
            capa_msg = f" CAPA {capa.get('capa_id')} opened." if capa.get("capa_triggered") else ""
            return OrchestratorDecision(
                next_agent="FINISH",
                instruction="All orchestra quality gates, QA Line Clearance, KPIs & CAPA checks complete. Close eBMR.",
                gate_decision="SIGNED_OFF",
                reviewed_stage="closure",
                audit_notes=(
                    f"Protocol SIGNED_OFF: Downtime {analyt.get('downtime_minutes')}m, Total Impact "
                    f"${analyt.get('total_cost_usd', 0):,.2f} ({analyt.get('damage_severity')}).{capa_msg}"
                ),
            )
        return OrchestratorDecision(
            next_agent="analyst_capa",
            instruction="Log closed incident analytics, check CAPA trigger, and export .xlsx report.",
            gate_decision="REJECTED",
            reviewed_stage="analyst",
            audit_notes="Gate 4 incomplete: missing SQLite analytics or Excel workbook.",
        )

    return OrchestratorDecision(
        next_agent="FINISH",
        instruction="Protocol closed.",
        gate_decision="SIGNED_OFF",
        reviewed_stage="closure",
        audit_notes="eBMR archived.",
    )


def _enforce_orchestra_interlocks(
    llm_decision: OrchestratorDecision, state: MaintenanceState
) -> OrchestratorDecision:
    """Enforce mandatory GMP-QA-010 safety & hierarchy interlocks on LLM Conductor decisions."""
    det = _evaluate_orchestra_state(state)

    if det.gate_decision in ("REJECTED", "ESCALATED", "AWAITING_HUMAN"):
        return det
    if (
        llm_decision.next_agent != det.next_agent
        or llm_decision.reviewed_stage != det.reviewed_stage
        or llm_decision.gate_decision != det.gate_decision
    ):
        return OrchestratorDecision(
            next_agent=det.next_agent,
            instruction=llm_decision.instruction or det.instruction,
            gate_decision=det.gate_decision,
            reviewed_stage=det.reviewed_stage,
            audit_notes=f"{llm_decision.audit_notes} [GMP Verified: {det.audit_notes}]",
        )
    return llm_decision


def _build_hitl_pause_payload(
    state: MaintenanceState,
    decision: OrchestratorDecision,
) -> Optional[Dict[str, Any]]:
    """Pause for Human QA authorization on SQL writes and Repair/Rework budgets."""
    if not state.get("hitl_enabled", True):
        return None

    is_tr = str(state.get("language", "en")).lower() == "tr"
    human_decisions = state.get("human_decisions", {})
    inv = state.get("inventory_report", {})
    diag = state.get("diagnostic_report", {})
    maint = state.get("maintenance_report", {})
    pr = state.get("purchase_requisition", {})

    # Gate A: SQL Database Write Authorization (fires reliably even under Live LLM)
    if (
        inv.get("awaiting_sql_approval")
        and human_decisions.get("sql_reservation", {}).get("decision") != "APPROVED"
    ):
        part_code = str(inv.get("proposed_part_code") or "")
        pr_note = (
            (f" Oto-Satınalma Taslağı: {pr['pr_id']} ({pr['order_qty']}x {pr['part_code']})." if is_tr else f" Auto-PO drafted: {pr['pr_id']} ({pr['order_qty']}x {pr['part_code']}).")
            if pr.get("created")
            else ""
        )
        return {
            "approval_id": "sql_reservation",
            "gate_type": "SQL_MUTATION",
            "title": "SQLite Envanter Yazma İzni" if is_tr else "SQLite Inventory Write Authorization",
            "summary": (
                f"Stok Ajanı {inv.get('bin_location')} rafından 1x {part_code} (${inv.get('unit_cost_usd', 0):,.2f}) "
                f"rezervasyonu istiyor (Adet {inv.get('current_stock_qty')} -> {inv.get('projected_stock_qty')}).{pr_note}"
                if is_tr
                else f"Stock Agent requests permission to reserve 1x {part_code} (${inv.get('unit_cost_usd', 0):,.2f}) "
                f"from {inv.get('bin_location')} (Qty {inv.get('current_stock_qty')} -> {inv.get('projected_stock_qty')}).{pr_note}"
            ),
            "sql_preview": inv.get("sql_preview", ""),
            "part_code": part_code,
            "requested_part_code": inv.get("requested_part_code", part_code),
            "is_alternative": bool(inv.get("used_alternative", False)),
            "purchase_requisition": pr,
            "unit_cost_usd": float(inv.get("unit_cost_usd", 0.0)),
            "bin_location": inv.get("bin_location", ""),
            "machine_id": state["machine_id"],
            "sop_ref": f"{diag.get('sop_code', 'SOP-MNT-001')} p.{diag.get('page_number', 1)}",
        }

    # Gate B: Pre-Repair Cost Authorization (before Tier 1 Technician Attempt #1)
    if (
        decision.next_agent == "field_technician"
        and inv.get("reserved")
        and not maint
        and human_decisions.get("repair_attempt_1", {}).get("decision") != "APPROVED"
    ):
        part_code = str(inv.get("reserved_part_code") or diag.get("required_part_code") or "BRG-4402")
        part_cost = float(inv.get("unit_cost_usd", 0.0))
        cost_est = estimate_repair_cost(state["machine_id"], part_cost, attempt_number=1)
        sop_ref = f"{diag.get('sop_code', 'SOP-MNT-001')} p.{diag.get('page_number', 2)}"
        return {
            "approval_id": "repair_attempt_1",
            "gate_type": "REPAIR_COST",
            "title": (
                f"Onarım İcrası ve Bütçe İzni ({state['machine_id']})"
                if is_tr
                else f"Repair Execution & Budget Authorization ({state['machine_id']})"
            ),
            "summary": (
                f"{state['machine_name']} üzerinde {sop_ref} uyarınca EKED ve 1. Kademe Teknisyen tarafından {part_code} değişimini onaylayın."
                if is_tr
                else f"Authorize LOTO & Tier-1 Technician replacement of {part_code} on {state['machine_name']} per {sop_ref}."
            ),
            "part_code": part_code,
            "is_alternative": bool(inv.get("used_alternative", False)),
            "machine_id": state["machine_id"],
            "sop_ref": sop_ref,
            "cost_breakdown": cost_est,
        }

    # Gate C: Rework Cost Authorization (after Tier 2 Engineer delivers corrective realignment spec)
    if (
        decision.next_agent == "field_technician"
        and state.get("engineering_replan_done")
        and int(maint.get("attempt_count", 0)) < 2
        and human_decisions.get("rework_attempt_2", {}).get("decision") != "APPROVED"
    ):
        part_code = str(inv.get("reserved_part_code") or "SL-8810")
        part_cost = float(inv.get("unit_cost_usd", 0.0))
        next_attempt = int(maint.get("attempt_count", 1)) + 1
        cost_est = estimate_repair_cost(state["machine_id"], part_cost, attempt_number=next_attempt)
        replan = state.get("engineering_replan", {})
        proc = replan.get("corrective_procedure", "Cold dual-beam laser shaft realignment <= 0.05 mm")
        return {
            "approval_id": "rework_attempt_2",
            "gate_type": "REWORK_COST",
            "title": (
                f"Mühendis Hizalama Revizyon İzni ({state['machine_id']})"
                if is_tr
                else f"Engineer Alignment Rework Authorization ({state['machine_id']})"
            ),
            "summary": (
                f"Güvenilirlik Mühendisi 1. Denemeyi ({maint.get('post_repair_vibration_rms')} mm/s > 2.5) inceledi ve reçete yazdı: {proc}."
                if is_tr
                else f"Reliability Engineer diagnosed Attempt #1 ({maint.get('post_repair_vibration_rms')} mm/s > 2.5) and prescribed: {proc}."
            ),
            "engineering_replan": replan,
            "part_code": part_code,
            "machine_id": state["machine_id"],
            "sop_ref": str(replan.get("sop_ref", "SOP-MNT-002 p.3")),
            "cost_breakdown": cost_est,
        }

    return None


def orchestrator_node(state: MaintenanceState) -> Dict[str, Any]:
    """Central GMP Conductor orchestrating parallel specialists, 3-tier hierarchy & HITL gates."""
    iteration = int(state.get("iteration_count", 0)) + 1
    max_iters = int(state.get("max_iterations", 22))
    effective_model = state["llm_model"]
    switch_note: Optional[str] = None

    human_decisions = state.get("human_decisions", {})
    inv_state = state.get("inventory_report", {})
    maint_state = state.get("maintenance_report", {})

    just_resumed_sql = bool(
        inv_state.get("awaiting_sql_approval")
        and human_decisions.get("sql_reservation", {}).get("decision") == "APPROVED"
    )
    just_resumed_repair = bool(
        inv_state.get("reserved")
        and not maint_state
        and human_decisions.get("repair_attempt_1", {}).get("decision") == "APPROVED"
    )
    just_resumed_rework = bool(
        state.get("engineering_replan_done")
        and int(maint_state.get("attempt_count", 0)) < 2
        and human_decisions.get("rework_attempt_2", {}).get("decision") == "APPROVED"
    )

    if iteration > max_iters:
        decision = OrchestratorDecision(
            next_agent="FINISH",
            instruction="Maximum orchestra iterations reached; escalating for QA review.",
            gate_decision="ESCALATED",
            reviewed_stage="closure",
            audit_notes=f"Safety loop limit ({max_iters}) reached.",
        )
    elif state.get("error_message") or state.get("execution_mode") == "offline_demo" or just_resumed_sql or just_resumed_repair or just_resumed_rework:
        decision = _evaluate_orchestra_state(state)
    else:
        try:
            snapshot = {
                "ticket_id": state["ticket_id"],
                "machine_id": state["machine_id"],
                "alarm_vibration_rms": state["alarm_vibration_rms"],
                "diagnostic_done": state.get("diagnostic_done"),
                "precheck_done": state.get("precheck_done"),
                "diagnostic_approved": state.get("diagnostic_approved"),
                "stock_approved": state.get("stock_approved"),
                "escalation_to_engineer": state.get("escalation_to_engineer"),
                "engineering_replan_done": state.get("engineering_replan_done"),
                "maintenance_approved": state.get("maintenance_approved"),
                "clearance_approved": state.get("clearance_approved"),
                "diagnostic_report": state.get("diagnostic_report"),
                "precheck_report": state.get("precheck_report"),
                "inventory_report": state.get("inventory_report"),
                "maintenance_report": state.get("maintenance_report"),
                "engineering_replan": state.get("engineering_replan"),
                "clearance_report": state.get("clearance_report"),
                "analytics_report": state.get("analytics_report"),
            }
            sys_prompt = (
                "You are the central GMP Quality Conductor (Orchestrator) managing a 3-tier factory hierarchy:\n"
                "- Initial intake -> 'parallel_intake' (dispatches Diagnostic Engineer + Batch/Stock Pre-Check concurrently).\n"
                "- After parallel intake -> verify SOP citation -> 'stock_agent'.\n"
                "- After stock reserved -> 'field_technician' (Tier 1).\n"
                "- If Technician post-repair > 2.5 mm/s -> REJECTED and escalate UP to 'diagnostic_engineer' (Tier 2) for engineering_replan, then back to 'field_technician'.\n"
                "- Once <= 2.5 mm/s -> 'qa_clearance' (Tier 3) -> 'analyst_capa' -> 'FINISH' (SIGNED_OFF).\n"
                "Return OrchestratorDecision JSON." + _lang_prompt_hint(state)
            )

            def _call_orch(model_id: str) -> OrchestratorDecision:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.0,
                )
                return invoke_structured_decision(
                    llm,
                    OrchestratorDecision,
                    [
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=f"Current Orchestra eBMR State:\n{json.dumps(snapshot, indent=2)}"),
                    ],
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            raw_decision, effective_model, switch_note = _invoke_live_with_model_fallback(state, _call_orch)
            decision = _enforce_orchestra_interlocks(raw_decision, state)
        except Exception as exc:
            err_msg = f"LLM Conductor error ({state['llm_provider']}/{state['llm_model']}): {exc}"
            audit_entry = append_gmp_audit_log(
                ticket_id=state["ticket_id"],
                step_number=_next_step_num(state),
                agent_name="GMP Conductor",
                action_summary="Autonomous LLM call failed",
                gate_decision="ESCALATED",
                audit_notes=err_msg,
            )
            return {
                "next_agent": "FINISH",
                "last_gate_decision": "ESCALATED",
                "escalated": True,
                "protocol_closed": True,
                "error_message": err_msg,
                "iteration_count": iteration,
                "audit_trail": [audit_entry],
                "agent_events": [
                    {
                        "timestamp_utc": audit_entry["timestamp_utc"],
                        "node": "orchestrator",
                        "next_agent": "FINISH",
                        "gate_decision": "ESCALATED",
                        "summary": err_msg,
                    }
                ],
            }

    diag_ok = bool(state.get("diagnostic_approved"))
    stock_ok = bool(state.get("stock_approved"))
    maint_ok = bool(state.get("maintenance_approved"))
    clear_ok = bool(state.get("clearance_approved"))
    analyt_ok = bool(state.get("analyst_approved"))
    escalate_eng = bool(state.get("escalation_to_engineer"))
    closed = bool(state.get("protocol_closed"))
    escalated = bool(state.get("escalated"))

    if decision.reviewed_stage == "diagnostic" and decision.gate_decision == "APPROVED":
        diag_ok = True
    elif decision.reviewed_stage == "stock" and decision.gate_decision == "APPROVED" and not inv_state.get("awaiting_sql_approval"):
        stock_ok = True
    elif decision.reviewed_stage == "stock" and decision.gate_decision == "ESCALATED":
        escalated = True
        closed = True
    elif decision.reviewed_stage == "maintenance":
        if decision.gate_decision == "APPROVED":
            maint_ok = True
            escalate_eng = False
        elif decision.gate_decision == "REJECTED":
            maint_ok = False
            escalate_eng = True
    elif decision.reviewed_stage == "clearance" and decision.gate_decision == "APPROVED":
        clear_ok = True
    elif decision.reviewed_stage in ("analyst", "closure") and decision.gate_decision == "SIGNED_OFF":
        analyt_ok = True
        closed = True
    elif decision.gate_decision == "ESCALATED":
        escalated = True
        closed = True

    hitl_pause = _build_hitl_pause_payload(state, decision)
    awaiting_human = hitl_pause is not None
    pending_payload = hitl_pause or {}

    if awaiting_human:
        next_agent_out = "FINISH"
        closed = False
        gate_out = "AWAITING_HUMAN"
        action_label = f"HITL Permission Pause [{hitl_pause.get('gate_type')}]"
        extra_note = f" Paused for Human QA approval: {hitl_pause.get('title')}."
    else:
        next_agent_out = decision.next_agent
        gate_out = decision.gate_decision
        if decision.next_agent == "FINISH":
            closed = True
        action_label = (
            "Dispatch Parallel Orchestra (Diagnostic + Pre-Check)"
            if decision.next_agent == "parallel_intake"
            else (
                f"Gate [{decision.reviewed_stage.upper()}] -> Route to {decision.next_agent}"
                if decision.next_agent != "FINISH"
                else f"Protocol Closure [{decision.gate_decision}]"
            )
        )
        extra_note = ""

    final_notes = (f"[{switch_note}] {decision.audit_notes}{extra_note}" if switch_note else f"{decision.audit_notes}{extra_note}").strip()
    audit_entry = append_gmp_audit_log(
        ticket_id=state["ticket_id"],
        step_number=_next_step_num(state),
        agent_name="GMP Conductor",
        action_summary=action_label,
        gate_decision=gate_out,
        audit_notes=final_notes,
    )

    event_item = {
        "timestamp_utc": audit_entry["timestamp_utc"],
        "node": "orchestrator",
        "reviewed_stage": decision.reviewed_stage,
        "next_agent": next_agent_out,
        "gate_decision": gate_out,
        "instruction": decision.instruction,
        "summary": final_notes,
    }

    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"

    if iteration == 1:
        cmd_recv = (
            f"{state['machine_id']} üzerinde Sınır Titreşim Alarmı ({state['alarm_vibration_rms']:.1f} mm/s RMS > {GMP_VIBRATION_LIMIT_RMS} limit)"
            if is_tr
            else f"Edge Vibration Alarm on {state['machine_id']} ({state['alarm_vibration_rms']:.1f} mm/s RMS > {GMP_VIBRATION_LIMIT_RMS} limit)"
        )
    else:
        cmd_recv = (
            f"'{state.get('last_worker', 'worker')}' ajanından dönüş ({decision.reviewed_stage} aşaması)"
            if is_tr
            else f"Callback from '{state.get('last_worker', 'worker')}' after '{decision.reviewed_stage}' stage"
        )

    if awaiting_human:
        orch_thinking = (
            f"SQLite güncellemesi veya saha maliyet harcaması öncesinde bağlayıcı İnsan QA Onayı [{hitl_pause.get('gate_type')}] bekleniyor."
            if is_tr
            else f"Gate requires binding Human QA approval [{hitl_pause.get('gate_type')}] before mutating SQLite or authorizing shop-floor spend. Pausing cleanly with full preview."
        )
        orch_delegate = f"human_qa ({hitl_pause.get('gate_type')})"
    elif next_agent_out == "parallel_intake":
        orch_thinking = (
            f"Sensör ISO 20816-3 Bölge C/D sınırını aştı ({state['alarm_vibration_rms']:.1f} mm/s). 2. Kademe Teşhis Mühendisi (SOP RAG) ve Parti/Stok Ön Kontrol eşzamanlı başlatılıyor."
            if is_tr
            else f"Continuous edge monitor breached ISO 20816-3 Zone D/C ({state['alarm_vibration_rms']:.1f} mm/s). Dispatching Tier 2 Diagnostic Engineer (Hybrid PDF RAG) concurrently with Batch & Stock Pre-Check via Send()."
        )
        orch_delegate = "diagnostic_engineer + stock_precheck (Paralel)" if is_tr else "diagnostic_engineer + stock_precheck (Parallel)"
    elif decision.reviewed_stage == "maintenance" and gate_out == "REJECTED":
        orch_thinking = (
            "Kapı 3 REDDEDİLDİ: Onarım sonrası titreşim 2.5 mm/s GMP limitini aştı. 1. Kademe Teknisyenden 2. Kademe Güvenilirlik Mühendisine YUKARI eskalasyon başlatılıyor."
            if is_tr
            else "Gate 3 REJECTED: Post-repair vibration exceeds 2.5 mm/s GMP release limit. Escalating UP from Tier 1 Technician to Tier 2 Reliability Engineer for corrective SOP procedure."
        )
        orch_delegate = "diagnostic_engineer (Yukarı Eskalasyon)" if is_tr else "diagnostic_engineer (Upward Escalation)"
    else:
        orch_thinking = (
            f"GMP-QA-010 gereği '{decision.reviewed_stage}' kapısı incelendi -> {gate_out}. {decision.instruction}"
            if is_tr
            else f"Evaluated '{decision.reviewed_stage}' gate evidence per GMP-QA-010 -> {gate_out}. {decision.instruction}"
        )
        orch_delegate = (
            ("FINISH (eBMR Arşivlendi)" if is_tr else "FINISH (Protocol Archived)")
            if next_agent_out == "FINISH"
            else next_agent_out
        )

    orch_trace = _make_trace(
        node="orchestrator",
        tier="Tier 3 Supervisory Conductor",
        agent_name="GMP Quality Conductor (Orchestrator)",
        command_received=cmd_recv,
        thinking=orch_thinking,
        actions_taken=[f"Quality Gate [{decision.reviewed_stage.upper()}] -> {gate_out}", f"SHA-256 Audit Step #{audit_entry['step_number']}"],
        output_verdict=final_notes,
        delegated_to=orch_delegate,
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "next_agent": next_agent_out,
        "current_instruction": decision.instruction,
        "last_gate_decision": gate_out,
        "awaiting_human_approval": awaiting_human,
        "pending_approval": pending_payload,
        "diagnostic_approved": diag_ok,
        "stock_approved": stock_ok,
        "escalation_to_engineer": escalate_eng,
        "maintenance_approved": maint_ok,
        "clearance_approved": clear_ok,
        "analyst_approved": analyt_ok,
        "protocol_closed": closed,
        "escalated": escalated,
        "iteration_count": iteration,
        "audit_trail": [audit_entry],
        "agent_events": [event_item],
        "agent_traces": [orch_trace],
    }


# ---------------------------------------------------------------------------
# Tier 2A: Reliability & Diagnostic Engineer (Dual Mode: Initial RAG + Escalation Replan)
# ---------------------------------------------------------------------------


def diagnostic_engineer_node(state: MaintenanceState) -> Dict[str, Any]:
    machine_id = state["machine_id"]
    effective_model = state["llm_model"]
    sc = DEMO_SCENARIOS.get(state["scenario_id"], {})
    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"
    new_tool_calls: List[Dict[str, Any]] = []

    # Escalation Mode: Technician Attempt #1 failed > 2.5 mm/s -> Engineer prescribes corrective alignment
    if state.get("escalation_to_engineer") and not state.get("engineering_replan_done"):
        maint = state.get("maintenance_report", {})
        measured_rms = maint.get("post_repair_vibration_rms", sc.get("fail_attempt1_rms") or 4.8)
        replan_sop = str(sc.get("replan_sop_ref") or "SOP-MNT-002 p.3")
        replan_proc_hint = str(sc.get("replan_procedure") or "Cold Dual-Beam Laser Shaft Realignment")
        query = f"{machine_id} {replan_sop} {replan_proc_hint} post-repair vibration > 2.5 mm/s"
        raw_llm_json: Dict[str, Any] = {}

        if state.get("execution_mode") == "offline_demo":
            rag = search_sop_manuals_impl(query, top_k=2)
            new_tool_calls.append(
                {
                    "agent": "Reliability Engineer (Tier 2)",
                    "tool": "search_sop_manuals",
                    "args": {"query": query[:140]},
                    "output_preview": json.dumps(rag)[:280],
                    "timestamp_utc": _utc_now(),
                }
            )
        else:
            try:
                sys_prompt = (
                    "You are the Tier 2 Reliability Engineer.\n"
                    "The Tier 1 Field Technician escalated a failed post-repair vibration check (> 2.5 mm/s).\n"
                    "Call search_sop_manuals(query) for corrective alignment/parallelism tolerances and output JSON with keys: "
                    "root_cause_of_failure, corrective_procedure, alignment_tolerance, torque_spec_nm, sop_ref."
                    + _lang_prompt_hint(state)
                )
                task_prompt = (
                    f"Machine: {machine_id} | Measured Post-Repair RMS: {measured_rms} mm/s (Limit 2.5 mm/s)\n"
                    f"Conductor Instruction: {state.get('current_instruction')}"
                )

                def _call_replan(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                    llm = get_chat_model(
                        provider=state["llm_provider"],
                        model=model_id,
                        api_key=state.get("api_key"),
                        temperature=0.0,
                    )
                    return _run_specialist_tool_loop(
                        llm=llm,
                        system_prompt=sys_prompt,
                        task_prompt=task_prompt,
                        tools=[search_sop_manuals],
                        pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                    )

                (raw_text, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_replan)
                for ex in executed:
                    new_tool_calls.append({"agent": "Reliability Engineer (Tier 2)", **ex})
                raw_llm_json = _safe_parse_json(raw_text)
            except Exception as exc:
                return {
                    "last_worker": "diagnostic_engineer",
                    "error_message": f"Reliability Engineer (Escalation) LLM failure: {exc}",
                    "tool_calls_log": new_tool_calls,
                }

        ctx = get_active_run_context()
        default_root = (
            f"160 deg C sealing platen thermal frame bow & cold bearing fit misalignment ({measured_rms} mm/s RMS)."
            if machine_id == "PK-02"
            else f"1x/2x axial coupling misalignment & motor soft-foot distortion ({measured_rms} mm/s RMS)."
        )
        replan = {
            "escalated_from_rms": measured_rms,
            "root_cause_of_failure": str(raw_llm_json.get("root_cause_of_failure") or default_root),
            "corrective_procedure": str(raw_llm_json.get("corrective_procedure") or replan_proc_hint),
            "alignment_tolerance": str(
                raw_llm_json.get("alignment_tolerance")
                or sc.get("replan_tolerance")
                or "Soft foot < 0.05 mm; Parallel <= 0.05 mm TIR; Angular <= 0.05 mm/100mm"
            ),
            "torque_spec_nm": str(
                raw_llm_json.get("torque_spec_nm")
                or ("38 Nm hot platen retorque" if machine_id == "PK-02" else "55 Nm calibrated star torque")
            ),
            "sop_ref": str(raw_llm_json.get("sop_ref") or replan_sop),
        }

        audit_entry = append_gmp_audit_log(
            ticket_id=state["ticket_id"],
            step_number=_next_step_num(state),
            agent_name="Reliability Engineer (Tier 2)",
            action_summary="Escalation RCA & Corrective Alignment Prescription",
            gate_decision="APPROVED",
            audit_notes=f"Diagnosed {measured_rms} mm/s failure -> Prescribed {replan['corrective_procedure']} ({replan['sop_ref']}).",
        )
        replan_trace = _make_trace(
            node="diagnostic_engineer",
            tier="Tier 2 Reliability Engineering",
            agent_name="Reliability Engineer (Escalation Mode)",
            command_received=(
                f"1. KADEME TEKNİSYENDEN YUKARI ESKALASYON: 1. Deneme {measured_rms} mm/s (> 2.5)"
                if is_tr
                else f"ESCALATED FROM TIER 1 TECHNICIAN via Conductor: Attempt #1 measured {measured_rms} mm/s (> 2.5)"
            ),
            thinking=(
                f"1. deneme hata spektrumu incelendi ({default_root}). Hibrit SOP RAG ({replan['sop_ref']}) sorgulanarak bağlayıcı tolerans belirlendi: '{replan['alignment_tolerance']}'."
                if is_tr
                else f"Analyzed Attempt #1 failure spectrum ({default_root}). Queried Hybrid SOP RAG ({replan['sop_ref']}) to author binding engineering tolerance '{replan['alignment_tolerance']}'."
            ),
            actions_taken=[
                f"search_sop_manuals('{query[:65]}...')",
                f"Issued Engineering Replan ({replan['sop_ref']})",
            ],
            output_verdict=f"Reçete / Prescribed: {replan['corrective_procedure']} [Tork: {replan['torque_spec_nm']}]",
            delegated_to="orchestrator -> field_technician (2. Deneme)" if is_tr else "orchestrator -> field_technician (Attempt #2)",
            timestamp_utc=audit_entry["timestamp_utc"],
            lang=lang_code,
        )
        return {
            "llm_model": effective_model,
            "last_worker": "diagnostic_engineer",
            "engineering_replan": replan,
            "engineering_replan_done": True,
            "rag_chunks": list(ctx.rag_chunks) if ctx else [],
            "tool_calls_log": new_tool_calls,
            "audit_trail": [audit_entry],
            "agent_events": [
                {
                    "timestamp_utc": audit_entry["timestamp_utc"],
                    "node": "diagnostic_engineer",
                    "gate_decision": "APPROVED",
                    "summary": f"Tier 2 Engineer prescribed corrective alignment ({replan['sop_ref']}).",
                }
            ],
            "agent_traces": [replan_trace],
        }

    # Initial Parallel Intake Mode: Multi-Sensor Telemetry + Hybrid PDF SOP RAG
    query = f"{machine_id} {state['alarm_type']} {state['symptom_summary']}"
    raw_llm_json = {}
    telemetry_snap: Dict[str, Any] = {}

    if state.get("execution_mode") == "offline_demo":
        telemetry_snap = read_sensor_telemetry_impl(machine_id)
        rag = search_sop_manuals_impl(query, top_k=3)
        new_tool_calls.extend(
            [
                {
                    "agent": "Reliability Engineer (Tier 2)",
                    "tool": "read_sensor_telemetry",
                    "args": {"machine_id": machine_id},
                    "output_preview": json.dumps(telemetry_snap)[:280],
                    "timestamp_utc": _utc_now(),
                },
                {
                    "agent": "Reliability Engineer (Tier 2)",
                    "tool": "search_sop_manuals",
                    "args": {"query": query[:120]},
                    "output_preview": json.dumps(rag)[:280],
                    "timestamp_utc": _utc_now(),
                },
            ]
        )
    else:
        try:
            sys_prompt = (
                "You are the Tier 2 Reliability & Diagnostic Engineer.\n"
                "1. Call read_sensor_telemetry(machine_id) and search_sop_manuals(query).\n"
                "2. Identify failure mode, cite SOP code, filename, and page number.\n"
                "3. Specify exact required_part_code (BRG-4402, LUB-3010, RLR-4450, SL-8810, IMP-8825, SHM-8805, BLT-2205, SRV-2240, or BRG-2290).\n"
                "Respond with pure JSON keys: failure_mode, root_cause, sop_code, sop_filename, "
                "page_number, required_part_code, required_tools." + _lang_prompt_hint(state)
            )
            task_prompt = (
                f"Ticket: {state['ticket_id']} | Machine: {machine_id} | Trigger: {state.get('trigger_source')}\n"
                f"Symptom: {state['symptom_summary']}"
            )

            def _call_diag(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.1,
                )
                return _run_specialist_tool_loop(
                    llm=llm,
                    system_prompt=sys_prompt,
                    task_prompt=task_prompt,
                    tools=[read_sensor_telemetry, search_sop_manuals],
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            (raw_text, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_diag)
            for ex in executed:
                new_tool_calls.append({"agent": "Reliability Engineer (Tier 2)", **ex})
            raw_llm_json = _safe_parse_json(raw_text)
        except Exception as exc:
            return {
                "last_worker": "diagnostic_engineer",
                "error_message": f"Diagnostic Engineer LLM failure: {exc}",
                "tool_calls_log": new_tool_calls,
            }

    ctx = get_active_run_context()
    if not telemetry_snap:
        telemetry_snap = (ctx.telemetry_readings[-1] if (ctx and ctx.telemetry_readings) else read_sensor_telemetry_impl(machine_id))
    rag_hits = list(ctx.rag_chunks) if ctx and ctx.rag_chunks else search_sop_manuals_impl(query, top_k=3)["results"]
    top_hit = rag_hits[0] if rag_hits else {}
    expected_part = str(sc.get("expected_part_code", "BRG-4402"))

    part_match = re.search(
        r"\b(BRG-4402(?:-ALT)?|LUB-3010|RLR-4450(?:-ALT)?|SL-8810(?:-ALT)?|IMP-8825|SHM-8805|BLT-2205(?:-ALT)?|SRV-2240|BRG-2290)\b",
        str(raw_llm_json.get("required_part_code", "")).upper(),
    )
    required_part = part_match.group(1) if part_match else expected_part

    sop_code = str(raw_llm_json.get("sop_code") or top_hit.get("sop_code") or sc.get("expected_sop_code") or "SOP-MNT-001")
    sop_filename = str(raw_llm_json.get("sop_filename") or top_hit.get("filename") or "")
    page_num = int(raw_llm_json.get("page_number") or top_hit.get("page_number") or sc.get("expected_sop_page") or 2)

    def_fm = str(sc.get("failure_mode_title") or state.get("alarm_type") or "Vibration Anomaly")
    def_rc = str(sc.get("root_cause_summary") or state["symptom_summary"])
    def_tools = sc.get("required_tools") or ["Calibrated torque wrench", "Tri-axial vibration analyzer"]

    report = {
        "machine_id": machine_id,
        "failure_mode": str(raw_llm_json.get("failure_mode") or def_fm),
        "root_cause": str(raw_llm_json.get("root_cause") or def_rc),
        "sop_code": sop_code,
        "sop_filename": sop_filename,
        "page_number": page_num,
        "required_part_code": required_part,
        "required_tools": raw_llm_json.get("required_tools") or def_tools,
        "rag_evidence": rag_hits[:3],
        "telemetry_snapshot": telemetry_snap,
    }

    if ctx is not None:
        ctx.sop_citation = f"{sop_code} p.{page_num}"

    audit_entry = append_gmp_audit_log(
        ticket_id=state["ticket_id"],
        step_number=_next_step_num(state),
        agent_name="Reliability Engineer (Tier 2)",
        action_summary=f"Parallel SOP RAG ({sop_code} p.{page_num}) & Multi-Sensor Telemetry",
        gate_decision="PENDING",
        audit_notes=f"Diagnosed '{report['failure_mode']}'; part required: {required_part}.",
    )

    diag_trace = _make_trace(
        node="diagnostic_engineer",
        tier="Tier 2 Reliability Engineering",
        agent_name="Reliability & Diagnostic Engineer",
        command_received=str(state.get("current_instruction") or f"Diagnose {machine_id} vibration alarm via SOP RAG"),
        thinking=(
            f"Canlı sensör zarfı okundu (RMS={telemetry_snap.get('measured_vibration_rms', state['alarm_vibration_rms'])} mm/s, "
            f"Frekans={telemetry_snap.get('dominant_frequency_hz', 'N/A')} Hz, Kurtosis={telemetry_snap.get('kurtosis', 'N/A')}, "
            f"Sıcaklık={telemetry_snap.get('housing_temperature_c', 'N/A')} C). "
            f"Hibrit RAG {sop_code} s.{page_num} eşleştirdi: '{report['failure_mode']}' -> Gerekli parça {required_part}."
            if is_tr
            else (
                f"Inspected live sensor envelope (RMS={telemetry_snap.get('measured_vibration_rms', state['alarm_vibration_rms'])} mm/s, "
                f"Freq={telemetry_snap.get('dominant_frequency_hz', 'N/A')} Hz, Kurtosis={telemetry_snap.get('kurtosis', 'N/A')}, "
                f"Temp={telemetry_snap.get('housing_temperature_c', 'N/A')} C). "
                f"Retrieved {sop_code} p.{page_num} confirming '{report['failure_mode']}' -> requires {required_part}."
            )
        ),
        actions_taken=[
            f"read_sensor_telemetry('{machine_id}')",
            f"search_sop_manuals(...) -> Top hit {sop_code} p.{page_num} (score {top_hit.get('score', 0.0)})",
        ],
        output_verdict=f"Kök Neden / Root Cause: {report['failure_mode']} | Parça: {required_part} | Ref: {sop_code} p.{page_num}",
        delegated_to="orchestrator (Kapı 1 Teşhis Onayı)" if is_tr else "orchestrator (Gate 1 Diagnostic Verification)",
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "last_worker": "diagnostic_engineer",
        "diagnostic_done": True,
        "diagnostic_report": report,
        "rag_chunks": list(ctx.rag_chunks) if ctx else rag_hits[:3],
        "telemetry_readings": list(ctx.telemetry_readings) if ctx else [],
        "tool_calls_log": new_tool_calls,
        "audit_trail": [audit_entry],
        "agent_events": [
            {
                "timestamp_utc": audit_entry["timestamp_utc"],
                "node": "diagnostic_engineer",
                "gate_decision": "PENDING",
                "summary": f"Diagnosed {report['failure_mode']} ({sop_code} p.{page_num}) -> {required_part}",
            }
        ],
        "agent_traces": [diag_trace],
    }


# ---------------------------------------------------------------------------
# Tier 2B: Parallel Stock & Batch-Risk Pre-Check Agent
# ---------------------------------------------------------------------------


def stock_precheck_node(state: MaintenanceState) -> Dict[str, Any]:
    """Runs concurrently with Diagnostic Engineer via LangGraph Send() fan-out."""
    machine_id = state["machine_id"]
    effective_model = state["llm_model"]
    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"
    new_tool_calls: List[Dict[str, Any]] = []

    if state.get("execution_mode") == "offline_demo":
        sql_str = (
            f"SELECT part_code, part_name, stock_qty, min_reorder_qty, bin_location, compatible_alt_code "
            f"FROM spare_parts_inventory WHERE machine_id = '{machine_id}'"
        )
        inv_snap = query_inventory_sql_impl(sql_str)
        batch_risk = assess_batch_risk_impl(machine_id)
        new_tool_calls.extend(
            [
                {
                    "agent": "Batch & Stock Pre-Check",
                    "tool": "query_inventory_sql",
                    "args": {"sql_query": sql_str},
                    "output_preview": json.dumps(inv_snap)[:280],
                    "timestamp_utc": _utc_now(),
                },
                {
                    "agent": "Batch & Stock Pre-Check",
                    "tool": "assess_batch_risk",
                    "args": {"machine_id": machine_id},
                    "output_preview": json.dumps(batch_risk)[:280],
                    "timestamp_utc": _utc_now(),
                },
            ]
        )
    else:
        try:
            sys_prompt = (
                "You are the Parallel Batch-Risk & Stock Readiness Specialist.\n"
                "1. Call assess_batch_risk(machine_id).\n"
                "2. Call query_inventory_sql to inspect all spare parts for that machine_id.\n"
                "Return concise JSON with batch_id and low_stock_parts." + _lang_prompt_hint(state)
            )
            task_prompt = f"Machine: {machine_id} | Ticket: {state['ticket_id']}"

            def _call_precheck(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.0,
                )
                return _run_specialist_tool_loop(
                    llm=llm,
                    system_prompt=sys_prompt,
                    task_prompt=task_prompt,
                    tools=[assess_batch_risk, query_inventory_sql],
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            (_, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_precheck)
            for ex in executed:
                new_tool_calls.append({"agent": "Batch & Stock Pre-Check", **ex})
            batch_risk = assess_batch_risk_impl(machine_id)
            inv_snap = query_inventory_sql_impl(
                f"SELECT part_code, stock_qty, min_reorder_qty, compatible_alt_code FROM spare_parts_inventory WHERE machine_id = '{machine_id}'"
            )
        except Exception as exc:
            return {
                "error_message": f"Batch & Stock Pre-Check LLM failure: {exc}",
                "tool_calls_log": new_tool_calls,
            }

    low_stock = [r["part_code"] for r in inv_snap.get("rows", []) if int(r.get("stock_qty", 0)) <= int(r.get("min_reorder_qty", 0))]
    precheck_report = {
        **batch_risk,
        "machine_parts_checked": len(inv_snap.get("rows", [])),
        "low_stock_alerts": low_stock,
    }

    ctx = get_active_run_context()
    audit_entry = append_gmp_audit_log(
        ticket_id=state["ticket_id"],
        step_number=_next_step_num(state),
        agent_name="Batch & Stock Pre-Check",
        action_summary=f"Parallel Batch Risk ({batch_risk['active_batch_id']}) & Stock Snapshot",
        gate_decision="PENDING",
        audit_notes=f"Risk: {batch_risk['contamination_risk']}; Low-stock alerts: {low_stock or 'None'}.",
    )

    precheck_trace = _make_trace(
        node="stock_precheck",
        tier="Tier 2 Operations & Batch QA",
        agent_name="Batch & Stock Pre-Check",
        command_received=(
            f"Paralel Dağıtım Send(): {machine_id} için GMP parti riski ve SQLite stok hazırlığı kontrol et"
            if is_tr
            else f"Parallel Fan-Out via Send(): Assess GMP lot risk & SQLite readiness for {machine_id}"
        ),
        thinking=(
            f"Aktif parti {batch_risk['active_batch_id']} ({batch_risk['contamination_risk']} risk) eşzamanlı incelendi ve SQLite envanteri ({len(inv_snap.get('rows', []))} parça; kritik stok={low_stock or 'Yok'}) sorgulandı."
            if is_tr
            else (
                f"Concurrently inspected active batch {batch_risk['active_batch_id']} ({batch_risk['contamination_risk']} risk) "
                f"and queried SQLite inventory ({len(inv_snap.get('rows', []))} asset parts checked; low stock alerts={low_stock or 'None'})."
            )
        ),
        actions_taken=[
            f"assess_batch_risk('{machine_id}') -> {batch_risk['active_batch_id']} Quarantined",
            f"query_inventory_sql(...) -> Low stock: {low_stock or 'None'}",
        ],
        output_verdict=(
            f"Parti {batch_risk['active_batch_id']} QA Karantinasına Alındı | Kritik Stok: {low_stock or 'Yok'}"
            if is_tr
            else f"Batch {batch_risk['active_batch_id']} QA Hold Placed | Low Stock Alerts: {low_stock or 'None'}"
        ),
        delegated_to="orchestrator (Paralel Senkron Tamam)" if is_tr else "orchestrator (Parallel Sync Complete)",
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "precheck_done": True,
        "precheck_report": precheck_report,
        "sql_queries": list(ctx.sql_queries) if ctx else [],
        "tool_calls_log": new_tool_calls,
        "audit_trail": [audit_entry],
        "agent_events": [
            {
                "timestamp_utc": audit_entry["timestamp_utc"],
                "node": "stock_precheck",
                "gate_decision": "PENDING",
                "summary": f"Parallel Pre-Check: {batch_risk['active_batch_id']} quarantined; low stock={low_stock or 'None'}",
            }
        ],
        "agent_traces": [precheck_trace],
    }


# ---------------------------------------------------------------------------
# Tier 2C: Stock & Auto-Procurement Agent (HITL SQL Gate + Auto-PR)
# ---------------------------------------------------------------------------


def stock_agent_node(state: MaintenanceState) -> Dict[str, Any]:
    diag = state.get("diagnostic_report", {})
    req_part = str(diag.get("required_part_code") or DEMO_SCENARIOS[state["scenario_id"]]["expected_part_code"]).upper()
    ticket_id = state["ticket_id"]
    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"
    new_tool_calls: List[Dict[str, Any]] = []
    raw_llm_json: Dict[str, Any] = {}
    pr_result: Dict[str, Any] = dict(state.get("purchase_requisition", {}))

    hitl_on = bool(state.get("hitl_enabled", True))
    sql_approved = (not hitl_on) or (
        state.get("human_decisions", {}).get("sql_reservation", {}).get("decision") == "APPROVED"
    )
    resuming_approved_sql = bool(sql_approved and state.get("inventory_report", {}).get("awaiting_sql_approval"))
    effective_model = state["llm_model"]

    if resuming_approved_sql:
        target_part = str(state["inventory_report"].get("proposed_part_code") or req_part).upper()
        res = reserve_spare_part_impl(target_part, 1, ticket_id)
        new_tool_calls.append(
            {
                "agent": "Stock & Procurement",
                "tool": "reserve_spare_part",
                "args": {"part_code": target_part, "quantity": 1, "ticket_id": ticket_id},
                "output_preview": json.dumps(res)[:280],
                "timestamp_utc": _utc_now(),
            }
        )
    elif state.get("execution_mode") == "offline_demo":
        sql_str = (
            f"SELECT part_code, part_name, stock_qty, min_reorder_qty, bin_location, unit_cost_usd, compatible_alt_code "
            f"FROM spare_parts_inventory WHERE part_code = '{req_part}'"
        )
        inv_res = query_inventory_sql_impl(sql_str)
        new_tool_calls.append(
            {
                "agent": "Stock & Procurement",
                "tool": "query_inventory_sql",
                "args": {"sql_query": sql_str},
                "output_preview": json.dumps(inv_res)[:280],
                "timestamp_utc": _utc_now(),
            }
        )
        row = (inv_res.get("rows") or [{}])[0]
        stock_qty = int(row.get("stock_qty", 0))
        min_qty = int(row.get("min_reorder_qty", 0))

        if stock_qty <= min_qty:
            pr_result = create_purchase_requisition_impl(req_part, ticket_id)
            new_tool_calls.append(
                {
                    "agent": "Stock & Procurement",
                    "tool": "create_purchase_requisition",
                    "args": {"part_code": req_part, "ticket_id": ticket_id},
                    "output_preview": json.dumps(pr_result)[:280],
                    "timestamp_utc": _utc_now(),
                }
            )

        if stock_qty >= 1:
            if sql_approved:
                res = reserve_spare_part_impl(req_part, 1, ticket_id)
                new_tool_calls.append(
                    {
                        "agent": "Stock & Procurement",
                        "tool": "reserve_spare_part",
                        "args": {"part_code": req_part, "quantity": 1, "ticket_id": ticket_id},
                        "output_preview": json.dumps(res)[:280],
                        "timestamp_utc": _utc_now(),
                    }
                )
        else:
            alt_info = check_alternative_parts_impl(req_part)
            new_tool_calls.append(
                {
                    "agent": "Stock & Procurement",
                    "tool": "check_alternative_parts",
                    "args": {"part_code": req_part},
                    "output_preview": json.dumps(alt_info)[:280],
                    "timestamp_utc": _utc_now(),
                }
            )
            if alt_info.get("has_qualified_in_stock_alternative") and sql_approved:
                alt_code = str(alt_info["compatible_alt_code"])
                res = reserve_spare_part_impl(alt_code, 1, ticket_id)
                new_tool_calls.append(
                    {
                        "agent": "Stock & Procurement",
                        "tool": "reserve_spare_part",
                        "args": {"part_code": alt_code, "quantity": 1, "ticket_id": ticket_id},
                        "output_preview": json.dumps(res)[:280],
                        "timestamp_utc": _utc_now(),
                    }
                )
    else:
        try:
            active_tools = (
                [query_inventory_sql, check_alternative_parts, create_purchase_requisition, reserve_spare_part]
                if sql_approved
                else [query_inventory_sql, check_alternative_parts, create_purchase_requisition]
            )
            rule = (
                "If stock_qty >= 1, call reserve_spare_part(part_code, 1, ticket_id). "
                "If stock_qty == 0, call check_alternative_parts(part_code) & reserve qualified alternative."
                if sql_approved
                else "Do NOT call reserve_spare_part yet (Human QA approval required). "
                "Call check_alternative_parts if stock_qty == 0 and create_purchase_requisition if stock_qty <= min_reorder_qty."
            )
            sys_prompt = (
                "You are the Tier 2 Stock & Procurement SQL Specialist.\n"
                "1. Run query_inventory_sql (SELECT only) for the required part.\n"
                f"2. {rule}\n"
                "Return JSON summarizing part availability, PO status, and reservation."
                + _lang_prompt_hint(state)
            )
            task_prompt = f"Ticket: {ticket_id} | Required Part: {req_part}\nInstruction: {state.get('current_instruction')}"

            def _call_stock(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.0,
                )
                return _run_specialist_tool_loop(
                    llm=llm,
                    system_prompt=sys_prompt,
                    task_prompt=task_prompt,
                    tools=active_tools,
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            (raw_text, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_stock)
            for ex in executed:
                new_tool_calls.append({"agent": "Stock & Procurement", **ex})
            raw_llm_json = _safe_parse_json(raw_text)
        except Exception as exc:
            return {
                "last_worker": "stock_agent",
                "error_message": f"Stock Agent LLM failure: {exc}",
                "tool_calls_log": new_tool_calls,
            }

        pr_chk = create_purchase_requisition_impl(req_part, ticket_id)
        if pr_chk.get("created"):
            pr_result = pr_chk

    ctx = get_active_run_context()

    if not sql_approved:
        prim_q = query_inventory_sql_impl(
            f"SELECT part_code, part_name, stock_qty, min_reorder_qty, bin_location, unit_cost_usd, compatible_alt_code "
            f"FROM spare_parts_inventory WHERE UPPER(part_code) = '{req_part}'"
        )
        prim_row = (prim_q.get("rows") or [{}])[0]
        prim_qty = int(prim_row.get("stock_qty", 0))
        proposed_row: Optional[Dict[str, Any]] = None
        is_alt = False

        if prim_qty >= 1:
            proposed_row = prim_row
        else:
            if not pr_result.get("created"):
                pr_result = create_purchase_requisition_impl(req_part, ticket_id)
            alt_chk = check_alternative_parts_impl(req_part)
            if alt_chk.get("has_qualified_in_stock_alternative"):
                alt_code = str(alt_chk["compatible_alt_code"])
                alt_q = query_inventory_sql_impl(
                    f"SELECT part_code, part_name, stock_qty, bin_location, unit_cost_usd "
                    f"FROM spare_parts_inventory WHERE UPPER(part_code) = '{alt_code}'"
                )
                proposed_row = (alt_q.get("rows") or [{}])[0]
                is_alt = True

        if proposed_row and int(proposed_row.get("stock_qty", 0)) >= 1:
            prop_code = str(proposed_row["part_code"])
            cur_qty = int(proposed_row["stock_qty"])
            proj_qty = cur_qty - 1
            unit_cost = float(proposed_row["unit_cost_usd"])
            bin_loc = str(proposed_row["bin_location"])
            sql_preview = (
                f"UPDATE spare_parts_inventory\n"
                f"SET stock_qty = {proj_qty}  -- currently {cur_qty}\n"
                f"WHERE part_code = '{prop_code}';\n\n"
                f"INSERT INTO part_reservations (ticket_id, part_code, quantity, unit_cost_usd, bin_location, is_alternative)\n"
                f"VALUES ('{ticket_id}', '{prop_code}', 1, {unit_cost:.2f}, '{bin_loc}', {1 if is_alt else 0});"
            )
            report = {
                "requested_part_code": req_part,
                "reserved": False,
                "awaiting_sql_approval": True,
                "proposed_part_code": prop_code,
                "part_name": proposed_row.get("part_name", prop_code),
                "used_alternative": is_alt,
                "bin_location": bin_loc,
                "unit_cost_usd": unit_cost,
                "current_stock_qty": cur_qty,
                "projected_stock_qty": proj_qty,
                "sql_preview": sql_preview,
                "escalate_procurement": False,
            }
            pr_msg = f" Drafted PR {pr_result['pr_id']}." if pr_result.get("created") else ""
            action_sum = f"Propose SQL Write ({prop_code}) & Auto-PO Check"
            notes = (
                f"OEM {req_part} qty=0; proposing qualified alt {prop_code} (${unit_cost:.2f}).{pr_msg} Awaiting Human SQL approval."
                if is_alt
                else f"Proposing {prop_code} reservation in {bin_loc} (qty {cur_qty}->{proj_qty}, ${unit_cost:.2f}). Awaiting Human SQL approval."
            )
        else:
            report = {
                "requested_part_code": req_part,
                "reserved": False,
                "awaiting_sql_approval": False,
                "reserved_part_code": None,
                "used_alternative": False,
                "escalate_procurement": True,
                "notes": raw_llm_json.get("notes", f"Zero stock for {req_part} and compatible alternatives."),
            }
            action_sum = f"Stockout Escalation for {req_part}"
            notes = f"Zero inventory available for {req_part} or alternatives."
    else:
        latest_res = ctx.reservations[-1] if (ctx and ctx.reservations) else None
        if latest_res is None and state.get("execution_mode") == "live":
            alt_chk = check_alternative_parts_impl(req_part)
            if alt_chk.get("has_qualified_in_stock_alternative"):
                latest_res = reserve_spare_part_impl(str(alt_chk["compatible_alt_code"]), 1, ticket_id)
            else:
                direct = reserve_spare_part_impl(req_part, 1, ticket_id)
                if direct.get("status") == "RESERVED":
                    latest_res = direct

        if latest_res and latest_res.get("status") == "RESERVED":
            report = {
                "requested_part_code": req_part,
                "reserved": True,
                "awaiting_sql_approval": False,
                "reserved_part_code": latest_res["part_code"],
                "part_name": latest_res.get("part_name", latest_res["part_code"]),
                "used_alternative": bool(latest_res.get("is_alternative")),
                "bin_location": latest_res["bin_location"],
                "unit_cost_usd": float(latest_res["unit_cost_usd"]),
                "remaining_stock_qty": latest_res.get("remaining_stock_qty"),
                "escalate_procurement": False,
            }
            pr_msg = f" Drafted PR {pr_result['pr_id']}." if pr_result.get("created") else ""
            action_sum = (
                f"OEM {req_part} stockout -> Reserved alt {report['reserved_part_code']} ({report['bin_location']})"
                if report["used_alternative"]
                else f"Reserved {report['reserved_part_code']} ({report['bin_location']})"
            )
            notes = (
                f"Primary {req_part} qty=0; reserved alt {report['reserved_part_code']} "
                f"(rem qty={report['remaining_stock_qty']}, ${report['unit_cost_usd']:.2f}).{pr_msg}"
                if report["used_alternative"]
                else f"Reserved 1x {report['reserved_part_code']} (rem qty={report['remaining_stock_qty']}, ${report['unit_cost_usd']:.2f}).{pr_msg}"
            )
        else:
            report = {
                "requested_part_code": req_part,
                "reserved": False,
                "awaiting_sql_approval": False,
                "reserved_part_code": None,
                "used_alternative": False,
                "escalate_procurement": True,
                "notes": f"Zero stock for {req_part} and compatible alternatives.",
            }
            action_sum = f"Stockout Escalation for {req_part}"
            notes = f"Zero inventory available for {req_part} or alternatives."

    audit_entry = append_gmp_audit_log(
        ticket_id=ticket_id,
        step_number=_next_step_num(state),
        agent_name="Stock & Procurement",
        action_summary=action_sum,
        gate_decision="PENDING",
        audit_notes=notes,
    )

    if report.get("awaiting_sql_approval"):
        if is_tr:
            stock_thinking = (
                f"OEM {req_part} için SQLite stoğu sorgulandı. "
                + (
                    f"Stok=0 tespit edildi -> {report.get('bin_location')} rafında GMP muadil {report.get('proposed_part_code')} (Adet {report.get('current_stock_qty')}) bulundu. "
                    if report.get("used_alternative")
                    else f"{report.get('bin_location')} rafında {report.get('proposed_part_code')} (Adet {report.get('current_stock_qty')}) mevcut. "
                )
                + ("Düşük stok nedeniyle Oto-Satınalma (PR) açıldı. " if pr_result.get("created") else "")
                + "İnsan QA SQL onayı için atomik UPDATE/INSERT önizlemesi hazırlandı."
            )
            stock_delegate = "orchestrator -> İnsan QA (SQL_MUTATION)"
        else:
            stock_thinking = (
                f"Queried SQLite spare_parts_inventory for OEM {req_part}. "
                + (
                    f"Found stock=0 -> located qualified alternative {report.get('proposed_part_code')} "
                    f"in {report.get('bin_location')} (Qty {report.get('current_stock_qty')}). "
                    if report.get("used_alternative")
                    else f"Found {report.get('proposed_part_code')} in {report.get('bin_location')} (Qty {report.get('current_stock_qty')}). "
                )
                + ("Triggered Tier 2 Auto-PO due to reorder threshold. " if pr_result.get("created") else "")
                + "Drafted atomic SQL UPDATE/INSERT preview pending Human QA SQL permission."
            )
            stock_delegate = "orchestrator -> Human QA (SQL_MUTATION)"
    elif report.get("reserved"):
        if is_tr:
            stock_thinking = (
                f"İnsan SQL izni onaylandı. 1x {report.get('reserved_part_code')} (${report.get('unit_cost_usd', 0):,.2f}, {report.get('bin_location')}) "
                f"atomik olarak rezerve edildi. Kalan stok={report.get('remaining_stock_qty')}."
                + (f" Oto-PR {pr_result.get('pr_id')} açık." if pr_result.get("created") else "")
            )
            stock_delegate = "orchestrator -> field_technician (Maliyet Kapısı)"
        else:
            stock_thinking = (
                f"Human SQL permission confirmed. Executed atomic SQLite reservation for 1x {report.get('reserved_part_code')} "
                f"(${report.get('unit_cost_usd', 0):,.2f}, bin {report.get('bin_location')}). "
                f"Remaining qty={report.get('remaining_stock_qty')}."
                + (f" Auto-PO {pr_result.get('pr_id')} open." if pr_result.get("created") else "")
            )
            stock_delegate = "orchestrator -> field_technician (Cost Gate)"
    else:
        stock_thinking = f"Stok tükendi / Stock check failed: {req_part} = 0."
        stock_delegate = "orchestrator (ESCALATE)"

    stock_actions = [f"query_inventory_sql('{req_part}')"]
    if pr_result.get("created"):
        stock_actions.append(f"Auto-PO Drafted: {pr_result.get('pr_id')} ({pr_result.get('order_qty')}x)")
    if report.get("reserved"):
        stock_actions.append(f"SQLite Reserved 1x {report.get('reserved_part_code')} -> rem {report.get('remaining_stock_qty')}")
    elif report.get("awaiting_sql_approval"):
        stock_actions.append(f"Prepared SQL Write Preview ({report.get('proposed_part_code')})")

    stock_trace = _make_trace(
        node="stock_agent",
        tier="Tier 2 Inventory & Procurement",
        agent_name="Stock & Procurement Agent",
        command_received=str(state.get("current_instruction") or f"Verify SQLite inventory & reserve {req_part}"),
        thinking=stock_thinking,
        actions_taken=stock_actions,
        output_verdict=notes,
        delegated_to=stock_delegate,
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "last_worker": "stock_agent",
        "inventory_report": report,
        "purchase_requisition": pr_result,
        "sql_queries": list(ctx.sql_queries) if ctx else [],
        "tool_calls_log": new_tool_calls,
        "audit_trail": [audit_entry],
        "agent_events": [
            {
                "timestamp_utc": audit_entry["timestamp_utc"],
                "node": "stock_agent",
                "gate_decision": "PENDING",
                "summary": notes,
            }
        ],
        "agent_traces": [stock_trace],
    }


# ---------------------------------------------------------------------------
# Tier 1: Field Maintenance Technician (LOTO, Hardware Swap & Escalation)
# ---------------------------------------------------------------------------


def field_technician_node(state: MaintenanceState) -> Dict[str, Any]:
    machine_id = state["machine_id"]
    diag = state.get("diagnostic_report", {})
    inv = state.get("inventory_report", {})
    replan = state.get("engineering_replan", {})
    part_code = str(inv.get("reserved_part_code") or diag.get("required_part_code") or "BRG-4402")
    sop_ref = (
        str(replan.get("sop_ref"))
        if state.get("engineering_replan_done") and replan
        else f"{diag.get('sop_code', 'SOP-MNT-001')} p.{diag.get('page_number', 2)}"
    )
    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"
    new_tool_calls: List[Dict[str, Any]] = []
    effective_model = state["llm_model"]

    if state.get("execution_mode") == "offline_demo":
        rep_res = execute_part_replacement_impl(
            machine_id=machine_id,
            old_part=f"WORN-{diag.get('required_part_code', part_code)}",
            new_part=part_code,
            sop_ref=sop_ref,
        )
        vib_res = run_post_repair_vibration_test_impl(machine_id=machine_id)
        new_tool_calls.extend(
            [
                {
                    "agent": "Field Technician (Tier 1)",
                    "tool": "execute_part_replacement",
                    "args": {
                        "machine_id": machine_id,
                        "old_part": f"WORN-{diag.get('required_part_code', part_code)}",
                        "new_part": part_code,
                        "sop_ref": sop_ref,
                    },
                    "output_preview": json.dumps(rep_res)[:280],
                    "timestamp_utc": _utc_now(),
                },
                {
                    "agent": "Field Technician (Tier 1)",
                    "tool": "run_post_repair_vibration_test",
                    "args": {"machine_id": machine_id},
                    "output_preview": json.dumps(vib_res)[:280],
                    "timestamp_utc": _utc_now(),
                },
            ]
        )
    else:
        try:
            sys_prompt = (
                "You are the Tier 1 Field Maintenance Technician.\n"
                "1. Call execute_part_replacement(machine_id, old_part, new_part, sop_ref) following SOP or Engineer prescription.\n"
                "2. ALWAYS call run_post_repair_vibration_test(machine_id) afterwards.\n"
                "Return JSON with attempt_number and post_repair_vibration_rms."
                + _lang_prompt_hint(state)
            )
            task_prompt = (
                f"Ticket: {state['ticket_id']} | Machine: {machine_id} | Part: {part_code} | SOP: {sop_ref}\n"
                f"Instruction: {state.get('current_instruction')}"
            )

            def _call_tech(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.0,
                )
                return _run_specialist_tool_loop(
                    llm=llm,
                    system_prompt=sys_prompt,
                    task_prompt=task_prompt,
                    tools=[execute_part_replacement, run_post_repair_vibration_test],
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            (_, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_tech)
            for ex in executed:
                new_tool_calls.append({"agent": "Field Technician (Tier 1)", **ex})
        except Exception as exc:
            return {
                "last_worker": "field_technician",
                "error_message": f"Field Technician LLM failure: {exc}",
                "tool_calls_log": new_tool_calls,
            }

    ctx = get_active_run_context()
    prev_tests = int(state.get("vibration_test_count", 0))
    if ctx is not None and ctx.vibration_test_count <= prev_tests:
        if ctx.repair_attempt_count <= int(state.get("repair_attempt_count", 0)):
            execute_part_replacement_impl(machine_id, f"WORN-{part_code}", part_code, sop_ref)
        run_post_repair_vibration_test_impl(machine_id)

    attempts = list(ctx.maintenance_attempts) if ctx else []
    latest_attempt = attempts[-1] if attempts else {}
    latest_rms = float(ctx.latest_vibration_rms) if (ctx and ctx.latest_vibration_rms is not None) else 1.5
    attempt_count = int(ctx.vibration_test_count) if (ctx and ctx.vibration_test_count > 0) else max(1, len(attempts))
    passed = latest_rms <= GMP_VIBRATION_LIMIT_RMS

    report = {
        "machine_id": machine_id,
        "part_installed": part_code,
        "sop_ref": sop_ref,
        "attempt_count": attempt_count,
        "before_vibration_rms": state["alarm_vibration_rms"],
        "post_repair_vibration_rms": latest_rms,
        "gmp_limit_rms": GMP_VIBRATION_LIMIT_RMS,
        "passed_gmp_limit": passed,
        "procedure_executed": latest_attempt.get("procedure_executed", "Part replacement & verification completed"),
        "attempts_history": attempts,
    }

    if attempt_count > 1:
        rework_title = "Thermal Parallelism Alignment" if machine_id == "PK-02" else "Laser Shaft Realignment"
        action_label = f"{rework_title} (Attempt #{attempt_count}) & Vibration Test"
    else:
        action_label = f"Part Replacement ({part_code}) & Post-Repair Vibration Test"

    verdict_str = "PASS" if passed else "FAIL -> ESCALATE TO ENGINEER"
    notes = (
        f"Before: {state['alarm_vibration_rms']} mm/s -> Attempt #{attempt_count}: {latest_rms} mm/s RMS "
        f"(Limit <= {GMP_VIBRATION_LIMIT_RMS} -> {verdict_str})."
    )
    audit_entry = append_gmp_audit_log(
        ticket_id=state["ticket_id"],
        step_number=_next_step_num(state),
        agent_name="Field Technician (Tier 1)",
        action_summary=action_label,
        gate_decision="PENDING",
        audit_notes=notes,
    )

    tech_trace = _make_trace(
        node="field_technician",
        tier="Tier 1 Shop-Floor Execution",
        agent_name="Field Maintenance Technician",
        command_received=str(state.get("current_instruction") or f"Execute LOTO & {action_label}"),
        thinking=(
            f"Onaylı iş emri tamamlandı (Deneme #{attempt_count}): {report['procedure_executed']}. Onarım sonrası akselerometre doğrulaması: {latest_rms:.2f} mm/s RMS (Limit {GMP_VIBRATION_LIMIT_RMS}) "
            + ("-> cGMP UYGUN." if passed else "-> LİMİT AŞILDI! Güvenilirlik Mühendisine YUKARI eskalasyon başlatılıyor!")
            if is_tr
            else (
                f"Completed authorized work order (Attempt #{attempt_count}): {report['procedure_executed']}. "
                f"Ran post-repair tri-axial accelerometer verification -> {latest_rms:.2f} mm/s RMS vs {GMP_VIBRATION_LIMIT_RMS} limit "
                + ("(WITHIN cGMP ZONE A/B)." if passed else "(EXCEEDS cGMP LIMIT -> Halting handoff; escalating UP to Reliability Engineer!).")
            )
        ),
        actions_taken=[
            f"execute_part_replacement('{machine_id}', '{part_code}', '{sop_ref}')",
            f"run_post_repair_vibration_test('{machine_id}') -> {latest_rms:.2f} mm/s",
        ],
        output_verdict=notes,
        delegated_to=(
            ("orchestrator -> qa_clearance (Kapı 3B GEÇTİ)" if is_tr else "orchestrator -> qa_clearance (Gate 3B PASS)")
            if passed
            else ("orchestrator -> diagnostic_engineer (YUKARI ESKALASYON)" if is_tr else "orchestrator -> diagnostic_engineer (UPWARD ESCALATION)")
        ),
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "last_worker": "field_technician",
        "maintenance_report": report,
        "escalation_to_engineer": not passed,
        "telemetry_readings": list(ctx.telemetry_readings) if ctx else [],
        "maintenance_attempts": attempts,
        "vibration_test_count": int(ctx.vibration_test_count) if ctx else attempt_count,
        "repair_attempt_count": int(ctx.repair_attempt_count) if ctx else attempt_count,
        "tool_calls_log": new_tool_calls,
        "audit_trail": [audit_entry],
        "agent_events": [
            {
                "timestamp_utc": audit_entry["timestamp_utc"],
                "node": "field_technician",
                "gate_decision": "PENDING",
                "summary": notes,
            }
        ],
        "agent_traces": [tech_trace],
    }


# ---------------------------------------------------------------------------
# Tier 3: QA Line Clearance Officer
# ---------------------------------------------------------------------------


def qa_clearance_node(state: MaintenanceState) -> Dict[str, Any]:
    machine_id = state["machine_id"]
    ticket_id = state["ticket_id"]
    maint = state.get("maintenance_report", {})
    final_rms = float(maint.get("post_repair_vibration_rms", 1.5))
    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"
    new_tool_calls: List[Dict[str, Any]] = []
    effective_model = state["llm_model"]

    if state.get("execution_mode") == "offline_demo":
        clearance_res = run_qa_line_clearance_impl(machine_id, ticket_id, final_rms)
        new_tool_calls.append(
            {
                "agent": "QA Line Clearance",
                "tool": "run_qa_line_clearance",
                "args": {"machine_id": machine_id, "ticket_id": ticket_id, "final_vibration_rms": final_rms},
                "output_preview": json.dumps(clearance_res)[:280],
                "timestamp_utc": _utc_now(),
            }
        )
    else:
        try:
            sys_prompt = (
                "You are the Tier 3 QA Line Clearance Officer.\n"
                "Call run_qa_line_clearance(machine_id, ticket_id, final_vibration_rms) to verify LOTO removal, "
                "tool reconciliation, sanitation wipe/CIP, and 10-min stability. Return JSON summary."
                + _lang_prompt_hint(state)
            )
            task_prompt = f"Ticket: {ticket_id} | Machine: {machine_id} | Final RMS: {final_rms}"

            def _call_clear(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.0,
                )
                return _run_specialist_tool_loop(
                    llm=llm,
                    system_prompt=sys_prompt,
                    task_prompt=task_prompt,
                    tools=[run_qa_line_clearance],
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            (_, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_clear)
            for ex in executed:
                new_tool_calls.append({"agent": "QA Line Clearance", **ex})
            clearance_res = run_qa_line_clearance_impl(machine_id, ticket_id, final_rms)
        except Exception as exc:
            return {
                "last_worker": "qa_clearance",
                "error_message": f"QA Line Clearance LLM failure: {exc}",
                "tool_calls_log": new_tool_calls,
            }

    notes = (
        f"Line Clearance {clearance_res['line_status_after_clearance']}: "
        "LOTO cleared, 100% tools reconciled, sanitary wipe/CIP & stability passed."
    )
    audit_entry = append_gmp_audit_log(
        ticket_id=ticket_id,
        step_number=_next_step_num(state),
        agent_name="QA Line Clearance",
        action_summary=f"Post-Repair Line Clearance ({machine_id})",
        gate_decision="APPROVED" if clearance_res["clearance_approved"] else "REJECTED",
        audit_notes=notes,
    )

    clr_trace = _make_trace(
        node="qa_clearance",
        tier="Tier 3 Quality Assurance",
        agent_name="QA Line Clearance Officer",
        command_received=str(state.get("current_instruction") or f"Execute cGMP Line Clearance on {machine_id}"),
        thinking=(
            f"{machine_id} üzerinde onarım sonrası kararlılık ({final_rms:.2f} mm/s <= {GMP_VIBRATION_LIMIT_RMS}), 480V EKED kilit sökümü, %100 alet sayımı ve hijyenik IPA/CIP temizliği doğrulandı."
            if is_tr
            else (
                f"Audited post-repair stability ({final_rms:.2f} mm/s <= {GMP_VIBRATION_LIMIT_RMS}), 480V LOTO padlock removal, "
                f"100% tool accountability count, and cleanroom IPA/CIP flush completion on {machine_id}."
            )
        ),
        actions_taken=[f"run_qa_line_clearance('{machine_id}', final_rms={final_rms:.2f}) -> {clearance_res['line_status_after_clearance']}"],
        output_verdict=notes,
        delegated_to="orchestrator -> analyst_capa",
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "last_worker": "qa_clearance",
        "clearance_report": clearance_res,
        "clearance_approved": bool(clearance_res["clearance_approved"]),
        "tool_calls_log": new_tool_calls,
        "audit_trail": [audit_entry],
        "agent_events": [
            {
                "timestamp_utc": audit_entry["timestamp_utc"],
                "node": "qa_clearance",
                "gate_decision": "APPROVED" if clearance_res["clearance_approved"] else "REJECTED",
                "summary": notes,
            }
        ],
        "agent_traces": [clr_trace],
    }


# ---------------------------------------------------------------------------
# Tier 2/3: Quality, Cost & CAPA Analyst Agent
# ---------------------------------------------------------------------------


def analyst_capa_node(state: MaintenanceState) -> Dict[str, Any]:
    diag = state.get("diagnostic_report", {})
    inv = state.get("inventory_report", {})
    maint = state.get("maintenance_report", {})
    is_tr = str(state.get("language", "en")).lower() == "tr"
    lang_code = "tr" if is_tr else "en"
    new_tool_calls: List[Dict[str, Any]] = []
    effective_model = state["llm_model"]

    sop_ref = f"{diag.get('sop_code', 'SOP-MNT-001')} p.{diag.get('page_number', 2)}"
    failure_mode = str(diag.get("failure_mode", state["alarm_type"]))
    payload = {
        "ticket_id": state["ticket_id"],
        "machine_id": state["machine_id"],
        "failure_mode": failure_mode,
        "sop_ref": sop_ref,
        "part_used": inv.get("reserved_part_code", diag.get("required_part_code", "BRG-4402")),
        "used_alternative": bool(inv.get("used_alternative", False)),
        "part_cost_usd": float(inv.get("unit_cost_usd", 0.0)),
        "alarm_vibration_rms": float(state["alarm_vibration_rms"]),
        "final_vibration_rms": float(maint.get("post_repair_vibration_rms", 1.5)),
        "repair_attempts": int(maint.get("attempt_count", 1)),
    }

    analytics_res: Optional[Dict[str, Any]] = None
    capa_res: Optional[Dict[str, Any]] = None
    excel_res: Optional[Dict[str, Any]] = None

    if state.get("execution_mode") == "offline_demo":
        analytics_res = log_incident_analytics_impl(payload)
        capa_res = create_capa_if_repeat_failure_impl(state["machine_id"], state["ticket_id"], failure_mode, sop_ref)
        excel_res = generate_monthly_excel_report_impl()
        new_tool_calls.extend(
            [
                {
                    "agent": "Quality & CAPA Analyst",
                    "tool": "log_incident_analytics",
                    "args": {"ticket_data": payload},
                    "output_preview": json.dumps(analytics_res)[:280],
                    "timestamp_utc": _utc_now(),
                },
                {
                    "agent": "Quality & CAPA Analyst",
                    "tool": "create_capa_if_repeat_failure",
                    "args": {"machine_id": state["machine_id"], "ticket_id": state["ticket_id"]},
                    "output_preview": json.dumps(capa_res)[:280],
                    "timestamp_utc": _utc_now(),
                },
                {
                    "agent": "Quality & CAPA Analyst",
                    "tool": "generate_monthly_excel_report",
                    "args": {},
                    "output_preview": json.dumps(excel_res)[:280],
                    "timestamp_utc": _utc_now(),
                },
            ]
        )
    else:
        try:
            sys_prompt = (
                "You are the Quality, Cost & CAPA Analyst.\n"
                "1. Call log_incident_analytics(ticket_data).\n"
                "2. Call create_capa_if_repeat_failure(machine_id, ticket_id, failure_mode, sop_ref).\n"
                "3. Call generate_monthly_excel_report().\n"
                "Return JSON summary." + _lang_prompt_hint(state)
            )
            task_prompt = (
                f"Closed Ticket Payload:\n{json.dumps(payload, indent=2)}\n"
                f"Instruction: {state.get('current_instruction')}"
            )

            def _call_analyst(model_id: str) -> Tuple[str, List[Dict[str, Any]]]:
                llm = get_chat_model(
                    provider=state["llm_provider"],
                    model=model_id,
                    api_key=state.get("api_key"),
                    temperature=0.0,
                )
                return _run_specialist_tool_loop(
                    llm=llm,
                    system_prompt=sys_prompt,
                    task_prompt=task_prompt,
                    tools=[log_incident_analytics, create_capa_if_repeat_failure, generate_monthly_excel_report],
                    pacing_delay=float(state.get("pacing_delay_sec", 0.3)),
                )

            (_, executed), effective_model, _ = _invoke_live_with_model_fallback(state, _call_analyst)
            for ex in executed:
                new_tool_calls.append({"agent": "Quality & CAPA Analyst", **ex})
        except Exception as exc:
            return {
                "last_worker": "analyst_capa",
                "error_message": f"Analyst & CAPA LLM failure: {exc}",
                "tool_calls_log": new_tool_calls,
            }

    ctx = get_active_run_context()
    if analytics_res is None and ctx is not None and ctx.analytics_report is not None:
        analytics_res = ctx.analytics_report
    if analytics_res is None or not analytics_res.get("logged_to_db"):
        analytics_res = log_incident_analytics_impl(payload)
    if capa_res is None:
        capa_res = create_capa_if_repeat_failure_impl(state["machine_id"], state["ticket_id"], failure_mode, sop_ref)
    if excel_res is None:
        excel_res = {"excel_path": ctx.excel_path} if (ctx and ctx.excel_path) else generate_monthly_excel_report_impl()

    report = {
        "logged_to_db": True,
        "ticket_id": analytics_res["ticket_id"],
        "downtime_minutes": analytics_res["downtime_minutes"],
        "mttr_minutes": analytics_res["mttr_minutes"],
        "part_cost_usd": analytics_res["part_cost_usd"],
        "labor_cost_usd": analytics_res["labor_cost_usd"],
        "production_loss_usd": analytics_res["production_loss_usd"],
        "total_cost_usd": analytics_res["total_cost_usd"],
        "damage_severity": analytics_res["damage_severity"],
        "repair_attempts": analytics_res["repair_attempts"],
        "first_time_fix": analytics_res["first_time_fix"],
        "excel_report_path": excel_res["excel_path"],
        "monthly_kpis": analytics_res.get("monthly_kpis_after_update", {}),
    }

    capa_note = f" CAPA Triggered: {capa_res.get('capa_id')}." if capa_res.get("capa_triggered") else " Below CAPA threshold."
    notes = (
        f"Downtime={report['downtime_minutes']}m, Total Cost=${report['total_cost_usd']:,.2f} "
        f"(Parts ${report['part_cost_usd']:.0f} + Labor ${report['labor_cost_usd']:.0f} + "
        f"Loss ${report['production_loss_usd']:,.0f}), Severity={report['damage_severity']}.{capa_note} 4-sheet Excel exported."
    )
    audit_entry = append_gmp_audit_log(
        ticket_id=state["ticket_id"],
        step_number=_next_step_num(state),
        agent_name="Quality & CAPA Analyst",
        action_summary="KPI Logging, Auto-CAPA Check & 4-Sheet Excel Export",
        gate_decision="PENDING",
        audit_notes=notes,
    )

    analyst_trace = _make_trace(
        node="analyst_capa",
        tier="Tier 2/3 Financial & Quality Analytics",
        agent_name="Quality, Cost & CAPA Analyst",
        command_received=str(state.get("current_instruction") or f"Compute KPIs, check 30d repeat CAPA & export Excel for {state['ticket_id']}"),
        thinking=(
            f"Duruş={report['downtime_minutes']}dk (MTTR={report['mttr_minutes']}dk, Müdahale={report['repair_attempts']}). "
            f"Toplam Maliyet: ${report['part_cost_usd']:.2f} parça + ${report['labor_cost_usd']:.2f} işçilik + ${report['production_loss_usd']:,.2f} duruş = ${report['total_cost_usd']:,.2f} ({report['damage_severity']}). 30 günlük tekrar eşiği denetlendi."
            if is_tr
            else (
                f"Calculated Downtime={report['downtime_minutes']}m (MTTR={report['mttr_minutes']}m, "
                f"Attempts={report['repair_attempts']}). Financial sum: ${report['part_cost_usd']:.2f} part + "
                f"${report['labor_cost_usd']:.2f} labor + ${report['production_loss_usd']:,.2f} stoppage = "
                f"${report['total_cost_usd']:,.2f} ({report['damage_severity']}). Checked 30d asset history for repeat-failure CAPA."
            )
        ),
        actions_taken=[
            f"log_incident_analytics -> Total ${report['total_cost_usd']:,.2f}",
            f"create_capa_if_repeat_failure -> {capa_res.get('capa_id') if capa_res.get('capa_triggered') else 'No CAPA needed'}",
            "generate_monthly_excel_report() -> 4-Sheet .xlsx",
        ],
        output_verdict=notes,
        delegated_to="orchestrator (Kapı 4 Nihai İmza)" if is_tr else "orchestrator (Gate 4 Final Closure & Sign-Off)",
        timestamp_utc=audit_entry["timestamp_utc"],
        lang=lang_code,
    )

    return {
        "llm_model": effective_model,
        "last_worker": "analyst_capa",
        "analytics_report": report,
        "capa_report": capa_res,
        "tool_calls_log": new_tool_calls,
        "audit_trail": [audit_entry],
        "agent_events": [
            {
                "timestamp_utc": audit_entry["timestamp_utc"],
                "node": "analyst_capa",
                "gate_decision": "PENDING",
                "summary": notes,
            }
        ],
        "agent_traces": [analyst_trace],
    }
