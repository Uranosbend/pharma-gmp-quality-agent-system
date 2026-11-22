"""LangGraph 3-Tier Parallel Orchestra Graph, Streaming & HITL Resume Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents import (
    analyst_capa_node,
    diagnostic_engineer_node,
    field_technician_node,
    orchestrator_node,
    qa_clearance_node,
    stock_agent_node,
    stock_precheck_node,
)
from src.state import MaintenanceState
from src.tools import (
    RunContext,
    append_gmp_audit_log,
    clear_active_run_context,
    set_active_run_context,
)


def route_from_orchestrator(state: MaintenanceState) -> Union[str, List[Send]]:
    """Conditional Conductor router supporting parallel Send() fan-out and HITL pause."""
    if state.get("error_message") or state.get("protocol_closed") or state.get("awaiting_human_approval"):
        return END
    if int(state.get("iteration_count", 0)) >= int(state.get("max_iterations", 22)):
        return END

    nxt = state.get("next_agent", "FINISH")
    if nxt == "parallel_intake":
        return [
            Send("diagnostic_engineer", state),
            Send("stock_precheck", state),
        ]
    if nxt in ("diagnostic_engineer", "stock_agent", "field_technician", "qa_clearance", "analyst_capa"):
        return nxt
    return END


def build_maintenance_graph():
    """Compile the LangGraph 7-role Parallel Orchestra + 3-Tier Hierarchy graph."""
    workflow = StateGraph(MaintenanceState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("diagnostic_engineer", diagnostic_engineer_node)
    workflow.add_node("stock_precheck", stock_precheck_node)
    workflow.add_node("stock_agent", stock_agent_node)
    workflow.add_node("field_technician", field_technician_node)
    workflow.add_node("qa_clearance", qa_clearance_node)
    workflow.add_node("analyst_capa", analyst_capa_node)

    workflow.add_edge(START, "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        [
            "diagnostic_engineer",
            "stock_precheck",
            "stock_agent",
            "field_technician",
            "qa_clearance",
            "analyst_capa",
            END,
        ],
    )

    # All orchestra specialists & Tier-1 Technician report strictly back to the Conductor
    workflow.add_edge("diagnostic_engineer", "orchestrator")
    workflow.add_edge("stock_precheck", "orchestrator")
    workflow.add_edge("stock_agent", "orchestrator")
    workflow.add_edge("field_technician", "orchestrator")
    workflow.add_edge("qa_clearance", "orchestrator")
    workflow.add_edge("analyst_capa", "orchestrator")

    return workflow.compile()


def _hydrate_context_from_state(state: MaintenanceState) -> RunContext:
    """Rebuild RunContext from persisted MaintenanceState so HITL pause/resume is lossless."""
    inv = state.get("inventory_report", {})
    maint = state.get("maintenance_report", {})
    diag = state.get("diagnostic_report", {})

    sop_cite = ""
    if diag.get("sop_code"):
        sop_cite = f"{diag['sop_code']} p.{diag.get('page_number', 1)}"

    ctx = RunContext(
        ticket_id=state["ticket_id"],
        scenario_id=state["scenario_id"],
        machine_id=state["machine_id"],
        alarm_vibration_rms=float(state["alarm_vibration_rms"]),
        symptom_summary=state["symptom_summary"],
        inject_post_repair_fail=bool(state.get("inject_post_repair_fail", False)),
        vibration_test_count=int(state.get("vibration_test_count", 0)),
        repair_attempt_count=int(state.get("repair_attempt_count", 0)),
        latest_vibration_rms=maint.get("post_repair_vibration_rms"),
        part_used=inv.get("reserved_part_code") if inv.get("reserved") else None,
        used_alternative=bool(inv.get("used_alternative", False)),
        part_cost_usd=float(inv.get("unit_cost_usd", 0.0)) if inv.get("reserved") else 0.0,
        sop_citation=sop_cite,
        rag_chunks=list(state.get("rag_chunks", [])),
        sql_queries=list(state.get("sql_queries", [])),
        maintenance_attempts=list(state.get("maintenance_attempts", [])),
        telemetry_readings=list(state.get("telemetry_readings", [])),
        audit_step_counter=len(state.get("audit_trail", [])),
    )
    return ctx


def _merge_update_into_state(target: MaintenanceState, node_payload: Dict[str, Any]) -> None:
    """Apply LangGraph node diff respecting Annotated reducer list fields."""
    from src.state import _merge_append_unique

    reducer_list_keys = {
        "rag_chunks",
        "sql_queries",
        "telemetry_readings",
        "maintenance_attempts",
        "tool_calls_log",
        "audit_trail",
        "agent_events",
        "agent_traces",
    }
    for k, v in node_payload.items():
        if k in reducer_list_keys:
            target[k] = _merge_append_unique(target.get(k, []), v)  # type: ignore[literal-required]
        else:
            target[k] = v  # type: ignore[literal-required]


def stream_maintenance_workflow(
    initial_state: MaintenanceState,
) -> Iterator[Tuple[str, Dict[str, Any], MaintenanceState, RunContext]]:
    """Stream node-by-node LangGraph updates along with hydrated RunContext evidence."""
    graph = build_maintenance_graph()
    ctx = _hydrate_context_from_state(initial_state)
    set_active_run_context(ctx)

    merged_state: MaintenanceState = dict(initial_state)  # type: ignore[assignment]
    try:
        for update in graph.stream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": 50},
        ):
            for node_name, node_payload in update.items():
                if isinstance(node_payload, dict):
                    _merge_update_into_state(merged_state, node_payload)
                yield node_name, node_payload, merged_state, ctx
    finally:
        clear_active_run_context()


def apply_human_decision(
    state: MaintenanceState,
    approval_id: str,
    decision: Literal["APPROVED", "REJECTED"],
    reviewer: str = "QA Lead",
    comment: str = "",
) -> MaintenanceState:
    """Record a Human QA Supervisor's binding decision and prepare state for resume or hold."""
    updated: MaintenanceState = dict(state)  # type: ignore[assignment]
    pending = dict(state.get("pending_approval", {}))
    decisions = dict(state.get("human_decisions", {}))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    decisions[approval_id] = {
        "decision": decision,
        "reviewer": reviewer.strip() or "QA Lead",
        "comment": comment.strip(),
        "gate_type": pending.get("gate_type", approval_id),
        "timestamp_utc": ts,
    }
    updated["human_decisions"] = decisions
    updated["awaiting_human_approval"] = False
    updated["pending_approval"] = {}

    gate_label = pending.get("title", approval_id)
    note_suffix = f" Note: {comment.strip()}" if comment.strip() else ""
    audit_notes = f"Human QA ({reviewer.strip() or 'QA Lead'}) {decision} [{gate_label}].{note_suffix}"

    step_num = len(updated.get("audit_trail", [])) + 1
    audit_entry = append_gmp_audit_log(
        ticket_id=updated["ticket_id"],
        step_number=step_num,
        agent_name=f"Human QA ({reviewer.strip() or 'QA Lead'})",
        action_summary=f"HITL Authorization [{pending.get('gate_type', approval_id)}]",
        gate_decision=decision if decision == "APPROVED" else "HUMAN_REJECTED",
        audit_notes=audit_notes,
    )
    updated["audit_trail"] = list(updated.get("audit_trail", [])) + [audit_entry]
    updated["agent_events"] = list(updated.get("agent_events", [])) + [
        {
            "timestamp_utc": audit_entry["timestamp_utc"],
            "node": "human_qa",
            "gate_decision": audit_entry["gate_decision"],
            "summary": audit_notes,
        }
    ]
    is_tr = str(updated.get("language", "en")).lower() == "tr"
    human_trace = {
        "timestamp_utc": audit_entry["timestamp_utc"],
        "node": "human_qa",
        "tier": "İnsan Yönetişim Yetkisi" if is_tr else "Human Governance Authority",
        "agent_name": f"İnsan QA ({reviewer.strip() or 'QA Lideri'})" if is_tr else f"Human QA ({reviewer.strip() or 'QA Lead'})",
        "command_received": (
            f"HITL Kapısını Onayla [{pending.get('gate_type', approval_id)}]: {gate_label}"
            if is_tr
            else f"Authorize HITL Gate [{pending.get('gate_type', approval_id)}]: {gate_label}"
        ),
        "thinking": (
            f"Önerilen işlem incelendi ({pending.get('summary', '')}). "
            + (
                "GMP-QA-010 uyumu doğrulandı ve bir sonraki adım onaylandı."
                if decision == "APPROVED"
                else "Bağlayıcı QA yetkisi ile reddedildi; veritabanı ve donanım değişikliği engellendi."
            )
            if is_tr
            else (
                f"Reviewed proposed payload ({pending.get('summary', '')}). "
                + (
                    "Confirmed alignment with GMP-QA-010 controls and authorized next step."
                    if decision == "APPROVED"
                    else "Denied authorization under binding QA authority; blocking database & hardware changes."
                )
            )
        ),
        "actions_taken": [
            f"human_decisions['{approval_id}'] = {decision}",
            f"SHA-256 Audit Step #{step_num}",
        ],
        "output_verdict": f"{decision}: {audit_notes}",
        "delegated_to": (
            "orchestrator"
            if decision == "APPROVED"
            else ("FINISH (İNSAN REDDİ)" if is_tr else "FINISH (HUMAN_REJECTED)")
        ),
    }
    updated["agent_traces"] = list(updated.get("agent_traces", [])) + [human_trace]

    if decision == "REJECTED":
        updated["protocol_closed"] = True
        updated["escalated"] = True
        updated["last_gate_decision"] = "HUMAN_REJECTED"
        updated["next_agent"] = "FINISH"
    else:
        updated["last_gate_decision"] = "APPROVED"

    return updated


def run_maintenance_workflow(initial_state: MaintenanceState) -> Tuple[MaintenanceState, RunContext]:
    """Execute the LangGraph workflow until completion or Human QA pause."""
    final_state: MaintenanceState = initial_state
    final_ctx: Optional[RunContext] = None
    for _, _, merged, ctx in stream_maintenance_workflow(initial_state):
        final_state = merged
        final_ctx = ctx
    assert final_ctx is not None
    return final_state, final_ctx
