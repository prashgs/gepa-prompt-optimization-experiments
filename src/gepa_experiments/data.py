"""Dataset loading and synthetic BRG example definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_synthetic_examples() -> list[dict[str, Any]]:
    """Return ~24 synthetic stakeholder briefs with gold requirements + checklists."""
    seeds = [
        {
            "id": "brg_01",
            "brief": "Build an employee leave request portal where staff submit PTO, managers approve or reject, and HR sees a calendar of approved leave. Integrate with our existing Okta SSO.",
            "checklist": [
                "leave request submission",
                "manager approval",
                "HR calendar view",
                "Okta SSO",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_02",
            "brief": "We need a customer support ticket system. Customers open tickets by email or web form. Agents triage by priority and SLA. Supervisors need dashboards for open, breached, and resolved tickets.",
            "checklist": [
                "ticket intake email web",
                "priority triage",
                "SLA tracking",
                "supervisor dashboard",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_03",
            "brief": "Create a vendor onboarding workflow: collect W-9, insurance certificates, and bank details; compliance reviews documents; finance activates payment profiles in NetSuite.",
            "checklist": [
                "document collection",
                "compliance review",
                "NetSuite payment profile",
                "workflow statuses",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_04",
            "brief": "Add a self-service password reset for internal apps. Users verify identity via MFA, set a new password meeting complexity rules, and get an audit log entry for security.",
            "checklist": [
                "MFA verification",
                "password complexity",
                "audit logging",
                "self-service flow",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_05",
            "brief": "Build a product inventory sync between our Shopify storefront and the warehouse WMS. Stock levels must update within 5 minutes of a change. Handle SKU mismatches with an exception queue.",
            "checklist": [
                "Shopify WMS sync",
                "5 minute freshness",
                "SKU mismatch exceptions",
                "stock level updates",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_06",
            "brief": "We want a meeting room booking app for HQ. Employees reserve rooms by time slot, see capacity and AV equipment, and get Outlook calendar invites. Admins can block rooms for maintenance.",
            "checklist": [
                "room reservation",
                "capacity AV info",
                "Outlook invites",
                "admin maintenance blocks",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_07",
            "brief": "Implement expense report submission for field sales. Users upload receipts, categorize spend, managers approve under policy limits, and finance exports to Concur nightly.",
            "checklist": [
                "receipt upload",
                "categorization",
                "manager approval policy",
                "Concur export",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_08",
            "brief": "Create a patient appointment scheduling module for a clinic. Patients pick providers and slots online, receive SMS reminders 24h before, and staff can reschedule with conflict detection.",
            "checklist": [
                "online scheduling",
                "SMS reminders",
                "reschedule conflict detection",
                "provider slots",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_09",
            "brief": "Add multi-currency checkout to our B2B portal. Show prices in USD, EUR, and GBP using daily FX rates from ECB. Invoices must store both local and USD amounts.",
            "checklist": [
                "multi-currency display",
                "ECB FX rates",
                "invoice dual amounts",
                "checkout",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_10",
            "brief": "Build a document retention policy engine. Classify uploaded files by type, apply retention periods, notify owners 30 days before purge, and support legal hold overrides.",
            "checklist": [
                "document classification",
                "retention periods",
                "purge notifications",
                "legal hold",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_11",
            "brief": "We need an API rate-limiting gateway for partner integrations. Default 1000 req/min per API key, burst to 2000, return 429 with Retry-After, and expose usage metrics in a partner portal.",
            "checklist": [
                "rate limits per API key",
                "429 Retry-After",
                "burst handling",
                "usage metrics portal",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_12",
            "brief": "Create an onboarding checklist app for new hires. HR assigns role-based tasks, managers mark completion, IT provisions accounts, and day-1 readiness is tracked on a dashboard.",
            "checklist": [
                "role-based tasks",
                "manager completion",
                "IT provisioning",
                "readiness dashboard",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_13",
            "brief": "Add subscription pause/resume for our SaaS. Customers pause for 1–3 months, billing stops, data retained, and resume restores prior plan. Support agents can override pause length.",
            "checklist": [
                "pause resume subscription",
                "billing stop",
                "data retention",
                "agent override",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_14",
            "brief": "Build a warehouse pick-path optimizer. Given an order batch, suggest aisle sequence to minimize walk distance, support handheld barcode confirmation, and flag short-picks.",
            "checklist": [
                "pick path optimization",
                "barcode confirmation",
                "short-pick flags",
                "order batching",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_15",
            "brief": "Implement GDPR data-subject request handling. Track access, erasure, and portability requests with 30-day SLA, identity verification, and export packages in JSON.",
            "checklist": [
                "DSR types",
                "30-day SLA",
                "identity verification",
                "JSON export",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_16",
            "brief": "Create a feature-flag service for mobile apps. Product can target by % rollout, user cohort, and app version. Clients poll every 5 minutes; audit every flag change.",
            "checklist": [
                "percentage rollout",
                "cohort targeting",
                "version targeting",
                "audit trail",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_17",
            "brief": "Add a returns portal for ecommerce. Customers select order items, print prepaid labels, track RMA status, and get refunds within 5 business days of warehouse receipt.",
            "checklist": [
                "item selection returns",
                "prepaid labels",
                "RMA tracking",
                "5-day refunds",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_18",
            "brief": "Build internal knowledge-base search with RAG. Employees query SOP docs; answers cite source pages; admins reindex weekly; PII in docs must be redacted from answers.",
            "checklist": [
                "RAG search",
                "source citations",
                "weekly reindex",
                "PII redaction",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_19",
            "brief": "We need shift scheduling for a call center. Planners assign agents to shifts respecting skills and labor law rest rules; agents swap shifts with approval; publish schedules 2 weeks ahead.",
            "checklist": [
                "skill-based scheduling",
                "rest rule compliance",
                "shift swaps approval",
                "2-week publish",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_20",
            "brief": "Create a contract renewal tracker. Pull end dates from Salesforce, alert owners 90/60/30 days out, capture renewal decision, and escalate overdue renewals to VP Sales.",
            "checklist": [
                "Salesforce end dates",
                "90 60 30 alerts",
                "renewal decision capture",
                "escalation",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_21",
            "brief": "Add offline mode for our field inspection app. Inspectors capture photos and forms without network, sync when online with conflict resolution preferring latest server timestamp.",
            "checklist": [
                "offline capture",
                "photo forms sync",
                "conflict resolution",
                "field inspection",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_22",
            "brief": "Build a marketing email preference center. Users manage topic subscriptions, frequency caps, and global unsubscribe. Changes propagate to Braze within 15 minutes.",
            "checklist": [
                "topic preferences",
                "frequency caps",
                "global unsubscribe",
                "Braze sync 15 min",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_23",
            "brief": "Implement fraud-review queue for payments over $500. Flag high-risk scores, analysts approve/decline with reason codes, and declined users get templated emails.",
            "checklist": [
                "high-risk flagging",
                "analyst review queue",
                "reason codes",
                "decline emails",
                "acceptance criteria",
            ],
        },
        {
            "id": "brg_24",
            "brief": "Create a classroom waitlist for university registration. When a seat opens, notify the next student via email/SMS with a 12-hour claim window before offering to the next.",
            "checklist": [
                "waitlist ordering",
                "seat open notification",
                "12-hour claim window",
                "email SMS",
                "acceptance criteria",
            ],
        },
    ]

    examples: list[dict[str, Any]] = []
    for seed in seeds:
        checklist = seed["checklist"]
        gold = _gold_from_brief(seed["brief"], checklist)
        examples.append(
            {
                "id": seed["id"],
                "brief": seed["brief"],
                "gold_requirements": gold,
                "checklist": checklist,
                "split": "train",
            }
        )

    # Assign splits: 12 train / 6 val / 6 test
    for i, ex in enumerate(examples):
        if i < 12:
            ex["split"] = "train"
        elif i < 18:
            ex["split"] = "val"
        else:
            ex["split"] = "test"
    return examples


def _gold_from_brief(brief: str, checklist: list[str]) -> str:
    topics = ", ".join(checklist[:-1]) if checklist else "core needs"
    return (
        f"## User Stories\n"
        f"- As a stakeholder, I want the system to address: {topics}, "
        f"so that the brief is fully delivered.\n\n"
        f"## Functional Requirements\n"
        f"- The system shall implement capabilities described in: {brief}\n"
        f"- The system shall cover checklist topics: {', '.join(checklist)}\n\n"
        f"## Non-Functional Requirements\n"
        f"- The solution shall be usable by intended roles with clear status feedback.\n"
        f"- Critical flows shall include auditability where sensitive actions occur.\n\n"
        f"## Acceptance Criteria\n"
        + "\n".join(
            f"- Given a valid user, when exercising '{item}', then the expected outcome is observable and verifiable."
            for item in checklist
        )
    )


def write_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = make_synthetic_examples()
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return path


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_dataset(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = [e for e in examples if e.get("split") == "train"]
    val = [e for e in examples if e.get("split") == "val"]
    test = [e for e in examples if e.get("split") == "test"]
    if not train:
        # Fallback if split field missing
        n = len(examples)
        train, val, test = examples[: n // 2], examples[n // 2 : 3 * n // 4], examples[3 * n // 4 :]
    return train, val, test


def to_gepa_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape examples for GEPA DefaultAdapter-style fields + custom extras."""
    out = []
    for ex in examples:
        out.append(
            {
                "input": ex["brief"],
                "additional_context": {
                    "checklist": ex.get("checklist", []),
                    "id": ex.get("id", ""),
                },
                "answer": ex.get("gold_requirements", ""),
                "brief": ex["brief"],
                "gold_requirements": ex.get("gold_requirements", ""),
                "checklist": ex.get("checklist", []),
                "id": ex.get("id", ""),
            }
        )
    return out
