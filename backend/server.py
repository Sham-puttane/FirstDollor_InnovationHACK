#!/usr/bin/env python3
"""
First Dollar — Backend API Server

Chains: query_graph -> foo_engine -> personalize -> JSON response

Endpoints:
    POST /api/query            — Full pipeline (intake -> personalized queue + graph)
    POST /api/whatif            — What-If slider (re-runs FOO only, no GraphRAG, ~50ms)
    POST /api/search           — Semantic search (Q6 free text)
    GET  /api/community/<id>   — Community info with report
    GET  /api/node/<name>      — Node neighborhood (click-to-expand)
    GET  /api/health           — Health check
    GET  /api/personas         — Pre-built demo personas
"""

import sys
import os
import json
import copy
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from query_graph import GraphQuery
from foo_engine import order_actions
from personalize import personalize_steps

# ─── Initialize ──────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

print("Loading knowledge graph...", file=sys.stderr)
gq = GraphQuery()
print("Server ready.", file=sys.stderr)

# Session cache: stores last traversal so What-If doesn't re-run GraphRAG
_session_cache = {}


# ─── Pre-built demo personas ────────────────────────────────────────────────

DEMO_PERSONAS = {
    "maria": {
        "name": "Maria",
        "description": "Unbanked gig worker with credit card debt, no insurance, no savings",
        "args": {"q1": "no", "q2": "gig", "q3": "credit_card", "q4": "nothing", "q5": ["none"], "q6": None},
    },
    "james": {
        "name": "James",
        "description": "First-gen college student with student loans, has health insurance, minimal savings",
        "args": {"q1": "yes", "q2": "salary", "q3": "student", "q4": "under_500", "q5": ["health"], "q6": None},
    },
    "aisha": {
        "name": "Aisha",
        "description": "Recent immigrant, cash income, no debt, no savings, no insurance",
        "args": {"q1": "no", "q2": "cash", "q3": "none", "q4": "nothing", "q5": ["none"],
                 "q6": "I just moved to America and need help with money"},
    },
}

# ─── What-If Scenarios (per brief section 2.5) ──────────────────────────────
# Each scenario changes one profile variable. FOO re-runs, GraphRAG does NOT.

WHATIF_SCENARIOS = {
    "lose_job": {
        "label": "What if I lose my job?",
        "changes": {"income_type": "irregular", "savings_level": "nothing"},
    },
    "earn_more": {
        "label": "What if I earn $200 more per month?",
        "changes": {},  # income amount doesn't change FOO logic, but savings threshold may
    },
    "pay_off_debt": {
        "label": "What if I pay off my debt?",
        "changes": {"debt_type": "none"},
    },
    "get_renters_insurance": {
        "label": "What if I get renters insurance?",
        "changes": {"_add_insurance": "renters"},
    },
    "open_bank_account": {
        "label": "What if I open a bank account?",
        "changes": {"has_bank_account": "yes"},
    },
    "get_steady_job": {
        "label": "What if I get a steady job?",
        "changes": {"income_type": "salary"},
    },
}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "graph_nodes": gq.G.number_of_nodes(),
        "graph_edges": gq.G.number_of_edges(),
        "communities": len(gq.communities),
        "has_embeddings": gq.has_embeddings,
    })


@app.route("/api/personas")
def personas():
    return jsonify({
        name: {"name": p["name"], "description": p["description"], "args": p["args"]}
        for name, p in DEMO_PERSONAS.items()
    })


@app.route("/api/query", methods=["POST"])
def query():
    """Full pipeline: intake -> graph traversal -> FOO -> personalization."""
    data = request.json

    if "persona" in data and data["persona"] in DEMO_PERSONAS:
        args = DEMO_PERSONAS[data["persona"]]["args"].copy()
    else:
        args = {
            "q1": data.get("q1", "yes"), "q2": data.get("q2", "salary"),
            "q3": data.get("q3", "none"), "q4": data.get("q4", "nothing"),
            "q5": data.get("q5", ["none"]), "q6": data.get("q6"),
        }

    # Step 1: Graph traversal (this is the only GraphRAG call)
    traversal = gq.traverse_from_profile(**args)

    # Cache traversal for What-If slider (per brief: "full context block stored in memory")
    session_id = data.get("session_id", "default")
    _session_cache[session_id] = {
        "traversal": traversal,
        "args": args,
    }

    # Step 2: FOO ordering
    steps = order_actions(traversal)

    # Step 3: Gemini personalization
    if not data.get("skip_personalize"):
        steps = personalize_steps(
            steps, persona=traversal["persona"],
            profile=traversal["profile"], q6_text=args.get("q6"),
        )

    # Step 4: Build response
    viz = gq.get_graph_for_visualization(traversal["all_nodes"], traversal["edges"])
    community_info = [gq.get_community_info(cid) for cid in traversal["communities_touched"]
                      if gq.get_community_info(cid)]

    return jsonify({
        "session_id": session_id,
        "profile": traversal["profile"],
        "persona": traversal["persona"],
        "protection_gaps": traversal["protection_gaps"],
        "steps": steps,
        "graph": viz,
        "communities": community_info,
        "whatif_scenarios": {k: v["label"] for k, v in WHATIF_SCENARIOS.items()},
        "stats": {
            "total_nodes": len(traversal["all_nodes"]),
            "total_edges": len(traversal["edges"]),
            "total_steps": len(steps),
            "communities_touched": len(traversal["communities_touched"]),
        },
    })


@app.route("/api/whatif", methods=["POST"])
def whatif():
    """
    What-If slider: re-runs FOO only with one profile variable changed.
    Per brief: "GraphRAG runs exactly once. The slider only re-executes
    the JavaScript FOO rule engine with one profile variable changed.
    This takes ~50 milliseconds."

    Request: {"session_id": "...", "scenario": "lose_job"}
    """
    data = request.json
    session_id = data.get("session_id", "default")
    scenario_key = data.get("scenario")

    if session_id not in _session_cache:
        return jsonify({"error": "No active session. Call /api/query first."}), 400

    if scenario_key not in WHATIF_SCENARIOS:
        return jsonify({"error": f"Unknown scenario: {scenario_key}",
                        "available": list(WHATIF_SCENARIOS.keys())}), 400

    scenario = WHATIF_SCENARIOS[scenario_key]
    cached = _session_cache[session_id]
    traversal = cached["traversal"]

    # Deep copy the traversal and modify only the profile
    modified = copy.deepcopy(traversal)
    changes = scenario["changes"]

    # Apply profile changes
    for key, value in changes.items():
        if key == "_add_insurance":
            # Special: add insurance type to q5
            if value not in modified["profile"]["insurance_types"]:
                modified["profile"]["insurance_types"].append(value)
            if "none" in modified["profile"]["insurance_types"]:
                modified["profile"]["insurance_types"].remove("none")
            # Recalculate protection gaps
            all_types = {"renters", "health", "auto", "life"}
            has = set(modified["profile"]["insurance_types"]) - {"none"}
            modified["protection_gaps"] = sorted(all_types - has)
        elif key in modified["profile"]:
            modified["profile"][key] = value

    # Recalculate protection gaps if insurance changed
    if "has_bank_account" in changes or "debt_type" in changes or "income_type" in changes:
        pass  # protection_gaps don't change for these

    # Re-run FOO only (no GraphRAG, no Gemini) — this is ~50ms
    steps = order_actions(modified)

    return jsonify({
        "scenario": scenario_key,
        "scenario_label": scenario["label"],
        "profile": modified["profile"],
        "persona": modified["persona"],
        "protection_gaps": modified["protection_gaps"],
        "steps": steps,
        "stats": {"total_steps": len(steps)},
    })


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    results = gq.search(data.get("query", ""), top_k=data.get("top_k", 10))
    return jsonify({"query": data.get("query", ""), "results": results})


@app.route("/api/community/<int:cid>")
def community(cid):
    info = gq.get_community_info(cid)
    if info:
        return jsonify(info)
    return jsonify({"error": f"Community {cid} not found"}), 404


@app.route("/api/node/<path:name>")
def node(name):
    depth = request.args.get("depth", 1, type=int)
    result = gq.get_node_neighborhood(name.upper(), depth=min(depth, 3))
    if "error" in result:
        return jsonify(result), 404
    viz = gq.get_graph_for_visualization(result["all_nodes"], result["edges"])
    return jsonify(viz)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  First Dollar API Server")
    print("  http://localhost:5000")
    print("  Endpoints:")
    print("    POST /api/query            — Full pipeline")
    print("    POST /api/whatif           — What-If slider (FOO only, ~50ms)")
    print("    POST /api/search           — Semantic search")
    print("    GET  /api/community/<id>   — Community info")
    print("    GET  /api/node/<name>      — Node neighborhood")
    print("    GET  /api/health           — Health check")
    print("    GET  /api/personas         — Demo personas\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
