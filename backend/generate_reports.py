#!/usr/bin/env python3
"""
First Dollar — Generate Community Report Summaries
Makes 21 LLM calls to Gemini to create narrative summaries of each community.
These are used for graph visualization tooltips and global search context.
"""

import json
import sys
import time
import requests
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"

# API keys to rotate (free tier, 5 RPM each)
API_KEYS = [
    "AIzaSyAu5b6DScCd2x-KMv4kl88uEXJlltVw6WM",
    "AIzaSyAeW5k5qiD3C6vy5ojppmpUpll0W6KLYMY",
    "AIzaSyAnsPL7uEz6i_PuJAU6xoq9z8nDM-BwXrk",
    "AIzaSyDillXU2EzEcHanFbPFY3AWrTIuchDL1W8",
]

MODEL = "gemini-2.5-flash"


def call_gemini(prompt: str, key_index: int = 0) -> str | None:
    """Call Gemini API with key rotation."""
    key = API_KEYS[key_index % len(API_KEYS)]
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 500, "temperature": 0.3},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        elif resp.status_code == 429:
            print(f"    Key {key_index} rate limited, trying next...")
            time.sleep(2)
            return call_gemini(prompt, key_index + 1)
        else:
            print(f"    API error {resp.status_code}: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("  Generating Community Reports (21 LLM calls)")
    print("=" * 60)

    communities = json.loads(
        (OUTPUT_DIR / "communities.json").read_text(encoding="utf-8")
    )
    entities = json.loads(
        (OUTPUT_DIR / "entities.json").read_text(encoding="utf-8")
    )
    entity_lookup = {e["name"]: e for e in entities}

    reports = []

    for i, comm in enumerate(communities):
        print(f"\n  [{i+1}/{len(communities)}] Community {comm['id']} ({comm['size']} nodes)")

        # Build context for this community
        leaders = comm["leaders"][:8]
        leader_info = []
        for name in leaders:
            ent = entity_lookup.get(name, {})
            desc = ent.get("description", "")[:150]
            etype = ent.get("type", "?")
            leader_info.append(f"- {name} ({etype}): {desc}")

        types = comm.get("dominant_types", {})
        type_str = ", ".join(f"{t}: {c}" for t, c in list(types.items())[:5])

        prompt = f"""You are summarizing a community of related financial concepts for a financial wellness app targeting underserved populations (gig workers, immigrants, unbanked individuals).

This community has {comm['size']} entities. The dominant entity types are: {type_str}.

Key entities in this community:
{chr(10).join(leader_info)}

Write a 2-3 sentence summary of what this community covers and why it matters for someone starting their financial journey. Be specific and practical. Do not use jargon."""

        summary = call_gemini(prompt, key_index=i)

        if summary:
            print(f"    OK: {summary[:80]}...")
        else:
            # Fallback: generate from leader names
            summary = f"This community covers topics related to {', '.join(leaders[:3]).lower()} and related financial concepts. It contains {comm['size']} interconnected entities."
            print(f"    FALLBACK: {summary[:80]}...")

        reports.append({
            "community_id": comm["id"],
            "size": comm["size"],
            "leaders": leaders,
            "dominant_types": types,
            "summary": summary,
        })

        time.sleep(1)  # Polite delay

    # Save
    output_path = OUTPUT_DIR / "community_reports.json"
    output_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Saved {len(reports)} community reports to {output_path}")


if __name__ == "__main__":
    main()
