from __future__ import annotations

import json
from pathlib import Path

from personal_search_layer.router import route_query


def test_router_intent_accuracy_threshold() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "eval" / "router_intents.jsonl"
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))

    total = 0
    correct = 0
    for case in cases:
        query = case.get("query")
        expected = case.get("intent")
        if not query or not expected:
            continue
        total += 1
        if route_query(query).primary_intent.value == expected:
            correct += 1

    assert total > 0
    assert correct / total >= 0.8
