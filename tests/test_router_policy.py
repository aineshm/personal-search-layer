from __future__ import annotations

import json

import personal_search_layer.router as router
from personal_search_layer.router import PrimaryIntent


def test_router_policy_can_be_overridden(monkeypatch, tmp_path) -> None:
    policy = {
        "flags": {
            "definition": ["define"],
            "steps": ["steps"],
            "summary": ["summarize"],
        },
        "classification": {
            "lookup_explicit": ["exact"],
            "compare": ["compare"],
            "timeline": ["timeline"],
            "task": ["checklist"],
            "synthesis": ["synthesize"],
            "fact_words": ["which"],
            "short_lookup_word_count": 2,
            "question_mark_is_fact": False,
        },
        "pipeline_settings": {
            "lookup": {
                "k": 5,
                "lexical_weight": 0.9,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": False,
                "verifier_mode": "minimal",
                "max_repair_passes": 0,
            },
            "fact": {
                "k": 7,
                "lexical_weight": 0.5,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": True,
                "verifier_mode": "strict",
                "max_repair_passes": 1,
            },
            "synthesis": {
                "k": 7,
                "lexical_weight": 0.5,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": True,
                "verifier_mode": "strict",
                "max_repair_passes": 1,
            },
            "compare": {
                "k": 7,
                "lexical_weight": 0.5,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": True,
                "verifier_mode": "strict",
                "max_repair_passes": 1,
            },
            "timeline": {
                "k": 7,
                "lexical_weight": 0.5,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": True,
                "verifier_mode": "strict",
                "max_repair_passes": 1,
            },
            "task": {
                "k": 7,
                "lexical_weight": 0.5,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": True,
                "verifier_mode": "strict",
                "max_repair_passes": 1,
            },
            "other": {
                "k": 7,
                "lexical_weight": 0.5,
                "allow_multihop": 0,
                "use_rerank": False,
                "generate_answer": True,
                "verifier_mode": "strict",
                "max_repair_passes": 1,
            },
        },
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    monkeypatch.setenv("PSL_ROUTER_POLICY", str(path))
    router._load_policy.cache_clear()

    decision = router.route_query("hello there")
    assert decision.primary_intent == PrimaryIntent.LOOKUP

    router._load_policy.cache_clear()
