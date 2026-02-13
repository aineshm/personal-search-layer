from __future__ import annotations

import runpy


def test_render_summary_includes_metrics(tmp_path) -> None:
    report = {
        "cases": 2,
        "top_k_default": 5,
        "model_name": "demo-model",
        "model_revision": "rev1",
        "git_commit": "abc123",
        "metrics@k": {
            "lexical": {"recall": 1.0, "mrr": 1.0, "ndcg": 0.5},
            "vector": {"recall": 0.5, "mrr": 0.25, "ndcg": 0.2},
            "hybrid": {"recall": 1.0, "mrr": 1.0, "ndcg": 0.5},
        },
        "metrics_delta": {"lexical": {"recall": 0.1, "mrr": -0.05, "ndcg": 0.0}},
        "router_accuracy": {"correct": 1, "total": 2, "accuracy": 0.5},
    }
    module = runpy.run_path("eval/summarize_eval.py")
    render_summary = module["render_summary"]
    summary = render_summary(report)

    assert "Evaluation summary" in summary
    assert "router accuracy: 0.500 (1/2)" in summary
    assert "lexical" in summary
    assert "+0.100" in summary
