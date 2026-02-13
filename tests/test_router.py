from personal_search_layer.router import (
    PrimaryIntent,
    VerifierMode,
    default_pipeline_settings,
    route_query,
)


def test_route_query_lookup() -> None:
    decision = route_query('find the exact phrase "smoke corpus keyword"')
    assert decision.primary_intent == PrimaryIntent.LOOKUP


def test_route_query_compare() -> None:
    decision = route_query("compare version A vs version B")
    assert decision.primary_intent == PrimaryIntent.COMPARE


def test_route_query_timeline() -> None:
    decision = route_query("timeline of project milestones")
    assert decision.primary_intent == PrimaryIntent.TIMELINE


def test_route_query_task() -> None:
    decision = route_query("create a checklist for deployment")
    assert decision.primary_intent == PrimaryIntent.TASK


def test_route_query_fact() -> None:
    decision = route_query("when was the index built?")
    assert decision.primary_intent == PrimaryIntent.FACT


def test_route_query_definition_fact() -> None:
    decision = route_query("what is reciprocal rank fusion")
    assert decision.primary_intent == PrimaryIntent.FACT


def test_route_query_summary_synthesis() -> None:
    decision = route_query("summarize retrieval performance")
    assert decision.primary_intent == PrimaryIntent.SYNTHESIS


def test_route_flags() -> None:
    decision = route_query("What is a vector index? Provide a summary")
    assert decision.flags.wants_definition is True
    assert decision.flags.wants_summary is True


def test_default_pipeline_settings_lookup() -> None:
    settings = default_pipeline_settings(PrimaryIntent.LOOKUP)
    assert 5 <= settings.k <= 10
    assert settings.generate_answer is False
    assert settings.allow_multihop == 0
    assert settings.verifier_mode in {VerifierMode.OFF, VerifierMode.MINIMAL}


def test_default_pipeline_settings_synthesis_bounds() -> None:
    settings = default_pipeline_settings(PrimaryIntent.SYNTHESIS)
    assert settings.allow_multihop <= 1
    assert settings.max_repair_passes <= 1
