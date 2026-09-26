from __future__ import annotations

import json

import social_post.agents as agents_module
from crewai import LLM
import pytest
from social_post.agents import CrewAIAgentService, GroqCompatibleLLM
from social_post.models import TopicClarification

from conftest import FakeAgents


def test_configured_groq_model_initializes_through_litellm(settings):
    service = CrewAIAgentService(settings)
    agent = service._agent(
        role="Configuration test",
        goal="Verify that the configured provider can be initialized",
        backstory="A test-only agent that never makes an API request.",
    )

    assert agent.llm.model == "groq/openai/gpt-oss-20b"
    assert isinstance(agent.llm, GroqCompatibleLLM)


def test_structured_output_is_validated_locally_without_forced_tool_call(
    settings, monkeypatch
):
    captured = {}
    service = CrewAIAgentService(settings)

    def fake_call(messages, *args, **kwargs):
        captured["messages"] = messages
        return '''```json
        {
          "needs_clarification": true,
          "reason": "The acronym is ambiguous.",
          "likely_interpretations": ["Open Knowledge Foundation"],
          "recommended_interpretation": null
        }
        ```'''

    monkeypatch.setattr(service.llm, "call", fake_call)
    agent = service._agent("Clarifier", "Clarify", "Test agent")
    result = service._run(
        agent=agent,
        description="Clarify this topic.",
        expected_output="A clarification decision.",
        output_model=TopicClarification,
    )

    assert result.needs_clarification is True
    assert len(captured["messages"]) == 2
    assert captured["messages"][0]["role"] == "system"
    assert "Return exactly one JSON object" in captured["messages"][1]["content"]


def test_groq_adapter_strips_crewai_cache_breakpoints(settings, monkeypatch):
    captured = {}

    def fake_call(self, messages, *args, **kwargs):
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr(LLM, "call", fake_call)
    service = CrewAIAgentService(settings)
    original = [
        {
            "role": "system",
            "content": "Stable instructions",
            "cache_breakpoint": True,
        },
        {"role": "user", "content": "Topic", "cache_breakpoint": True},
    ]

    assert service.llm.call(original) == "ok"
    assert all("cache_breakpoint" not in item for item in captured["messages"])
    assert all("cache_breakpoint" in item for item in original)


def test_groq_adapter_waits_suggested_time_and_retries_same_call(
    settings, monkeypatch
):
    attempts = 0
    sleeps = []

    class RateLimitError(Exception):
        pass

    def fake_call(self, messages, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("Rate limit reached. Please try again in 1.5975s.")
        return "recovered"

    monkeypatch.setattr(LLM, "call", fake_call)
    monkeypatch.setattr(agents_module.time, "sleep", sleeps.append)
    service = CrewAIAgentService(settings)

    result = service.llm.call([{"role": "user", "content": "Continue"}])

    assert result == "recovered"
    assert attempts == 2
    assert sleeps == [pytest.approx(1.8475)]


def test_groq_adapter_does_not_retry_unrelated_errors(settings, monkeypatch):
    def fake_call(self, messages, *args, **kwargs):
        raise ValueError("Invalid prompt")

    monkeypatch.setattr(LLM, "call", fake_call)
    service = CrewAIAgentService(settings)

    with pytest.raises(ValueError, match="Invalid prompt"):
        service.llm.call([{"role": "user", "content": "Continue"}])


def test_research_searches_deterministically_then_uses_tool_free_agent(
    settings, monkeypatch
):
    queries = []
    captured = {}

    class FakeSearchTool:
        def __init__(self, n_results):
            assert n_results == 5

        def run(self, *, search_query):
            queries.append(search_query)
            return {
                "organic": [
                    {
                        "title": "Official source",
                        "link": "https://example.com/source",
                        "snippet": "Evidence",
                    }
                ]
            }

    monkeypatch.setattr(agents_module, "SerperDevTool", FakeSearchTool)
    service = CrewAIAgentService(settings)
    expected = FakeAgents().research(
        agents_module.TopicRequest(
            original_idea="RAG vs OKF",
            confirmed_interpretation="RAG vs Open Knowledge Framework",
        )
    )

    def fake_run(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(service, "_run", fake_run)
    result = service.research(
        agents_module.TopicRequest(
            original_idea="RAG vs OKF",
            confirmed_interpretation="RAG vs Open Knowledge Framework",
        )
    )

    assert result == expected
    assert len(queries) == 3
    assert captured["agent"].tools == []
    assert "SERPER EVIDENCE" in captured["description"]


def test_research_payload_is_grounded_to_actual_search_sources():
    evidence_sources = [
        {
            "title": "Official source",
            "url": "https://example.com/official",
            "publisher": "example.com",
        }
    ]
    payload = {
        "claims": [
            {
                "claim": "Supported",
                "source_urls": ["https://example.com/official"],
                "time_sensitive": False,
            },
            {
                "claim": "Invented",
                "source_urls": ["https://invented.example/fake"],
                "time_sensitive": False,
            },
        ],
        "sources": [{"title": "Invented", "url": "https://invented.example/fake"}],
    }

    grounded = CrewAIAgentService._ground_research_payload(payload, evidence_sources)

    assert grounded["sources"] == evidence_sources
    assert [claim["claim"] for claim in grounded["claims"]] == ["Supported"]


def test_structured_output_validation_error_is_repaired(settings, monkeypatch):
    service = CrewAIAgentService(settings)
    valid = FakeAgents().research(
        agents_module.TopicRequest(original_idea="RAG versus knowledge graphs")
    ).model_dump(mode="json")
    invalid = {**valid, "comparison_points": valid["comparison_points"][:1]}
    responses = iter([json.dumps(invalid), json.dumps(valid)])
    calls = []

    def fake_call(messages, *args, **kwargs):
        calls.append(messages.copy())
        return next(responses)

    monkeypatch.setattr(service.llm, "call", fake_call)
    agent = service._agent("Researcher", "Research", "Test agent")
    result = service._run(
        agent=agent,
        description="Compare the two systems.",
        expected_output="A complete research brief.",
        output_model=agents_module.ResearchBrief,
    )

    assert len(result.comparison_points) == 3
    assert len(calls) == 2
    assert "at least 3 items" in calls[1][-1]["content"]
