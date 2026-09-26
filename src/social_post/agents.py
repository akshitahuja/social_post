from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Protocol, TypeVar
from urllib.parse import urlparse

from crewai import Agent, LLM
from crewai_tools import SerperDevTool
from json_repair import loads as load_repaired_json
from pydantic import BaseModel, ValidationError

from .config import Settings
from .models import (
    LinkedInDraft,
    ResearchBrief,
    TopicClarification,
    TopicRequest,
    VisualBrief,
)


class AgentService(Protocol):
    def clarify(self, request: TopicRequest) -> TopicClarification: ...

    def research(self, request: TopicRequest) -> ResearchBrief: ...

    def write(
        self,
        request: TopicRequest,
        research: ResearchBrief,
        previous: LinkedInDraft | None = None,
        feedback: str = "",
    ) -> LinkedInDraft: ...

    def design(
        self,
        request: TopicRequest,
        research: ResearchBrief,
        previous: VisualBrief | None = None,
        feedback: str = "",
    ) -> VisualBrief: ...


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class GroqCompatibleLLM(LLM):
    """LiteLLM adapter that removes CrewAI-only message metadata.

    CrewAI marks stable prompt prefixes with a top-level ``cache_breakpoint``
    field. Native providers strip or translate it, but the LiteLLM fallback in
    CrewAI 1.15 forwards it to Groq, whose OpenAI-compatible API rejects it.
    """

    rate_limit_max_retries: int = 12
    rate_limit_max_wait_seconds: float = 60.0
    rate_limit_safety_buffer_seconds: float = 0.25

    @staticmethod
    def _clean_messages(messages):
        if not isinstance(messages, list):
            return messages
        return [
            {key: value for key, value in message.items() if key != "cache_breakpoint"}
            if isinstance(message, dict)
            else message
            for message in messages
        ]

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        return (
            exc.__class__.__name__ == "RateLimitError"
            or "rate limit" in str(exc).lower()
            or "rate_limit_exceeded" in str(exc).lower()
        )

    def _retry_delay(self, exc: Exception, retry_number: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                suggested = float(retry_after)
            except (TypeError, ValueError):
                suggested = 0.0
            if suggested > 0:
                return min(
                    suggested + self.rate_limit_safety_buffer_seconds,
                    self.rate_limit_max_wait_seconds,
                )

        patterns = (
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
            r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        )
        message = str(exc)
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return min(
                    float(match.group(1)) + self.rate_limit_safety_buffer_seconds,
                    self.rate_limit_max_wait_seconds,
                )
        return min(2 ** max(retry_number - 1, 0), self.rate_limit_max_wait_seconds)

    def call(self, messages, *args, **kwargs):
        cleaned_messages = self._clean_messages(messages)
        retry_number = 0
        while True:
            try:
                return super().call(cleaned_messages, *args, **kwargs)
            except Exception as exc:
                if not self._is_rate_limit_error(exc):
                    raise
                if retry_number >= self.rate_limit_max_retries:
                    raise
                retry_number += 1
                time.sleep(self._retry_delay(exc, retry_number))


class CrewAIAgentService:
    """Runs one focused CrewAI specialist for each pipeline stage."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.groq_api_key:
            os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
        if settings.serper_api_key:
            os.environ.setdefault("SERPER_API_KEY", settings.serper_api_key)
        self.llm = GroqCompatibleLLM(
            model=settings.model,
            api_key=settings.groq_api_key or None,
            rate_limit_max_retries=settings.rate_limit_max_retries,
            rate_limit_max_wait_seconds=settings.rate_limit_max_wait_seconds,
        )

    def _run(
        self,
        *,
        agent: Agent,
        description: str,
        expected_output: str,
        output_model: type[OutputModel],
        payload_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> OutputModel:
        schema = json.dumps(output_model.model_json_schema(), indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {agent.role}.\nGoal: {agent.goal}\n"
                    f"Background: {agent.backstory}\n"
                    "Follow the user's evidence and output requirements exactly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{description}\n\n"
                    f"Expected output: {expected_output}\n\n"
                    "Return exactly one JSON object and no markdown or explanatory text. "
                    "The JSON must conform to this schema:\n"
                    f"{schema}"
                ),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(self.settings.structured_output_retries + 1):
            raw = self.llm.call(messages)
            try:
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("The model returned an empty response")
                return self._parse_output(raw, output_model, payload_transform)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                if attempt >= self.settings.structured_output_retries:
                    raise
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": raw if isinstance(raw, str) else "",
                        },
                        {
                            "role": "user",
                            "content": (
                                "Your previous JSON did not satisfy the required schema. "
                                "Correct every validation error below and return the complete "
                                "JSON object again. Do not return a patch, markdown, commentary, "
                                "or a tool call. Preserve valid evidence and do not invent sources.\n\n"
                                f"VALIDATION ERRORS:\n{str(exc)[:4000]}"
                            ),
                        },
                    ]
                )
        raise RuntimeError("Structured output repair failed") from last_error

    @staticmethod
    def _parse_output(
        raw: str,
        output_model: type[OutputModel],
        payload_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> OutputModel:
        """Repair minor JSON formatting issues, then enforce the Pydantic contract.

        CrewAI's ``output_pydantic`` uses a required synthetic tool call. Some Groq
        models instead return valid JSON as text and Groq rejects that generation.
        Local validation keeps structured outputs without forcing a tool invocation.
        """
        candidate = raw.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        first, last = candidate.find("{"), candidate.rfind("}")
        if first >= 0 and last > first:
            candidate = candidate[first : last + 1]
        parsed = load_repaired_json(candidate)
        if payload_transform is not None:
            if not isinstance(parsed, dict):
                raise ValueError("Structured model output must be a JSON object")
            parsed = payload_transform(parsed)
        return output_model.model_validate(parsed)

    @staticmethod
    def _evidence_sources(search_results: list[dict[str, Any]]) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        seen: set[str] = set()
        for batch in search_results:
            result = batch.get("results", {})
            if not isinstance(result, dict):
                continue
            candidates: list[Any] = []
            for key in ("organic", "organic_results", "news"):
                value = result.get(key, [])
                if isinstance(value, list):
                    candidates.extend(value)
            knowledge_graph = result.get("knowledgeGraph")
            if isinstance(knowledge_graph, dict):
                candidates.append(
                    {
                        "title": knowledge_graph.get("title"),
                        "link": knowledge_graph.get("website")
                        or knowledge_graph.get("descriptionLink"),
                    }
                )
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                url = item.get("link") or item.get("url")
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    continue
                if url in seen:
                    continue
                seen.add(url)
                host = urlparse(url).netloc.removeprefix("www.")
                sources.append(
                    {
                        "title": str(item.get("title") or host or url)[:300],
                        "url": url,
                        "publisher": host[:150],
                    }
                )
                if len(sources) == 12:
                    return sources
        return sources

    @staticmethod
    def _ground_research_payload(
        payload: dict[str, Any],
        evidence_sources: list[dict[str, str]],
    ) -> dict[str, Any]:
        allowed = {source["url"] for source in evidence_sources}
        grounded_claims = []
        for claim in payload.get("claims", []):
            if not isinstance(claim, dict):
                continue
            urls = [str(url) for url in claim.get("source_urls", []) if str(url) in allowed]
            if urls:
                grounded_claims.append({**claim, "source_urls": urls})
        payload["claims"] = grounded_claims
        payload["sources"] = evidence_sources
        return payload

    def _agent(self, role: str, goal: str, backstory: str, tools=None) -> Agent:
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,
            tools=tools or [],
            verbose=False,
            allow_delegation=False,
            memory=False,
            max_retry_limit=2,
            respect_context_window=True,
        )

    def clarify(self, request: TopicRequest) -> TopicClarification:
        agent = self._agent(
            "Topic Clarifier",
            "Detect ambiguous terminology before expensive research begins",
            (
                "You are a careful technical editor. You identify acronyms, likely "
                "typos, and phrases with multiple materially different meanings. You "
                "do not block familiar, unambiguous technical terms."
            ),
        )
        return self._run(
            agent=agent,
            description=(
                "Review this proposed LinkedIn topic and decide whether research can "
                "safely proceed. Topic request:\n"
                f"{request.model_dump_json(indent=2)}\n\n"
                "Ask for clarification only when different interpretations would lead "
                "to substantially different posts. When clarification is unnecessary, "
                "set recommended_interpretation to a clean version of the topic."
            ),
            expected_output=(
                "A TopicClarification object with a decision, concise reason, no more "
                "than four likely interpretations, and a recommended interpretation."
            ),
            output_model=TopicClarification,
        )

    def research(self, request: TopicRequest) -> ResearchBrief:
        if not self.settings.serper_api_key:
            raise RuntimeError("SERPER_API_KEY is required for live research")
        topic = request.confirmed_interpretation or request.original_idea
        search_tool = SerperDevTool(n_results=5)
        queries = [
            topic,
            f"{topic} official documentation",
            f"{topic} comparison use cases limitations",
        ]
        search_results = []
        for query in queries:
            result = search_tool.run(search_query=query)
            search_results.append({"query": query, "results": result})
        evidence = json.dumps(search_results, default=str, ensure_ascii=False)
        evidence_sources = self._evidence_sources(search_results)
        if not evidence_sources:
            raise RuntimeError("Serper returned no usable web sources for this topic")
        agent = self._agent(
            "Evidence-focused Technology Researcher",
            "Create a concise, well-sourced brief for an accurate social post",
            (
                "You are a skeptical technology researcher. You prefer primary sources, "
                "official documentation, and reputable technical publications. You never "
                "invent URLs or retain unsupported claims."
            ),
        )
        return self._run(
            agent=agent,
            description=(
                "Synthesize a research brief for the confirmed topic and audience using "
                "only the supplied Serper evidence. Do not invent facts, sources, URLs, "
                "publishers, or expansions of acronyms. Prefer primary and official "
                "sources present in the evidence.\n"
                f"{request.model_dump_json(indent=2)}\n\n"
                f"SERPER EVIDENCE:\n{evidence}\n\n"
                "Return 3-8 useful comparison points, practical examples, important "
                "caveats, and explicit factual claims. Every claim must cite one or more "
                "URLs that also appear in sources. Exclude facts you cannot support. Mark "
                "claims whose truth can change over time as time_sensitive."
            ),
            expected_output=(
                "A ResearchBrief with a clear interpretation, summary, comparison points, "
                "examples, caveats, claims with source URLs, and 1-12 real sources. Prefer "
                "one supportable source over inventing extra citations."
            ),
            output_model=ResearchBrief,
            payload_transform=lambda payload: self._ground_research_payload(
                payload, evidence_sources
            ),
        )

    def write(
        self,
        request: TopicRequest,
        research: ResearchBrief,
        previous: LinkedInDraft | None = None,
        feedback: str = "",
    ) -> LinkedInDraft:
        agent = self._agent(
            "LinkedIn Technology Writer",
            "Turn verified research into an engaging, useful LinkedIn post",
            (
                "You write like an experienced practitioner: confident but not hyped, "
                "conversational but precise. You optimize for clarity, saves, and genuine "
                "discussion rather than empty engagement bait."
            ),
        )
        revision_context = ""
        if previous:
            revision_context = (
                "\nThis is a targeted revision. Preserve everything that does not conflict "
                f"with the feedback. Previous draft:\n{previous.model_dump_json(indent=2)}"
                f"\nHuman feedback:\n{feedback}"
            )
        return self._run(
            agent=agent,
            description=(
                f"Write a LinkedIn post for this request:\n{request.model_dump_json(indent=2)}\n"
                f"Use only this verified research:\n{research.model_dump_json(indent=2)}\n"
                "Use an expert-but-conversational voice, a specific non-clickbait hook, "
                "short mobile-readable paragraphs, a useful comparison, a practical "
                "takeaway, and a genuine closing question. Keep the post under 3,000 "
                "characters. Return at most five relevant hashtags separately. Do not put "
                "source URLs in the body and do not make claims absent from the brief."
                f"{revision_context}"
            ),
            expected_output=(
                "A LinkedInDraft containing the post body, up to five hashtags, and the "
                "research source URLs used as research_references."
            ),
            output_model=LinkedInDraft,
        )

    def design(
        self,
        request: TopicRequest,
        research: ResearchBrief,
        previous: VisualBrief | None = None,
        feedback: str = "",
    ) -> VisualBrief:
        agent = self._agent(
            "Social Infographic Designer",
            "Create a concise visual specification for a readable LinkedIn graphic",
            (
                "You design information-first social graphics. You minimize text, use "
                "parallel phrasing, and never place unsupported claims in a visual."
            ),
        )
        revision_context = ""
        if previous:
            revision_context = (
                "\nRevise this visual specification only as requested. Previous visual:\n"
                f"{previous.model_dump_json(indent=2)}\nHuman feedback:\n{feedback}"
            )
        return self._run(
            agent=agent,
            description=(
                f"Create a 1080x1080 infographic specification for:\n"
                f"{request.model_dump_json(indent=2)}\n"
                f"Verified research:\n{research.model_dump_json(indent=2)}\n"
                "Prefer a two-column comparison. Each side must have 2-4 short, parallel "
                "points, each ideally under 65 characters. Use the supplied brand colors: "
                f"primary {self.settings.primary_color}, accent {self.settings.accent_color}. "
                "Write descriptive alt text under 120 characters."
                f"{revision_context}"
            ),
            expected_output=(
                "A VisualBrief with concise labels and points, valid hex colors, and useful "
                "alt text. Do not return image data or markdown."
            ),
            output_model=VisualBrief,
        )
