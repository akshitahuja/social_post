from __future__ import annotations

from pathlib import Path

import pytest

from social_post.config import Settings
from social_post.models import (
    ComparisonPoint,
    LinkedInDraft,
    ResearchBrief,
    ResearchClaim,
    Source,
    TopicClarification,
    TopicRequest,
    VisualBrief,
)


class FakeAgents:
    def __init__(self, ambiguous: bool = False):
        self.ambiguous = ambiguous
        self.write_calls = 0
        self.design_calls = 0

    def clarify(self, request: TopicRequest) -> TopicClarification:
        if self.ambiguous:
            return TopicClarification(
                needs_clarification=True,
                reason="OKF has multiple possible meanings.",
                likely_interpretations=[
                    "RAG versus the Open Knowledge Framework",
                    "RAG versus a knowledge graph",
                ],
                recommended_interpretation=None,
            )
        return TopicClarification(
            needs_clarification=False,
            recommended_interpretation=request.original_idea,
        )

    def research(self, request: TopicRequest) -> ResearchBrief:
        return ResearchBrief(
            interpretation=request.confirmed_interpretation or request.original_idea,
            summary=(
                "RAG retrieves source passages at query time, while knowledge graphs "
                "represent explicit entities and relationships for connected reasoning."
            ),
            comparison_points=[
                ComparisonPoint(
                    aspect="Data model",
                    left="Unstructured passages and embeddings",
                    right="Entities and explicit relationships",
                ),
                ComparisonPoint(
                    aspect="Best at",
                    left="Grounding answers in documents",
                    right="Traversing connected facts",
                ),
                ComparisonPoint(
                    aspect="Trade-off",
                    left="Retrieval quality affects the answer",
                    right="Graph construction needs governance",
                ),
            ],
            practical_examples=["Product documentation assistant"],
            caveats=["The approaches can be combined."],
            claims=[
                ResearchClaim(
                    claim="RAG adds retrieved context to generation.",
                    source_urls=["https://example.com/rag"],
                )
            ],
            sources=[
                Source(title="RAG paper", url="https://example.com/rag"),
                Source(title="Knowledge graphs", url="https://example.com/kg"),
            ],
        )

    def write(
        self,
        request: TopicRequest,
        research: ResearchBrief,
        previous: LinkedInDraft | None = None,
        feedback: str = "",
    ) -> LinkedInDraft:
        self.write_calls += 1
        suffix = f"\n\nRevision note: {feedback}" if feedback else ""
        return LinkedInDraft(
            body=(
                "RAG and knowledge graphs solve different context problems.\n\n"
                "RAG finds relevant passages when a question arrives. Knowledge graphs "
                "make entities and their relationships explicit.\n\n"
                "Use RAG for grounding in changing documents. Use a graph when connected "
                "relationships are central. In many systems, the strongest answer is both.\n\n"
                "Which context problem are you solving?"
                + suffix
            ),
            hashtags=["RAG", "KnowledgeGraphs", "AIEngineering"],
            research_references=[source.url for source in research.sources],
        )

    def design(
        self,
        request: TopicRequest,
        research: ResearchBrief,
        previous: VisualBrief | None = None,
        feedback: str = "",
    ) -> VisualBrief:
        self.design_calls += 1
        headline = "RAG vs Knowledge Graphs"
        if feedback:
            headline = "Choosing Your Context Layer"
        return VisualBrief(
            headline=headline,
            subheadline="Two ways to give AI useful context",
            left_label="RAG",
            right_label="Knowledge graph",
            left_points=["Retrieves passages", "Works with documents", "Query-time context"],
            right_points=["Connects entities", "Models relationships", "Structured context"],
            alt_text="Comparison of RAG and knowledge graphs for adding context to AI systems.",
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        groq_api_key="test-groq",
        serper_api_key="test-serper",
        linkedin_client_id="test-client",
        linkedin_client_secret="test-secret",
        linkedin_redirect_uri="http://test/auth/linkedin/callback",
    )

