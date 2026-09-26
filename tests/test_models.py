from __future__ import annotations

import pytest

from social_post.models import (
    ComparisonPoint,
    LinkedInDraft,
    ResearchBrief,
    ResearchClaim,
    ReviewAction,
    ReviewDecision,
    Source,
)


def test_linkedin_draft_normalizes_hashtags_and_counts_full_text():
    draft = LinkedInDraft(
        body="A useful technical post that is deliberately longer than eighty characters for validation.",
        hashtags=["AI", "#AI", "Knowledge Graphs"],
    )
    assert draft.hashtags == ["#AI", "#KnowledgeGraphs"]
    assert draft.character_count == len(draft.full_text)


def test_revision_requires_feedback():
    with pytest.raises(ValueError, match="Revision feedback is required"):
        ReviewDecision(action=ReviewAction.REVISE_IMAGE, revision=1)


def test_post_rejects_linkedin_character_overflow():
    with pytest.raises(ValueError, match="3,000"):
        LinkedInDraft(body="x" * 2_800, hashtags=["y" * 250])


def test_research_rejects_claims_with_unlisted_sources():
    with pytest.raises(ValueError, match="claim URL"):
        ResearchBrief(
            interpretation="RAG versus knowledge graphs",
            summary="A sufficiently detailed summary comparing two context architectures.",
            comparison_points=[
                ComparisonPoint(aspect=str(i), left="Left", right="Right")
                for i in range(3)
            ],
            claims=[
                ResearchClaim(
                    claim="Unsupported claim",
                    source_urls=["https://unlisted.example/claim"],
                )
            ],
            sources=[
                Source(title="One", url="https://example.com/one"),
                Source(title="Two", url="https://example.com/two"),
            ],
        )


def test_research_accepts_one_credible_source_for_niche_topics():
    brief = ResearchBrief(
        interpretation="A niche framework",
        summary="A sufficiently detailed summary based on one available primary source.",
        comparison_points=[
            ComparisonPoint(aspect=str(i), left="Left", right="Right")
            for i in range(3)
        ],
        sources=[Source(title="Primary", url="https://example.com/primary")],
    )

    assert len(brief.sources) == 1
