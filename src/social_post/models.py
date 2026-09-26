from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowStatus(StrEnum):
    CLARIFYING = "clarifying"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    RESEARCHING = "researching"
    WRITING = "writing"
    DESIGNING = "designing"
    RENDERING = "rendering"
    AWAITING_REVIEW = "awaiting_review"
    REVISING_CONTENT = "revising_content"
    REVISING_IMAGE = "revising_image"
    REVISING_BOTH = "revising_both"
    APPROVED = "approved"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TopicRequest(BaseModel):
    original_idea: str = Field(min_length=3, max_length=500)
    confirmed_interpretation: str | None = Field(default=None, max_length=500)
    audience: str = Field(
        default="Technology professionals and business leaders",
        min_length=3,
        max_length=300,
    )
    context: str = Field(default="", max_length=2_000)


class TopicClarification(BaseModel):
    needs_clarification: bool
    reason: str = ""
    likely_interpretations: list[str] = Field(default_factory=list, max_length=4)
    recommended_interpretation: str | None = None


class Source(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    url: HttpUrl
    publisher: str = Field(default="", max_length=150)
    accessed_at: datetime = Field(default_factory=utc_now)


class ResearchClaim(BaseModel):
    claim: str = Field(min_length=1, max_length=700)
    source_urls: list[HttpUrl] = Field(min_length=1)
    time_sensitive: bool = False


class ComparisonPoint(BaseModel):
    aspect: str = Field(min_length=1, max_length=120)
    left: str = Field(min_length=1, max_length=350)
    right: str = Field(min_length=1, max_length=350)


class ResearchBrief(BaseModel):
    interpretation: str = Field(min_length=3, max_length=500)
    summary: str = Field(min_length=20, max_length=2_000)
    comparison_points: list[ComparisonPoint] = Field(min_length=3, max_length=8)
    practical_examples: list[str] = Field(default_factory=list, max_length=6)
    caveats: list[str] = Field(default_factory=list, max_length=6)
    claims: list[ResearchClaim] = Field(default_factory=list, max_length=12)
    sources: list[Source] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def claims_must_reference_listed_sources(self) -> "ResearchBrief":
        listed = {str(source.url) for source in self.sources}
        for claim in self.claims:
            unlisted = {str(url) for url in claim.source_urls} - listed
            if unlisted:
                raise ValueError(
                    "Every claim URL must also appear in sources: "
                    + ", ".join(sorted(unlisted))
                )
        return self


class LinkedInDraft(BaseModel):
    body: str = Field(min_length=80, max_length=2_800)
    hashtags: list[str] = Field(default_factory=list, max_length=5)
    research_references: list[HttpUrl] = Field(default_factory=list)
    character_count: int = 0

    @model_validator(mode="after")
    def normalize(self) -> "LinkedInDraft":
        normalized: list[str] = []
        for tag in self.hashtags:
            clean = "#" + tag.strip().lstrip("#").replace(" ", "")
            if len(clean) > 1 and clean not in normalized:
                normalized.append(clean)
        self.hashtags = normalized[:5]
        self.character_count = len(self.full_text)
        if self.character_count > 3_000:
            raise ValueError("LinkedIn post must not exceed 3,000 characters")
        return self

    @property
    def full_text(self) -> str:
        tags = " ".join(self.hashtags)
        return f"{self.body.strip()}\n\n{tags}".strip()


class VisualBrief(BaseModel):
    layout: Literal["comparison", "process"] = "comparison"
    headline: str = Field(min_length=3, max_length=80)
    subheadline: str = Field(default="", max_length=110)
    left_label: str = Field(min_length=1, max_length=30)
    right_label: str = Field(min_length=1, max_length=30)
    left_points: list[str] = Field(min_length=2, max_length=4)
    right_points: list[str] = Field(min_length=2, max_length=4)
    primary_color: str = "#0A66C2"
    accent_color: str = "#7C3AED"
    background_color: str = "#F7F9FC"
    alt_text: str = Field(min_length=20, max_length=120)

    @field_validator("primary_color", "accent_color", "background_color")
    @classmethod
    def colors_must_be_hex(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
            raise ValueError("Colors must use six-digit hex notation")
        return value.upper()


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REVISE_CONTENT = "revise_content"
    REVISE_IMAGE = "revise_image"
    REVISE_BOTH = "revise_both"
    CANCEL = "cancel"


class ReviewDecision(BaseModel):
    action: ReviewAction
    feedback: str = Field(default="", max_length=2_000)
    revision: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def feedback_required_for_revision(self) -> "ReviewDecision":
        if self.action in {
            ReviewAction.REVISE_CONTENT,
            ReviewAction.REVISE_IMAGE,
            ReviewAction.REVISE_BOTH,
        } and not self.feedback.strip():
            raise ValueError("Revision feedback is required")
        return self


class PublishResult(BaseModel):
    post_urn: str = Field(min_length=3)
    published_revision: int = Field(ge=1)
    published_at: datetime = Field(default_factory=utc_now)
    api_status: str = "published"


class WorkflowState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    request: TopicRequest
    status: WorkflowStatus = WorkflowStatus.CLARIFYING
    clarification: TopicClarification | None = None
    research: ResearchBrief | None = None
    draft: LinkedInDraft | None = None
    visual: VisualBrief | None = None
    image_path: str | None = None
    revision: int = 1
    approved_revision: int | None = None
    feedback_history: list[ReviewDecision] = Field(default_factory=list)
    publish_result: PublishResult | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_busy(self) -> bool:
        return self.status in {
            WorkflowStatus.CLARIFYING,
            WorkflowStatus.RESEARCHING,
            WorkflowStatus.WRITING,
            WorkflowStatus.DESIGNING,
            WorkflowStatus.RENDERING,
            WorkflowStatus.REVISING_CONTENT,
            WorkflowStatus.REVISING_IMAGE,
            WorkflowStatus.REVISING_BOTH,
            WorkflowStatus.PUBLISHING,
        }

    @property
    def can_publish(self) -> bool:
        return (
            self.status == WorkflowStatus.APPROVED
            and self.approved_revision == self.revision
            and self.publish_result is None
        )
