from __future__ import annotations

from social_post.models import (
    ReviewAction,
    ReviewDecision,
    TopicRequest,
    WorkflowStatus,
)
from social_post.renderer import InfographicRenderer
from social_post.storage import WorkflowRepository
from social_post.workflow import PipelineEngine

from conftest import FakeAgents


def make_engine(settings, agents):
    settings.ensure_directories()
    repository = WorkflowRepository(settings.workflows_dir)
    return (
        PipelineEngine(
            settings,
            repository,
            agents,
            InfographicRenderer("Test footer"),
        ),
        repository,
    )


def test_ambiguous_topic_pauses_before_research(settings):
    agents = FakeAgents(ambiguous=True)
    engine, repository = make_engine(settings, agents)
    state = engine.create(TopicRequest(original_idea="Difference between RAG and OKF"))

    engine.prepare(state.id)

    stored = repository.get(state.id)
    assert stored.status == WorkflowStatus.AWAITING_CLARIFICATION
    assert stored.research is None
    assert stored.clarification.likely_interpretations


def test_mocked_end_to_end_with_targeted_revisions(settings):
    agents = FakeAgents()
    engine, repository = make_engine(settings, agents)
    state = engine.create(
        TopicRequest(original_idea="RAG versus knowledge graphs")
    )

    engine.prepare(state.id)
    generated = repository.get(state.id)
    assert generated.status == WorkflowStatus.AWAITING_REVIEW
    assert generated.image_path
    original_image = generated.image_path
    original_body = generated.draft.body

    content_decision = ReviewDecision(
        action=ReviewAction.REVISE_CONTENT,
        feedback="Make the practical recommendation sharper.",
        revision=1,
    )
    engine.request_revision(state.id, content_decision)
    engine.revise(state.id, content_decision)
    content_revised = repository.get(state.id)
    assert content_revised.revision == 2
    assert content_revised.image_path == original_image
    assert content_revised.draft.body != original_body

    body_before_image_revision = content_revised.draft.body
    image_decision = ReviewDecision(
        action=ReviewAction.REVISE_IMAGE,
        feedback="Use a more outcome-focused headline.",
        revision=2,
    )
    engine.request_revision(state.id, image_decision)
    engine.revise(state.id, image_decision)
    image_revised = repository.get(state.id)
    assert image_revised.revision == 3
    assert image_revised.draft.body == body_before_image_revision
    assert image_revised.image_path != original_image

    approved = engine.approve(state.id, revision=3)
    assert approved.status == WorkflowStatus.APPROVED
    assert approved.can_publish

