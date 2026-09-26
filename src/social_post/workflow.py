from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from pydantic import BaseModel, Field

from .agents import AgentService
from .config import Settings
from .models import (
    LinkedInDraft,
    ResearchBrief,
    ReviewAction,
    ReviewDecision,
    TopicRequest,
    VisualBrief,
    WorkflowState,
    WorkflowStatus,
)
from .renderer import InfographicRenderer
from .storage import WorkflowRepository


class PipelineFlowState(BaseModel):
    id: str
    request: TopicRequest
    workflow_id: str
    research: ResearchBrief | None = None
    draft: LinkedInDraft | None = None
    visual: VisualBrief | None = None
    image_path: str | None = None


@persist()
class InitialGenerationFlow(Flow[PipelineFlowState]):
    services: Any = Field(exclude=True)
    repository: Any = Field(exclude=True)
    renderer: Any = Field(exclude=True)
    images_dir: Path = Field(exclude=True)

    @start()
    def research_topic(self) -> ResearchBrief:
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "status", WorkflowStatus.RESEARCHING),
        )
        self.state.research = self.services.research(self.state.request)
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "research", self.state.research),
        )
        return self.state.research

    @listen(research_topic)
    def write_post(self, _: ResearchBrief) -> LinkedInDraft:
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "status", WorkflowStatus.WRITING),
        )
        self.state.draft = self.services.write(self.state.request, self.state.research)
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "draft", self.state.draft),
        )
        return self.state.draft

    @listen(write_post)
    def design_visual(self, _: LinkedInDraft) -> VisualBrief:
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "status", WorkflowStatus.DESIGNING),
        )
        self.state.visual = self.services.design(self.state.request, self.state.research)
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "visual", self.state.visual),
        )
        return self.state.visual

    @listen(design_visual)
    def render_visual(self, _: VisualBrief) -> str:
        self.repository.update(
            self.state.workflow_id,
            lambda state: setattr(state, "status", WorkflowStatus.RENDERING),
        )
        image_path = self.images_dir / f"{self.state.workflow_id}-r1.png"
        self.renderer.render(self.state.visual, image_path)
        self.state.image_path = str(image_path.resolve())

        def finish(state: WorkflowState) -> None:
            state.research = self.state.research
            state.draft = self.state.draft
            state.visual = self.state.visual
            state.image_path = self.state.image_path
            state.status = WorkflowStatus.AWAITING_REVIEW
            state.error = None

        self.repository.update(self.state.workflow_id, finish)
        return self.state.image_path


class RevisionFlowState(BaseModel):
    id: str
    workflow_id: str
    request: TopicRequest
    research: ResearchBrief
    draft: LinkedInDraft
    visual: VisualBrief
    action: ReviewAction
    feedback: str
    revision: int
    image_path: str | None = None


@persist()
class RevisionFlow(Flow[RevisionFlowState]):
    services: Any = Field(exclude=True)
    repository: Any = Field(exclude=True)
    renderer: Any = Field(exclude=True)
    images_dir: Path = Field(exclude=True)

    @start()
    def revise(self) -> str:
        if self.state.action in {
            ReviewAction.REVISE_CONTENT,
            ReviewAction.REVISE_BOTH,
        }:
            self.state.draft = self.services.write(
                self.state.request,
                self.state.research,
                previous=self.state.draft,
                feedback=self.state.feedback,
            )
        if self.state.action in {
            ReviewAction.REVISE_IMAGE,
            ReviewAction.REVISE_BOTH,
        }:
            self.state.visual = self.services.design(
                self.state.request,
                self.state.research,
                previous=self.state.visual,
                feedback=self.state.feedback,
            )
            image_path = self.images_dir / (
                f"{self.state.workflow_id}-r{self.state.revision}.png"
            )
            self.renderer.render(self.state.visual, image_path)
            self.state.image_path = str(image_path.resolve())

        def finish(state: WorkflowState) -> None:
            state.draft = self.state.draft
            state.visual = self.state.visual
            if self.state.image_path:
                state.image_path = self.state.image_path
            state.status = WorkflowStatus.AWAITING_REVIEW
            state.error = None

        self.repository.update(self.state.workflow_id, finish)
        return "awaiting_review"


class PipelineEngine:
    def __init__(
        self,
        settings: Settings,
        repository: WorkflowRepository,
        services: AgentService,
        renderer: InfographicRenderer,
    ):
        self.settings = settings
        self.repository = repository
        self.services = services
        self.renderer = renderer
        self.persistence = SQLiteFlowPersistence(str(settings.flow_db_path))

    def create(self, request: TopicRequest) -> WorkflowState:
        state = WorkflowState(request=request)
        return self.repository.save(state)

    def prepare(self, workflow_id: str) -> None:
        try:
            if not self.settings.groq_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is missing. Add it to .env and restart the app."
                )
            state = self.repository.get(workflow_id)
            clarification = self.services.clarify(state.request)

            def apply_clarification(current: WorkflowState) -> None:
                current.clarification = clarification
                if clarification.needs_clarification:
                    current.status = WorkflowStatus.AWAITING_CLARIFICATION
                else:
                    current.request.confirmed_interpretation = (
                        clarification.recommended_interpretation
                        or current.request.original_idea
                    )

            state = self.repository.update(workflow_id, apply_clarification)
            if not clarification.needs_clarification:
                self.generate(workflow_id)
        except Exception as exc:
            self._fail(workflow_id, exc)

    def confirm_interpretation(self, workflow_id: str, interpretation: str) -> None:
        if not interpretation.strip():
            raise ValueError("A confirmed interpretation is required")

        def confirm(state: WorkflowState) -> None:
            if state.status != WorkflowStatus.AWAITING_CLARIFICATION:
                raise ValueError("This workflow is not waiting for clarification")
            state.request.confirmed_interpretation = interpretation.strip()
            state.status = WorkflowStatus.RESEARCHING
            state.error = None

        self.repository.update(workflow_id, confirm)

    def generate(self, workflow_id: str) -> None:
        try:
            if not self.settings.generation_ready:
                raise RuntimeError(
                    "GROQ_API_KEY and SERPER_API_KEY are required for generation."
                )
            state = self.repository.get(workflow_id)
            if not state.request.confirmed_interpretation:
                raise ValueError("Confirm the topic interpretation before research")
            flow_state = PipelineFlowState(
                id=state.id,
                workflow_id=state.id,
                request=state.request,
            )
            flow = InitialGenerationFlow(
                initial_state=flow_state,
                services=self.services,
                repository=self.repository,
                renderer=self.renderer,
                images_dir=self.settings.images_dir,
                persistence=self.persistence,
            )
            flow.kickoff()
        except Exception as exc:
            self._fail(workflow_id, exc)

    def request_revision(self, workflow_id: str, decision: ReviewDecision) -> None:
        status_for_action = {
            ReviewAction.REVISE_CONTENT: WorkflowStatus.REVISING_CONTENT,
            ReviewAction.REVISE_IMAGE: WorkflowStatus.REVISING_IMAGE,
            ReviewAction.REVISE_BOTH: WorkflowStatus.REVISING_BOTH,
        }
        if decision.action not in status_for_action:
            raise ValueError("Use approve() or cancel() for this review action")

        def begin(state: WorkflowState) -> None:
            if state.status != WorkflowStatus.AWAITING_REVIEW:
                raise ValueError("Only a reviewable workflow can be revised")
            if decision.revision != state.revision:
                raise ValueError("The review is stale; refresh before submitting feedback")
            state.feedback_history.append(decision)
            state.revision += 1
            state.approved_revision = None
            state.status = status_for_action[decision.action]
            state.error = None

        self.repository.update(workflow_id, begin)

    def revise(self, workflow_id: str, decision: ReviewDecision) -> None:
        try:
            state = self.repository.get(workflow_id)
            if not all((state.research, state.draft, state.visual)):
                raise ValueError("The workflow has no complete draft to revise")
            flow_state = RevisionFlowState(
                id=state.id,
                workflow_id=state.id,
                request=state.request,
                research=state.research,
                draft=state.draft,
                visual=state.visual,
                image_path=state.image_path,
                action=decision.action,
                feedback=decision.feedback,
                revision=state.revision,
            )
            RevisionFlow(
                initial_state=flow_state,
                services=self.services,
                repository=self.repository,
                renderer=self.renderer,
                images_dir=self.settings.images_dir,
                persistence=self.persistence,
            ).kickoff()
        except Exception as exc:
            self._fail(workflow_id, exc)

    def approve(self, workflow_id: str, revision: int) -> WorkflowState:
        def apply(state: WorkflowState) -> None:
            if state.status != WorkflowStatus.AWAITING_REVIEW:
                raise ValueError("Only a reviewable workflow can be approved")
            if revision != state.revision:
                raise ValueError("The approval is stale; refresh before approving")
            state.feedback_history.append(
                ReviewDecision(action=ReviewAction.APPROVE, revision=revision)
            )
            state.approved_revision = revision
            state.status = WorkflowStatus.APPROVED

        return self.repository.update(workflow_id, apply)

    def cancel(self, workflow_id: str, revision: int) -> WorkflowState:
        def apply(state: WorkflowState) -> None:
            if state.status in {WorkflowStatus.PUBLISHING, WorkflowStatus.PUBLISHED}:
                raise ValueError("A publishing or published workflow cannot be cancelled")
            state.feedback_history.append(
                ReviewDecision(action=ReviewAction.CANCEL, revision=revision)
            )
            state.status = WorkflowStatus.CANCELLED

        return self.repository.update(workflow_id, apply)

    def _fail(self, workflow_id: str, exc: Exception) -> None:
        def fail(state: WorkflowState) -> None:
            state.status = WorkflowStatus.FAILED
            state.error = str(exc)[:2_000]

        self.repository.update(workflow_id, fail)
