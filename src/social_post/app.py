from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agents import CrewAIAgentService
from .config import Settings, get_settings
from .linkedin import LinkedInClient, LinkedInError, PublishingService
from .models import ReviewAction, ReviewDecision, TopicRequest
from .renderer import InfographicRenderer
from .storage import WorkflowNotFoundError, WorkflowRepository
from .workflow import PipelineEngine


PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_directories()
    repository = WorkflowRepository(settings.workflows_dir)
    agents = CrewAIAgentService(settings)
    renderer = InfographicRenderer(settings.footer)
    engine = PipelineEngine(settings, repository, agents, renderer)
    linkedin = LinkedInClient(settings)
    publisher = PublishingService(repository, linkedin)

    app = FastAPI(title="LinkedIn Content Studio", version="0.1.0")
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    app.state.settings = settings
    app.state.repository = repository
    app.state.engine = engine
    app.state.linkedin = linkedin
    app.state.publisher = publisher

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "workflows": repository.list_recent(),
                "generation_ready": settings.generation_ready,
                "linkedin": linkedin.connection(),
            },
        )

    @app.post("/workflows")
    def create_workflow(
        background_tasks: BackgroundTasks,
        idea: str = Form(...),
        audience: str = Form("Technology professionals and business leaders"),
        context: str = Form(""),
    ):
        try:
            state = engine.create(
                TopicRequest(original_idea=idea, audience=audience, context=context)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        background_tasks.add_task(engine.prepare, state.id)
        return RedirectResponse(f"/workflows/{state.id}", status_code=303)

    @app.get("/workflows/{workflow_id}", response_class=HTMLResponse)
    def workflow_page(request: Request, workflow_id: str):
        state = _get(repository, workflow_id)
        return templates.TemplateResponse(
            request,
            "workflow.html",
            {
                "state": state,
                "linkedin": linkedin.connection(),
                "generation_ready": settings.generation_ready,
            },
        )

    @app.get("/api/workflows/{workflow_id}")
    def workflow_status(workflow_id: str):
        state = _get(repository, workflow_id)
        payload = state.model_dump(mode="json")
        payload["is_busy"] = state.is_busy
        payload["can_publish"] = state.can_publish
        if state.image_path:
            payload["image_url"] = f"/workflows/{state.id}/image?v={state.revision}"
        return payload

    @app.get("/workflows/{workflow_id}/image")
    def workflow_image(workflow_id: str):
        state = _get(repository, workflow_id)
        if not state.image_path or not Path(state.image_path).is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(state.image_path, media_type="image/png")

    @app.post("/workflows/{workflow_id}/clarify")
    def clarify(
        workflow_id: str,
        background_tasks: BackgroundTasks,
        interpretation: str = Form(""),
        custom_interpretation: str = Form(""),
    ):
        try:
            selected = custom_interpretation.strip() or interpretation.strip()
            engine.confirm_interpretation(workflow_id, selected)
            background_tasks.add_task(engine.generate, workflow_id)
        except (ValueError, WorkflowNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)

    @app.post("/workflows/{workflow_id}/review")
    def review(
        workflow_id: str,
        background_tasks: BackgroundTasks,
        action: ReviewAction = Form(...),
        revision: int = Form(...),
        feedback: str = Form(""),
    ):
        try:
            decision = ReviewDecision(
                action=action,
                revision=revision,
                feedback=feedback,
            )
            if action == ReviewAction.APPROVE:
                engine.approve(workflow_id, revision)
            elif action == ReviewAction.CANCEL:
                engine.cancel(workflow_id, revision)
            else:
                engine.request_revision(workflow_id, decision)
                background_tasks.add_task(engine.revise, workflow_id, decision)
        except (ValueError, WorkflowNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)

    @app.get("/auth/linkedin/start")
    def linkedin_start(return_to: str = "/"):
        try:
            url = linkedin.authorization_url(return_to=return_to)
        except LinkedInError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url, status_code=302)

    @app.get("/auth/linkedin/callback")
    def linkedin_callback(code: str, state: str):
        try:
            return_to = linkedin.exchange_code(code, state)
        except LinkedInError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(return_to, status_code=303)

    @app.post("/workflows/{workflow_id}/publish")
    def publish(workflow_id: str):
        try:
            publisher.publish(workflow_id)
        except (LinkedInError, WorkflowNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)

    return app


def _get(repository: WorkflowRepository, workflow_id: str):
    try:
        return repository.get(workflow_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workflow not found") from exc


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("social_post.app:app", host=settings.host, port=settings.port, reload=False)
