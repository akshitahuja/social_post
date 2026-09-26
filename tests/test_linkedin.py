from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from social_post.linkedin import (
    LinkedInClient,
    LinkedInError,
    LinkedInTokenStore,
    PublishingService,
)
from social_post.models import (
    LinkedInDraft,
    TopicRequest,
    VisualBrief,
    WorkflowState,
    WorkflowStatus,
)
from social_post.storage import WorkflowRepository


def approved_state(tmp_path: Path) -> WorkflowState:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png-data")
    return WorkflowState(
        request=TopicRequest(original_idea="RAG versus knowledge graphs"),
        status=WorkflowStatus.APPROVED,
        draft=LinkedInDraft(
            body="A sufficiently detailed LinkedIn post body that explains a useful technical comparison clearly.",
            hashtags=["AI"],
        ),
        visual=VisualBrief(
            headline="RAG vs graphs",
            left_label="RAG",
            right_label="Graphs",
            left_points=["Retrieves text", "Query time"],
            right_points=["Connects facts", "Structured"],
            alt_text="A visual comparison of RAG and graph-based context systems.",
        ),
        image_path=str(image_path),
        approved_revision=1,
    )


def connected_store(settings) -> LinkedInTokenStore:
    store = LinkedInTokenStore(settings.linkedin_token_path)
    store.save(
        {
            "access_token": "token",
            "person_id": "person-123",
            "name": "Test User",
        }
    )
    return store


def test_partial_image_upload_failure_never_creates_post(settings, tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "initializeUpload" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "value": {
                        "uploadUrl": "https://upload.example/image",
                        "image": "urn:li:image:123",
                    }
                },
            )
        if request.url.host == "upload.example":
            return httpx.Response(500, text="upload failed")
        return httpx.Response(500, text="post must not be called")

    client = LinkedInClient(
        settings,
        connected_store(settings),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    state = approved_state(tmp_path)
    with pytest.raises(LinkedInError, match="image upload failed"):
        client.publish_image_post(
            text=state.draft.full_text,
            image_path=Path(state.image_path),
            alt_text=state.visual.alt_text,
            revision=1,
        )
    assert not any(url.endswith("/posts") for url in calls)


def test_publish_is_idempotent_after_success(settings, tmp_path):
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if "initializeUpload" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "value": {
                        "uploadUrl": "https://upload.example/image",
                        "image": "urn:li:image:123",
                    }
                },
            )
        if request.url.host == "upload.example":
            return httpx.Response(201)
        if request.url.path.endswith("/posts"):
            post_calls += 1
            return httpx.Response(201, headers={"x-restli-id": "urn:li:share:999"})
        return httpx.Response(404)

    repository = WorkflowRepository(settings.workflows_dir)
    state = repository.save(approved_state(tmp_path))
    client = LinkedInClient(
        settings,
        connected_store(settings),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    publishing = PublishingService(repository, client)

    first = publishing.publish(state.id)
    second = publishing.publish(state.id)

    assert first.post_urn == "urn:li:share:999"
    assert second == first
    assert post_calls == 1
    assert repository.get(state.id).status == WorkflowStatus.PUBLISHED
