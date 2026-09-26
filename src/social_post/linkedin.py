from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx

from .config import Settings
from .models import PublishResult, WorkflowState, WorkflowStatus
from .storage import WorkflowRepository


class LinkedInError(RuntimeError):
    pass


class LinkedInTokenStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, self.path)
        os.chmod(self.path, 0o600)

    def connected(self) -> bool:
        data = self.load()
        return bool(data.get("access_token") and data.get("person_id"))


class LinkedInClient:
    authorize_url = "https://www.linkedin.com/oauth/v2/authorization"
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    userinfo_url = "https://api.linkedin.com/v2/userinfo"
    api_root = "https://api.linkedin.com/rest"

    def __init__(
        self,
        settings: Settings,
        token_store: LinkedInTokenStore | None = None,
        http: httpx.Client | None = None,
    ):
        self.settings = settings
        self.token_store = token_store or LinkedInTokenStore(
            settings.linkedin_token_path
        )
        self.http = http or httpx.Client(timeout=30.0, follow_redirects=True)

    def authorization_url(self, return_to: str = "/") -> str:
        if not self.settings.linkedin_ready:
            raise LinkedInError(
                "Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET before connecting LinkedIn."
            )
        state = secrets.token_urlsafe(32)
        existing = self.token_store.load()
        existing.update({"oauth_state": state, "return_to": return_to})
        self.token_store.save(existing)
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.settings.linkedin_client_id,
                "redirect_uri": self.settings.linkedin_redirect_uri,
                "state": state,
                "scope": "openid profile w_member_social",
            }
        )
        return f"{self.authorize_url}?{query}"

    def exchange_code(self, code: str, state: str) -> str:
        saved = self.token_store.load()
        if not state or not secrets.compare_digest(state, saved.get("oauth_state", "")):
            raise LinkedInError("LinkedIn OAuth state did not match; start the connection again")
        response = self.http.post(
            self.token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.settings.linkedin_client_id,
                "client_secret": self.settings.linkedin_client_secret,
                "redirect_uri": self.settings.linkedin_redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self._raise(response, "LinkedIn token exchange failed")
        token_data = response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise LinkedInError("LinkedIn did not return an access token")

        profile_response = self.http.get(
            self.userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        self._raise(profile_response, "LinkedIn profile lookup failed")
        profile = profile_response.json()
        person_id = profile.get("sub")
        if not person_id:
            raise LinkedInError("LinkedIn profile response did not include a member ID")
        self.token_store.save(
            {
                "access_token": access_token,
                "expires_in": token_data.get("expires_in"),
                "person_id": person_id,
                "name": profile.get("name", "LinkedIn member"),
                "return_to": saved.get("return_to", "/"),
            }
        )
        return saved.get("return_to", "/")

    def connection(self) -> dict[str, Any]:
        data = self.token_store.load()
        return {
            "connected": self.token_store.connected(),
            "name": data.get("name", ""),
            "configured": self.settings.linkedin_ready,
        }

    def publish_image_post(
        self,
        *,
        text: str,
        image_path: Path,
        alt_text: str,
        revision: int,
    ) -> PublishResult:
        token = self.token_store.load()
        if not self.token_store.connected():
            raise LinkedInError("Connect a LinkedIn account before publishing")
        if not image_path.is_file():
            raise LinkedInError("The approved infographic file is missing")
        access_token = token["access_token"]
        author = f"urn:li:person:{token['person_id']}"
        headers = self._api_headers(access_token)

        initialize = self.http.post(
            f"{self.api_root}/images?action=initializeUpload",
            headers=headers,
            json={"initializeUploadRequest": {"owner": author}},
        )
        self._raise(initialize, "LinkedIn image initialization failed")
        value = initialize.json().get("value", {})
        upload_url = value.get("uploadUrl")
        image_urn = value.get("image")
        if not upload_url or not image_urn:
            raise LinkedInError("LinkedIn returned an incomplete image upload response")

        with image_path.open("rb") as image_file:
            upload = self.http.put(
                upload_url,
                content=image_file.read(),
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "image/png",
                },
            )
        self._raise(upload, "LinkedIn image upload failed")

        create = self.http.post(
            f"{self.api_root}/posts",
            headers=headers,
            json={
                "author": author,
                "commentary": text,
                "visibility": "PUBLIC",
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": [],
                },
                "content": {
                    "media": {
                        "id": image_urn,
                        "altText": alt_text,
                    }
                },
                "lifecycleState": "PUBLISHED",
                "isReshareDisabledByAuthor": False,
            },
        )
        self._raise(create, "LinkedIn post creation failed")
        post_urn = create.headers.get("x-restli-id")
        if not post_urn:
            raise LinkedInError("LinkedIn created the post but did not return its URN")
        return PublishResult(post_urn=post_urn, published_revision=revision)

    def _api_headers(self, access_token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Linkedin-Version": self.settings.linkedin_api_version,
            "X-Restli-Protocol-Version": "2.0.0",
        }

    @staticmethod
    def _raise(response: httpx.Response, message: str) -> None:
        if response.is_success:
            return
        detail = response.text[:500]
        raise LinkedInError(f"{message} ({response.status_code}): {detail}")


class PublishingService:
    def __init__(
        self,
        repository: WorkflowRepository,
        client: LinkedInClient,
    ):
        self.repository = repository
        self.client = client

    def publish(self, workflow_id: str) -> PublishResult:
        state = self.repository.get(workflow_id)
        if state.publish_result is not None:
            return state.publish_result
        if not state.can_publish:
            raise LinkedInError("Only the currently approved revision can be published")
        if not all((state.draft, state.visual, state.image_path)):
            raise LinkedInError("The approved post package is incomplete")

        self.repository.update(
            workflow_id,
            lambda item: setattr(item, "status", WorkflowStatus.PUBLISHING),
        )
        try:
            result = self.client.publish_image_post(
                text=state.draft.full_text,
                image_path=Path(state.image_path),
                alt_text=state.visual.alt_text,
                revision=state.revision,
            )

            def finish(item: WorkflowState) -> None:
                item.publish_result = result
                item.status = WorkflowStatus.PUBLISHED
                item.error = None

            self.repository.update(workflow_id, finish)
            return result
        except Exception as exc:
            def restore(item: WorkflowState) -> None:
                item.status = WorkflowStatus.APPROVED
                item.error = str(exc)[:2_000]

            self.repository.update(workflow_id, restore)
            raise
