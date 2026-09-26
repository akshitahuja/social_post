from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Callable

from .models import WorkflowState, utc_now


class WorkflowNotFoundError(KeyError):
    pass


class WorkflowRepository:
    """Small, atomic JSON repository suitable for a single-user local app."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, workflow_id: str) -> Path:
        safe = "".join(ch for ch in workflow_id if ch.isalnum() or ch in "-_")
        if not safe or safe != workflow_id:
            raise WorkflowNotFoundError(workflow_id)
        return self.root / f"{safe}.json"

    def save(self, state: WorkflowState) -> WorkflowState:
        state.updated_at = utc_now()
        path = self._path(state.id)
        temp = path.with_suffix(".tmp")
        payload = state.model_dump_json(indent=2)
        with self._lock:
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, path)
        return state

    def get(self, workflow_id: str) -> WorkflowState:
        path = self._path(workflow_id)
        with self._lock:
            if not path.exists():
                raise WorkflowNotFoundError(workflow_id)
            return WorkflowState.model_validate_json(path.read_text(encoding="utf-8"))

    def update(
        self,
        workflow_id: str,
        updater: Callable[[WorkflowState], None],
    ) -> WorkflowState:
        with self._lock:
            state = self.get(workflow_id)
            updater(state)
            return self.save(state)

    def list_recent(self, limit: int = 20) -> list[WorkflowState]:
        states: list[WorkflowState] = []
        with self._lock:
            for path in self.root.glob("*.json"):
                try:
                    states.append(
                        WorkflowState.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
                    )
                except (OSError, ValueError):
                    continue
        return sorted(states, key=lambda item: item.updated_at, reverse=True)[:limit]
