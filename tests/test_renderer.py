from __future__ import annotations

from PIL import Image
import pytest

from social_post.models import VisualBrief
from social_post.renderer import InfographicRenderError, InfographicRenderer


def make_brief(**updates) -> VisualBrief:
    data = {
        "headline": "RAG vs Knowledge Graphs",
        "subheadline": "Choose the right context architecture",
        "left_label": "RAG (Retrieval-Augmented)",
        "right_label": "OKF (Open Knowledge Format)",
        "left_points": ["Retrieves passages", "Uses documents", "Query-time context"],
        "right_points": ["Connects entities", "Models relations", "Structured context"],
        "alt_text": "A comparison between RAG and knowledge graphs for AI context.",
    }
    data.update(updates)
    return VisualBrief(**data)


def test_renderer_creates_valid_square_png(tmp_path):
    path = InfographicRenderer("Example author").render(
        make_brief(), tmp_path / "card.png"
    )
    with Image.open(path) as image:
        assert image.size == (1080, 1080)
        assert image.format == "PNG"


def test_renderer_rejects_unreadable_background(tmp_path):
    brief = make_brief(background_color="#172033")
    with pytest.raises(InfographicRenderError, match="contrast"):
        InfographicRenderer().render(brief, tmp_path / "bad.png")
