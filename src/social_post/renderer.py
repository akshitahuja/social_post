from __future__ import annotations

from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont

from .models import VisualBrief


HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _normalize_display_text(value: str) -> str:
    """Keep generated copy compatible with the bundled cross-platform fonts."""
    replacements = str.maketrans(
        {
            "\u00a0": " ",
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
        }
    )
    return value.translate(replacements)


class InfographicRenderError(ValueError):
    pass


def _color(value: str, fallback: str) -> str:
    return value if HEX_COLOR.fullmatch(value) else fallback


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    converted = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * converted[0] + 0.7152 * converted[1] + 0.0722 * converted[2]


def _contrast(left: str, right: str) -> float:
    high, low = sorted((_luminance(left), _luminance(right)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default(size=size)


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font,
    fill: str,
    width_chars: int,
    spacing: int = 8,
) -> int:
    lines = textwrap.wrap(text, width=width_chars, break_long_words=False) or [text]
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        draw.text((center_x - (box[2] - box[0]) / 2, y), line, font=font, fill=fill)
        y += box[3] - box[1] + spacing
    return y


def _fit_single_line_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int = 31,
    min_size: int = 16,
):
    for size in range(start_size, min_size - 1, -1):
        font = _font(size, bold=True)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= max_width:
            return font
    return _font(min_size, bold=True)


class InfographicRenderer:
    width = 1080
    height = 1080

    def __init__(self, footer: str = ""):
        self.footer = footer

    def render(self, brief: VisualBrief, output_path: Path) -> Path:
        headline = _normalize_display_text(brief.headline)
        subheadline = _normalize_display_text(brief.subheadline)
        left_label = _normalize_display_text(brief.left_label)
        right_label = _normalize_display_text(brief.right_label)
        left_points = [_normalize_display_text(point) for point in brief.left_points]
        right_points = [_normalize_display_text(point) for point in brief.right_points]
        primary = _color(brief.primary_color, "#0A66C2")
        accent = _color(brief.accent_color, "#7C3AED")
        background = _color(brief.background_color, "#F7F9FC")
        if _contrast(background, "#172033") < 4.5:
            raise InfographicRenderError("Background does not provide readable contrast")

        image = Image.new("RGB", (self.width, self.height), background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((55, 42, 1025, 1038), radius=34, fill="#FFFFFF")
        draw.rounded_rectangle((55, 42, 1025, 62), radius=10, fill=primary)

        title_font = _font(56, bold=True)
        subtitle_font = _font(25)
        point_font = _font(25)
        footer_font = _font(19)

        y = _draw_centered_lines(
            draw, headline, 540, 92, title_font, "#172033", 28, spacing=10
        )
        if y > 250:
            raise InfographicRenderError("Headline is too long for the template")
        if subheadline:
            y = _draw_centered_lines(
                draw, subheadline, 540, y + 4, subtitle_font, "#526079", 52
            )
        content_top = max(y + 30, 260)
        content_bottom = 955

        columns = [
            (82, 514, primary, left_label, left_points),
            (566, 998, accent, right_label, right_points),
        ]
        for left, right, color, label, points in columns:
            draw.rounded_rectangle(
                (left, content_top, right, content_bottom),
                radius=26,
                fill="#F4F7FB",
                outline=color,
                width=4,
            )
            draw.rounded_rectangle(
                (left + 20, content_top + 20, right - 20, content_top + 86),
                radius=18,
                fill=color,
            )
            label_font = _fit_single_line_font(
                draw,
                label,
                max_width=(right - left) - 64,
            )
            label_box = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_box[2] - label_box[0]
            label_height = label_box[3] - label_box[1]
            draw.text(
                (
                    (left + right - label_width) / 2,
                    content_top + 53 - label_height / 2 - label_box[1],
                ),
                label,
                font=label_font,
                fill="#FFFFFF",
            )

            point_y = content_top + 126
            for point in points:
                wrapped = textwrap.wrap(point, width=27, break_long_words=False)
                needed = len(wrapped) * 34 + 30
                if point_y + needed > content_bottom - 18:
                    raise InfographicRenderError(
                        "Visual copy overflows the infographic; shorten the points"
                    )
                draw.ellipse((left + 32, point_y + 8, left + 46, point_y + 22), fill=color)
                line_y = point_y
                for line in wrapped:
                    draw.text((left + 62, line_y), line, font=point_font, fill="#263247")
                    line_y += 34
                point_y += needed

        if self.footer:
            footer = _normalize_display_text(self.footer[:80])
            draw.text((82, 990), footer, font=footer_font, fill="#64748B")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
        with Image.open(output_path) as check:
            if check.size != (1080, 1080) or check.format != "PNG":
                raise InfographicRenderError("Rendered image failed output validation")
        return output_path
