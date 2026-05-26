"""Convert a Markdown file into a PDF using ReportLab.

Usage:
    python scripts/build_pdf.py docs/PIPELINE_GUIDE.md docs/PIPELINE_GUIDE.pdf

Supports the markdown subset used in our docs:
  - ATX headings (# .. ######)
  - Paragraphs
  - Bullet lists (- or *)
  - Fenced code blocks (```)
  - Pipe tables (| a | b |\n| --- | --- |\n| 1 | 2 |)
  - Inline: **bold**, *italic*, `code`, [text](url)
  - Horizontal rule (---)
"""
from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontSize=22,
                                leading=26, spaceAfter=18, textColor=colors.HexColor("#1f2937")),
        "h1":    ParagraphStyle("H1", parent=body, fontSize=18, leading=22,
                                spaceBefore=18, spaceAfter=10, textColor=colors.HexColor("#1f2937"),
                                fontName="Helvetica-Bold"),
        "h2":    ParagraphStyle("H2", parent=body, fontSize=14, leading=18,
                                spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1f2937"),
                                fontName="Helvetica-Bold"),
        "h3":    ParagraphStyle("H3", parent=body, fontSize=12, leading=15,
                                spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#374151"),
                                fontName="Helvetica-Bold"),
        "h4":    ParagraphStyle("H4", parent=body, fontSize=11, leading=14,
                                spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#374151"),
                                fontName="Helvetica-Bold"),
        "body":  body,
        "code":  ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=8.5,
                                leading=11, backColor=colors.HexColor("#f3f4f6"),
                                borderColor=colors.HexColor("#e5e7eb"), borderWidth=0.5,
                                borderPadding=6, leftIndent=0, rightIndent=0,
                                spaceBefore=6, spaceAfter=8),
        "table_header": ParagraphStyle("TH", parent=body, fontName="Helvetica-Bold",
                                       fontSize=9, leading=12, textColor=colors.white),
        "table_cell":   ParagraphStyle("TD", parent=body, fontSize=9, leading=12,
                                       spaceAfter=0),
    }


# ---------------------------------------------------------------------------
# Inline markdown → ReportLab markup (a small HTML subset Paragraph understands)
# ---------------------------------------------------------------------------
def render_inline(text: str) -> str:
    # Escape HTML special chars first so user `<` `>` `&` survive.
    out = escape(text, quote=False)
    # `code` → <font face="Courier" ...>code</font>
    out = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font face="Courier" size="9" backColor="#f3f4f6">{m.group(1)}</font>',
        out,
    )
    # **bold**
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    # *italic* / _italic_  (skip if adjacent to non-space, to avoid eating snake_case)
    out = re.sub(r"(?<![A-Za-z0-9_])\*([^*\n]+)\*(?![A-Za-z0-9_])", r"<i>\1</i>", out)
    out = re.sub(r"(?<![A-Za-z0-9_])_([^_\n]+)_(?![A-Za-z0-9_])", r"<i>\1</i>", out)
    # [text](url) → <link href="url" color="blue">text</link>
    out = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#2563eb"><u>\1</u></link>',
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Block-level parser
# ---------------------------------------------------------------------------
def parse_blocks(md: str) -> list[tuple]:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[tuple] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # blank line
        if line.strip() == "":
            i += 1
            continue

        # fenced code block
        if line.startswith("```"):
            i += 1
            code: list[str] = []
            while i < n and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # consume the closing fence (if any)
            blocks.append(("code", "\n".join(code)))
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            blocks.append((f"h{level}", m.group(2).strip()))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}\s*$", line):
            blocks.append(("hr", ""))
            i += 1
            continue

        # pipe table
        if line.lstrip().startswith("|") and i + 1 < n:
            sep = lines[i + 1].strip()
            if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", sep):
                rows: list[list[str]] = []
                while i < n and lines[i].lstrip().startswith("|"):
                    raw = lines[i].strip().strip("|")
                    rows.append([c.strip() for c in raw.split("|")])
                    i += 1
                # rows[0] = header, rows[1] = separator, rest = data
                header = rows[0]
                data = rows[2:]
                blocks.append(("table", (header, data)))
                continue

        # unordered list
        if re.match(r"^\s*[-*+]\s+", line):
            items: list[str] = []
            while i < n and re.match(r"^\s*[-*+]\s+", lines[i]):
                item_text = re.sub(r"^\s*[-*+]\s+", "", lines[i])
                # continuation lines indented
                i += 1
                while i < n and lines[i].startswith(("  ", "\t")) and lines[i].strip():
                    item_text += " " + lines[i].strip()
                    i += 1
                items.append(item_text)
            blocks.append(("ul", items))
            continue

        # paragraph: collect lines until blank, special start, or EOF
        para_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            ln = lines[i]
            if (
                ln.startswith("#")
                or ln.startswith("```")
                or re.match(r"^\s*[-*+]\s+", ln)
                or (ln.lstrip().startswith("|") and i + 1 < n
                    and re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", lines[i + 1].strip()))
                or re.match(r"^-{3,}\s*$", ln)
            ):
                break
            para_lines.append(ln)
            i += 1
        if para_lines:
            blocks.append(("p", " ".join(s.strip() for s in para_lines)))

    return blocks


# ---------------------------------------------------------------------------
# Block → ReportLab flowable
# ---------------------------------------------------------------------------
def render_blocks(blocks: list[tuple], styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    for kind, content in blocks:
        if kind == "h1":
            story.append(Paragraph(render_inline(content), styles["h1"]))
        elif kind == "h2":
            story.append(Paragraph(render_inline(content), styles["h2"]))
        elif kind == "h3":
            story.append(Paragraph(render_inline(content), styles["h3"]))
        elif kind in ("h4", "h5", "h6"):
            story.append(Paragraph(render_inline(content), styles["h4"]))
        elif kind == "p":
            story.append(Paragraph(render_inline(content), styles["body"]))
        elif kind == "ul":
            items = [
                ListItem(Paragraph(render_inline(t), styles["body"]),
                         leftIndent=12, value="bullet")
                for t in content
            ]
            story.append(ListFlowable(items, bulletType="bullet",
                                      leftIndent=14, bulletFontName="Helvetica",
                                      bulletFontSize=9, spaceBefore=4, spaceAfter=8))
        elif kind == "code":
            # Preformatted preserves whitespace and uses a monospace style.
            story.append(Preformatted(content, styles["code"]))
        elif kind == "hr":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#d1d5db"), spaceBefore=4, spaceAfter=8))
        elif kind == "table":
            header, rows = content
            data = [[Paragraph(render_inline(c), styles["table_header"]) for c in header]]
            for r in rows:
                # pad short rows so the table is rectangular
                if len(r) < len(header):
                    r = r + [""] * (len(header) - len(r))
                data.append([Paragraph(render_inline(c), styles["table_cell"]) for c in r])
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f9fafb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t)
            story.append(Spacer(1, 8))
    return story


# ---------------------------------------------------------------------------
# Page header/footer
# ---------------------------------------------------------------------------
def make_canvas_callback(title: str):
    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, title)
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.2 * cm, f"PADS v0.3.0")
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()
    return _on_page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_pdf(md_path: Path, pdf_path: Path) -> None:
    md = md_path.read_text(encoding="utf-8")
    blocks = parse_blocks(md)
    styles = make_styles()

    # Extract first H1 as document title for the running header.
    doc_title = "PADS Pipeline Guide"
    for kind, content in blocks:
        if kind == "h1":
            doc_title = content
            break

    story = render_blocks(blocks, styles)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=doc_title, author="Julen Berrueta",
    )
    on_page = make_canvas_callback(doc_title)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_pdf.py <input.md> <output.pdf>")
        sys.exit(2)
    build_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote {sys.argv[2]}")
