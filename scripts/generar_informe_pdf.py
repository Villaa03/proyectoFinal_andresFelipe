"""Genera el informe PDF desde docs/informe_final.md con tablas e imágenes."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "informe_final.md"
OUTPUT = ROOT / "docs" / "informe_final.pdf"
PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
MARGIN_X = 15 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X


def inline_markup(text: str) -> str:
    """Escapa texto Markdown y convierte énfasis, código y enlaces simples."""
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`|\[[^\]]+\]\([^)]+\))")
    parts = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            parts.append(f"<b>{html.escape(token[2:-2])}</b>")
        elif token.startswith("`"):
            parts.append(f"<font name='Courier'>{html.escape(token[1:-1])}</font>")
        else:
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            label, href = link.groups()
            parts.append(
                f"<link href='{html.escape(href, quote=True)}' color='#28627f'>"
                f"{html.escape(label)}</link>"
            )
        cursor = match.end()
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def parse_table(lines: list[str], styles) -> LongTable:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r"[-: ]+", cell or "-") for cell in cells):
            continue
        style = styles["TableHeader"] if not rows else styles["TableCell"]
        rows.append([Paragraph(inline_markup(cell), style) for cell in cells])
    columns = max(len(row) for row in rows)
    for row in rows:
        row.extend([Paragraph("", styles["TableCell"]) for _ in range(columns - len(row))])
    widths = [CONTENT_WIDTH / columns] * columns
    if columns == 7:
        weights = [1.8, 0.55, 1.0, 1.08, 0.68, 0.73, 0.55]
        widths = [CONTENT_WIDTH * weight / sum(weights) for weight in weights]
    elif columns == 4:
        weights = [2.3, 0.8, 0.8, 0.8]
        widths = [CONTENT_WIDTH * weight / sum(weights) for weight in weights]
    table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9.2),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7f9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f5")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cad4dc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_story(markdown: str):
    base = getSampleStyleSheet()
    styles = {
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=13, spaceAfter=6, textColor=colors.HexColor("#24313a"),
        ),
        "Title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
            leading=26, textColor=colors.HexColor("#173c55"), spaceAfter=10,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=14,
            leading=17, textColor=colors.HexColor("#24445b"), spaceBefore=11, spaceAfter=6,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.8,
            leading=13, textColor=colors.HexColor("#315d75"), spaceBefore=8, spaceAfter=4,
            keepWithNext=True,
        ),
        "TableCell": ParagraphStyle(
            "TableCell", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5,
            leading=9.2, spaceAfter=0,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.5,
            leading=9.2, textColor=colors.white, spaceAfter=0,
        ),
        "Quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], leftIndent=12, borderColor=colors.HexColor("#d77a36"),
            borderWidth=2, borderPadding=6, fontSize=9, leading=12, spaceAfter=8,
        ),
    }
    story = []
    lines = markdown.splitlines()
    index = 0
    pending_list: list[str] = []
    ordered = False

    def flush_list():
        nonlocal pending_list, ordered
        if not pending_list:
            return
        for item_index, item in enumerate(pending_list, 1):
            prefix = f"{item_index}. " if ordered else "- "
            story.append(Paragraph(prefix + inline_markup(item), styles["Body"]))
        pending_list = []
        ordered = False

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush_list()
            index += 1
            continue
        if re.match(r"^\|.*\|$", line):
            flush_list()
            table_lines = []
            while index < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[index]):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(parse_table(table_lines, styles))
            story.append(Spacer(1, 8))
            continue
        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            flush_list()
            alt, relative = image_match.groups()
            image_path = (SOURCE.parent / relative).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"No se encontró la imagen del informe: {image_path}")
            image = Image(str(image_path))
            image._restrictSize(CONTENT_WIDTH * 0.78, 260)
            image.hAlign = "CENTER"
            story.append(Paragraph(html.escape(alt), styles["H3"]))
            story.append(image)
            story.append(Spacer(1, 8))
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_list()
            level, content = len(heading.group(1)), heading.group(2)
            if content == "4. Resultados en el test":
                story.append(PageBreak())
            style = "Title" if level == 1 else ("H2" if level == 2 else "H3")
            story.append(Paragraph(inline_markup(content), styles[style]))
            index += 1
            continue
        if line.startswith("---"):
            flush_list()
            story.append(Spacer(1, 5))
            index += 1
            continue
        if line.startswith("> "):
            flush_list()
            story.append(Paragraph(inline_markup(line[2:]), styles["Quote"]))
            index += 1
            continue
        bullet = re.match(r"^(?:-\s+|\*\s+)(.*)$", line)
        number = re.match(r"^\d+\.\s+(.*)$", line)
        if bullet or number:
            item = (bullet or number).group(1)
            is_ordered = bool(number)
            if pending_list and ordered != is_ordered:
                flush_list()
            ordered = is_ordered
            pending_list.append(item)
            index += 1
            continue
        flush_list()
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^(#{1,3}\s|\||!\[|> |(?:-\s+|\*\s+)|\d+\.\s+)", lines[index].strip()
        ):
            paragraph.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph)), styles["Body"]))
    flush_list()
    return story


def add_page_chrome(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c6d1d8"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, PAGE_HEIGHT - 11 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 11 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#536773"))
    canvas.drawString(MARGIN_X, PAGE_HEIGHT - 8 * mm, "Proyecto Adult Income ML | Informe académico")
    canvas.line(MARGIN_X, 10 * mm, PAGE_WIDTH - MARGIN_X, 10 * mm)
    canvas.drawString(MARGIN_X, 6 * mm, "Fuente: UCI Adult (1994) | Uso educativo")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 6 * mm, f"Página {document.page}")
    canvas.restoreState()


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    story = build_story(markdown)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=PAGE_SIZE,
        rightMargin=MARGIN_X,
        leftMargin=MARGIN_X,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
        title="Informe final - Proyecto Adult Income ML",
        author="Andrés Felipe Villanueva Parra",
        subject="Evaluación académica de clasificación y análisis ético",
    )
    document.build(story, onFirstPage=add_page_chrome, onLaterPages=add_page_chrome)
    print(f"PDF generado: {OUTPUT}")


if __name__ == "__main__":
    main()
