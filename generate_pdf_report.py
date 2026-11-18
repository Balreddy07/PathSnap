"""
generate_pdf_report.py
======================
Generates an academic, publication-quality PDF report for PathSnap using ReportLab.
Reads outputs/reports/PathSnap_Final_Project_Report.md and compiles it into:
outputs/reports/PathSnap_Final_Project_Report.pdf
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "PathSnap_Final_Project_Report.md")
REPORT_PDF_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "PathSnap_Final_Project_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with total page counts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(45, 11 * inch - 36, "PathSnap: Distributed Map-Matching & Route Reconstruction")
            self.drawRightString(8.5 * inch - 45, 11 * inch - 36, "Machine Learning Final Project Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(45, 11 * inch - 40, 8.5 * inch - 45, 11 * inch - 40)

        # Footer (All pages)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 45, 30, footer_text)
        self.drawString(45, 30, "PES University • Department of Computer Science & Engineering • Semester 5 ML")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(45, 40, 8.5 * inch - 45, 40)

        self.restoreState()


def build_pdf():
    os.makedirs(os.path.dirname(REPORT_PDF_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        REPORT_PDF_PATH,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=48,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#0f172a")     # Slate 900
    c_accent  = colors.HexColor("#0284c7")     # Sky 600
    c_sub     = colors.HexColor("#334155")     # Slate 700
    c_bg_call = colors.HexColor("#f8fafc")     # Slate 50
    c_border  = colors.HexColor("#e2e8f0")     # Slate 200

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_sub,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_accent,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_primary,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=c_sub,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=c_primary,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0369a1"),
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("PathSnap: Distributed Map-Matching & Route Reconstruction", title_style))
    story.append(Paragraph("Machine Learning & Hidden Markov Model Hybrid Architecture • Final Project Report", subtitle_style))

    meta_text = (
        "<b>Course:</b> UE24CS352A - Machine Learning | <b>Institution:</b> PES University, RR Campus<br/>"
        "<b>Repository:</b> https://github.com/Balreddy07/PathSnap | <b>Branch:</b> main | <b>Date:</b> November 18, 2026"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=8, spaceAfter=10))

    # Executive Summary Box
    exec_summary_text = (
        "<b>Executive Summary:</b> PathSnap is an end-to-end distributed machine learning system that reconstructs "
        "continuous vehicle road-network routes from noisy GPS logs. Real-world GPS observations suffer from multipath "
        "reflections and urban canyon distortions, causing geometric snapping to fail. PathSnap introduces a two-stage hybrid framework: "
        "(1) an 11-feature gradient-boosted spatial classifier (XGBoost / Random Forest) estimating local candidate matching probabilities "
        "<i>P(O<sub>t</sub> | S<sub>t</sub>)</i>, and (2) a Hidden Markov Model with directed shortest-path Viterbi decoding that enforces "
        "road network topological continuity. Evaluated on 80 trips (1.23M records across 208k GPS fixes), PathSnap achieves an "
        "average <b>Route F1-score of 0.7187</b> and <b>89.15% Precision</b>, significantly outperforming geometric snapping (0.6095) "
        "and ML-only snapping (0.6752)."
    )
    callout_table = Table([[Paragraph(exec_summary_text, callout_style)]], colWidths=[7.2 * inch])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg_call),
        ('BOX', (0, 0), (-1, -1), 1, c_accent),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # Read Markdown Report
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Parse sections
    lines = md_text.splitlines()
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()

        # Skip main headers already added
        if stripped.startswith("# PathSnap:") or stripped.startswith("## Comprehensive Final") or stripped.startswith("> **"):
            continue
        if stripped == "---":
            continue

        # Table detection
        if stripped.startswith("|") and stripped.endswith("|"):
            if "---" in stripped:
                continue  # separator line
            parts = [p.strip() for p in stripped.split("|")[1:-1]]
            table_rows.append(parts)
            in_table = True
            continue
        elif in_table:
            # Render accumulated table
            if len(table_rows) > 0:
                header = table_rows[0]
                body_rows = table_rows[1:]
                col_w = (7.2 * inch) / len(header)

                t_data = [[Paragraph(f"<b>{c}</b>", table_header_style) for c in header]]
                for r in body_rows:
                    # Clean markdown bold/formatting in cells
                    cleaned_r = []
                    for c in r:
                        c_clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", c)
                        c_clean = c_clean.replace("`", "")
                        cleaned_r.append(Paragraph(c_clean, table_cell_style))
                    t_data.append(cleaned_r)

                t = Table(t_data, colWidths=[col_w] * len(header))
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), c_accent),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, c_border),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg_call]),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
            table_rows = []
            in_table = False

        # Section Headers
        if stripped.startswith("## "):
            sec_title = stripped.replace("## ", "").strip()
            story.append(Paragraph(sec_title, h1_style))
        elif stripped.startswith("### "):
            sub_title = stripped.replace("### ", "").strip()
            story.append(Paragraph(sub_title, h2_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = stripped[2:].strip()
            bullet_clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", bullet_text)
            bullet_clean = re.sub(r"`(.*?)`", r"<code>\1</code>", bullet_clean)
            story.append(Paragraph(f"• {bullet_clean}", bullet_style))
        elif stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. "):
            item_text = stripped[3:].strip()
            item_clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", item_text)
            item_clean = re.sub(r"`(.*?)`", r"<code>\1</code>", item_clean)
            num = stripped[:3]
            story.append(Paragraph(f"<b>{num}</b> {item_clean}", bullet_style))
        elif stripped.startswith("```"):
            continue
        elif len(stripped) > 0:
            clean_p = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", stripped)
            clean_p = re.sub(r"`(.*?)`", r"<code>\1</code>", clean_p)
            clean_p = clean_p.replace("$$", "").replace("$", "")
            story.append(Paragraph(clean_p, body_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Report generated successfully at: {REPORT_PDF_PATH}")
    print(f"File size: {os.path.getsize(REPORT_PDF_PATH):,} bytes")


if __name__ == "__main__":
    build_pdf()
