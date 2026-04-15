"""Generate synthetic PDF and DOCX test fixtures using PyMuPDF and python-docx.

Run once to create fixtures in tests/fixtures/:
    uv run python tests/generate_fixtures.py
"""

import os

import fitz  # PyMuPDF
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def make_simple_report_pdf():
    """Create a 3-page PDF with headings, paragraphs, and a footnote."""
    doc = fitz.open()

    # Page 1: Title + intro
    page1 = doc.new_page(width=595, height=842)  # A4

    # Title / heading
    page1.insert_text(
        (72, 80),
        "Annual Performance Report 2024",
        fontsize=20,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Section heading
    page1.insert_text(
        (72, 140),
        "Executive Summary",
        fontsize=15,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Body paragraphs
    body_text_1 = (
        "This report summarises the financial and operational performance of the organisation "
        "during fiscal year 2024. Overall revenue grew by 12% compared to the prior year, "
        "driven primarily by expansion in the Asia-Pacific region."
    )
    _insert_wrapped_text(page1, (72, 180), body_text_1, fontsize=11, max_width=450)

    body_text_2 = (
        "Operating costs remained stable at approximately 68% of total revenue, consistent "
        "with the five-year average. Capital expenditure increased by 8% due to the new "
        "datacenter investment completed in Q3."
    )
    _insert_wrapped_text(page1, (72, 260), body_text_2, fontsize=11, max_width=450)

    # Footnote at bottom of page
    page1.insert_text(
        (72, 800),
        "1 Footnote: All figures audited by external auditors as of December 2024.",
        fontsize=8,
        fontname="helv",
        color=(0.3, 0.3, 0.3),
    )

    # Page 2: Another section
    page2 = doc.new_page(width=595, height=842)

    page2.insert_text(
        (72, 80),
        "Financial Highlights",
        fontsize=15,
        fontname="helv",
        color=(0, 0, 0),
    )

    body_text_3 = (
        "Total revenue for fiscal 2024 was $4.2 billion, representing a 12% year-over-year "
        "increase. Net income reached $620 million, a 15% improvement versus the prior year. "
        "Earnings per share rose to $3.45 from $3.00 in fiscal 2023."
    )
    _insert_wrapped_text(page2, (72, 140), body_text_3, fontsize=11, max_width=450)

    body_text_4 = (
        "The board approved a dividend increase of 10 cents per share, bringing the annual "
        "dividend to $1.20. Share buyback programme repurchased 2.1 million shares at an "
        "average price of $42.50 during the year."
    )
    _insert_wrapped_text(page2, (72, 240), body_text_4, fontsize=11, max_width=450)

    # Page 3: Conclusion
    page3 = doc.new_page(width=595, height=842)

    page3.insert_text(
        (72, 80),
        "Conclusion and Outlook",
        fontsize=15,
        fontname="helv",
        color=(0, 0, 0),
    )

    body_text_5 = (
        "Management remains confident in the organisation's growth trajectory for fiscal 2025. "
        "Guidance for next year projects revenue growth of 8-10% with stable operating margins."
    )
    _insert_wrapped_text(page3, (72, 140), body_text_5, fontsize=11, max_width=450)

    out_path = os.path.join(FIXTURES_DIR, "simple_report.pdf")
    doc.save(out_path)
    doc.close()
    print(f"Created: {out_path}")


def make_multi_column_report_pdf():
    """Create a 2-page PDF with 2-column layout."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Title spanning full width
    page.insert_text(
        (72, 60),
        "Multi-Column Research Summary",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Left column
    left_texts = [
        ("Methodology", 15),
        ("The study employed a mixed-methods approach combining quantitative surveys with qualitative interviews.", 11),
        ("A total of 450 participants were recruited from five metropolitan areas.", 11),
        ("Data collection occurred over six months from January to June 2024.", 11),
    ]
    y = 120
    for text, size in left_texts:
        _insert_wrapped_text(page, (72, y), text, fontsize=size, max_width=210)
        y += 80

    # Right column
    right_texts = [
        ("Key Findings", 15),
        ("Participant satisfaction with digital services increased from 62% to 78% over the period.", 11),
        ("Response time improvements of 34% were recorded across all service categories.", 11),
        ("Cost efficiency gains averaged 22% when new protocols were adopted.", 11),
    ]
    y = 120
    for text, size in right_texts:
        _insert_wrapped_text(page, (310, y), text, fontsize=size, max_width=210)
        y += 80

    out_path = os.path.join(FIXTURES_DIR, "multi_column_report.pdf")
    doc.save(out_path)
    doc.close()
    print(f"Created: {out_path}")


def make_report_with_tables_pdf():
    """Create a PDF with structured data tables."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Title
    page.insert_text(
        (72, 60),
        "Quarterly Results with Data Tables",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Introduction paragraph
    page.insert_text(
        (72, 110),
        "The following table presents quarterly revenue and profit data for fiscal year 2024.",
        fontsize=11,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Draw a table manually using rectangles and text
    table_data = [
        ["Quarter", "Revenue ($M)", "Profit ($M)", "Margin (%)"],
        ["Q1 2024", "980", "142", "14.5"],
        ["Q2 2024", "1050", "158", "15.0"],
        ["Q3 2024", "1080", "168", "15.6"],
        ["Q4 2024", "1090", "152", "13.9"],
    ]

    col_x = [72, 180, 310, 440]
    row_height = 28
    start_y = 160

    for row_idx, row in enumerate(table_data):
        y = start_y + row_idx * row_height
        # Draw row background for header
        if row_idx == 0:
            page.draw_rect(
                fitz.Rect(72, y - 4, 530, y + row_height - 4),
                color=(0.8, 0.8, 0.8),
                fill=(0.9, 0.9, 0.9),
            )
        else:
            page.draw_rect(
                fitz.Rect(72, y - 4, 530, y + row_height - 4),
                color=(0.85, 0.85, 0.85),
            )
        for col_idx, cell_text in enumerate(row):
            page.insert_text(
                (col_x[col_idx] + 4, y + 12),
                cell_text,
                fontsize=10 if row_idx > 0 else 11,
                fontname="helv",
                color=(0, 0, 0),
            )

    # Another text block after the table
    page.insert_text(
        (72, 330),
        "Analysis of Results",
        fontsize=14,
        fontname="helv",
        color=(0, 0, 0),
    )
    _insert_wrapped_text(
        page,
        (72, 360),
        "The data shows consistent revenue growth throughout the year, with Q4 showing "
        "slightly compressed margins due to increased holiday season operational costs.",
        fontsize=11,
        max_width=450,
    )

    out_path = os.path.join(FIXTURES_DIR, "report_with_tables.pdf")
    doc.save(out_path)
    doc.close()
    print(f"Created: {out_path}")


def make_simple_report_docx():
    """Create a DOCX with headings, paragraphs, and a table."""
    doc = Document()

    # Title
    doc.add_heading("Annual Performance Report 2024", level=0)

    # Section 1
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "This report summarises the financial and operational performance of the organisation "
        "during fiscal year 2024. Overall revenue grew by 12% compared to the prior year, "
        "driven primarily by expansion in the Asia-Pacific region."
    )
    doc.add_paragraph(
        "Operating costs remained stable at approximately 68% of total revenue. "
        "Capital expenditure increased by 8% due to the new datacenter investment."
    )

    # Section 2
    doc.add_heading("Financial Highlights", level=1)
    doc.add_paragraph(
        "Total revenue for fiscal 2024 was $4.2 billion, representing a 12% year-over-year "
        "increase. Net income reached $620 million, a 15% improvement."
    )

    # Sub-section
    doc.add_heading("Quarterly Breakdown", level=2)
    doc.add_paragraph(
        "Q1 delivered strong results with revenue of $980M. Q2 continued the trend with "
        "$1,050M. Q3 and Q4 maintained similar performance levels."
    )

    # Table
    table = doc.add_table(rows=5, cols=4)
    table.style = "Table Grid"
    headers = ["Quarter", "Revenue ($M)", "Profit ($M)", "Margin (%)"]
    data_rows = [
        ["Q1 2024", "980", "142", "14.5"],
        ["Q2 2024", "1050", "158", "15.0"],
        ["Q3 2024", "1080", "168", "15.6"],
        ["Q4 2024", "1090", "152", "13.9"],
    ]
    for col_idx, header in enumerate(headers):
        table.rows[0].cells[col_idx].text = header
    for row_idx, row_data in enumerate(data_rows, start=1):
        for col_idx, cell_text in enumerate(row_data):
            table.rows[row_idx].cells[col_idx].text = cell_text

    # Section 3
    doc.add_heading("Conclusion", level=1)
    doc.add_paragraph(
        "Management remains confident in the organisation's growth trajectory for fiscal 2025. "
        "Guidance for next year projects revenue growth of 8-10% with stable operating margins."
    )

    out_path = os.path.join(FIXTURES_DIR, "simple_report.docx")
    doc.save(out_path)
    print(f"Created: {out_path}")


def _insert_wrapped_text(
    page: fitz.Page,
    origin: tuple,
    text: str,
    fontsize: int = 11,
    max_width: int = 450,
    fontname: str = "helv",
) -> None:
    """Insert text with basic word-wrapping into a PDF page."""
    x, y = origin
    words = text.split()
    lines = []
    current_line: list[str] = []
    # Approximate: each char ~fontsize * 0.5 width
    char_width = fontsize * 0.55

    for word in words:
        current_line.append(word)
        line_width = len(" ".join(current_line)) * char_width
        if line_width > max_width:
            if len(current_line) > 1:
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]
            else:
                lines.append(current_line[0])
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    line_height = fontsize * 1.4
    for i, line in enumerate(lines):
        page.insert_text(
            (x, y + i * line_height),
            line,
            fontsize=fontsize,
            fontname=fontname,
            color=(0, 0, 0),
        )


if __name__ == "__main__":
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    make_simple_report_pdf()
    make_multi_column_report_pdf()
    make_report_with_tables_pdf()
    make_simple_report_docx()
    print("All fixtures generated successfully.")
