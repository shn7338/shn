from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: render_pdf_page.py <pdf> <page-number-1-based> <output-png>")
    pdf_path = Path(sys.argv[1])
    page_number = int(sys.argv[2])
    output_path = Path(sys.argv[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(str(pdf_path))
    if not 1 <= page_number <= len(document):
        raise ValueError(f"Page {page_number} outside 1..{len(document)}")
    page = document[page_number - 1]
    bitmap = page.render(scale=1.7)
    bitmap.to_pil().save(output_path)


if __name__ == "__main__":
    main()
