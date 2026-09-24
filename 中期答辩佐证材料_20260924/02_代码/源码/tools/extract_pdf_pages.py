from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader


def parse_pages(spec: str, maximum: int) -> list[int]:
    output: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            output.extend(range(start, end + 1))
        else:
            output.append(int(part))
    return [page for page in output if 1 <= page <= maximum]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Usage: extract_pdf_pages.py <pdf> <pages> [max-chars-per-page]")
    pdf_path = Path(sys.argv[1])
    reader = PdfReader(str(pdf_path))
    maximum = int(sys.argv[3]) if len(sys.argv) == 4 else 4500
    for page_number in parse_pages(sys.argv[2], len(reader.pages)):
        text = reader.pages[page_number - 1].extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        print(f"\n===== PAGE {page_number} =====\n{text[:maximum]}")


if __name__ == "__main__":
    main()
