from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


TARGET_NAMES = [
    "基于深度神经网络的电磁频谱地图构建方法研究_李双宁.pdf",
    "基于压缩感知的电磁频谱重构技术研究_张黎微.pdf",
    "基于深度学习的低空信噪比地图高效估计方法研究_李峰.pdf",
    "基于无线电地图重构的智能无人机路径导航技术研究_郝晴.pdf",
    "基于无线电环境地图认知引擎关键技术研究_邓宇兵.pdf",
    "基于无线指纹的Radio_Map预测与生成研究_卢嘉王男.pdf",
    "面向城区环境的无线传播深度学习模型研究_郑毅.pdf",
    "基于深度学习的城市无线电地图构建方法研究_丁志文.pdf",
]

KEYWORDS = [
    "稀疏",
    "采样率",
    "基站位置",
    "发射机位置",
    "建筑高度",
    "建筑物",
    "输入",
    "掩码",
    "深度补全",
    "深度图像先验",
    "压缩感知",
    "生成对抗",
    "迁移学习",
]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippets(text: str, keyword: str, limit: int = 2, radius: int = 180) -> list[str]:
    output: list[str] = []
    start = 0
    while len(output) < limit:
        index = text.find(keyword, start)
        if index < 0:
            break
        output.append(compact(text[max(0, index - radius) : index + len(keyword) + radius]))
        start = index + len(keyword)
    return output


def analyze_pdf(pdf_path: Path) -> dict:
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    extraction_errors: list[int] = []
    for index, page in enumerate(reader.pages):
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            extraction_errors.append(index + 1)
            page_texts.append("")

    joined = "\n".join(page_texts)
    hits = {}
    for keyword in KEYWORDS:
        pages = [index + 1 for index, text in enumerate(page_texts) if keyword in text]
        if pages:
            hits[keyword] = {
                "pages": pages[:12],
                "count_pages": len(pages),
                "snippets": snippets(joined, keyword),
            }

    return {
        "file": str(pdf_path),
        "size": pdf_path.stat().st_size,
        "pages": len(reader.pages),
        "encrypted": bool(reader.is_encrypted),
        "extractable_characters": len(joined),
        "first_page_characters": len(page_texts[0]) if page_texts else 0,
        "extraction_error_pages": extraction_errors,
        "keywords": hits,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) != 2:
        raise SystemExit("Usage: analyze_cnki_pdfs.py <directory>")
    root = Path(sys.argv[1])
    results = []
    for name in TARGET_NAMES:
        path = root / name
        if not path.exists():
            results.append({"file": str(path), "error": "missing"})
            continue
        results.append(analyze_pdf(path))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
