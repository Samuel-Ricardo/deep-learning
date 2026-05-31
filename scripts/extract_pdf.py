#!/usr/bin/env python
import sys
import os

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def extract_text(pdf_path, out_path):
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text:
            texts.append(text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(texts))
    print(f"Extracted {len(texts)} pages to {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_pdf.py <pdf_path> [out_path]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(pdf_path)[0] + '.txt'
    if not os.path.isfile(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(2)
    if PdfReader is None:
        print("PyPDF2 not installed. Install with: pip install PyPDF2")
        sys.exit(3)
    extract_text(pdf_path, out_path)


if __name__ == "__main__":
    main()
