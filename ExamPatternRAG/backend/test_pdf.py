import pymupdf
from pathlib import Path


pdf_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "papers"
    / "jee-2026.pdf"
)

document = pymupdf.open(pdf_path)

print("=" * 80)
print("PDF:", pdf_path)
print("=" * 80)

for page_number in range(min(3, len(document))):

    page = document[page_number]

    text = page.get_text("text")

    print("\n")
    print("=" * 80)
    print(f"PAGE {page_number + 1}")
    print("=" * 80)

    print(text[:6000])

document.close()