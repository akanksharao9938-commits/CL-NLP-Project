import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text.strip()
        })

    document.close()

    return pages


def find_all_pdfs(pdf_folder):
    """
    Recursively find every PDF inside the papers folder,
    including PDFs inside year/session subfolders.
    """

    pdf_folder = Path(pdf_folder)

    pdf_files = list(pdf_folder.rglob("*.pdf"))

    return sorted(pdf_files)


def process_all_pdfs(pdf_folder):
    """Process every PDF recursively."""

    pdf_folder = Path(pdf_folder)

    if not pdf_folder.exists():
        print(f"ERROR: Folder does not exist:")
        print(pdf_folder.resolve())
        return []

    pdf_files = find_all_pdfs(pdf_folder)

    if not pdf_files:
        print("No PDF files found.")
        print(f"Searched in:")
        print(pdf_folder.resolve())
        return []

    print(f"Found {len(pdf_files)} PDF files.")

    all_documents = []

    for pdf_file in pdf_files:

        print("\n" + "=" * 70)
        print(f"Processing: {pdf_file}")
        print("=" * 70)

        try:
            pages = extract_text_from_pdf(pdf_file)

            # Try to identify year from the file path
            year = None

            for part in pdf_file.parts:
                if part.isdigit() and len(part) == 4:
                    year = part
                    break

            # If year isn't a folder, try filename
            if year is None:
                for possible_year in range(2020, 2027):
                    if str(possible_year) in pdf_file.name:
                        year = str(possible_year)
                        break

            document = {
                "filename": pdf_file.name,
                "filepath": str(pdf_file),
                "year": year,
                "pages": pages
            }

            all_documents.append(document)

            print(f"Year: {year}")
            print(f"Pages extracted: {len(pages)}")

        except Exception as e:
            print(f"ERROR processing {pdf_file.name}: {e}")

    return all_documents


if __name__ == "__main__":

    # Project root/data/papers
    pdf_folder = Path(__file__).resolve().parent.parent / "data" / "papers"

    documents = process_all_pdfs(pdf_folder)

    print("\n" + "=" * 70)
    print("PDF PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Successfully processed: {len(documents)} PDFs")

    for document in documents:
        print(
            f"{document['filename']} → "
            f"{len(document['pages'])} pages "
            f"(Year: {document['year']})"
        )