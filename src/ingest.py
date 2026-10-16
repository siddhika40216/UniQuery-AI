
from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_FOLDER = PROJECT_ROOT / "data" / "documents"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "extracted"


def extract_text_from_pdf(pdf_path):
    extracted_pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page": page_number,
                    "text": text
                })

    return extracted_pages


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DOCUMENTS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        print(f"Add a PDF to: {DOCUMENTS_FOLDER}")
        return

    for pdf_path in pdf_files:
        try:
            pages = extract_text_from_pdf(pdf_path)

            if not pages:
                print(f"No extractable text: {pdf_path.name}")
                print("This PDF may need OCR.")
                continue

            output_path = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

            with output_path.open("w", encoding="utf-8") as file:
                for page in pages:
                    file.write(f"\n--- Page {page['page']} ---\n")
                    file.write(page["text"] + "\n")

            print(f"\nProcessed: {pdf_path.name}")
            print(f"Pages extracted: {len(pages)}")
            print(f"Saved text: {output_path}")

        except Exception as error:
            print(f"Error processing {pdf_path.name}: {error}")


if __name__ == "__main__":
    main()
