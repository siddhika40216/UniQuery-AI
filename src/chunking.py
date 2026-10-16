
from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FOLDER = PROJECT_ROOT / "data" / "extracted"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "chunks"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_into_chunks(text, source, page_number):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append({
                "source": source,
                "page": page_number,
                "text": chunk_text
            })

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    text_files = list(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        print("No extracted text files found.")
        return

    for text_file in text_files:
        content = text_file.read_text(encoding="utf-8")
        pages = re.split(r"--- Page (\d+) ---", content)

        for index in range(1, len(pages), 2):
            page_number = int(pages[index])
            page_text = pages[index + 1].strip()

            if page_text:
                all_chunks.extend(
                    split_into_chunks(
                        page_text,
                        text_file.name,
                        page_number
                    )
                )

    output_path = OUTPUT_FOLDER / "chunks.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    print(f"Pages processed into chunks: {len(all_chunks)}")
    print(f"Chunks saved to: {output_path}")

    if all_chunks:
        print("\nSample chunk:")
        print(all_chunks[0]["text"][:350])


if __name__ == "__main__":
    main()
