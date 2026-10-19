
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "extracted"
OUTPUT_DIR = BASE_DIR / "data" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

# Course codes such as GEOG 203L, STAT 101, HSC 324
COURSE_PATTERN = re.compile(
    r"(?m)^[ \t]*([A-Z]{2,}\s+\d{3}[A-Z]?)\s+(.+?)\s*$"
)

PAGE_PATTERN = re.compile(r"--- Page (\d+) ---")


def clean_text(text):
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for file_path in INPUT_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8", errors="replace")

        # Split the extracted document into physical PDF pages.
        page_parts = PAGE_PATTERN.split(text)

        for i in range(1, len(page_parts), 2):
            page_number = int(page_parts[i])
            page_text = page_parts[i + 1]

            matches = list(COURSE_PATTERN.finditer(page_text))

            for j, match in enumerate(matches):
                start = match.start()
                end = (
                    matches[j + 1].start()
                    if j + 1 < len(matches)
                    else len(page_text)
                )

                course_text = clean_text(page_text[start:end])

                code = re.sub(r"\s+", " ", match.group(1)).strip()
                title = match.group(2).strip()

                # Ignore incomplete or very short records.
                if len(course_text) < 80:
                    continue

                all_chunks.append({
                    "course_code": code,
                    "course_title": title,
                    "source": file_path.name,
                    "page": page_number,
                    "text": course_text
                })

    # Remove exact duplicate course records from repeated syllabus sections.
    unique_chunks = []
    seen = set()

    for chunk in all_chunks:
        key = (
            chunk["course_code"].upper(),
            re.sub(r"\s+", " ", chunk["text"]).strip().lower()
        )

        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(unique_chunks, file, ensure_ascii=False, indent=2)

    print("Course records created:", len(unique_chunks))
    print("Saved to:", OUTPUT_FILE)

    matches = [
        chunk for chunk in unique_chunks
        if chunk["course_code"].upper() == "GEOG 203L"
    ]

    print("\nGEOG 203L records after deduplication:", len(matches))

    for chunk in matches[:2]:
        print("\nCourse:", chunk["course_code"], chunk["course_title"])
        print("PDF page:", chunk["page"])
        print(chunk["text"][:1200])


if __name__ == "__main__":
    main()
