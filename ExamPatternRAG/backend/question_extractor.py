import re
import json
from pathlib import Path
import pymupdf


def extract_pdf_text(pdf_path):
    """
    Extract complete text from a PDF.
    All pages are combined into one continuous text.
    """

    document = pymupdf.open(pdf_path)

    all_text = []

    for page in document:
        text = page.get_text("text")
        all_text.append(text)

    document.close()

    return "\n".join(all_text)


def clean_text(text):
    """
    Clean unnecessary PDF formatting while preserving
    question content.
    """

    # Remove repeated publisher headers
    text = re.sub(
        r"CAREER POINT\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"CAREER POINT Ltd\.,.*?(?=\n)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove website
    text = re.sub(
        r"www\.careerpoint\.ac\.in",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated paper header
    text = re.sub(
        r"JEE Main Online Paper\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove page numbers that appear alone
    text = re.sub(
        r"\n\s*\d+\s*\n",
        "\n",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Reduce excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text
    )

    return text.strip()


def detect_question_format(text):
    """
    Detect whether the PDF uses:

    Q.1
    Q.2
    ...

    or:

    1.
    2.
    3.
    ...
    """

    q_format = re.findall(
        r"\nQ\.\s*\d+\s*",
        text
    )

    numbered_format = re.findall(
        r"\n\s*\d+\.\s+",
        text
    )

    if len(q_format) >= 2:
        return "Q"

    if len(numbered_format) >= 2:
        return "NUMBER"

    return None


def extract_questions(text, year):
    """
    Extract questions from either:

    Q.1
    Q.2
    ...

    or:

    1.
    2.
    3.
    ...

    format.
    """

    text = clean_text(text)

    question_format = detect_question_format(text)

    print(f"Detected question format: {question_format}")

    if question_format == "Q":

        question_pattern = re.compile(
            r"\nQ\.\s*(\d+)\s*",
            re.MULTILINE
        )

    elif question_format == "NUMBER":

        question_pattern = re.compile(
            r"\n\s*(\d+)\.\s+",
            re.MULTILINE
        )

    else:
        print("WARNING: Could not detect question format.")
        return []

    matches = list(
        question_pattern.finditer(text)
    )

    questions = []

    for i, match in enumerate(matches):

        question_number = int(
            match.group(1)
        )

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        question_block = text[
            start:end
        ].strip()

        # --------------------------------------------------
        # Extract answer
        # --------------------------------------------------

        answer_match = re.search(
            r"\bAns\.\s*\[?\s*([1-4])\s*\]?",
            question_block,
            re.IGNORECASE
        )

        answer = None

        if answer_match:
            answer = int(
                answer_match.group(1)
            )

        # --------------------------------------------------
        # Extract solution
        # --------------------------------------------------

        solution = ""

        solution_match = re.search(
            r"\bSol\.\s*(.*)",
            question_block,
            re.IGNORECASE | re.DOTALL
        )

        if solution_match:

            solution = solution_match.group(1).strip()

            question_part = question_block[
                :solution_match.start()
            ].strip()

        else:

            question_part = question_block

        # --------------------------------------------------
        # Extract options
        # --------------------------------------------------

        option_matches = list(
            re.finditer(
                r"\((1|2|3|4)\)\s*",
                question_part
            )
        )

        options = []

        if option_matches:

            question_text = question_part[
                :option_matches[0].start()
            ].strip()

            for j, option_match in enumerate(
                option_matches
            ):

                option_start = option_match.end()

                if j + 1 < len(option_matches):

                    option_end = (
                        option_matches[j + 1].start()
                    )

                else:

                    option_end = len(
                        question_part
                    )

                option_text = question_part[
                    option_start:option_end
                ].strip()

                # Remove accidental answer text
                option_text = re.split(
                    r"\bAns\.",
                    option_text,
                    flags=re.IGNORECASE
                )[0].strip()

                options.append(
                    option_text
                )

        else:

            question_text = question_part

        # --------------------------------------------------
        # Remove answer from question text
        # --------------------------------------------------

        question_text = re.sub(
            r"\bAns\.\s*\[?\s*[1-4]\s*\]?",
            "",
            question_text,
            flags=re.IGNORECASE
        ).strip()

        # --------------------------------------------------
        # Remove obvious section headings
        # --------------------------------------------------

        question_text = re.sub(
            r"^(SECTION-A|SECTION-B)\s*",
            "",
            question_text,
            flags=re.IGNORECASE
        ).strip()

        # --------------------------------------------------
        # Create structured record
        # --------------------------------------------------

        question = {
            "year": year,
            "question_number": question_number,
            "question": question_text,
            "options": options,
            "answer": answer,
            "solution": solution
        }

        questions.append(question)

    return questions


def process_pdf(pdf_path, year):
    """
    Process one PDF.
    """

    print("\n" + "=" * 70)
    print(f"Processing: {pdf_path}")
    print("=" * 70)

    text = extract_pdf_text(pdf_path)

    print(
        f"Extracted characters: {len(text)}"
    )

    questions = extract_questions(
        text,
        year
    )

    print(
        f"Questions extracted: {len(questions)}"
    )

    return questions


def save_questions(
    questions,
    output_path
):
    """
    Save questions to JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            questions,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    pdf_folder = (
        project_root /
        "data" /
        "papers"
    )

    output_file = (
        project_root /
        "data" /
        "processed" /
        "questions.json"
    )

    all_questions = []

    pdf_files = sorted(
        pdf_folder.glob("*.pdf")
    )

    print("=" * 70)
    print("JEE QUESTION EXTRACTOR")
    print("=" * 70)

    print(
        f"PDF files found: {len(pdf_files)}"
    )

    for pdf_file in pdf_files:

        year_match = re.search(
            r"(20\d{2})",
            pdf_file.name
        )

        if year_match:

            year = int(
                year_match.group(1)
            )

        else:

            year = None

        questions = process_pdf(
            pdf_file,
            year
        )

        all_questions.extend(
            questions
        )

    save_questions(
        all_questions,
        output_file
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Total questions: "
        f"{len(all_questions)}"
    )

    print("\nQuestions by year:")

    years = {}

    for question in all_questions:

        year = question["year"]

        years[year] = (
            years.get(year, 0) + 1
        )

    for year in sorted(years):

        print(
            f"{year}: {years[year]} questions"
        )

    print("\nSample questions:")

    for question in all_questions[-3:]:

        print("-" * 70)

        print(
            f"Year: "
            f"{question['year']}"
        )

        print(
            f"Question No: "
            f"{question['question_number']}"
        )

        print(
            f"Question: "
            f"{question['question'][:300]}"
        )

        print(
            f"Options: "
            f"{question['options']}"
        )

        print(
            f"Answer: "
            f"{question['answer']}"
        )