import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "questions.json"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
)

INDEX_FILE = VECTORSTORE_DIR / "questions.index"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# ---------------------------------------------------------
# Load questions
# ---------------------------------------------------------

def load_questions():

    print("Loading questions...")

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    print(
        f"Loaded {len(questions)} questions."
    )

    return questions


# ---------------------------------------------------------
# Prepare text for embedding
# ---------------------------------------------------------

def create_embedding_text(question):

    year = question.get("year", "")

    question_number = question.get(
        "question_number",
        ""
    )

    question_text = question.get(
        "question",
        ""
    )

    options = question.get(
        "options",
        []
    )

    # Combine options
    options_text = ""

    if options:

        options_text = "\n".join(
            [
                f"Option {i + 1}: {option}"
                for i, option in enumerate(options)
            ]
        )

    # We deliberately don't put the complete solution
    # into the embedding text.
    #
    # The question is what we want to retrieve
    # for trend analysis and similarity search.

    embedding_text = f"""
Year: {year}

Question Number: {question_number}

Question:
{question_text}

{options_text}
""".strip()

    return embedding_text


# ---------------------------------------------------------
# Create embeddings
# ---------------------------------------------------------

def create_embeddings(
    questions,
    model
):

    print("\nPreparing text for embeddings...")

    texts = [
        create_embedding_text(question)
        for question in questions
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} questions..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings


# ---------------------------------------------------------
# Create FAISS index
# ---------------------------------------------------------

def create_faiss_index(embeddings):

    print("\nCreating FAISS index...")

    dimension = embeddings.shape[1]

    # Because embeddings are normalized,
    # Inner Product behaves like cosine similarity.

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype(
            np.float32
        )
    )

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors."
    )

    return index


# ---------------------------------------------------------
# Save index and metadata
# ---------------------------------------------------------

def save_vectorstore(
    index,
    questions
):

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            questions,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nVector store saved.")

    print(
        f"FAISS index: {INDEX_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("EXAM PATTERN RAG - EMBEDDING CREATION")
    print("=" * 70)

    # Load questions
    questions = load_questions()

    # Load embedding model
    print(
        "\nLoading Sentence Transformer model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "Embedding model loaded."
    )

    # Create embeddings
    embeddings = create_embeddings(
        questions,
        model
    )

    # Create FAISS index
    index = create_faiss_index(
        embeddings
    )

    # Save
    save_vectorstore(
        index,
        questions
    )

    print("\n" + "=" * 70)
    print("EMBEDDING CREATION COMPLETE")
    print("=" * 70)

    print(
        f"Total questions: {len(questions)}"
    )

    print(
        f"Vector dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal}"
    )