import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
)

INDEX_FILE = VECTORSTORE_DIR / "questions.index"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_vectorstore():
    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"FAISS index loaded successfully."
    )
    print(
        f"Total vectors: {index.ntotal}"
    )

    print("\nLoading metadata...")

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print(
        f"Metadata loaded: {len(metadata)} questions."
    )

    return index, metadata


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    query,
    model
):
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding.astype(
        np.float32
    )


# ============================================================
# SEARCH SIMILAR QUESTIONS
# ============================================================

def search_questions(
    query,
    index,
    metadata,
    model,
    top_k=5
):

    query_embedding = create_query_embedding(
        query,
        model
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        question = metadata[
            index_position
        ]

        result = {
            "score": float(score),
            "year": question.get(
                "year"
            ),
            "question_number": question.get(
                "question_number"
            ),
            "question": question.get(
                "question"
            ),
            "options": question.get(
                "options",
                []
            ),
            "answer": question.get(
                "answer"
            ),
            "solution": question.get(
                "solution"
            )
        }

        results.append(result)

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n" + "=" * 70)
    print("SEARCH QUERY")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)
    print("RETRIEVED QUESTIONS")
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Year: "
            f"{result['year']}"
        )

        print(
            f"Question Number: "
            f"{result['question_number']}"
        )

        print(
            f"Question:\n"
            f"{result['question']}"
        )

        if result["options"]:

            print("\nOptions:")

            for option_number, option in enumerate(
                result["options"],
                start=1
            ):

                print(
                    f"  ({option_number}) "
                    f"{option}"
                )

        print(
            f"\nAnswer: "
            f"{result['answer']}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("EXAM PATTERN RAG - RETRIEVER")
    print("=" * 70)

    # Load FAISS index and metadata
    index, metadata = load_vectorstore()

    # Load the same embedding model
    print(
        "\nLoading Sentence Transformer model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------

    query = input(
        "\nEnter your question/topic to search: "
    )

    # Search
    results = search_questions(
        query=query,
        index=index,
        metadata=metadata,
        model=model,
        top_k=5
    )

    # Display results
    display_results(
        query,
        results
    )