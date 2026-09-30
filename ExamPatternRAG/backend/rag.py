import os
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# VECTORSTORE PATHS
# ============================================================

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
)

INDEX_FILE = (
    VECTORSTORE_DIR
    / "questions.index"
)

METADATA_FILE = (
    VECTORSTORE_DIR
    / "metadata.json"
)


# ============================================================
# LOAD FAISS VECTORSTORE
# ============================================================

def load_vectorstore():

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"FAISS index loaded: "
        f"{index.ntotal} vectors"
    )

    print(
        "Loading metadata..."
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"Metadata loaded: "
        f"{len(metadata)} questions"
    )

    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "Embedding model loaded."
    )

    return model


# ============================================================
# RETRIEVE HISTORICAL QUESTIONS
# ============================================================

def retrieve_questions(
    query,
    index,
    metadata,
    model,
    top_k=5
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        np.float32
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

        results.append(
            {
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
        )

    return results


# ============================================================
# FORMAT RETRIEVED QUESTIONS
# ============================================================

def format_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        year = result[
            "year"
        ]

        question_number = result[
            "question_number"
        ]

        question_text = result[
            "question"
        ]

        options = result[
            "options"
        ]

        answer = result[
            "answer"
        ]

        context = (
            f"Historical Question {i}\n"
            f"Year: {year}\n"
            f"Question Number: "
            f"{question_number}\n"
            f"Question: "
            f"{question_text}\n"
        )

        if options:

            context += (
                "Options:\n"
            )

            for option_number, option in enumerate(
                options,
                start=1
            ):

                context += (
                    f"({option_number}) "
                    f"{option}\n"
                )

        context += (
            f"Answer: {answer}\n"
        )

        context_parts.append(
            context
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# CREATE GEMINI MODEL
# ============================================================

def create_llm():

    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "Please add your Gemini API key "
            "to the .env file."
        )

    print(
        "\nLoading Gemini model..."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key
    )

    print(
        "Gemini model loaded."
    )

    return llm


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_response(response):

    """
    Extract only the text generated by Gemini.

    This prevents internal response information,
    such as signatures or metadata, from being
    displayed in the terminal.
    """

    content = response.content

    # Case 1:
    # Normal string response
    if isinstance(
        content,
        str
    ):

        return content.strip()

    # Case 2:
    # Gemini returns a list of content blocks
    if isinstance(
        content,
        list
    ):

        text_parts = []

        for part in content:

            # Dictionary content block
            if isinstance(
                part,
                dict
            ):

                if part.get(
                    "type"
                ) == "text":

                    text = part.get(
                        "text",
                        ""
                    )

                    if text:
                        text_parts.append(
                            text
                        )

            # Direct string content
            elif isinstance(
                part,
                str
            ):

                text_parts.append(
                    part
                )

        return "\n".join(
            text_parts
        ).strip()

    # Fallback
    return str(
        content
    ).strip()


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_answer(
    query,
    results,
    llm
):

    context = format_context(
        results
    )

    prompt = f"""
You are an AI assistant for an academic
exam pattern analysis system.

The user has asked:

{query}

Below are historical questions retrieved
from previous JEE Main question papers.

Use ONLY the retrieved historical questions
as the factual basis for your analysis.

Retrieved historical questions:

{context}

Analyze the retrieved questions and provide:

1. Main Topic
   Identify the main academic topic related
   to the user's query.

2. Historical Appearance
   Explain how the topic appears in the
   retrieved questions.

3. Years
   Mention the years in which relevant
   questions appeared.

4. Recurring Concepts
   Identify concepts, formulas, or question
   patterns that occur repeatedly.

5. Student Preparation
   Explain what concepts a student should
   focus on based on these retrieved
   questions.

6. Trend Adequacy
   State whether the retrieved sample is
   sufficient to identify a reliable trend.

Important rules:

- Use only the retrieved questions.
- Do not invent questions.
- Do not invent years.
- Do not invent statistics.
- Do not claim a trend from insufficient evidence.
- Clearly distinguish direct topic matches
  from related topics.
- Keep the explanation clear and useful
  for students.
"""

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    # IMPORTANT:
    # Return only the actual text.
    return clean_response(
        response
    )


# ============================================================
# DISPLAY RETRIEVED QUESTIONS
# ============================================================

def display_retrieved_questions(
    results
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVED HISTORICAL QUESTIONS"
    )

    print(
        "=" * 70
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Question {i} ---"
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
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Question:\n"
            f"{result['question']}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "EXAM PATTERN RAG - "
        "RETRIEVAL AUGMENTED GENERATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load FAISS index and metadata
    # --------------------------------------------------------

    index, metadata = (
        load_vectorstore()
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # Load Gemini
    # --------------------------------------------------------

    llm = create_llm()

    # --------------------------------------------------------
    # Ask user for query
    # --------------------------------------------------------

    query = input(
        "\nEnter your topic/question: "
    )

    # --------------------------------------------------------
    # Retrieve relevant questions
    # --------------------------------------------------------

    print(
        "\nSearching historical questions..."
    )

    results = retrieve_questions(
        query=query,
        index=index,
        metadata=metadata,
        model=embedding_model,
        top_k=5
    )

    # --------------------------------------------------------
    # Display retrieved questions
    # --------------------------------------------------------

    display_retrieved_questions(
        results
    )

    # --------------------------------------------------------
    # Generate RAG analysis
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING RAG ANALYSIS..."
    )

    print(
        "=" * 70
    )

    answer = generate_answer(
        query=query,
        results=results,
        llm=llm
    )

    # --------------------------------------------------------
    # Display final answer
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )