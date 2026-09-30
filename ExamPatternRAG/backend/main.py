# ============================================================
# EXAM PATTERN RAG PIPELINE - FASTAPI BACKEND
# ============================================================

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

QUESTIONS_FILE = PROCESSED_DIR / "questions.json"
INDEX_FILE = VECTORSTORE_DIR / "questions.index"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# Create directories if they don't exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# We directly use this model.
# No Gemini 3.6 fallback/retry logic is used.
GEMINI_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Exam Pattern RAG API",
    description="RAG-based analysis of historical JEE question papers",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n==========================================")
print("   EXAM PATTERN RAG BACKEND")
print("==========================================")
print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ============================================================
# LOAD QUESTION DATA
# ============================================================

questions_data = []

if QUESTIONS_FILE.exists():
    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions_data = json.load(f)

        print(f"Loaded {len(questions_data)} questions.")

    except Exception as e:
        print(f"Error loading questions.json: {e}")

else:
    print("WARNING: questions.json not found.")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

faiss_index = None
metadata = []

if INDEX_FILE.exists() and METADATA_FILE.exists():

    try:
        faiss_index = faiss.read_index(str(INDEX_FILE))

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"FAISS index loaded: {faiss_index.ntotal} vectors.")
        print(f"Metadata loaded: {len(metadata)} records.")

    except Exception as e:
        print(f"Error loading FAISS vector store: {e}")

else:
    print("WARNING: FAISS vector store not found.")


# ============================================================
# TOPIC KEYWORDS
# ============================================================

TOPIC_KEYWORDS = {

    "Physics - Mechanics": [
        "motion",
        "kinematics",
        "newton",
        "force",
        "friction",
        "work",
        "energy",
        "power",
        "momentum",
        "collision",
        "rotational",
        "rotation",
        "gravitation",
        "center of mass",
        "simple harmonic",
        "oscillation",
        "mechanical"
    ],

    "Physics - Electricity": [
        "electric field",
        "electric potential",
        "potential",
        "capacitor",
        "capacitance",
        "current",
        "resistance",
        "resistor",
        "circuit",
        "ohm",
        "kirchhoff",
        "electric charge",
        "electrostatic"
    ],

    "Physics - Magnetism": [
        "magnetic field",
        "magnetic force",
        "magnetic flux",
        "ampere",
        "biot",
        "savart",
        "cyclotron",
        "lorentz",
        "magnetism",
        "magnetic moment"
    ],

    "Physics - Optics": [
        "lens",
        "mirror",
        "refraction",
        "reflection",
        "interference",
        "diffraction",
        "polarization",
        "optics",
        "prism",
        "focal length"
    ],

    "Physics - Modern Physics": [
        "photoelectric",
        "photon",
        "bohr",
        "atom",
        "nuclear",
        "radioactive",
        "radioactivity",
        "de broglie",
        "matter wave",
        "semiconductor",
        "nucleus",
        "binding energy",
        "mass defect"
    ],

    "Physics - Thermodynamics": [
        "thermodynamic",
        "thermodynamics",
        "entropy",
        "enthalpy",
        "heat",
        "temperature",
        "gas",
        "kinetic theory",
        "thermal",
        "isothermal",
        "adiabatic"
    ],

    "Chemistry - Physical": [
        "mole",
        "molarity",
        "molality",
        "thermochemistry",
        "chemical kinetics",
        "equilibrium",
        "ionic equilibrium",
        "electrochemistry",
        "solution",
        "colligative",
        "rate constant",
        "ph",
        "osmotic"
    ],

    "Chemistry - Organic": [
        "alkane",
        "alkene",
        "alkyne",
        "alcohol",
        "aldehyde",
        "ketone",
        "amine",
        "amine",
        "ether",
        "ester",
        "benzene",
        "phenol",
        "organic",
        "hydrocarbon",
        "reaction mechanism",
        "grignard"
    ],

    "Chemistry - Inorganic": [
        "periodic",
        "coordination",
        "transition",
        "metallurgy",
        "s-block",
        "p-block",
        "d-block",
        "f-block",
        "halogen",
        "noble gas",
        "inorganic",
        "oxidation state"
    ],

    "Mathematics - Algebra": [
        "quadratic",
        "sequence",
        "series",
        "progression",
        "matrix",
        "determinant",
        "complex number",
        "permutation",
        "combination",
        "binomial",
        "equation",
        "inequality",
        "logarithm",
        "algebra"
    ],

    "Mathematics - Calculus": [
        "limit",
        "continuity",
        "differentiation",
        "derivative",
        "integration",
        "integral",
        "differential equation",
        "calculus",
        "maximum",
        "minimum",
        "tangent",
        "normal"
    ],

    "Mathematics - Coordinate Geometry": [
        "straight line",
        "circle",
        "parabola",
        "ellipse",
        "hyperbola",
        "coordinate",
        "conic",
        "distance",
        "slope",
        "section formula"
    ],

    "Mathematics - Probability & Statistics": [
        "probability",
        "mean",
        "median",
        "mode",
        "variance",
        "standard deviation",
        "statistics",
        "random variable"
    ],

    "Mathematics - Vectors & 3D": [
        "vector",
        "three dimensional",
        "3d",
        "direction cosine",
        "direction ratio",
        "line in space",
        "plane",
        "dot product",
        "cross product"
    ]
}


# ============================================================
# REQUEST MODELS
# ============================================================

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_gemini_text(response) -> str:
    """
    Convert different Gemini/LangChain response formats
    into one clean text string.
    """

    if response is None:
        return ""

    # Normal string response
    if isinstance(response, str):
        return response.strip()

    # LangChain AIMessage.content
    content = getattr(response, "content", None)

    if content is not None:

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            text_parts = []

            for block in content:

                if isinstance(block, str):
                    text_parts.append(block)

                elif isinstance(block, dict):
                    text_value = block.get("text")

                    if isinstance(text_value, str):
                        text_parts.append(text_value)

                else:
                    text_value = getattr(block, "text", None)

                    if isinstance(text_value, str):
                        text_parts.append(text_value)

            return "\n".join(text_parts).strip()

    # Direct list response
    if isinstance(response, list):

        text_parts = []

        for block in response:

            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):

                text_value = block.get("text")

                if isinstance(text_value, str):
                    text_parts.append(text_value)

            else:

                text_value = getattr(block, "text", None)

                if isinstance(text_value, str):
                    text_parts.append(text_value)

        return "\n".join(text_parts).strip()

    # Last fallback
    return str(response).strip()


def classify_question(question_text: str):
    """
    Keyword-based topic classification.
    """

    text = question_text.lower()

    matched_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text:
                matched_topics.append(topic)
                break

    return matched_topics


def get_all_topic_counts():

    topic_counts = {}

    for topic in TOPIC_KEYWORDS:
        topic_counts[topic] = 0

    year_counts = {}

    for item in questions_data:

        year = item.get("year")

        question_text = item.get("question", "")

        matched_topics = classify_question(question_text)

        if year not in year_counts:
            year_counts[year] = 0

        year_counts[year] += 1

        for topic in matched_topics:

            topic_counts[topic] += 1

    return topic_counts, year_counts


def get_topic_year_counts():

    topic_year_counts = {}

    for topic in TOPIC_KEYWORDS:
        topic_year_counts[topic] = {}

    for item in questions_data:

        year = item.get("year")

        question_text = item.get("question", "")

        matched_topics = classify_question(question_text)

        for topic in matched_topics:

            if year not in topic_year_counts[topic]:
                topic_year_counts[topic][year] = 0

            topic_year_counts[topic][year] += 1

    return topic_year_counts


def forecast_topic(topic_data):

    """
    Simple linear trend forecast for the next year.
    """

    if len(topic_data) < 2:
        return {
            "forecast": 0,
            "slope": 0,
            "trend": "Insufficient data"
        }

    years = sorted(topic_data.keys())

    x = np.array(years, dtype=float)

    y = np.array(
        [topic_data[year] for year in years],
        dtype=float
    )

    slope, intercept = np.polyfit(x, y, 1)

    next_year = max(years) + 1

    forecast_value = slope * next_year + intercept

    forecast_value = max(0, forecast_value)

    if slope > 0.5:
        trend = "Increasing"

    elif slope < -0.5:
        trend = "Decreasing"

    else:
        trend = "Relatively Stable"

    return {
        "forecast": round(float(forecast_value), 1),
        "slope": round(float(slope), 2),
        "trend": trend
    }


def create_rag_prompt(query: str, retrieved_questions: list) -> str:

    question_text = ""

    for i, q in enumerate(retrieved_questions, start=1):

        question_text += f"""
Question {i}
Year: {q.get("year")}
Question Number: {q.get("question_number")}

{q.get("question")}

Options:
{q.get("options", [])}

Answer: {q.get("answer")}

Similarity Score: {q.get("score")}
----------------------------------------
"""

    prompt = f"""
You are an academic exam-pattern analysis assistant.

You are analyzing historical JEE question papers using
retrieved questions from a Retrieval-Augmented Generation
(RAG) system.

Student Query:
{query}

Retrieved Historical Questions:
{question_text}

Based ONLY on the retrieved questions, provide a concise
and useful analysis.

Use exactly these sections:

Main Topic:
State the main topic identified from the query.

Historical Appearance:
Explain which retrieved questions are directly related
to the requested topic. Distinguish direct matches from
only broadly related questions.

Years:
List the years in which directly relevant questions
appear in the retrieved data.

Recurring Concepts:
List important concepts appearing in the relevant
questions.

Student Preparation:
Explain what concepts or question types students should
practice based on the retrieved evidence.

Trend Adequacy:
Explain whether the retrieved sample is sufficient to
identify a strong historical trend. Be honest when the
sample size is small.

Important:
Do not claim that a topic appeared in a year unless it is
supported by the retrieved questions.
Do not invent questions or statistics.
"""

    return prompt


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Exam Pattern RAG API is running",
        "version": "1.0.0",
        "gemini_model": GEMINI_MODEL,
        "endpoints": [
            "/upload",
            "/process",
            "/query",
            "/trends",
            "/topics",
            "/forecast"
        ]
    }


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_path = PAPERS_DIR / file.filename

    try:

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "message": "PDF uploaded successfully.",
            "filename": file.filename,
            "path": str(file_path)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


# ============================================================
# PROCESS ENDPOINT
# ============================================================

@app.post("/process")
def process_papers():

    try:

        scripts = [
            BASE_DIR / "backend" / "pdf_processor.py",
            BASE_DIR / "backend" / "question_extractor.py",
            BASE_DIR / "backend" / "embeddings.py"
        ]

        outputs = []

        for script in scripts:

            if not script.exists():

                return {
                    "success": False,
                    "message": f"Script not found: {script}"
                }

            print(f"\nRunning: {script.name}")

            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True
            )

            outputs.append({
                "script": script.name,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })

            if result.returncode != 0:

                return {
                    "success": False,
                    "message": f"{script.name} failed.",
                    "outputs": outputs
                }

        return {
            "success": True,
            "message": "PDF processing, question extraction and embeddings completed.",
            "outputs": outputs
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Processing failed: {str(e)}"
        }


# ============================================================
# QUERY ENDPOINT - MAIN RAG FUNCTION
# ============================================================

@app.post("/query")
def query_rag(request: QueryRequest):

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if faiss_index is None:

        raise HTTPException(
            status_code=500,
            detail="FAISS vector store is not available."
        )

    if not metadata:

        raise HTTPException(
            status_code=500,
            detail="Vector metadata is not available."
        )

    # --------------------------------------------------------
    # STEP 1: Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # --------------------------------------------------------
    # STEP 2: Search FAISS
    # --------------------------------------------------------

    top_k = min(
        max(request.top_k or 5, 1),
        faiss_index.ntotal
    )

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    # --------------------------------------------------------
    # STEP 3: Prepare retrieved questions
    # --------------------------------------------------------

    retrieved_questions = []

    for score, index in zip(scores[0], indices[0]):

        if index < 0 or index >= len(metadata):
            continue

        item = metadata[index]

        retrieved_questions.append({
            "year": item.get("year"),
            "question_number": item.get(
                "question_number",
                item.get("number")
            ),
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "answer": item.get("answer"),
            "score": round(float(score), 4)
        })

    # --------------------------------------------------------
    # STEP 4: Gemini RAG analysis
    # --------------------------------------------------------

    analysis = ""
    gemini_available = False

    if not GOOGLE_API_KEY:

        analysis = (
            "Gemini API key is not configured. "
            "Please add GOOGLE_API_KEY to the .env file."
        )

    else:

        try:

            print("\n==========================================")
            print("RAG QUERY")
            print("==========================================")
            print(f"Query: {query}")
            print(f"Retrieved: {len(retrieved_questions)} questions")
            print(f"Gemini Model: {GEMINI_MODEL}")

            prompt = create_rag_prompt(
                query,
                retrieved_questions
            )

            print("Generating Gemini analysis...")

            # ------------------------------------------------
            # DIRECTLY USE GEMINI 3.5 FLASH LITE
            # ------------------------------------------------

            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )

            response = llm.invoke(prompt)

            analysis = extract_gemini_text(response)

            if analysis:

                gemini_available = True

                print("Gemini generation successful.")

            else:

                analysis = (
                    "Gemini returned an empty response."
                )

        except Exception as e:

            print(f"Gemini error: {e}")

            analysis = (
                "Gemini analysis could not be generated.\n\n"
                f"Error: {str(e)}"
            )

    # --------------------------------------------------------
    # STEP 5: Return clean JSON
    # --------------------------------------------------------

    return {
        "success": True,
        "query": query,
        "retrieved_questions": retrieved_questions,
        "analysis": analysis,
        "model": GEMINI_MODEL,
        "gemini_available": gemini_available
    }


# ============================================================
# TOPICS ENDPOINT
# ============================================================

@app.get("/topics")
def get_topics():

    return {
        "success": True,
        "topics": list(TOPIC_KEYWORDS.keys()),
        "total_topics": len(TOPIC_KEYWORDS)
    }


# ============================================================
# TRENDS ENDPOINT
# ============================================================

@app.get("/trends")
def get_trends():

    topic_counts, year_counts = get_all_topic_counts()

    sorted_topics = sorted(
        topic_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    total_questions = len(questions_data)

    topic_results = []

    for topic, count in sorted_topics:

        percentage = 0

        if total_questions > 0:
            percentage = (count / total_questions) * 100

        topic_results.append({
            "topic": topic,
            "count": count,
            "percentage": round(percentage, 1)
        })

    return {
        "success": True,
        "total_questions": total_questions,
        "year_counts": year_counts,
        "topics": topic_results
    }


# ============================================================
# FORECAST ENDPOINT
# ============================================================

@app.get("/forecast")
def get_forecast():

    topic_year_counts = get_topic_year_counts()

    forecasts = []

    for topic, year_data in topic_year_counts.items():

        result = forecast_topic(year_data)

        forecasts.append({
            "topic": topic,
            "historical_data": year_data,
            "forecast_year": 2027,
            "forecast_count": result["forecast"],
            "slope": result["slope"],
            "trend": result["trend"]
        })

    forecasts.sort(
        key=lambda x: x["forecast_count"],
        reverse=True
    )

    return {
        "success": True,
        "forecast_year": 2027,
        "method": "Linear trend baseline",
        "note": (
            "Forecast values are statistical estimates "
            "based on historical keyword-classified counts "
            "and should not be treated as guaranteed exam predictions."
        ),
        "forecasts": forecasts
    }


# ============================================================
# RUN SERVER DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )