import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = BASE_DIR / "data" / "processed" / "questions.json"


# ============================================================
# TOPIC KEYWORDS
# ============================================================

TOPIC_KEYWORDS = {
    "Physics - Mechanics": [
        "motion", "velocity", "acceleration", "force",
        "friction", "momentum", "collision", "projectile",
        "work", "energy", "power", "gravitation",
        "rotation", "rotational", "torque", "angular"
    ],

    "Physics - Electricity": [
        "electric field", "electric potential", "capacitor",
        "capacitance", "electric charge", "resistance",
        "resistor", "current", "ohm", "circuit", "kirchhoff"
    ],

    "Physics - Magnetism": [
        "magnetic field", "magnetic force", "magnetic flux",
        "electromagnetic induction", "induced emf",
        "faraday", "lenz", "inductor", "inductance"
    ],

    "Physics - Optics": [
        "lens", "mirror", "reflection", "refraction",
        "interference", "diffraction", "polarization",
        "optics", "focal length"
    ],

    "Physics - Modern Physics": [
        "photoelectric", "de broglie", "bohr", "nucleus",
        "radioactive", "radioactivity", "half life",
        "nuclear", "atom", "quantum", "photon",
        "semiconductor"
    ],

    "Physics - Thermodynamics": [
        "thermodynamics", "temperature", "heat", "entropy",
        "gas", "isothermal", "adiabatic", "kinetic theory"
    ],

    "Chemistry - Physical": [
        "mole", "molarity", "molar", "equilibrium",
        "kinetics", "rate constant", "electrochemistry",
        "osmotic", "colligative", "enthalpy", "entropy"
    ],

    "Chemistry - Organic": [
        "alkane", "alkene", "alkyne", "alcohol",
        "aldehyde", "ketone", "ether", "amine",
        "amide", "benzene", "organic", "hydrocarbon",
        "polymer", "isomer"
    ],

    "Chemistry - Inorganic": [
        "periodic", "coordination", "ligand",
        "transition element", "metal", "non-metal",
        "halogen", "alkali", "oxidation state",
        "salt", "compound"
    ],

    "Mathematics - Algebra": [
        "quadratic", "equation", "sequence", "series",
        "progression", "matrix", "determinant",
        "complex number", "permutation", "combination",
        "binomial"
    ],

    "Mathematics - Calculus": [
        "derivative", "differentiation", "integration",
        "integral", "limit", "continuity",
        "differential equation", "maximum", "minimum"
    ],

    "Mathematics - Coordinate Geometry": [
        "straight line", "circle", "parabola",
        "ellipse", "hyperbola", "coordinate",
        "slope", "conic"
    ],

    "Mathematics - Probability & Statistics": [
        "probability", "mean", "median", "variance",
        "standard deviation", "random variable",
        "distribution"
    ],

    "Mathematics - Vectors & 3D": [
        "vector", "three dimensional", "3d",
        "direction cosine", "plane", "line in space"
    ]
}


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    if not QUESTIONS_FILE.exists():

        print("ERROR: questions.json was not found.")
        print("Expected location:")
        print(QUESTIONS_FILE)

        return []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# CLASSIFY QUESTION
# ============================================================

def classify_question(question):

    text = clean_text(question)

    matched_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text:

                matched_topics.append(topic)

                break

    return matched_topics


# ============================================================
# CREATE YEAR-TOPIC DATA
# ============================================================

def create_year_topic_data(questions):

    year_topic_counts = defaultdict(Counter)

    for question in questions:

        year = question.get("year")

        question_text = question.get(
            "question",
            ""
        )

        topics = classify_question(
            question_text
        )

        for topic in topics:

            year_topic_counts[year][topic] += 1

    return year_topic_counts


# ============================================================
# LINEAR REGRESSION
# ============================================================

def calculate_slope(values):

    n = len(values)

    if n < 2:
        return 0

    x_values = list(range(n))

    x_mean = sum(x_values) / n
    y_mean = sum(values) / n

    numerator = 0
    denominator = 0

    for x, y in zip(x_values, values):

        numerator += (
            (x - x_mean) *
            (y - y_mean)
        )

        denominator += (
            (x - x_mean) ** 2
        )

    if denominator == 0:
        return 0

    return numerator / denominator


# ============================================================
# FORECAST NEXT YEAR
# ============================================================

def forecast_topic(
    years,
    values
):

    slope = calculate_slope(values)

    last_value = values[-1]

    forecast = last_value + slope

    # A question count cannot be negative
    forecast = max(
        0,
        forecast
    )

    return slope, forecast


# ============================================================
# TREND LABEL
# ============================================================

def get_trend_label(slope):

    if slope > 1:
        return "Increasing"

    elif slope < -1:
        return "Decreasing"

    else:
        return "Relatively Stable"


# ============================================================
# DISPLAY FORECAST
# ============================================================

def display_forecast(
    year_topic_counts
):

    years = sorted(
        year_topic_counts.keys()
    )

    next_year = max(years) + 1

    print("\n" + "=" * 80)

    print(
        f"FORECAST FOR {next_year}"
    )

    print("=" * 80)

    results = []

    all_topics = set()

    for year in years:

        all_topics.update(
            year_topic_counts[year].keys()
        )

    for topic in all_topics:

        values = []

        for year in years:

            count = year_topic_counts[
                year
            ].get(
                topic,
                0
            )

            values.append(count)

        slope, forecast = forecast_topic(
            years,
            values
        )

        trend = get_trend_label(
            slope
        )

        results.append(
            (
                topic,
                values,
                slope,
                forecast,
                trend
            )
        )

    # Sort by forecast frequency
    results.sort(
        key=lambda x: x[3],
        reverse=True
    )

    print()

    for rank, result in enumerate(
        results,
        start=1
    ):

        topic = result[0]
        values = result[1]
        slope = result[2]
        forecast = result[3]
        trend = result[4]

        print(
            f"{rank:2d}. {topic}"
        )

        print(
            f"    Historical counts: {values}"
        )

        print(
            f"    Trend: {trend}"
        )

        print(
            f"    Slope: {slope:.2f}"
        )

        print(
            f"    Forecast for {next_year}: "
            f"{forecast:.1f}"
        )

        print()


# ============================================================
# DISPLAY YEAR-WISE TABLE
# ============================================================

def display_year_table(
    year_topic_counts
):

    years = sorted(
        year_topic_counts.keys()
    )

    print("\n" + "=" * 80)

    print(
        "YEAR-WISE TOPIC COUNTS"
    )

    print("=" * 80)

    topics = set()

    for year in years:

        topics.update(
            year_topic_counts[
                year
            ].keys()
        )

    topics = sorted(topics)

    for topic in topics:

        values = []

        for year in years:

            count = year_topic_counts[
                year
            ].get(
                topic,
                0
            )

            values.append(count)

        print(
            f"{topic:<40} : "
            f"{values}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)

    print(
        "        EXAM PATTERN RAG - FORECASTING"
    )

    print("=" * 80)

    # Load questions

    questions = load_questions()

    if not questions:

        return

    print(
        f"\nLoaded {len(questions)} questions."
    )

    # Create topic data

    year_topic_counts = (
        create_year_topic_data(
            questions
        )
    )

    # Display historical data

    display_year_table(
        year_topic_counts
    )

    # Generate forecast

    display_forecast(
        year_topic_counts
    )

    print("=" * 80)

    print(
        "FORECASTING COMPLETE"
    )

    print("=" * 80)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()