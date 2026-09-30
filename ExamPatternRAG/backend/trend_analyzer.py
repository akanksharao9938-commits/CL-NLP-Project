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

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:

        questions = json.load(file)

    return questions


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

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
# YEAR-WISE ANALYSIS
# ============================================================

def analyze_years(questions):

    year_counts = Counter()

    for question in questions:

        year = question.get("year")

        if year:

            year_counts[year] += 1

    return dict(sorted(year_counts.items()))


# ============================================================
# TOPIC FREQUENCY
# ============================================================

def analyze_topics(questions):

    topic_counts = Counter()

    for question in questions:

        question_text = question.get("question", "")

        topics = classify_question(question_text)

        for topic in topics:

            topic_counts[topic] += 1

    return topic_counts


# ============================================================
# YEAR + TOPIC ANALYSIS
# ============================================================

def analyze_year_topic(questions):

    year_topic_counts = defaultdict(Counter)

    for question in questions:

        year = question.get("year")

        question_text = question.get("question", "")

        topics = classify_question(question_text)

        for topic in topics:

            year_topic_counts[year][topic] += 1

    return year_topic_counts


# ============================================================
# DISPLAY YEAR STATISTICS
# ============================================================

def display_year_statistics(year_counts):

    print("\n" + "=" * 70)

    print("YEAR-WISE QUESTION DISTRIBUTION")

    print("=" * 70)

    total = sum(year_counts.values())

    for year, count in year_counts.items():

        percentage = (count / total) * 100

        print(
            f"{year}: {count} questions ({percentage:.1f}%)"
        )

    print("-" * 70)

    print(f"TOTAL QUESTIONS: {total}")


# ============================================================
# DISPLAY TOPIC STATISTICS
# ============================================================

def display_topic_statistics(topic_counts):

    print("\n" + "=" * 70)

    print("MOST FREQUENT TOPICS")

    print("=" * 70)

    if not topic_counts:

        print("No topics detected.")

        return

    for rank, (topic, count) in enumerate(
        topic_counts.most_common(),
        start=1
    ):

        print(
            f"{rank:2d}. {topic:<40} {count}"
        )


# ============================================================
# DISPLAY YEAR-WISE TOPICS
# ============================================================

def display_year_topic_statistics(year_topic_counts):

    print("\n" + "=" * 70)

    print("YEAR-WISE TOPIC DISTRIBUTION")

    print("=" * 70)

    for year in sorted(year_topic_counts):

        print(f"\nYEAR {year}")

        print("-" * 50)

        topics = year_topic_counts[year]

        if not topics:

            print("No topics detected.")

            continue

        for topic, count in topics.most_common():

            print(
                f"{topic:<40} {count}"
            )


# ============================================================
# FIND RECURRING TOPICS
# ============================================================

def find_recurring_topics(year_topic_counts):

    topic_years = defaultdict(set)

    for year, topics in year_topic_counts.items():

        for topic in topics:

            topic_years[topic].add(year)

    recurring = []

    for topic, years in topic_years.items():

        if len(years) >= 2:

            recurring.append(
                (
                    topic,
                    len(years),
                    sorted(years)
                )
            )

    recurring.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return recurring


# ============================================================
# DISPLAY RECURRING TOPICS
# ============================================================

def display_recurring_topics(recurring):

    print("\n" + "=" * 70)

    print("RECURRING TOPICS")

    print("=" * 70)

    if not recurring:

        print("No recurring topics detected.")

        return

    for rank, (topic, year_count, years) in enumerate(
        recurring,
        start=1
    ):

        years_text = ", ".join(
            str(year) for year in years
        )

        print(
            f"{rank:2d}. {topic:<40} "
            f"{year_count} years -> {years_text}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print("       EXAM PATTERN RAG - TREND ANALYZER")

    print("=" * 70)

    questions = load_questions()

    if not questions:

        return

    print(f"\nLoaded {len(questions)} questions.")

    # 1. Year-wise analysis

    year_counts = analyze_years(questions)

    display_year_statistics(year_counts)

    # 2. Topic frequency

    topic_counts = analyze_topics(questions)

    display_topic_statistics(topic_counts)

    # 3. Year + topic analysis

    year_topic_counts = analyze_year_topic(questions)

    display_year_topic_statistics(
        year_topic_counts
    )

    # 4. Recurring topics

    recurring_topics = find_recurring_topics(
        year_topic_counts
    )

    display_recurring_topics(
        recurring_topics
    )

    print("\n" + "=" * 70)

    print("TREND ANALYSIS COMPLETE")

    print("=" * 70)


if __name__ == "__main__":

    main()