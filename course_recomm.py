"""
Course Recommendation System  (Field Project)
----------------------------------------------
Frontend : Streamlit (glassmorphism CSS, gradient background)
Backend  : SQLite3  (users.db)
ML Engine: bag-of-words (count vectors) + Cosine Similarity, written by hand (no scikit-learn)
Data     : pandas  +  data/course.csv

Run:  streamlit run course_recomm.py

Folder layout:
    course_recommend/
    |-- course_recomm.py
    |-- users.db            (created automatically)
    `-- data/
        `-- course.csv      (coursetitle, subject, level, num_lectures, content_duration)
"""

import html
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------- PATHS (relative, so it runs on any PC) -----------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "course.csv"
DB_PATH = BASE_DIR / "users.db"

# ----------------- DROPDOWN OPTIONS -----------------
FIELDS = ["Computer Science", "Data Science", "Business", "Arts", "Engineering", "Other"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]
DURATIONS = ["<1 month", "1-3 months", "3-6 months", "6+ months"]
MODES = ["Online", "Offline", "Hybrid"]

# Content-hours range that each duration preference is matched against.
# (Assumption - adjust to suit your dataset.)
DURATION_HOURS = {
    "<1 month": (0, 10),
    "1-3 months": (10, 30),
    "3-6 months": (30, 60),
    "6+ months": (60, float("inf")),
}

# Extra words added to the search term so each field matches more course titles.
FIELD_KEYWORDS = {
    "Computer Science": "computer programming python java",
    "Data Science": "data science machine learning python",
    "Business": "business management marketing finance",
    "Arts": "arts design drawing music photography",
    "Engineering": "engineering electrical mechanical civil",
    "Other": "",
}
# The Udemy-style dataset calls the top level "Expert".
LEVEL_KEYWORDS = {"Beginner": "beginner", "Intermediate": "intermediate", "Advanced": "advanced expert"}


# ----------------- DATABASE -----------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_users_table():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            field_of_study TEXT,
            level TEXT,
            duration TEXT,
            mode TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_user(name, field_of_study, level, duration, mode):
    conn = get_connection()
    conn.execute(
        "INSERT INTO users (name, field_of_study, level, duration, mode) VALUES (?, ?, ?, ?, ?)",
        (name, field_of_study, level, duration, mode),
    )
    conn.commit()
    conn.close()


# ----------------- LOAD & PREPARE COURSES -----------------
def parse_hours(value):
    """Turn values like '31.5 hours', '40 Hours', '4 Weeks', '90 mins' into hours."""
    text = str(value).lower()
    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return float("nan")
    number = float(match.group())
    if "week" in text:
        return number * 5      # assume ~5 study hours per week
    if "min" in text:
        return number / 60
    return number


@st.cache_data
def load_courses():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept the column names used in the report as well as the code
    df = df.rename(columns={"lectures": "num_lectures", "duration": "content_duration"})

    required = ["coursetitle", "subject", "level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"course.csv is missing column(s): {', '.join(missing)}")

    for col in ("num_lectures", "content_duration"):
        if col not in df.columns:
            df[col] = ""

    df[required] = df[required].fillna("").astype(str)
    df = df.drop_duplicates(subset="coursetitle").reset_index(drop=True)

    # Feature engineering (as in the report)
    df["combined_features"] = df["coursetitle"] + " " + df["subject"] + " " + df["level"]
    df["hours"] = df["content_duration"].apply(parse_hours)
    return df


# ----------------- TEXT VECTORIZER (same idea as CountVectorizer) -----------------
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "to", "with", "your", "you", "all", "level",
}


def tokenize(text):
    """Lower-case words, removing English stop words."""
    words = re.findall(r"[a-z0-9+#]+", str(text).lower())
    return [w for w in words if w not in STOP_WORDS]


def count_vector(text):
    """Bag-of-words: {word: how many times it appears}."""
    return Counter(tokenize(text))


def cosine(a, b):
    """Cosine similarity between two count vectors."""
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(word, 0) for word, count in a.items())
    norm = math.sqrt(sum(c * c for c in a.values())) * math.sqrt(sum(c * c for c in b.values()))
    return dot / norm if norm else 0.0


@st.cache_resource
def build_model(features: tuple):
    """Vectorize every course once instead of on every button click."""
    return [count_vector(text) for text in features]


# ----------------- RECOMMENDER -----------------
def recommend(df, field_of_study, level, duration, num_recs):
    vectors = build_model(tuple(df["combined_features"]))

    search_term = f"{field_of_study} {level} {FIELD_KEYWORDS[field_of_study]} {LEVEL_KEYWORDS[level]}"
    search_vector = count_vector(search_term)

    scored = df.copy()
    scored["similarity"] = [cosine(search_vector, v) for v in vectors]

    # Small bonus (not a hard filter) for courses that fit the preferred duration
    low, high = DURATION_HOURS[duration]
    fits = scored["hours"].between(low, high, inclusive="left")
    scored["score"] = scored["similarity"] + fits.astype(float) * 0.05

    return scored.sort_values("score", ascending=False).head(num_recs)


def format_duration(value):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "N/A"
    # Add "hours" only when the value is a bare number (avoids "4 Weeks hours")
    return f"{text} hours" if re.fullmatch(r"\d+(\.\d+)?", text) else text


# ----------------- BACKGROUND & UI STYLE -----------------
def apply_custom_style():
    st.markdown(
        """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1, h2, h3, h4, label, p {
        color: white !important;
    }
    div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.12);
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(8px);
    }
    button[kind="primary"] {
        background-color: #ff9800;
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config(page_title="Course Recommender", layout="centered")
    apply_custom_style()

    st.title("🎓 Course Recommendation System")

    create_users_table()

    try:
        df = load_courses()
    except FileNotFoundError:
        st.error(f"course.csv not found. Place it at: {CSV_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Could not load course.csv: {e}")
        st.stop()

    st.subheader("Welcome! Fill in your details to get recommendations")

    name = st.text_input("Your Name")
    field_of_study = st.selectbox("Field of Study", FIELDS)
    level = st.selectbox("Level", LEVELS)
    duration = st.selectbox("Duration Preference", DURATIONS)
    mode = st.radio("Mode of Study", MODES)
    num_recs = st.slider("Number of Recommendations", 1, 20, 5)

    if st.button("Recommend Courses", type="primary"):
        if not name.strip():
            st.warning("Please enter your name")
            return

        save_user(name.strip(), field_of_study, level, duration, mode)
        st.success(f"Thanks {name.strip()}! Here are your course recommendations:")

        recommendations = recommend(df, field_of_study, level, duration, num_recs)

        if recommendations["similarity"].max() == 0:
            st.info("No close matches found - showing the most relevant courses available.")

        st.subheader("📚 Recommended Courses")
        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="font-size:13px; font-style:italic;">
                    📘 <b>{html.escape(row['coursetitle'])}</b><br>
                    Subject: {html.escape(row['subject'])}<br>
                    Level: {html.escape(row['level'])}<br>
                    Number of Lectures: {html.escape(str(row['num_lectures']))}<br>
                    Content Duration: {html.escape(format_duration(row['content_duration']))}<br>
                    Match Score: {row['similarity']:.2f}
                    <hr>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
