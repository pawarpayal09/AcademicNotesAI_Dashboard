"""
data_prep.py  -  Loading + cleaning + feature engineering for the
AcademicNotesAI fused dataset.  Imported by eda.py and dashboard.py.
"""
import re
import numpy as np
import pandas as pd

DATA_FILE = "data.xlsx"          # keep the Excel file in the same folder
NA_TOKEN = "Not Applicable"

# --------------------------------------------------------------------------
# Topic classifier: the raw 'topic' column is mostly "Not Applicable", so we
# derive a clean 'topic_group' from every text field of the record.
# Order matters: the first matching rule wins.
# --------------------------------------------------------------------------
TOPIC_RULES = [
    ("Data Science",      r"data science"),
    ("Machine Learning",  r"machine learning|unsupervised|\bml\b"),
    ("Cloud Computing",   r"cloud|iaas"),
    ("Cyber Security",    r"cyber"),
    ("DBMS",              r"dbms|database|normali[sz]ation"),
    ("Operating Systems", r"operating system"),
    ("Data Structures",   r"data stu"),
    ("Python",            r"python"),
    ("AI & LLMs",         r"\bai\b|artificial|large language|\bllm\b"),
]
TEXT_COLS = ["topic", "chat_title", "question", "description",
             "instruction", "image_name", "sources"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _classify(row):
    parts = []
    for c in TEXT_COLS:
        v = str(row[c])
        if v not in (NA_TOKEN, "Academic Notes", "nan"):
            parts.append(v)
    text = re.sub(r"[-_.]", " ", " ".join(parts).lower())
    for name, pat in TOPIC_RULES:
        if re.search(pat, text):
            return name
    return "General / Unspecified"


def load_data(path=DATA_FILE):
    df = pd.read_excel(path)

    # 1. Dates: three different formats were fused together -> parse 'mixed'
    df["date_parsed"] = pd.to_datetime(
        df["date"].astype(str).str.replace(" UTC", "", regex=False),
        format="mixed", errors="coerce")
    df["has_time"] = df["date"].astype(str).str.len() > 10

    # 2. 'Not Applicable' placeholders -> real missing values (for EDA only)
    # 3. Friendly labels
    df["activity"] = df["activity_type"].replace({
        "chat": "AI Chat", "youtube": "YouTube Search", "image": "Image Study",
        "quiz": "Quiz", "saved_note": "Saved Note", "user": "Sign-up"})
    df["record_label"] = df["record_type"].str.replace("_", " ").str.title()

    # 4. User identity: same person has several UIDs -> use name where known
    uid2name = (df[df["user_name"] != NA_TOKEN]
                .drop_duplicates("user_uid").set_index("user_uid")["user_name"])
    df["user"] = df["user_uid"].map(uid2name).fillna("Unlinked account")
    df["user_short"] = df["user"].str.split().str[0]

    # 5. Feature engineering
    df["topic_group"] = df.apply(_classify, axis=1)

    # primary text of the record (what the user typed / asked)
    def prompt(r):
        for c in ["question", "chat_title", "description", "instruction", "topic"]:
            v = str(r[c])
            if v not in (NA_TOKEN, "Academic Notes", "nan"):
                return v
        return ""
    df["prompt_text"] = df.apply(prompt, axis=1)
    df["prompt_words"] = df["prompt_text"].str.split().str.len().fillna(0).astype(int)
    df["prompt_chars"] = df["prompt_text"].str.len()

    # time features (hour is only reliable when the timestamp has a time part)
    df["weekday"] = pd.Categorical(df["date_parsed"].dt.day_name().str[:3],
                                   categories=WEEKDAYS, ordered=True)
    df["month"] = df["date_parsed"].dt.to_period("M").astype(str)
    df["week"] = df["date_parsed"].dt.to_period("W").dt.start_time
    df["hour"] = np.where(df["has_time"], df["activity_hour"], np.nan)
    df["day_part"] = pd.cut(df["hour"], [-1, 5, 11, 17, 23],
                            labels=["Night (0-5)", "Morning (6-11)",
                                    "Afternoon (12-17)", "Evening (18-23)"])
    # quiz extras
    df["quiz_pct"] = np.where(df["is_quiz"] == 1, df["percentage"], np.nan)
    return df


if __name__ == "__main__":
    d = load_data()
    print(d.shape)
    print(d["topic_group"].value_counts())
    print(d["user"].value_counts())
    print(d[["record_label", "topic_group"]].value_counts().head(40))
    print(d["week"].nunique(), d["date_parsed"].isna().sum())
