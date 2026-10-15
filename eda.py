"""
eda.py - Exploratory Data Analysis for the AcademicNotesAI fused dataset.
Run: python eda.py      (prints a full EDA report to the console)
"""
import pandas as pd
from data_prep import load_data, NA_TOKEN

pd.set_option("display.width", 200, "display.max_columns", 50)
raw = pd.read_excel("data.xlsx")
df = load_data()

def h(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

h("1. SHAPE & TYPES")
print("rows, columns:", raw.shape)
print(raw.dtypes.value_counts())
print("duplicate rows:", raw.duplicated().sum(), "| duplicate record_id:", raw.record_id.duplicated().sum())

h("2. MISSING DATA (true NaN vs the 'Not Applicable' placeholder)")
na = pd.DataFrame({"true_NaN": raw.isna().sum(),
                   "not_applicable": (raw == NA_TOKEN).sum(),
                   "zero_values": (raw.select_dtypes("number") == 0).sum()}).fillna(0).astype(int)
na["pct_empty"] = ((na.not_applicable + na.zero_values) / len(raw) * 100).round(1)
print(na.sort_values("pct_empty", ascending=False).head(20))
print("\nConstant columns (no information):",
      [c for c in raw.columns if raw[c].nunique() == 1])

h("3. RECORD TYPES (the 7 source files that were fused)")
print(df.record_label.value_counts())

h("4. USERS")
print(df.groupby("user").agg(records=("record_id", "count"),
                             accounts=("user_uid", "nunique")))
print("\nActivity by user:\n", pd.crosstab(df.user, df.activity))

h("5. TIME")
print("range:", df.date_parsed.min(), "->", df.date_parsed.max())
print("records per month:\n", df.month.value_counts().sort_index())
print("records per weekday:\n", df.weekday.value_counts().sort_index())
print("\nhour distribution (records with a clock time only):\n",
      df.hour.dropna().astype(int).value_counts().sort_index())
print("day-part:\n", df.day_part.value_counts())

h("6. TOPICS")
print(df.topic_group.value_counts())
print("\nTopic x Activity:\n", pd.crosstab(df.topic_group, df.activity))

h("7. NUMERIC SUMMARY BY ACTIVITY")
cols = ["prompt_words", "answer_length", "question_length", "message_count", "score", "percentage"]
print(df.groupby("activity")[cols].mean().round(1))

h("8. QUIZ")
q = df[df.is_quiz == 1][["date_parsed", "topic", "difficulty", "score", "total_questions", "percentage"]]
print(q.to_string(index=False))
print("avg %:", q.percentage.mean(), "| pass(>=60) rate:", (q.percentage >= 60).mean())
print(q.groupby("difficulty").percentage.mean())

h("9. SAVED NOTES")
n = df[df.is_saved_note == 1]
print(n[["question_length", "answer_length"]].describe().round(1))
print("corr(question_length, answer_length) =", round(n.question_length.corr(n.answer_length), 3))

h("10. CHAT")
c = df[df.is_chat == 1]
print(c.message_count.value_counts().sort_index())
print("most repeated chat prompts:\n", c.prompt_text.str.lower().str.strip("?. ").value_counts().head(8))

h("11. OUTLIERS (IQR rule)")
for col in ["answer_length", "prompt_words", "message_count"]:
    s = df.loc[df[col] > 0, col]
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    out = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
    print(f"{col}: n={len(s)}, outliers={len(out)}, values={out.tolist()}")

h("12. CORRELATIONS (non-constant numeric columns)")
num = ["prompt_words", "question_length", "answer_length", "message_count",
       "results_count", "score", "total_questions", "percentage"]
print(df[num].corr().round(2))
