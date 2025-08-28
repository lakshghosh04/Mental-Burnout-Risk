import pandas as pd, numpy as np, re

sm = pd.read_csv("data/Social Media and Mental Health Dataset.csv")
sl = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

def num_from_text(x):
    if pd.isna(x): return np.nan
    m = re.search(r"\d+(\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

sm2 = pd.DataFrame({
    "age": pd.to_numeric(sm["1. What is your age?"], errors="coerce"),
    "gender": sm["2. Gender"].astype(str),
    "hours_social": sm["8. What is the average time you spend on social media every day?"].apply(num_from_text),
    "sleep_hours": np.nan,
    "work_hours": np.nan
})

for col in [
    "13. On a scale of 1 to 5, how much are you bothered by worries?",
    "18. How often do you feel depressed or down?",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?"
]:
    sm[col] = pd.to_numeric(sm[col], errors="coerce")

sm2["target"] = (
    (sm["13. On a scale of 1 to 5, how much are you bothered by worries?"] >= 4) |
    (sm["18. How often do you feel depressed or down?"] >= 4) |
    (sm["20. On a scale of 1 to 5, how often do you face issues regarding sleep?"] >= 4)
).astype(int)

sl2 = pd.DataFrame({
    "age": pd.to_numeric(sl["Age"], errors="coerce"),
    "gender": sl["Gender"].astype(str),
    "hours_social": np.nan,
    "sleep_hours": pd.to_numeric(sl["Sleep Duration"], errors="coerce"),
    "work_hours": pd.to_numeric(sl["Work Hours"], errors="coerce") if "Work Hours" in sl.columns else np.nan,
    "target": pd.to_numeric(sl["Stress Level"], errors="coerce").apply(lambda x: 1 if x>=7 else 0)
})

df = pd.concat([sm2, sl2], ignore_index=True)
df = df.dropna(subset=["target"])
df.to_csv("data/burnout_unified.csv", index=False)
print("Saved data/burnout_unified.csv", df.shape)
