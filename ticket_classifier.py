import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

DATA_PATH = "all_tickets_processed_improved_v3.csv"

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]

print(f"Loaded {len(df)} tickets")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# CORRECTED: Use actual column names from your data
TEXT_COL = "Document"
CATEGORY_COL = "Topic_group"

print(f"\nUsing TEXT_COL: {TEXT_COL}")
print(f"Using CATEGORY_COL: {CATEGORY_COL}")

df = df.dropna(subset=[TEXT_COL, CATEGORY_COL]).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

print("\nCleaning text...")
df["clean_text"] = df[TEXT_COL].apply(clean_text)

df["priority"] = df["clean_text"].apply(
    lambda x: "High" if any(w in x for w in ["urgent", "asap", "critical", "down", "broken", "crash", "emergency", "immediately"]) else
    "Medium" if any(w in x for w in ["issue", "problem", "error", "bug", "slow", "delay"]) else "Low"
)

print(f"\nCategories: {df[CATEGORY_COL].nunique()} unique")
print(df[CATEGORY_COL].value_counts())
print("\nPriorities:")
print(df["priority"].value_counts())

X = df["clean_text"]
y = df[CATEGORY_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

plt.figure(figsize=(10, 8))
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nSaved: confusion_matrix.png")

print("\n" + "="*60)
print("TOP KEYWORDS PER CATEGORY")
print("="*60)
feature_names = np.array(vectorizer.get_feature_names_out())
for i, cls in enumerate(model.classes_):
    top_idx = np.argsort(model.coef_[i])[-10:][::-1]
    print(f"{cls}: {', '.join(feature_names[top_idx])}")

summary = f"""
TICKET CLASSIFICATION SUMMARY
------------------------------------------------
Total tickets: {len(df)}
Categories: {df[CATEGORY_COL].nunique()}
Accuracy: {accuracy:.3f}

HOW THIS SYSTEM HELPS BUSINESS:
- Automatically categorizes incoming tickets
- Assigns priority levels (High/Medium/Low)
- Reduces manual sorting time
- Routes tickets to the right team
- Improves response times
"""
print(summary)

with open("classification_summary.txt", "w") as f:
    f.write(summary)

print("\n✅ All done! Files generated: confusion_matrix.png, classification_summary.txt")