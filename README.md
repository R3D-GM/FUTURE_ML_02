# FUTURE_ML_02 - Support Ticket Classification & Prioritization

## 📌 Project Overview
This project builds an ML system that automatically classifies customer support tickets into 8 categories and assigns priority levels (High/Medium/Low). This helps businesses respond faster, reduce backlog, and improve customer satisfaction.

## 📊 Dataset Used
- **Source:** IT Service Ticket Classification Dataset (Kaggle)
- **Total Tickets:** 47,837
- **Categories:** 8

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- NLTK (text preprocessing)
- Scikit-learn (TF-IDF, Logistic Regression)
- Matplotlib, Seaborn (visualization)

## 📈 Model Performance
- **Accuracy:** 85.2%
- **Best Category:** Purchase (96% precision)
- **Category Breakdown:**
  - Access: 92% precision, 88% recall
  - Administrative Rights: 88% precision, 61% recall
  - HR Support: 86% precision, 87% recall
  - Hardware: 79% precision, 88% recall
  - Internal Project: 93% precision, 82% recall
  - Miscellaneous: 83% precision, 82% recall
  - Purchase: 96% precision, 86% recall
  - Storage: 93% precision, 83% recall

## 📋 How It Works
1. **Text Cleaning:** Lowercasing, punctuation removal
2. **Tokenization:** NLTK word tokenization
3. **Stopword Removal:** Removes common English words
4. **Lemmatization:** Reduces words to base form
5. **Feature Extraction:** TF-IDF vectorization (1-2 grams)
6. **Classification:** Logistic Regression

## 🔮 Priority Assignment
- **High:** Keywords like urgent, asap, critical, down, broken, crash, emergency, immediately
- **Medium:** Keywords like issue, problem, error, bug, slow, delay
- **Low:** Everything else

## 📂 Files in Repository
- `ticket_classifier.py` - Main Python script
- `all_tickets_processed_improved_v3.csv` - Dataset
- `confusion_matrix.png` - Model evaluation
- `classification_summary.txt` - Business insights
- `README.md` - This file
- `requirements.txt` - Dependencies

## 🚀 How to Run
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run: `python ticket_classifier.py`

## 👤 Author
Rediet Girma
Machine Learning Intern - Future Interns"# FUTURE_ML_02" 
