import pandas as pd
import joblib
import re
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', str(text).lower())
    doc = nlp(text[:50000]) 
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

if not os.path.exists('data'):
    os.makedirs('data')

csv_path = os.path.join("data", "resumes.csv")

if os.path.exists(csv_path):
    print(f"Loading real data from {csv_path}...")
    df = pd.read_csv(csv_path)
    text_col = 'Resume' if 'Resume' in df.columns else df.columns[0]
    raw_corpus = df[text_col].tolist()
else:
    print("Real CSV not found. Generating synthetic training data...")
    raw_corpus = [
        "python java sql hadoop spark data engineer pipeline etl" * 10,
        "react nodejs javascript html css frontend developer redux" * 10,
        "aws azure devops docker kubernetes ci cd jenkins terraform" * 10,
        "machine learning python scikit learn tensorflow keras ai data scientist" * 10,
        "project manager agile scrum jira leadership business analysis" * 10
    ] * 20

corpus = [clean_text(text) for text in raw_corpus]

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
vectorizer.fit(corpus)

joblib.dump(vectorizer, "data/resume_model.pkl")
print(f"Model trained on {len(corpus)} resumes and saved successfully.")