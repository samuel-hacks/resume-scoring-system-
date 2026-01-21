import streamlit as st
import pdfplumber
import joblib
import re
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

st.set_page_config(page_title="Resume Auditor Pro", layout="wide")

st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #007bff; color: white; font-size: 20px; font-weight: bold;
        border-radius: 10px; padding: 15px 30px; width: 100%; border: none;
    }
    div.stButton > button:hover { background-color: #0056b3; color: white; }
    .main-score { font-size: 60px; font-weight: 900; color: #007bff; text-align: center; line-height: 1; }
    .score-label { font-size: 14px; text-transform: uppercase; color: #6c757d; text-align: center; letter-spacing: 1px; }
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #dee2e6; }
    .metric-val { font-size: 24px; font-weight: bold; color: #212529; }
    .metric-title { font-size: 12px; color: #6c757d; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

SKILL_DB = {
    "AI Engineer": "python pytorch tensorflow scikit-learn nlp computer vision deep learning machine learning sql aws azure google cloud hugging face transformers rag generative ai",
    "Data Scientist": "python r sql pandas numpy matplotlib seaborn tableau powerbi statistics probability modeling a/b testing machine learning data cleaning big data spark",
    "Full Stack Developer": "javascript typescript react angular vue nodejs express django flask html css sql mongodb postgresql docker aws git rest api graphql",
    "DevOps Engineer": "docker kubernetes jenkins gitlab ci cd terraform ansible aws azure linux bash scripting networking monitoring prometheus grafana",
    "Cybersecurity Analyst": "network security penetration testing ethical hacking firewall siem splunk wireshark linux python cryptography risk assessment compliance iso 27001",
    "Product Manager": "agile scrum jira user stories roadmap product lifecycle stakeholder management market research data analysis leadership communication",
}

@st.cache_resource
def load_resources():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    if os.path.exists("data/resume_model.pkl"):
        vectorizer = joblib.load("data/resume_model.pkl")
    else:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        vectorizer.fit(list(SKILL_DB.values()) * 10) 
    
    return nlp, vectorizer

nlp, vectorizer = load_resources()

def clean_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    doc = nlp(text[:200000]) 
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def extract_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            return " ".join([p.extract_text() or "" for p in pdf.pages])
    except:
        return ""

def get_keywords(text):
    doc = nlp(text.lower())
    return set([token.text for token in doc if not token.is_stop and not token.is_punct])

def score_resume(text, target_role_text):
    clean_jd = clean_text(target_role_text)
    clean_resume = clean_text(text)
    
    vectors = vectorizer.transform([clean_jd, clean_resume])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    semantic_score = min(cosine_sim * 350, 100) 

    jd_keywords = get_keywords(target_role_text)
    resume_keywords = get_keywords(text)
    
    matched_skills = jd_keywords.intersection(resume_keywords)
    missing_skills = jd_keywords - resume_keywords
    
    if len(jd_keywords) > 0:
        skill_score = (len(matched_skills) / len(jd_keywords)) * 100
    else:
        skill_score = 0
        
    impact_matches = re.findall(r'(\d+%|\$\d+|\d+\+ years|reduced by \d+|improved by \d+)', text.lower())
    impact_score = min(len(impact_matches) * 15, 100)

    final_score = (semantic_score * 0.4) + (skill_score * 0.4) + (impact_score * 0.2)
    
    return {
        "final_score": round(final_score, 1),
        "semantic_score": round(semantic_score, 1),
        "skill_score": round(skill_score, 1),
        "impact_score": round(impact_score, 1),
        "matched": list(matched_skills),
        "missing": list(missing_skills),
        "impact_count": len(impact_matches)
    }

st.title("Resume Scoring System")

selected_role = st.selectbox("Select Target Role", list(SKILL_DB.keys()) + ["Custom"])

if selected_role == "Custom":
    target_skills_text = st.text_area("Enter Target Skills (comma-separated)")
else:
    target_skills_text = SKILL_DB[selected_role]

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

st.divider()

if uploaded_files and st.button(f"Analyze {len(uploaded_files)} Candidates"):
    
    progress = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        text = extract_text(file)
        if not text:
            st.error(f"Could not read {file.name}")
            continue

        result = score_resume(text, target_skills_text)
        
        with st.expander(f"{file.name} | Score: {result['final_score']}%", expanded=True):
            
            col_main, col_details = st.columns([1, 3])
            
            with col_main:
                st.markdown(f"<div class='main-score'>{result['final_score']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='score-label'>Fit Score</div>", unsafe_allow_html=True)
                
                if result['final_score'] > 75:
                    st.success("Strong Candidate")
                elif result['final_score'] > 50:
                    st.warning("Potential Candidate")
                else:
                    st.error("Weak Match")

            with col_details:
                st.subheader("Score Breakdown")
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{result['skill_score']}%</div>
                        <div class="metric-title">Skill Match</div>
                    </div>""", unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{result['semantic_score']}%</div>
                        <div class="metric-title">Role Relevance</div>
                    </div>""", unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{result['impact_score']}%</div>
                        <div class="metric-title">Project Value</div>
                    </div>""", unsafe_allow_html=True)
                    
            st.divider()
            
            tab1, tab2 = st.tabs(["Matched Skills", "Missing Skills"])
            
            with tab1:
                if result['matched']:
                    st.write("Found in resume:")
                    st.markdown(" ".join([f"`{s}`" for s in result['matched']]))
                else:
                    st.warning("No specific keywords matched.")
            
            with tab2:
                if result['missing']:
                    st.write("Not found in resume:")
                    st.markdown(" ".join([f"`{s}`" for s in result['missing']]))
                else:
                    st.success("All target skills present!")

        progress.progress((idx + 1) / len(uploaded_files))