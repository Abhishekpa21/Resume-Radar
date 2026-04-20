# 🎯 ResumeRadar — AI Resume Match Analyzer

> Upload resume + paste job description → get match score, missing skills, and AI-rewritten bullets.

---

## Quick Start (VS Code)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set Anthropic API key for AI rewrites
set ANTHROPIC_API_KEY=your_key_here        # Windows
export ANTHROPIC_API_KEY=your_key_here     # Mac/Linux

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## Project Structure

```
resumeradar/
├── app.py           ← Flask backend (NLP + scoring + AI)
├── dashboard.html   ← Frontend (served by Flask)
├── requirements.txt ← pip dependencies
└── README.md
```

---

## Features

| Feature | Description |
|---------|-------------|
| PDF Upload | Extracts text from uploaded resume PDFs |
| Skill Matching | Matches 100+ skills across 8 categories |
| Match Score | Overall % score (NLP similarity + skill + keyword) |
| Missing Skills | Skills the JD needs that your resume lacks |
| Category Breakdown | Bar chart: programming, web, data, cloud, etc. |
| AI Rewrites | Claude rewrites 3 bullets using JD language |
| Keyword Guide | Copy these keywords into your resume |

---

## How Scoring Works

```
Overall Score = (Similarity × 35%) + (Skill Match × 45%) + (Keyword Match × 20%)
```

- **Similarity** — TF-IDF cosine similarity between resume and JD text
- **Skill Match** — % of JD skills found in resume
- **Keyword Match** — % of JD top keywords appearing in resume

---

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serves dashboard.html |
| `/api/analyze` | POST | Runs full analysis, returns JSON |
| `/api/health` | GET | Check if PDF + AI are enabled |

---

## Deploy to Render

```bash
# Add Procfile with:
web: gunicorn app:app

# Set env var in Render dashboard:
ANTHROPIC_API_KEY = your_key_here

# Push to GitHub → connect Render → deploy
```

---

## CV Description

**ResumeRadar — AI-Powered Resume Match Analyzer**
- Built NLP pipeline using TF-IDF + cosine similarity to score resume-JD alignment
- Integrated Claude AI API to intelligently rewrite resume bullets for specific job roles
- Engineered skill extraction system covering 100+ skills across 8 technical categories
- Designed full-stack Flask app with PDF parsing, real-time scoring, and Chart.js visualizations
