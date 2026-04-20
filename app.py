"""
ResumeRadar — AI Resume vs Job Description Match Scorer
=======================================================
Run:
    pip install -r requirements.txt
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import re
import json
import requests
import tempfile
from collections import Counter

# PDF extraction
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# ─── Skill Database ───────────────────────────────────────────────────────────

SKILL_CATEGORIES = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "kotlin",
        "swift", "go", "rust", "ruby", "php", "scala", "r", "matlab",
        "dart", "flutter", "xml"
    ],
    "web": [
        "html", "css", "react", "angular", "vue", "nodejs", "express",
        "django", "flask", "fastapi", "spring", "bootstrap", "tailwind",
        "next.js", "nuxt", "graphql", "rest", "api"
    ],
    "data": [
        "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "tensorflow",
        "pytorch", "keras", "opencv", "nlp", "machine learning", "deep learning",
        "data analysis", "data science", "statistics", "tableau", "power bi",
        "excel", "sql", "nosql", "hadoop", "spark"
    ],
    "mobile": [
        "android", "ios", "react native", "flutter", "kotlin", "swift",
        "firebase", "xml layout", "jetpack compose", "room database",
        "retrofit", "mvvm", "mvp"
    ],
    "cloud": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "jenkins", "github actions", "heroku", "vercel", "netlify",
        "linux", "bash", "devops"
    ],
    "database": [
        "mysql", "postgresql", "mongodb", "sqlite", "redis", "cassandra",
        "dynamodb", "oracle", "firebase", "supabase", "prisma"
    ],
    "tools": [
        "git", "github", "gitlab", "jira", "postman", "figma", "vs code",
        "intellij", "android studio", "xcode", "agile", "scrum"
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "project management",
        "collaboration", "adaptability", "creativity"
    ]
}

# Flatten for easy lookup
ALL_SKILLS = {}
for cat, skills in SKILL_CATEGORIES.items():
    for skill in skills:
        ALL_SKILLS[skill] = cat


# ─── PDF Extractor ────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes):
    """Extract raw text from PDF bytes using PyMuPDF."""
    if not PDF_SUPPORT:
        return None, "PyMuPDF not installed. Run: pip install pymupdf"
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip(), None
    except Exception as e:
        return None, str(e)


# ─── Skill Extractor ──────────────────────────────────────────────────────────

def extract_skills(text):
    """Extract skills from text by matching against skill database."""
    text_lower = text.lower()
    found = {}
    for skill, category in ALL_SKILLS.items():
        # Use word boundary matching for accuracy
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found[skill] = category
    return found


def extract_keywords(text, top_n=30):
    """Extract top TF-IDF keywords from text."""
    try:
        # Clean text
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        # Filter stopwords (basic)
        stopwords = {
            'the','a','an','and','or','but','in','on','at','to','for',
            'of','with','by','from','is','are','was','were','be','been',
            'have','has','had','do','does','did','will','would','could',
            'should','may','might','this','that','these','those','i','we',
            'you','he','she','they','my','our','your','his','her','their',
            'it','its','as','if','so','up','out','no','not','can','all',
            'also','more','over','than','then','when','where','who','how',
            'what','which','into','through','during','before','after','am'
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        freq = Counter(keywords)
        return [word for word, _ in freq.most_common(top_n)]
    except Exception:
        return []


# ─── NLP Matcher ─────────────────────────────────────────────────────────────

def compute_similarity(resume_text, jd_text):
    """Compute cosine similarity between resume and JD using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(sim) * 100, 1)
    except Exception:
        return 0.0


def compute_skill_match(resume_skills, jd_skills):
    """Calculate skill overlap score."""
    if not jd_skills:
        return 100.0
    resume_set = set(resume_skills.keys())
    jd_set = set(jd_skills.keys())
    matched = resume_set & jd_set
    score = (len(matched) / len(jd_set)) * 100
    return round(score, 1)


def compute_keyword_match(resume_text, jd_keywords):
    """Check how many JD keywords appear in resume."""
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    if not jd_keywords:
        return 100.0
    return round((len(matched) / len(jd_keywords)) * 100, 1)


# ─── Claude AI Rewriter ───────────────────────────────────────────────────────

def ai_rewrite_bullets(resume_text, jd_text, missing_skills):
    """
    Use Claude API to rewrite resume bullets to match the job description.
    Falls back to rule-based suggestions if API is unavailable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return rule_based_suggestions(resume_text, jd_text, missing_skills)

    prompt = f"""You are an expert resume writer. Given a resume and a job description, rewrite 3 resume bullet points to better match the job.

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{jd_text[:1500]}

MISSING SKILLS TO INCORPORATE: {', '.join(list(missing_skills)[:8])}

Instructions:
- Extract 3 existing bullet points or achievements from the resume
- Rewrite each one using the job description's language and keywords
- Make them more impactful with numbers/metrics where possible
- Keep them truthful — don't invent experience

Respond ONLY with valid JSON in this exact format:
{{
  "rewrites": [
    {{"original": "original bullet text", "rewritten": "improved version", "reason": "why this is better"}},
    {{"original": "original bullet text", "rewritten": "improved version", "reason": "why this is better"}},
    {{"original": "original bullet text", "rewritten": "improved version", "reason": "why this is better"}}
  ],
  "summary": "One sentence on the biggest gap between this resume and the job"
}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1200,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        raw = data["content"][0]["text"]
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"AI rewrite failed: {e}, using rule-based fallback")
        return rule_based_suggestions(resume_text, jd_text, missing_skills)


def rule_based_suggestions(resume_text, jd_text, missing_skills):
    """Fallback when Claude API is not configured."""
    lines = [l.strip() for l in resume_text.split('\n') if len(l.strip()) > 40]
    bullets = lines[:3] if lines else [
        "Developed software applications",
        "Worked on team projects",
        "Completed coursework and assignments"
    ]

    jd_lower = jd_text.lower()
    action_verbs = {
        "developed": "Engineered", "built": "Architected",
        "worked": "Collaborated", "created": "Designed",
        "made": "Delivered", "helped": "Facilitated",
        "used": "Leveraged", "learned": "Mastered"
    }

    rewrites = []
    for i, bullet in enumerate(bullets[:3]):
        improved = bullet
        for weak, strong in action_verbs.items():
            improved = re.sub(r'\b' + weak + r'\b', strong, improved, flags=re.IGNORECASE)
        if missing_skills and i < len(list(missing_skills)):
            skill = list(missing_skills)[i]
            improved = f"{improved}, incorporating {skill} for enhanced performance"
        rewrites.append({
            "original": bullet,
            "rewritten": improved,
            "reason": f"Stronger action verb + added '{list(missing_skills)[i] if missing_skills else 'relevant keywords'}' from JD"
        })

    return {
        "rewrites": rewrites,
        "summary": f"Your resume lacks {len(missing_skills)} key skills from this JD. Focus on adding: {', '.join(list(missing_skills)[:4])}."
    }


# ─── Experience Extractor ────────────────────────────────────────────────────

def extract_experience_years(text):
    """Rough heuristic to estimate years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
    ]
    for p in patterns:
        match = re.search(p, text.lower())
        if match:
            return int(match.group(1))

    # Count year ranges like 2021-2023
    year_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)', text.lower())
    total = 0
    current_year = 2025
    for start, end in year_ranges:
        end_year = current_year if end in ['present', 'current'] else int(end)
        total += max(0, end_year - int(start))
    return total if total > 0 else None


def extract_education(text):
    """Extract education level from resume."""
    text_lower = text.lower()
    if any(w in text_lower for w in ['ph.d', 'phd', 'doctorate']):
        return "PhD"
    if any(w in text_lower for w in ['m.tech', 'mtech', 'm.e.', 'master', 'mba', 'msc', 'm.s.']):
        return "Masters"
    if any(w in text_lower for w in ['b.tech', 'btech', 'b.e.', 'bachelor', 'bsc', 'b.s.', 'b.a.']):
        return "Bachelors"
    if any(w in text_lower for w in ['diploma', '12th', 'higher secondary']):
        return "Diploma"
    return "Not detected"


# ─── Main Scorer ─────────────────────────────────────────────────────────────

def analyze_resume(resume_text, jd_text):
    """Full analysis pipeline."""

    # 1. Extract skills from both
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    # 2. Matched / missing / extra skills
    resume_set  = set(resume_skills.keys())
    jd_set      = set(jd_skills.keys())
    matched     = sorted(resume_set & jd_set)
    missing     = sorted(jd_set - resume_set)
    extra       = sorted(resume_set - jd_set)

    # 3. Extract JD keywords
    jd_keywords = extract_keywords(jd_text, top_n=25)

    # 4. Scores
    similarity_score  = compute_similarity(resume_text, jd_text)
    skill_score       = compute_skill_match(resume_skills, jd_skills)
    keyword_score     = compute_keyword_match(resume_text, jd_keywords)
    overall_score     = round(
        similarity_score * 0.35 +
        skill_score      * 0.45 +
        keyword_score    * 0.20, 1
    )

    # 5. Category breakdown
    category_scores = {}
    for cat in SKILL_CATEGORIES:
        cat_jd    = {s for s, c in jd_skills.items() if c == cat}
        cat_res   = {s for s, c in resume_skills.items() if c == cat}
        if cat_jd:
            category_scores[cat] = round((len(cat_res & cat_jd) / len(cat_jd)) * 100)
        elif cat_res:
            category_scores[cat] = 100
        else:
            category_scores[cat] = 0

    # 6. AI bullet rewrites
    ai_result = ai_rewrite_bullets(resume_text, jd_text, set(missing))

    # 7. Experience & education
    exp_years = extract_experience_years(resume_text)
    education = extract_education(resume_text)

    # 8. Verdict
    if overall_score >= 80:
        verdict = "Excellent Match"
        verdict_color = "green"
        advice = "Your resume is well-aligned. Apply with confidence!"
    elif overall_score >= 60:
        verdict = "Good Match"
        verdict_color = "blue"
        advice = "A few tweaks will make your application much stronger."
    elif overall_score >= 40:
        verdict = "Partial Match"
        verdict_color = "amber"
        advice = "Significant gaps exist. Add missing skills and tailor your bullets."
    else:
        verdict = "Low Match"
        verdict_color = "red"
        advice = "This role may need different skills. Consider upskilling or targeting better-fit roles."

    return {
        "overall_score":    overall_score,
        "similarity_score": similarity_score,
        "skill_score":      skill_score,
        "keyword_score":    keyword_score,
        "verdict":          verdict,
        "verdict_color":    verdict_color,
        "advice":           advice,
        "matched_skills":   matched,
        "missing_skills":   missing,
        "extra_skills":     extra[:10],
        "jd_keywords":      jd_keywords[:20],
        "category_scores":  category_scores,
        "resume_skill_count": len(resume_skills),
        "jd_skill_count":     len(jd_skills),
        "experience_years":   exp_years,
        "education":          education,
        "ai_rewrites":        ai_result.get("rewrites", []),
        "ai_summary":         ai_result.get("summary", ""),
    }


# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("dashboard.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    POST with:
      - resume_file: PDF file upload  OR  resume_text: plain text
      - jd_text: job description text
    """
    jd_text = request.form.get("jd_text", "").strip()
    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    # Get resume text
    resume_text = ""
    if "resume_file" in request.files:
        file = request.files["resume_file"]
        if file.filename.endswith(".pdf"):
            file_bytes = file.read()
            resume_text, err = extract_text_from_pdf(file_bytes)
            if err:
                return jsonify({"error": f"PDF error: {err}"}), 400
        else:
            resume_text = file.read().decode("utf-8", errors="ignore")
    elif "resume_text" in request.form:
        resume_text = request.form.get("resume_text", "").strip()

    if not resume_text:
        return jsonify({"error": "Resume text or PDF is required"}), 400

    if len(resume_text) < 50:
        return jsonify({"error": "Resume text is too short. Please provide full resume."}), 400

    try:
        result = analyze_resume(resume_text, jd_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "pdf_support": PDF_SUPPORT,
        "ai_enabled": bool(os.environ.get("ANTHROPIC_API_KEY"))
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print("\n" + "="*50)
    print("  🎯 ResumeRadar is running!")
    print(f"  Open: http://localhost:{port}")
    print(f"  PDF Support: {'✅' if PDF_SUPPORT else '❌ install pymupdf'}")
    print(f"  AI Rewrites: {'✅' if os.environ.get('ANTHROPIC_API_KEY') else '⚠ set ANTHROPIC_API_KEY for AI rewrites'}")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
