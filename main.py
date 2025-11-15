from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import io
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from groq import Groq
import logging

# Load environment variables
load_dotenv()

app = FastAPI(title="Resume Scanner API")

# CORS Configuration
# WARNING: Allowing all origins is insecure for production.
# Consider restricting this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf"""
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")

# --- AI Provider Functions ---
# Each function attempts analysis and returns a dict on success or None on failure.


def analyze_with_groq(cv_text: str, job_description: str) -> dict | None:
    """Analyze with Groq API. Returns analysis dict on success, None on failure."""
    if not GROQ_API_KEY:
        logging.info("Groq API key not found, skipping.")
        return None

    logging.info("🚀 Attempting AI analysis with: Groq")
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""You are an expert ATS (Applicant Tracking System). Analyze the CV against the Job Description.

Job Description:
{job_description[:2000]}

Candidate's CV:
{cv_text[:2500]}

Provide your analysis in this EXACT JSON format:
{{
  "match_percentage": <number between 0-100>,
  "missing_keywords": ["keyword1", "keyword2"],
  "summary": "Brief 2-3 sentence summary",
  "tips": ["tip1", "tip2", "tip3"]
}}

Respond ONLY with valid JSON, no other text."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert ATS system that only responds with JSON."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024,
        )
        
        response_text = chat_completion.choices[0].message.content
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start:json_end]
        
        parsed = json.loads(json_str)
        logging.info("✅ AI Success: Groq responded with valid JSON.")
        
        return {
            "ai_provider": "Groq (llama-3.1-8b-instant)",
            "match_percentage": int(parsed.get("match_percentage", 50)),
            "missing_keywords": parsed.get("missing_keywords", [])[:10],
            "summary": parsed.get("summary", "Analysis completed successfully."),
            "tips": parsed.get("tips", ["Tailor your resume to match job requirements"])[:5]
        }
    except Exception as e:
        logging.error(f"🚨 Groq analysis failed: {e}. Falling back to other providers.")
        return None

def analyze_with_gemini(cv_text: str, job_description: str) -> dict | None:
    """Analyze with Google Gemini API. Returns analysis dict on success, None on failure."""
    if not GOOGLE_PROJECT_ID:
        logging.info("Google Project ID not found, skipping Gemini.")
        return None

    logging.info("✨ Attempting AI analysis with: Google Gemini")
    
    # The prompt is the same, so we can reuse it
    prompt = f"""You are an expert ATS (Applicant Tracking System). Analyze the CV against the Job Description.

Job Description:
{job_description[:2000]}

Candidate's CV:
{cv_text[:2500]}

Provide your analysis in this EXACT JSON format:
{{
  "match_percentage": <number between 0-100>,
  "missing_keywords": ["keyword1", "keyword2"],
  "summary": "Brief 2-3 sentence summary",
  "tips": ["tip1", "tip2", "tip3"]
}}

Respond ONLY with valid JSON, no other text."""

    try:
        vertexai.init(project=GOOGLE_PROJECT_ID)
        model = GenerativeModel("gemini-1.5-flash-001")
        
        response = model.generate_content(prompt)
        
        response_text = response.text
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start:json_end]
        
        parsed = json.loads(json_str)
        logging.info("✅ AI Success: Google Gemini responded with valid JSON.")
        
        return {
            "ai_provider": "Google (Gemini 1.5 Flash)",
            "match_percentage": int(parsed.get("match_percentage", 50)),
            "missing_keywords": parsed.get("missing_keywords", [])[:10],
            "summary": parsed.get("summary", "Analysis completed successfully."),
            "tips": parsed.get("tips", ["Tailor your resume to match job requirements"])[:5]
        }
    except Exception as e:
        logging.error(f"🚨 Google Gemini analysis failed: {e}. Falling back to other providers.")
        return None

def create_fallback_analysis(cv_text: str, job_description: str) -> dict:
    """Create a basic keyword-based analysis as fallback"""
    
    # Convert to lowercase for comparison
    cv_lower = cv_text.lower()
    jd_lower = job_description.lower()
    
    # Common tech keywords to check
    keywords = [
        "python", "javascript", "java", "react", "node.js", "sql", "aws", 
        "docker", "kubernetes", "git", "api", "rest", "agile", "scrum",
        "typescript", "mongodb", "postgresql", "redis", "jenkins", "ci/cd"
    ]
    
    # Find keywords in job description
    jd_keywords = [kw for kw in keywords if kw in jd_lower]
    
    # Find missing keywords
    missing = [kw for kw in jd_keywords if kw not in cv_lower]
    
    # Calculate simple match percentage
    if jd_keywords:
        match_pct = int(((len(jd_keywords) - len(missing)) / len(jd_keywords)) * 100)
    else:
        match_pct = 60
    
    return {
        "ai_provider": "Local (keyword analyzer - fallback)",
        "match_percentage": max(40, min(85, match_pct)),
        "missing_keywords": missing[:8],
        "summary": f"Your CV shows relevant experience. Found {len(jd_keywords) - len(missing)} out of {len(jd_keywords)} key skills from the job description.",
        "tips": [
            "Add specific metrics and quantifiable achievements to your experience",
            "Include relevant keywords from the job description naturally throughout your CV",
            "Highlight projects that directly relate to the job requirements",
            "Ensure your CV clearly demonstrates the required technical skills"
        ]
    }

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = Form(...),
    job_description: str = Form(...)
):
    """
    Analyze a resume against a job description using FREE Hugging Face AI
    """
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate job description
    if len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is too short (minimum 20 characters)")
    
    try:
        # Read and extract text from PDF
        file_bytes = await file.read()
        cv_text = extract_text_from_pdf(file_bytes)
        
        # Validate extracted text
        if len(cv_text) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract sufficient text from PDF. Please ensure the PDF contains readable text."
            )
        
        # --- AI Analysis Chain ---
        # 1. Try Groq (fastest)
        result = analyze_with_groq(cv_text, job_description)
        # 2. If Groq fails, try Google Gemini
        if not result:
            result = analyze_with_gemini(cv_text, job_description)
        # 3. If all AI providers fail, use the local fallback
        if not result:
            logging.warning("🤷 All AI providers failed. Using local fallback analysis.")
            result = create_fallback_analysis(cv_text, job_description)
        
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "AI Resume Scanner API - FREE Edition",
        "version": "2.0.0",
        "ai_providers": ["Groq", "Google Gemini", "Local Fallback"],
        "cost": "100% FREE - No API costs!",
        "endpoints": {
            "/analyze": "POST - Analyze resume against job description"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)