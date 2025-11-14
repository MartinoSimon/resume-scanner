from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import io
import json
import requests

# Load environment variables
load_dotenv()

app = FastAPI(title="Resume Scanner API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_CANDIDATES = [
    os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"),
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "https://api-inference.huggingface.co/models/google/flan-t5-large"
]
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Optional, recommended but not required

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


# Hugging Face API Configuration (FREE!)
# Primary candidate + fallbacks (will try each; if none work, use local fallback -> ALWAYS FREE)
HF_API_CANDIDATES = [
    os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"),
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "https://api-inference.huggingface.co/models/google/flan-t5-large"
]
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Optional, recommended but not required

def analyze_with_huggingface(cv_text: str, job_description: str) -> dict:
    """Use Hugging Face's free inference API if available; otherwise fallback locally"""
    
    prompt = f"""You are an expert ATS (Applicant Tracking System). Analyze the CV against the Job Description.

Job Description:
{job_description[:1500]}

Candidate's CV:
{cv_text[:2000]}

Provide your analysis in this EXACT JSON format:
{{
  "match_percentage": <number between 0-100>,
  "missing_keywords": ["keyword1", "keyword2"],
  "summary": "Brief 2-3 sentence summary",
  "tips": ["tip1", "tip2", "tip3"]
}}

Respond ONLY with valid JSON, no other text."""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    # Try each candidate HF endpoint; if all fail, return local fallback (always free)
    for url in HF_API_CANDIDATES:
        headers = {}
        if HF_API_KEY:
            headers["Authorization"] = f"Bearer {HF_API_KEY}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            # If model is loading or gone, try next candidate
            if response.status_code in (503, 410, 404):
                continue
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "") or result[0].get("generated_text", "")
            else:
                generated_text = result.get("generated_text", "") or ""
            
            # Try to extract JSON from the response
            try:
                json_start = generated_text.find("{")
                json_end = generated_text.rfind("}") + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    return {
                        "match_percentage": int(parsed.get("match_percentage", 50)),
                        "missing_keywords": parsed.get("missing_keywords", [])[:10],
                        "summary": parsed.get("summary", "Analysis completed successfully."),
                        "tips": parsed.get("tips", ["Tailor your resume to match job requirements"])[:5]
                    }
                else:
                    # If HF responded but didn't return JSON as expected, fallback to local analysis
                    return create_fallback_analysis(cv_text, job_description)
            except (json.JSONDecodeError, ValueError):
                return create_fallback_analysis(cv_text, job_description)
        
        except requests.exceptions.Timeout:
            # Try next candidate on timeout
            continue
        except requests.exceptions.RequestException:
            # Try next candidate on other network/HTTP errors
            continue
    
    # If no HF endpoint succeeded, always return the free local fallback
    return create_fallback_analysis(cv_text, job_description)


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
    file: UploadFile,
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
        
        # Analyze using Hugging Face (FREE)
        result = analyze_with_huggingface(cv_text, job_description)
        
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
        "ai_provider": "Hugging Face (Mixtral-8x7B)",
        "cost": "100% FREE - No API costs!",
        "endpoints": {
            "/analyze": "POST - Analyze resume against job description"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)