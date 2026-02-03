import time
import logging
from typing import Optional
import cohere
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_score(resume_text, job_description):
    vect = CountVectorizer().fit_transform([resume_text, job_description])
    score = cosine_similarity(vect)[0][1]
    return round(score * 100, 2)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "command-a-03-2025"
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

def analyze_resume_with_llm(
    resume_text: str,
    job_role: str,
    job_description: str,
    cohere_client: cohere.Client
) -> Optional[str]:
    """
    Analyze a resume against a job role and description using Cohere's Command R+ model.
    """
    # Input validation
    if not resume_text.strip():
        raise ValueError("Resume text cannot be empty")
    if not job_role.strip():
        raise ValueError("Job role cannot be empty")

    # Prompt construction
    prompt = f"""
    As an expert resume reviewer, analyze this resume for the target job role.

    JOB ROLE: {job_role}
    JOB DESCRIPTION: {job_description if job_description else 'Not provided'}

    RESUME CONTENT:
    {resume_text[:8000]}

    Provide structured feedback in these sections:

    1. Missing Skills
    2. Formatting Improvements  
    3. Content Suggestions
    4. Experience Tailoring
    5. Overall Recommendations

    Be specific and actionable. Use bullet points.
    """

    # Retry logic for API calls
    for attempt in range(MAX_RETRIES):
        try:
            response = cohere_client.chat(
                message=prompt,
                model=DEFAULT_MODEL
            )
            return response.text
        except Exception as e:
            logger.warning(f"Cohere API call failed (attempt {attempt + 1}): {str(e)}")
            time.sleep(RETRY_DELAY)
    

    raise Exception("Failed to analyze resume after multiple attempts.")

