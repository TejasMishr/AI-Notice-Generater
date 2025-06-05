import os
import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Gemini model to use (change to your own fine-tuned if needed)
MODEL_NAME = "gemma-3-12b-it"

# Initialize FastAPI app
app = FastAPI(
    title="School Notice Generator",
    description="API for generating formal school notices using Gemini LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["serp.indigle.com","http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class NoticeRequest(BaseModel):
    school_name: str = Field(default="Indigle Public School", description="Name of the school")
    notice_type: str = Field(default="Annual Function", description="Type of notice")
    event_date: Optional[str] = Field(default=None, description="Event date (e.g. 'June 20, 2025')")
    key_details: str = Field(default="Annual cultural program and prize distribution", description="Key details or summary")
    recipient: Optional[str] = Field(default="All Students and Parents", description="Intended recipients")
    venue: Optional[str] = Field(default="School Auditorium", description="Venue for the event")
    time: Optional[str] = Field(default="5:00 PM", description="Event time")
    contact_info: Optional[str] = Field(default="office@indigle.edu | (555) 987-6543", description="Contact information")
    signature_title: str = Field(default="Principal", description="Authority signing the notice")

def generate_raw_notice(input_fields: dict, model: str = MODEL_NAME) -> str:
    """
    Generate a formal school notice in HTML using Gemini/Gemma LLM.
    """
    system_instructions = (
        "You are a professional administrative assistant for a school. Generate a polished, formal school notice "
        "in the following EXACT raw message format:\n\n"
        "<p><strong>[SCHOOL NAME] [NOTICE TYPE] NOTICE</strong></p>\n"
        "<p><br></p>\n"
        "<p>[Body content]</p>\n\n"
        "RULES:\n"
        "1. Use ONLY the specified HTML tags: <p>, <strong>, <br> (no other tags)\n"
        "2. First line must be: <p><strong>[SCHOOL NAME] [NOTICE TYPE] NOTICE</strong></p>\n"
        "3. Second line must be: <p><br></p>\n"
        "4. Content must include:\n"
        "   - Date of issue\n"
        "   - Recipient\n"
        "   - Event date, time, venue\n"
        "   - Key details\n"
        "   - Contact information\n"
        "   - Signature block\n"
        "5. Keep entire notice under 120 words\n"
        "6. Use formal academic tone\n"
        "7. Replace missing information with [Placeholder]\n"
        "8. Output ONLY the raw message text with NO additional commentary\n"
        "9. Format all content in paragraph tags (<p>content</p>)\n"
        "10. Use line breaks (<br>) only within paragraphs when absolutely necessary"
    )

    today = datetime.date.today().strftime("%B %d, %Y")
    # Default event_date to two weeks from today if not provided
    event_date = input_fields.get('event_date') or (datetime.date.today() + datetime.timedelta(days=14)).strftime("%B %d, %Y")

    # Helper to replace missing info with [Placeholder]
    def safe_get(key, default="[Placeholder]"):
        val = input_fields.get(key)
        return val if val and str(val).strip() else default

    input_block = (
        f"SCHOOL NOTICE DETAILS:\n"
        f"- School Name: {safe_get('school_name')}\n"
        f"- Notice Type: {safe_get('notice_type')} Notice\n"
        f"- Key Details: {safe_get('key_details')}\n"
        f"- Event Date: {event_date}\n"
        f"- Event Time: {safe_get('time')}\n"
        f"- Venue: {safe_get('venue')}\n"
        f"- Recipients: {safe_get('recipient')}\n"
        f"- Contact: {safe_get('contact_info')}\n"
        f"- Issued On: {today}\n"
        f"- Authority: {safe_get('signature_title')}"
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key missing. Set GEMINI_API_KEY in environment variables."
        )

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(role="user", parts=[types.Part(text=system_instructions)]),
        types.Content(role="user", parts=[types.Part(text=input_block)]),
    ]
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        response_mime_type="text/plain",
        temperature=0.8,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="No response generated by the Gemini model."
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error communicating with Gemini: {str(e)}"
        )

@app.post("/generate-notice", summary="Generate a school notice", response_model=dict)
async def create_notice(notice_request: NoticeRequest):
    """
    Generate a formal, HTML-formatted school notice.
    """
    notice = generate_raw_notice(notice_request.dict())
    return {"notice": notice}

@app.get("/", summary="Health check")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "School Notice Generator API is running."}
