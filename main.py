import os
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

app = FastAPI(
    title="School Notice Generator",
    description="API for generating formal school notices using Gemini LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://serp.indigle.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoticeRequest(BaseModel):
    notice_type: str = Field(default="Annual Function", description="Type of notice")
    event_date: Optional[str] = Field(
        default=None,
        description="Event date (e.g. 'June 20, 2025'); if omitted, output placeholder"
    )
    key_details: str = Field(
        default="Annual cultural program and prize distribution",
        description="Key details or summary"
    )
    recipient: Optional[str] = Field(
        default=None,
        description="Intended recipients (e.g. 'All Students and Parents')"
    )
    venue: Optional[str] = Field(
        default=None,
        description="Venue for the event (e.g. 'School Auditorium')"
    )
    time: Optional[str] = Field(
        default=None,
        description="Event time (e.g. '5:00 PM')"
    )
    contact_info: Optional[str] = Field(
        default=None,
        description="Contact information (e.g. 'office@school.edu | (555) 123-4567')"
    )
    signature_title: str = Field(default="Principal", description="Authority signing the notice")

def generate_raw_notice(input_fields: dict, model: str = MODEL_NAME) -> str:

    # Generating a formal school notice in HTML using Gemini/Gemma LLM. If the user has not provided event_date or contact_info, the output will show <strong>[Placeholder]</strong> instead of inventing dates.


    system_instructions = (
        "You are a professional administrative assistant for a school. "
        "Generate a polished, formal school notice following these EXACT specifications:\n\n"
        "OUTPUT FORMAT:\n"
        "1. Title: <p><strong>[NOTICE TYPE] NOTICE</strong></p>\n"
        "2. Spacing: <p><br></p>\n"
        "3. Body Structure (in order):\n"
        "   - Opening: <p>This is to inform [Recipient] that...</p>\n"
        "   - Event Details: <p>The [Event Type] will be held on [Date] at [Time] in [Venue].</p>\n"
        "   - Description: <p>[Key Details]</p>\n"
        "   - Contact: <p>For further information, please contact: [Contact Info]</p>\n"
        "   - Closing: <p>[Signature Title]</p>\n\n"
        "STRICT RULES:\n"
        "1. NEVER include school name or date of issue\n"
        "2. NEVER invent dates or contact information\n"
        "3. Use ONLY these HTML tags: <p>, <strong>, <br>\n"
        "4. Replace missing information with <strong>[Placeholder]</strong>\n"
        "5. Keep total length under 120 words\n"
        "6. Use formal, academic language\n"
        "7. Each section must be in its own <p> tags\n"
        "8. Use <strong> for emphasis on important terms\n"
        "9. Maintain consistent spacing between paragraphs\n"
        "10. No bullet points or numbered lists\n"
        "11. No additional commentary or formatting\n\n"
        "TONE AND STYLE:\n"
        "1. Professional and authoritative\n"
        "2. Clear and concise\n"
        "3. Formal but not overly complex\n"
        "4. Direct and informative\n"
        "5. Appropriate for educational context"
    )

    def safe_get(key: str) -> str:
        """
        If the user provided a nonempty string for `key`, return it.
        Otherwise, return the placeholder wrapped in <strong>.
        """
        val = input_fields.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        return "[Placeholder]"


    # If not provided, safe_get will cause the model to see "[Placeholder]".

    input_block = (
        "SCHOOL NOTICE DETAILS:\n"
        f"- Notice Type: {safe_get('notice_type')} Notice\n"
        f"- Key Details: {safe_get('key_details')}\n"
        f"- Event Date: {safe_get('event_date')}\n"
        f"- Event Time: {safe_get('time')}\n"
        f"- Venue: {safe_get('venue')}\n"
        f"- Recipient: {safe_get('recipient')}\n"
        f"- Contact Info: {safe_get('contact_info')}\n"
        f"- Signature Title: {safe_get('signature_title')}"
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
        temperature=0.7,
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
   
    notice_html = generate_raw_notice(notice_request.dict())
    return {"notice": notice_html}

@app.get("/", summary="Health check")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "School Notice Generator API is running."}
