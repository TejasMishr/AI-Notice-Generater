import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "gemma-3-12b-it"

# ----------- NOTICE TYPE PROMPTS -----------

NOTICE_TYPE_PROMPTS = {
    "Prize Distribution": (
        "Compose a formal notice announcing a prize distribution ceremony. "
        "Recognize the achievements of students in academics, sports, and extracurricular activities. "
        "Encourage all relevant recipients to attend the event as scheduled."
    ),
    "Holiday Notice": (
        "Compose a professional notice to inform students, staff, and parents about an upcoming school holiday "
        "as authorized by the administration. Clearly state the reason and the date of the holiday, if available."
    ),
    "Exam Schedule": (
        "Draft a formal notice announcing the schedule of forthcoming examinations. "
        "Inform recipients about the importance of punctuality and adherence to exam guidelines."
    ),
    "Parent Meeting": (
        "Prepare a notice inviting parents or guardians to attend a meeting with teachers or administration. "
        "State the purpose as discussing student progress or addressing relevant concerns."
    ),
    "Sports Event": (
        "Draft a notice announcing a school sports event. "
        "Encourage participation and promote values of sportsmanship and healthy competition among students."
    ),
    "Cultural Program": (
        "Compose an invitation to a school cultural program. "
        "Highlight performances, activities, or special guests, and encourage attendance by the school community."
    ),
    "Fee Payment": (
        "Issue a reminder regarding the upcoming or overdue payment of school fees. "
        "Provide clear instructions on payment deadlines and methods if applicable."
    ),
    "Admission Notice": (
        "Announce the opening of admissions for the upcoming academic session. "
        "Include instructions for interested applicants regarding eligibility, documentation, or deadlines as available."
    ),
    "Result Declaration": (
        "Notify students and parents about the declaration of academic results. "
        "Direct recipients on where or how to access the results."
    ),
    "School Closure": (
        "Announce a temporary closure of the school due to administrative orders or unforeseen circumstances. "
        "Include, if applicable, any instructions regarding resumption or further communication."
    ),
}

# ----------- FASTAPI SETUP -----------

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

# ----------- REQUEST SCHEMA -----------

class NoticeRequest(BaseModel):
    notice_type: str = Field(default="General Notice", description="Type of notice")
    event_date: Optional[str] = Field(default=None, description="Event date")
    key_details: str = Field(default="Annual cultural program and prize distribution", description="Key details or summary")
    recipient: Optional[str] = Field(default=None, description="Intended recipients")
    venue: Optional[str] = Field(default=None, description="Venue for the event")
    time: Optional[str] = Field(default=None, description="Event time")
    contact_info: Optional[str] = Field(default=None, description="Contact information")
    signature_title: str = Field(default="Principal", description="Authority signing the notice")

# ----------- UTILITIES -----------

def extract_details_from_text(text: str) -> dict:
    date_pattern = r'\b(?:on\s*)?([A-Z][a-z]+\s\d{1,2},\s\d{4})\b'
    time_pattern = r'\b(?:at\s*)?(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b'
    venue_pattern = r'(?:in|at)\s+([A-Z][A-Za-z0-9\s\-]*(Auditorium|Hall|Ground|Room|Center|Block|Building|Lab|Field))'
    recipient_pattern = r'\b(?:for|to|all|for all)\s+([A-Za-z\s&]+?)(?:\.|,|;|$)'
    contact_pattern = r'([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-_]+\.[a-zA-Z]+|\(\d{3}\)\s?\d{3}[-.\s]?\d{4})'

    return {
        "event_date": re.search(date_pattern, text).group(1) if re.search(date_pattern, text) else None,
        "time": re.search(time_pattern, text).group(1) if re.search(time_pattern, text) else None,
        "venue": re.search(venue_pattern, text).group(1) if re.search(venue_pattern, text) else None,
        "recipient": re.search(recipient_pattern, text).group(1).strip().title() if re.search(recipient_pattern, text) else None,
        "contact_info": re.search(contact_pattern, text).group(1) if re.search(contact_pattern, text) else None,
    }

def preprocess_notice_fields(fields: dict) -> dict:
    details = extract_details_from_text(fields.get("key_details", ""))
    for k, v in details.items():
        if not fields.get(k) and v:
            fields[k] = v
    return fields

def build_system_instructions(notice_type, custom_prompt):
    return (
        f"You are an expert administrative assistant for a school, drafting formal notices.\n\n"
        "Your task:\n"
        "- Generate a concise, formal school notice of the specified type (<strong>[NOTICE TYPE]</strong>).\n"
        "- STRICTLY follow the OUTPUT FORMAT and RULES below.\n"
        "- If any required field is missing, extract from Key Details or use <strong>[Placeholder]</strong>.\n"
        "- NEVER invent or assume information not present in input.\n\n"
        "OUTPUT FORMAT:\n"
        "1. Title: <p><strong>[NOTICE TYPE] NOTICE</strong></p>\n"
        "2. Spacing: <p><br></p>\n"
        "3. Body:\n"
        "<p>This is to inform [Recipient] that [Event Type/Notice Subject]...</p>\n"
        "<p>The event will take place on [Date] at [Time] in [Venue].</p>\n"
        "<p>[Extra Instructions]</p>\n"
        "<p>For further information, please contact: [Contact Info]</p>\n"
        "<p><strong>Regards,<br> </strong>\n"
        "<strong>[Signature Title]</strong></p>\n\n"
        "STRICT RULES:\n"
        "1. No school name, issue date, or bullet points.\n"
        "2. Only use <p>, <strong>, <br> tags.\n"
        "3. Max 250 words.\n"
        "4. Use formal and professional tone.\n\n"
        f"NOTICE TYPE GUIDANCE:\n{custom_prompt}"
    )

# ----------- NOTICE GENERATOR -----------

def generate_raw_notice(input_fields: dict, model: str = MODEL_NAME) -> str:
    notice_type = input_fields.get("notice_type", "General Notice")
    custom_prompt = NOTICE_TYPE_PROMPTS.get(notice_type, "")

    system_instructions = build_system_instructions(notice_type, custom_prompt)

    def safe_get(key: str) -> str:
        val = input_fields.get(key)
        return val.strip() if isinstance(val, str) and val.strip() else "[Placeholder]"

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
            notice_body = response.text.strip()
            full_html = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>{safe_get('notice_type')} Notice</title>
                </head>
                <body>
                    {notice_body}
                </body>
                </html>
            """
            return full_html
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

# ----------- API ROUTES -----------

@app.post("/generate-notice", summary="Generate a school notice", response_class=HTMLResponse)
async def create_notice(notice_request: NoticeRequest):
    fields = preprocess_notice_fields(notice_request.dict())
    notice_html = generate_raw_notice(fields)
    return HTMLResponse(content=notice_html, status_code=200)

@app.get("/", summary="Health check")
async def root():
    return {"message": "School Notice Generator API is running."}
