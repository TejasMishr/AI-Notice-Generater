import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from google import genai
from google.genai import types


load_dotenv()

MODEL_NAME = "gemma-3-12b-it"
MAX_NOTICE_WORDS = 200
PLACEHOLDER = "<strong>[Insert here]</strong>"

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
    notice_type: str = Field(default="General Notice", description="Type of notice")
    event_date: Optional[str] = Field(default=None, description="Event date (e.g. 'June 20, 2025')")
    key_details: str = Field(default="", description="Key details or summary")
    recipient: Optional[str] = Field(default=None, description="Intended recipients (e.g. 'All Students and Parents')")
    venue: Optional[str] = Field(default=None, description="Venue for the event (e.g. 'School Auditorium')")
    time: Optional[str] = Field(default=None, description="Event time (e.g. '5:00 PM')")
    contact_info: Optional[str] = Field(default=None, description="Contact information (e.g. 'office@school.edu | (555) 123-4567')")
    signature_title: str = Field(default="Principal", description="Authority signing the notice")

def extract_details_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    Attempt to extract event details from the given text using regex.
    """
    # Patterns for different fields (improved for variety)
    date_pattern = r'\b(?:on\s*)?([A-Z][a-z]+\s\d{1,2},\s\d{4})\b'
    time_pattern = r'\b(?:at\s*)?(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b'
    venue_pattern = r'(?:in|at)\s+([A-Z][A-Za-z0-9\s\-]*(?:Auditorium|Hall|Ground|Room|Center|Block|Building|Lab|Field))'
    recipient_pattern = r'\b(?:for|to|all|for all)\s+([A-Za-z\s&]+?)(?:\.|,|;|$)'
    contact_pattern = r'([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-_]+\.[a-zA-Z]+|\(\d{3}\)\s?\d{3}[-.\s]?\d{4})'

    return {
        "event_date": re.search(date_pattern, text).group(1) if re.search(date_pattern, text) else None,
        "time": re.search(time_pattern, text).group(1) if re.search(time_pattern, text) else None,
        "venue": re.search(venue_pattern, text).group(1) if re.search(venue_pattern, text) else None,
        "recipient": re.search(recipient_pattern, text).group(1).strip().title() if re.search(recipient_pattern, text) else None,
        "contact_info": re.search(contact_pattern, text).group(1) if re.search(contact_pattern, text) else None,
    }

def preprocess_notice_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill missing fields by extracting them from 'key_details', if possible.
    """
    details = extract_details_from_text(fields.get("key_details", "") or "")
    for k, v in details.items():
        if not fields.get(k) and v:
            fields[k] = v
    return fields

def build_system_instructions(notice_type: str, custom_prompt: str) -> str:
    """
    Build a strict, detailed system instruction prompt for the LLM.
    """
    return (
        f"You are an expert administrative assistant for a school, drafting formal notices.\n\n"
        "Your task:\n"
        "- Generate a concise, formal school notice of the specified type (<strong>[NOTICE TYPE]</strong>).\n"
        "- STRICTLY follow the OUTPUT FORMAT and RULES below.\n"
        f"- If any required field (Date, Time, Venue, Recipient, Contact Info) is missing, carefully examine the 'Key Details' text for possible information. "
        f"Only fill in if confidently found; otherwise, use {PLACEHOLDER}.\n"
        "- NEVER invent or assume information not present in input.\n\n"
        "OUTPUT FORMAT:\n"
        "1. Title: <p><strong>[NOTICE TYPE] NOTICE</strong></p>\n"
        "2. Spacing: <p><br></p>\n"
        "3. Body Structure (each in separate <p> tags):\n"
        "   - Opening: <p>This is to inform [Recipient] that [Event Type/Notice Subject]...</p>\n"
        "   - Main Information: <p>[Provide the main message, including reasons or instructions, e.g., 'The event will take place on [Date] at [Time] in [Venue].']</p>\n"
        "   - Additional Details: <p>[Key Details, such as highlights, specific requirements, or important instructions]</p>\n"
        "   - Contact: <p>For further information, please contact: [Contact Info]</p>\n"
        "   - Authority: <p><strong>[Signature Title]</strong></p>\n"
        "STRICT RULES:\n"
        "1. NEVER include school name or date of issue.\n"
        "2. NEVER invent details not given or extractable from 'Key Details'.\n"
        "3. Use ONLY <p>, <strong>, <br> tags.\n"
        f"4. Replace missing information with {PLACEHOLDER}.\n"
        f"5. Max total notice: {MAX_NOTICE_WORDS} words.\n"
        "6. Language: Formal, professional, clear, academic.\n"
        "7. No bullet points or extra formatting.\n"
        "8. No comments, explanations, or meta-output.\n\n"
        "EXTRACTION INSTRUCTIONS:\n"
        "- For each field (Date, Time, Venue, Recipient, Contact Info), first use the direct field value.\n"
        "- If missing, extract from 'Key Details' if clearly stated (e.g., 'The meeting will be at 10:00 AM in the Main Hall on June 14.').\n"
        f"- If you cannot find it, use {PLACEHOLDER}.\n\n"
        f"NOTICE TYPE GUIDANCE:\n{custom_prompt}\n"
        "If the notice type is not recognized, use a formal, generic tone."
    )

def safe_get(fields: Dict[str, Any], key: str, custom_prompt: str) -> str:
    """
    Get the field value or use placeholder or custom prompt.
    """
    val = fields.get(key)
    if isinstance(val, str) and val.strip():
        return val.strip()
    if key == "key_details" and custom_prompt:
        return custom_prompt
    return PLACEHOLDER

def generate_raw_notice(input_fields: Dict[str, Any], model: str = MODEL_NAME) -> str:
    """
    Generate a formal school notice using the Gemini LLM.
    """
    notice_type = input_fields.get("notice_type", "General Notice")
    custom_prompt = NOTICE_TYPE_PROMPTS.get(notice_type, "")

    system_instructions = build_system_instructions(notice_type, custom_prompt)

    input_block = (
        "SCHOOL NOTICE DETAILS:\n"
        f"- Notice Type: {safe_get(input_fields, 'notice_type', custom_prompt)} Notice\n"
        f"- Key Details: {safe_get(input_fields, 'key_details', custom_prompt)}\n"
        f"- Event Date: {safe_get(input_fields, 'event_date', custom_prompt)}\n"
        f"- Event Time: {safe_get(input_fields, 'time', custom_prompt)}\n"
        f"- Venue: {safe_get(input_fields, 'venue', custom_prompt)}\n"
        f"- Recipient: {safe_get(input_fields, 'recipient', custom_prompt)}\n"
        f"- Contact Info: {safe_get(input_fields, 'contact_info', custom_prompt)}\n"
        f"- Signature Title: {safe_get(input_fields, 'signature_title', custom_prompt)}"
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
        temperature=0.4,
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
    Generate a school notice based on user input, auto-extracting fields from key_details if necessary.
    """
    fields = preprocess_notice_fields(notice_request.dict())
    notice_html = generate_raw_notice(fields)
    return {"notice": notice_html}

@app.get("/", summary="Health check")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "School Notice Generator API is running."}
