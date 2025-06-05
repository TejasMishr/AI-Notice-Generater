# School Notice Generator

This Python application generates formal school notices using Google's Generative AI (Gemini).

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

You can get your API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

Run the script:

```bash
python main.py
```

The script will generate a sample notice using the example fields defined in `main.py`. You can modify the `example_fields` dictionary in `main.py` to generate different notices.

## Customizing Notices

To generate a custom notice, modify the `example_fields` dictionary in `main.py` with your desired values:

```python
example_fields = {
    "School_Name": "Your School Name",
    "Notice_Type": "Type of Notice",
    "Title": "Notice Title",
    "Date_of_Issue": "Date",
    "Recipient": "Target Audience",
    "Event_Date/Deadline": "Event Date and Time",
    "Venue": "Location",
    "Main_Purpose": "Purpose of Notice",
    "Key_Details": [
        "Detail 1",
        "Detail 2",
        "Detail 3"
    ],
    "Contact_Info": "Contact Information",
    "Signature_Name": "Signatory Name",
    "Signature_Title": "Signatory Title"
}
```
