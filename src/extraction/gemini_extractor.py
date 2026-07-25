import os
import json
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from google import genai

# ---------------------------------
# Load Environment Variables
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ---------------------------------
# Prompt
# ---------------------------------

PROMPT = """
You are an expert invoice parser.

Extract the following information from the invoice.

Return ONLY valid JSON.

{
  "invoice_number":"",
  "vendor_name":"",
  "gst_number":"",
  "invoice_date":"",
  "due_date":"",
  "invoice_amount":0,
  "tax_amount":0,
  "payment_method":"",
  "invoice_category":""
}

Rules:

- Output JSON only.
- No markdown.
- Missing values should be "".
- Amounts must be numeric.
"""

# ---------------------------------
# Gemini Extraction
# ---------------------------------

def extract_invoice(image_path):

    image_path = Path(image_path)

    print(f"Reading image: {image_path}")

    image = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PROMPT,
            image
        ]
    )

    text = response.text.strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return json.loads(text)


# ---------------------------------
# Testing
# ---------------------------------

if __name__ == "__main__":

    print("BASE_DIR:", BASE_DIR)
    print("IMAGE_PATH:", BASE_DIR / "uploads" / "sample_invoice1.png")

    from pathlib import Path
    print("Exists:", Path(BASE_DIR / "uploads" / "sample_invoice1.png").exists())

    IMAGE_PATH = BASE_DIR / "uploads" / "sample_invoice1.png"

    invoice = extract_invoice(IMAGE_PATH)

    print("\nExtracted Invoice\n")

    print(json.dumps(invoice, indent=4))