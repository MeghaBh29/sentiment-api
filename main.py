import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Connect to Gemini using your key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Request shape ---
class CommentRequest(BaseModel):
    comment: str

# --- Response shape ---
class SentimentResponse(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    rating: int     # 1 to 5

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                f"Analyze the sentiment of this comment: \"{request.comment}\"\n\n"
                "Return:\n"
                "- sentiment: exactly one of 'positive', 'negative', or 'neutral'\n"
                "- rating: an integer from 1 (very negative) to 5 (very positive)"
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SentimentResponse,  # 👈 Enforces structured output!
            ),
        )

        import json
        data = json.loads(response.text)
        return SentimentResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
