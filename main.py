import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/comment")
async def get_comment():
    return {"message": "Send a POST request with a comment field"}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f"Analyze the sentiment of this comment: \"{request.comment}\"\n\n"
                "Return:\n"
                "- sentiment: exactly one of 'positive', 'negative', or 'neutral'\n"
                "- rating: an integer from 1 (very negative) to 5 (very positive)"
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SentimentResponse,
            ),
        )
        data = json.loads(response.text)
        return SentimentResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
```

Then commit it on GitHub → Render will auto-redeploy → try submitting again with the full URL:
```
https://sentiment-api-2mgs.onrender.com/comment
