import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load your secret API key from .env file
load_dotenv()

# Create the FastAPI app
app = FastAPI()

# Connect to OpenAI using your key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Define what the REQUEST looks like ---
class CommentRequest(BaseModel):
    comment: str  # The user sends a "comment" field (a string)

# --- Define what the RESPONSE looks like ---
class SentimentResponse(BaseModel):
    sentiment: str  # "positive", "negative", or "neutral"
    rating: int     # A number from 1 to 5

# --- The actual API endpoint ---
@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    
    # Basic validation — don't process empty comments
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    
    try:
        # Ask OpenAI to analyze the comment using STRUCTURED OUTPUT
        response = client.responses.parse(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis tool. "
                        "Analyze the given comment and return:\n"
                        "- sentiment: exactly one of 'positive', 'negative', or 'neutral'\n"
                        "- rating: an integer from 1 (very negative) to 5 (very positive)\n"
                        "Be consistent and accurate."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            text_format=SentimentResponse,  # 👈 This enforces structured output!
        )

        # Extract the parsed result
        result = response.output_parsed
        return result

    except Exception as e:
        # If OpenAI fails for any reason, return a clean error
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
