from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from typing import Optional
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI and Anthropic
app = FastAPI()
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str
    child_name: Optional[str] = None
    scenario: Optional[str] = "general"

SCENARIOS = {
    "morning": "getting ready for the day",
    "bedtime": "preparing for bed",
    "homework": "doing homework",
    "meals": "eating meals",
    "chores": "doing chores",
    "general": "general encouragement"
}

SYSTEM_PROMPT = """You are Captain Jack the Watchful, a friendly and wise pirate who helps parents guide their children. 
Your responses should be:
1. Warm and encouraging, never scolding
2. Use child-friendly pirate language
3. Brief (2-3 sentences maximum)
4. Include specific praise or gentle guidance
5. Use nautical/pirate themes to make tasks fun

Always maintain a positive, supportive tone while staying in character as Captain Jack."""

@app.get("/")
def read_root():
    return {"status": "Captain Jack is ready to help!"}

@app.post("/message")
async def process_message(request: MessageRequest):
    try:
        # Build the prompt
        scenario_context = SCENARIOS.get(request.scenario, SCENARIOS["general"])
        child_context = f" for {request.child_name}" if request.child_name else ""
        
        # Generate response using Claude
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=150,  # Keep responses concise
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a response about {scenario_context}{child_context}. Parent's message: {request.message}"
                }
            ]
        )
        
        return {
            "message": response.content,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))
