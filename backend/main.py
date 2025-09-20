# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv

# Import the chat query processor
from chat_query import ChatQueryProcessor

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    relevant_links: List[Dict[str, str]]

# Initialize chat processor
chat_processor = ChatQueryProcessor(api_key=os.getenv("GEMINI_API_KEY"))

# Single chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat messages and return AI responses"""
    try:
        # Process the query
        result = chat_processor.process_query(request.message)
        
        return ChatResponse(
            response=result["response"],
            relevant_links=result["relevant_links"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Portfolio Chat Backend...")
    print(f"Gemini API Key configured: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)