import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from backend.graph import app as graph_app
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SupportVector Training Coach API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    thoughts: List[str]
    sources: List[dict]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Initial state
    inputs = {
        "question": request.message, 
        "original_question": request.message, # Explicitly track the user's first query
        "retry_count": 0, 
        "thoughts": []
    }
    
    # Run the graph
    result = graph_app.invoke(inputs)
    
    # Extract sources from documents
    sources = []
    if "documents" in result:
        for doc in result["documents"]:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path)
            sources.append({
                "page": doc.metadata.get("page", "N/A"),
                "source": source_name,
                "content": doc.page_content[:200] + "..."
            })
            
    return {
        "answer": result["generation"],
        "thoughts": result["thoughts"],
        "sources": sources
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
