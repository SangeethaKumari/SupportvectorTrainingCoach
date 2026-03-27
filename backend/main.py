import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
try:
    from backend.graph import app as graph_app
except ImportError:
    from graph import app as graph_app
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SupportVector Training Coach API")
print("--- BACKEND VERSION 3.0 (Gemini 3 Support) STARTED ---")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "SupportVector Training Coach API is running"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_generator():
        try:
            inputs = {
                "question": request.message, 
                "original_question": request.message, 
                "retry_count": 0, 
                "thoughts": []
            }
            
            async for event in graph_app.astream_events(inputs, version="v2"):
                kind = event["event"]
                # AGGRESSIVE DEBUG: Print every event or use a sample
                # print(f"[EVENT DEBUG] kind={kind}, name={event.get('name')}, node_meta={event.get('metadata', {}).get('langgraph_node')}")
                
                # Check for node end events
                node_name = event.get("metadata", {}).get("langgraph_node") or event.get("name")
                
                # Check for node/chain end events to capture output data
                is_end_event = kind in ["on_node_end", "on_chain_end"]
                node_name = event.get("metadata", {}).get("langgraph_node") or event.get("name")
                
                if is_end_event:
                    data = event["data"].get("output")
                    if data and isinstance(data, dict):
                        # Catch thoughts
                        if "thoughts" in data and data["thoughts"]:
                            yield f"data: {json.dumps({'type': 'thought', 'thought': data['thoughts'][-1]})}\n\n"
                        
                        # Catch sources from grade_documents or similar (LangChain docs format)
                        if "documents" in data:
                            doc_sources = []
                            for doc in data["documents"]:
                                try:
                                    path = doc.metadata.get("source", "Unknown")
                                    doc_sources.append({
                                        "page": doc.metadata.get("page", "N/A"),
                                        "source": os.path.basename(path),
                                        "content": doc.page_content[:200] + "..."
                                    })
                                except:
                                    continue
                            if doc_sources:
                                print(f"[DEBUG] Yielding docs from {node_name}")
                                yield f"data: {json.dumps({'type': 'metadata', 'sources': doc_sources})}\n\n"
                        
                        # Catch sources from generate node (our custom format)
                        if "sources" in data and data["sources"]:
                            print(f"[DEBUG] Yielding sources from {node_name}")
                            yield f"data: {json.dumps({'type': 'metadata', 'sources': data['sources']})}\n\n"

                # 2. Capture Tokens from the 'generate' node
                elif kind in ["on_chat_model_stream", "on_parser_stream"]:
                    tags = event.get("tags", [])
                    if node_name == "generate" and "final_answer" in tags:
                        chunk = event["data"].get("chunk")
                        raw_content = ""
                        
                        # 1. Extract raw content from chunk
                        if hasattr(chunk, "content"):
                            raw_content = chunk.content
                        elif isinstance(chunk, str):
                            raw_content = chunk
                        elif isinstance(chunk, list):
                            raw_content = chunk
                            
                        # 2. Extract string text from raw content (handling Gemini 3's list format)
                        text_to_send = ""
                        if isinstance(raw_content, str):
                            text_to_send = raw_content
                        elif isinstance(raw_content, list) and len(raw_content) > 0:
                            # Gemini 3 often sends a list of dicts: [{'type': 'text', 'text': '...'}]
                            item = raw_content[0]
                            if isinstance(item, dict) and 'text' in item:
                                text_to_send = item['text']
                        
                        if text_to_send:
                            # print(f"DEBUG: SENDING TEXT: {text_to_send[:10]}...")
                            yield f"data: {json.dumps({'type': 'token', 'token': text_to_send})}\n\n"

            yield "data: [DONE]\n\n"
            print("---STREAMING COMPLETE---")
        except Exception as e:
            print(f"STREAM ERROR: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
