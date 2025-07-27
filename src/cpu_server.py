# src/cpu_server.py

import argparse
import json
import uvicorn
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from llama_cpp import Llama
from typing import Optional, List

# --- Pydantic Models for OpenAI Compatibility ---
# These models define the structure of the API requests and responses,
# matching the OpenAI API specification. This is crucial for Open WebUI integration.
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]

# --- Global variables ---
llm: Optional[Llama] = None
app = FastAPI()
MODEL_NAME = "cpu-model" # This will be updated based on the loaded model file

# --- Helper Functions ---

def load_model(model_path: str, n_ctx: int, n_threads: int):
    """Loads the GGUF model using llama-cpp-python."""
    global llm, MODEL_NAME
    print(f"Loading GGUF model from {model_path}...")
    
    # Extract the model name from the file path for the API response
    MODEL_NAME = model_path.split('/')[-1]

    # Initialize the Llama model
    # n_gpu_layers=0 ensures it runs entirely on the CPU
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,  # Explicitly set to 0 for CPU-only inference
        verbose=True
    )
    print(f"CPU Model '{MODEL_NAME}' loaded successfully.")


# --- FastAPI Endpoints ---

@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return Response(status_code=200)

@app.get("/v1/models")
async def show_available_models():
    """Adds compatibility for Open WebUI model discovery."""
    model_card = {
        "id": MODEL_NAME,
        "object": "model",
        "owned_by": "user",
        "permission": []
    }
    return {"object": "list", "data": [model_card]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main endpoint for generating chat completions using llama.cpp."""
    if not llm:
        return JSONResponse(status_code=503, content={"error": "CPU Model is not loaded yet."})
    
    # The `llama-cpp-python` library has a convenient `create_chat_completion`
    # method that directly accepts OpenAI-like message structures.
    completion_generator = llm.create_chat_completion(
        messages=request.dict()["messages"], # Pass the messages directly
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream
    )

    if request.stream:
        # Stream the response
        async def stream_generator():
            for chunk in completion_generator:
                # The chunk format from llama-cpp-python is already OpenAI-compatible
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Return a single, non-streamed response
        return JSONResponse(content=completion_generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to the GGUF model file.")
    parser.add_argument('--n_ctx', type=int, default=4096, help="Context size for the model.")
    parser.add_argument('--n_threads', type=int, default=16, help="Number of CPU threads to use.")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()

    try:
        load_model(model_path=args.model, n_ctx=args.n_ctx, n_threads=args.n_threads)
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        print(f"Failed to load model and start server: {e}")