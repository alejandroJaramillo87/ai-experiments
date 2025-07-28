# src/cpu_server.py

import argparse
import json
import uvicorn
import time
import uuid
import asyncio  # <<< ADDED: For async operations
from concurrent.futures import ThreadPoolExecutor  # <<< ADDED: For non-blocking inference

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from llama_cpp import Llama
from typing import Optional, List, Dict, Any

# --- Pydantic Models for OpenAI Compatibility ---
# (This section is perfect, no changes needed)
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

# --- Global variables for the application state ---
llm: Optional[Llama] = None
executor: Optional[ThreadPoolExecutor] = None
app = FastAPI()

# <<< ADDED: Globals to hold args for the startup event
MODEL_PATH_FOR_STARTUP = ""
N_CTX_FOR_STARTUP = 4096
N_THREADS_FOR_STARTUP = 16
N_BATCH_FOR_STARTUP = 2048
MODEL_NAME = "cpu-model"



# <<< ADDED: FastAPI lifespan events to manage resources ---
@app.on_event("startup")
async def startup_event():
    """Loads the model and initializes the thread pool on server startup."""
    global llm, executor, MODEL_NAME

    # Use a thread pool to run blocking inference calls.
    executor = ThreadPoolExecutor(max_workers=4)

    print(f"Loading GGUF model from {MODEL_PATH_FOR_STARTUP}...")
    MODEL_NAME = MODEL_PATH_FOR_STARTUP.split('/')[-1]

    # Run the blocking Llama() constructor in the thread pool
    loop = asyncio.get_running_loop()
    llm = await loop.run_in_executor(
        executor,
        Llama,
        MODEL_PATH_FOR_STARTUP,
        n_ctx=N_CTX_FOR_STARTUP,
        n_threads=N_THREADS_FOR_STARTUP,
        n_gpu_layers=0,
        # Performance optimizations
        mlock=True,
        n_batch=N_BATCH_FOR_STARTUP,
        verbose=True
    )
    print(f"CPU Model '{MODEL_NAME}' loaded successfully. Server is ready.")

@app.on_event("shutdown")
def shutdown_event():
    """Shuts down the thread pool on server exit."""
    if executor:
        executor.shutdown(wait=True)


# --- FastAPI Endpoints ---

@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    if not llm:
        return Response(status_code=503) # Service Unavailable
    return Response(status_code=200)

@app.get("/v1/models")
async def show_available_models():
    """Adds compatibility for Open WebUI model discovery."""
    model_card = {"id": MODEL_NAME, "object": "model", "owned_by": "user", "permission": []}
    return {"object": "list", "data": [model_card]}


# <<< REWRITTEN: The chat completions endpoint is now fully async and non-blocking
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main endpoint for generating chat completions using llama.cpp."""
    if not llm or not executor:
        return JSONResponse(status_code=503, content={"error": "CPU Model is not loaded yet."})

    loop = asyncio.get_running_loop()

    # Create a synchronous function for the blocking call
    def generate():
        return llm.create_chat_completion(
            messages=[m.dict() for m in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream
        )

    # Run the blocking function in the thread pool
    completion_or_generator = await loop.run_in_executor(executor, generate)

    if request.stream:
        # For streaming, we consume the generator in an async generator
        async def stream_generator():
            for chunk in completion_or_generator:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # For non-streaming, the result is already a complete dictionary
        return JSONResponse(content=completion_or_generator)


# <<< REWRITTEN: This block is now for parsing args and local debugging.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama.cpp OpenAI-Compatible REST server")
    parser.add_argument('--model', type=str, required=True, help="Path to the GGUF model file.")
    parser.add_argument('--n_ctx', type=int, default=4096, help="Context size for the model.")
    parser.add_argument('--n_threads', type=int, default=16, help="Number of CPU threads to use.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host for the server.")
    parser.add_argument('--port', type=int, default=8001, help="Port for the server.")
    args = parser.parse_args()

    # Store args in global variables for the startup event to access
    MODEL_PATH_FOR_STARTUP = args.model
    N_CTX_FOR_STARTUP = args.n_ctx
    N_THREADS_FOR_STARTUP = args.n_threads

    print("--- Starting development server with Uvicorn ---")
    print("--- For production, run with Gunicorn ---")
    uvicorn.run(app, host=args.host, port=args.port)