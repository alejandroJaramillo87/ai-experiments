import os
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from llama_cpp import Llama, LlamaCache

from pydantic import BaseModel, Field

# --- Pydantic Models for OpenAI Compatibility ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    # top_k: Optional[int] = 40
    # repeat_penalty: Optional[float] = 1.1
    # seed: Optional[int] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    # usage: Optional[Dict[str, int]] = None

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


# --- Global Variables ---
llm: Optional[Llama] = None
executor: Optional[ThreadPoolExecutor] = None
app = FastAPI()
MODEL_NAME = "llama-cpp-model"

# --- Application Lifecycle (Lifespan) ---
@app.on_event("startup")
async def startup_event():
    """Initializes the Llama.cpp model and thread pool on server startup."""
    global llm, executor, MODEL_NAME

    # Read configuration from environment variables
    model_path = os.environ.get("MODEL_PATH", "/app/models/model.gguf")
    n_ctx = int(os.environ.get("N_CTX", "8192"))
    
    # --- Performance & Core Settings ---
    n_threads = int(os.environ.get("N_THREADS", "4"))
    n_threads_batch = int(os.environ.get("N_THREADS_BATCH", "4"))  # Number of CPU threads for parallel prompt processing.
    n_batch = int(os.environ.get("N_BATCH", "2048")) # Size of the batch for prompt processing. Higher is faster for long prompts.
    n_ubatch = int(os.environ.get("N_UBATCH", str(n_batch)))
    

    MODEL_NAME = Path(model_path).name
    print(f"Loading GGUF model from: {model_path}")
    print(f"Context size: {n_ctx}, Threads: {n_threads}, Batch size: {n_batch}")

    # The thread pool offloads all blocking I/O and CPU-bound inference
    # from the main asyncio event loop, keeping the server responsive.
    executor = ThreadPoolExecutor(max_workers=4)

    # Offload the blocking Llama() constructor to the thread pool
    loop = asyncio.get_running_loop()
    
    try:
        llm = await loop.run_in_executor(
            executor,
            lambda: Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                n_gpu_layers=0,  # Explicitly set for CPU-only inference
                use_mlock=True,   # Lock model in memory for performance
                numa=True               # Enables NUMA optimizations for your multi-chiplet AMD CPU.
            )
        )
        
        print(f"✅ Model '{MODEL_NAME}' loaded successfully. Server is ready.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():
    """Cleans up resources on server shutdown."""
    if executor:
        executor.shutdown(wait=True)
    print("Server has been shut down.")

# --- Helper Function ---
def generate_completion_sync(request: ChatCompletionRequest):
    """
    A synchronous wrapper for the blocking llm.create_chat_completion call.
    This is designed to be run in the ThreadPoolExecutor.
    """
    return llm.create_chat_completion(
        messages=[m.dict() for m in request.messages],
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

# --- FastAPI Endpoints ---
@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    if llm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Model not loaded"}
        )
    return {"status": "healthy", "model": MODEL_NAME}

@app.get("/v1/models")
async def show_available_models():
    """Provides a model card for compatibility with clients like Open WebUI."""
    model_card = {"id": MODEL_NAME, "object": "model", "owned_by": "user", "permission": []}
    return {"object": "list", "data": [model_card]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handles OpenAI-compatible chat requests using a non-blocking architecture."""
    if not llm or not executor:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded or ready."})
    
    # Validate request
    if not request.messages:
        return JSONResponse(
            status_code=400,
            content={"error": "Messages cannot be empty."}
        )

    # Enforce that the client requests the model that is actually loaded.
    if request.model != MODEL_NAME:
        return JSONResponse(
            status_code=400, # Bad Request
            content={
                "error": f"Incorrect model specified. This server is running '{MODEL_NAME}', but your request specified '{request.model}'."
            }
        )

    loop = asyncio.get_running_loop()
    try:
        # Process in thread pool to avoid blocking
        completion_or_generator = await loop.run_in_executor(
            executor, generate_completion_sync, request
        )

        if request.stream:
            # Streaming response with proper error handling
            async def stream_generator():
                try:
                    for chunk in completion_or_generator:
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_chunk = {
                        "error": {"message": str(e), "type": "server_error"}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                stream_generator(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # Disable Nginx buffering
                }
            )
        else:
            # Non-streaming response
            return JSONResponse(content=completion_or_generator)
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}}
        )


# This block is now only for local debugging, as Gunicorn will import 'app'.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)