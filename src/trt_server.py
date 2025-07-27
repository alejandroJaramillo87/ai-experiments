# rest_api_server.py
import argparse
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from transformers import AutoTokenizer, PreTrainedTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

from pydantic import BaseModel, Field

# --- Pydantic Models (No changes needed) ---
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
runner: Optional[ModelRunner] = None
tokenizer: Optional[PreTrainedTokenizer] = None
executor: Optional[ThreadPoolExecutor] = None
app = FastAPI()
MODEL_NAME = "your-model-name" 

# --- Application Lifecycle Hooks ---
# Use FastAPI's lifespan events to initialize the model and thread pool
# on startup and clean them up on shutdown. This is cleaner than global loads.
@app.on_event("startup")
async def startup_event():
    """Initializes the model and tokenizer on server startup."""
    global runner, tokenizer, executor, MODEL_NAME
    
    # This assumes the script is launched by gunicorn from the Docker CMD
    # and that those arguments are not available here. We'll parse them in main
    # and store them for the app to access, or hardcode for containerization.
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    # Gunicorn handles host and port, so we don't need them here.
    
    # This is a bit of a trick to get args when run by gunicorn
    import sys
    # Find the --engine_dir and --tokenizer_dir args from the CMD
    args, _ = parser.parse_known_args(sys.argv)

    engine_dir = args.engine_dir
    tokenizer_dir = args.tokenizer_dir
    MODEL_NAME = Path(engine_dir).name

    print(f"Loading engine from {engine_dir}...")
    # OPTIMIZATION: Set max_queue_size to handle backpressure from concurrent requests.
    runner = ModelRunner.from_dir(
        engine_dir=engine_dir,
        rank=0,
        debug_mode=False,
        max_queue_size=4 # Should be >= gunicorn worker count
    )
    
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        print("Warning: tokenizer.pad_token_id is None. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        raise ValueError("FATAL: tokenizer.eos_token_id is not defined.")
        
    # OPTIMIZATION: Initialize a thread pool to run blocking inference calls.
    executor = ThreadPoolExecutor(max_workers=4) # Should be >= gunicorn worker count
    print(f"Model '{MODEL_NAME}' loaded successfully. Server is ready.")

@app.on_event("shutdown")
def shutdown_event():
    """Shuts down the thread pool on server exit."""
    if executor:
        executor.shutdown()

# --- Helper Functions (format_prompt is unchanged) ---
def format_prompt(messages: List[ChatMessage]) -> str:
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def generate_sync(prompt_tokens: torch.Tensor, sampling_config: SamplingConfig) -> 'generator':
    """A synchronous wrapper for the blocking generate call."""
    return runner.generate(prompt_tokens, sampling_config=sampling_config, streaming=True)

# --- FastAPI Endpoints ---
@app.get("/v1/health")
async def health():
    return Response(status_code=200)

@app.get("/v1/models")
async def show_available_models():
    model_card = {"id": MODEL_NAME, "object": "model", "owned_by": "user", "permission": []}
    return {"object": "list", "data": [model_card]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Main endpoint for chat completions. Now fully async and non-blocking.
    """
    req_body = await request.json()
    chat_request = ChatCompletionRequest.parse_obj(req_body)
    
    if not runner or not tokenizer or not executor:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded yet."})

    prompt = format_prompt(chat_request.messages)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.int32)
    
    sampling_config = SamplingConfig(
        end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id,
        max_new_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature, top_p=chat_request.top_p
    )

    # OPTIMIZATION: Run the blocking `generate` call in the thread pool.
    # This frees the main event loop to handle other requests.
    loop = asyncio.get_running_loop()
    output_generator = await loop.run_in_executor(
        executor, generate_sync, input_ids, sampling_config
    )
    
    # Streaming and non-streaming responses are handled based on the generator
    if chat_request.stream:
        async def stream_generator():
            # Run the consumption of the generator in the thread pool as well
            while True:
                try:
                    # This is the async-friendly way to get the next item
                    output_batch = await loop.run_in_executor(executor, next, output_generator)
                    # The rest of your streaming logic is efficient and correct
                    output_ids = output_batch[0, 0, :]
                    # This slicing logic is correct and more efficient than re-decoding
                    last_token_count = input_ids.shape[1]
                    new_tokens = output_ids[last_token_count:]
                    input_ids = output_ids.unsqueeze(0) # Update for next iter
                    
                    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    if new_text:
                        stream_choice = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content=new_text))
                        chunk = ChatCompletionStreamResponse(model=chat_request.model, choices=[stream_choice])
                        yield f"data: {chunk.json()}\n\n"
                except StopIteration:
                    break # Generator is finished
            
            # Send final stop message
            final_choice = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")
            final_chunk = ChatCompletionStreamResponse(model=chat_request.model, choices=[final_choice])
            yield f"data: {final_chunk.json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # For non-streaming, we consume the whole generator in the thread pool
        def consume_generator():
            full_output_ids = None
            for output_batch in output_generator:
                full_output_ids = output_batch[0, 0, :]
            return full_output_ids
        
        full_output_ids = await loop.run_in_executor(executor, consume_generator)

        output_text = ""
        if full_output_ids is not None:
            output_text = tokenizer.decode(full_output_ids[input_ids.shape[1]:], skip_special_tokens=True)
        
        response_choice = ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=output_text))
        response = ChatCompletionResponse(model=chat_request.model, choices=[response_choice])
        return JSONResponse(content=response.dict())

# This block is now only for local debugging, as gunicorn will import 'app'.
if __name__ == "__main__":
    print("This block is for local debugging only. For production, use Gunicorn.")
    # You would need to manually create a startup event loop for local running.
    # Example: uvicorn.run("rest_api_server:app", host="0.0.0.0", port=8000, reload=True)
    # But the lifespan events won't fire without a proper server setup.
    # The Gunicorn command in the Dockerfile is the intended way to run.
    pass