# rest_api_server.py
import argparse
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator, List, Optional
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from transformers import AutoTokenizer, PreTrainedTokenizer
# Import SamplingConfig to correctly pass generation parameters
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

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
runner: Optional[ModelRunner] = None
tokenizer: Optional[PreTrainedTokenizer] = None
app = FastAPI()
MODEL_NAME = "mistral-7b-v0.3-int4-awq" # Define model name globally

# --- Helper Functions ---

def load_model(engine_dir: str, tokenizer_dir: str):
    """Loads the TensorRT-LLM engine and tokenizer from the specified directories."""
    global runner, tokenizer
    print(f"Loading engine from {engine_dir}...")
    runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0, debug_mode=False)
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False, trust_remote_code=True)
    
    # Add defensive checks for missing tokens
    if tokenizer.pad_token_id is None:
        print("Warning: tokenizer.pad_token_id is None. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if tokenizer.eos_token_id is None:
        raise ValueError("FATAL: tokenizer.eos_token_id is not defined. The model cannot run without an end-of-sequence token.")
        
    print("Model and tokenizer loaded successfully.")


def format_prompt(messages: List[ChatMessage]) -> str:
    """Formats a list of chat messages into a single string prompt for the model."""
    # This uses the ChatML format, which is common for many models.
    # It clearly delineates roles and content.
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt



# --- FastAPI Endpoints ---

@app.get("/v1/health")
async def health():
    """Health check endpoint for Docker and other services."""
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
    """
    This is the main endpoint for generating chat completions.
    It's designed to be compatible with the OpenAI API standard.
    """
    # 1. Check if the model is loaded and ready to serve requests.
    if not runner or not tokenizer:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded yet."})

    # 2. Format the incoming messages into a single prompt string for the model.
    prompt = format_prompt(request.messages)
    
    # 3. Tokenize the prompt. This converts the text string into a sequence of integer IDs.
    #    The .to(torch.int32) is necessary because the TensorRT-LLM engine expects
    #    input tensors to be of this specific data type, regardless of the model's weight precision.
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.int32)

    # 4. Create a SamplingConfig object to hold all generation parameters.
    #    This is the correct way to pass parameters like end_id, temperature, etc.
    #
    #    Call the TensorRT-LLM engine to start generating tokens.
    #    The `runner.generate` function is a Python generator, meaning it yields
    #    results incrementally rather than all at once. We always set `streaming=True`
    #    on the backend because it's more efficient to process tokens as they arrive.
    
    
    sampling_config = SamplingConfig(
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    # 5. Call the TensorRT-LLM engine to start generating tokens.
    #    We pass the config object directly to the generate call.
    output_generator = runner.generate(
        input_ids,
        sampling_config=sampling_config,
        streaming=True
    )

    # 6. Handle the response based on whether the client requested a streaming or non-streaming response.
    if request.stream:
        # 6a. STREAMING RESPONSE: Send back tokens as they are generated.
        async def stream_generator() -> AsyncGenerator[str, None]:
            last_token_count = 0
            for output_batch in output_generator:
                output_ids = output_batch[0, 0, :]
                new_tokens = output_ids[last_token_count:]
                new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                last_token_count = len(output_ids)

                if new_text:
                    stream_choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=new_text)
                    )
                    chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        choices=[stream_choice]
                    )
                    yield f"data: {chunk.json()}\n\n"

            final_choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop"
            )
            final_chunk = ChatCompletionStreamResponse(
                model=request.model,
                choices=[final_choice]
            )
            yield f"data: {final_chunk.json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # 6b. NON-STREAMING RESPONSE: Wait for the full generation and return it in one block.
        full_output_ids = None
        for output_batch in output_generator:
            full_output_ids = output_batch[0, 0, :]

        if full_output_ids is not None:
            output_text = tokenizer.decode(full_output_ids[input_ids.shape[1]:], skip_special_tokens=True)
        else:
            output_text = ""

        response_choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=output_text)
        )
        response = ChatCompletionResponse(
            model=request.model,
            choices=[response_choice]
        )
        return JSONResponse(content=response.dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    try:
        load_model(args.engine_dir, args.tokenizer_dir)
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        print(f"Failed to load model and start server: {e}")

