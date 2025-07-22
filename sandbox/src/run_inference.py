# src/run_inference.py
# A simple script to load the compiled TensorRT-LLM engine and run inference.
# Based on the run.py example from the NVIDIA TensorRT-LLM repository.

import json
from pathlib import Path
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner

# --- Configuration ---
ENGINE_DIR = "/app/trt_engine"
TOKENIZER_DIR = "/app/tokenizer"
PROMPT_TEXT = "Explain the importance of low-latency in AI inference."
MAX_OUTPUT_LEN = 250

def get_engine_name(model: str, dtype: str, tp_size: int, rank: int) -> str:
    """Gets the engine name from the config.json file."""
    # This function logic is derived from the TensorRT-LLM examples.
    with open(Path(ENGINE_DIR) / "config.json", 'r') as f:
        config = json.load(f)
    
    builder_config = config['builder_config']
    
    # Check for quantization mode to correctly name the engine
    quant_mode = builder_config.get('quant_mode', 'none')
    if quant_mode and 'fp4' in quant_mode:
        # For FP4, the name format is often different
        # Let's assume a standard naming convention if not explicitly found
        return f"model_fp4_kv_int8.engine"

    # Fallback for other dtypes if needed, though we are focused on FP4
    return f"model_{dtype}_tp{tp_size}_rank{rank}.engine"


def run_generation():
    """
    Loads the engine and tokenizer, and runs text generation.
    """
    engine_dir_path = Path(ENGINE_DIR)
    if not engine_dir_path.exists():
        print(f"Engine directory not found at {ENGINE_DIR}")
        return

    # Load the tokenizer
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    
    # The runner is the main entry point for inference.
    # For a single GPU run, the rank is 0.
    runner = ModelRunner.from_dir(
        engine_dir=ENGINE_DIR,
        rank=0,
        debug_mode=False # Set to True for more verbose output
    )

    print("Engine loaded. Running inference...")

    # Prepare the input tokens
    input_ids = tokenizer.encode(PROMPT_TEXT, return_tensors="pt").to("cuda")

    # Generate output
    # The generate method handles the inference loop.
    output_ids = runner.generate(
        input_ids,
        max_new_tokens=MAX_OUTPUT_LEN,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and print the output
    output_text = tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
    
    print("--- Prompt ---")
    print(PROMPT_TEXT)
    print("\n--- Generation ---")
    print(output_text)


if __name__ == "__main__":
    try:
        run_generation()
    except Exception as e:
        print(f"An error occurred during inference: {e}")