Of course. Here is a markdown document created from your notes with the requested explanations.

-----

# Running Optimized LLMs with TensorRT-LLM and Docker

This guide outlines the process for taking a pre-trained large language model from Hugging Face, optimizing it with NVIDIA's TensorRT-LLM, and running it in a containerized environment using Docker. The workflow involves quantizing the model to a lower precision, building a high-performance TensorRT engine, and then executing inference within a streamlined Docker container.

This process is ideal for creating a production-ready, high-throughput inference service on your local hardware.

-----

## Step 1: Download the Model

First, you need to download the model weights and tokenizer files from the Hugging Face Hub. The `TensorRT-LLM` repository includes a helper script for this.

**Note:** Before running, ensure you have cloned the [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM) locally, as the scripts used in the following steps are from that repo.

```bash
python TensorRT-LLM/examples/llama/download_model.py \
  --repo-id "mistralai/Mistral-7B-v0.3" \
  --output-dir "/mnt/ai-data/models/hf/Mistral-7B-v0.3"
```

This command downloads the specified model, in this case `Mistral-7B-v0.3`, into your target directory. This directory will contain the model weights, tokenizer configuration, and other necessary files.

-----

## Step 2: Quantize the Model

**Quantization** is a process that reduces the precision of the model's weights (e.g., from 16-bit floating-point to 4-bit integer). This significantly reduces the model's memory footprint (both on disk and in VRAM) and can lead to faster inference with minimal impact on accuracy.

Navigate to the `examples/quantization` directory within your local `TensorRT-LLM` repository and run the quantization script.

```bash
python quantize.py \
  --model_dir /mnt/ai-data/models/hf/Mistral-7B-v0.3 \
  --qformat int4_awq \
  --awq_block_size 64 \
  --output_dir /mnt/ai-data/models/quantized/Mistral-7B-v0.3-int4-awq \
  --tp_size 1
```

### Argument Breakdown:

  * `--model_dir`: Path to the original Hugging Face model downloaded in the previous step.
  * `--qformat int4_awq`: Specifies the quantization format. Here we're using **INT4** precision with the **Activation-aware Weight Quantization (AWQ)** algorithm, which is known for preserving model quality well.
  * `--awq_block_size`: A parameter specific to the AWQ algorithm that defines the group size for quantization. `64` is a common choice.
  * `--output_dir`: The directory where the quantized model checkpoints will be saved.
  * `--tp_size 1`: Sets the **Tensor Parallelism** size to 1. Since you're running on a single RTX 5090, a value of 1 is appropriate. For multi-GPU inference, this value would be increased.

-----

## Step 3: Build the TensorRT Engine

This is the core optimization step. The `trtllm-build` command takes the quantized model checkpoints and compiles them into a highly optimized TensorRT engine. This engine is tailored specifically to your GPU architecture (RTX 5090) and the parameters you define.

```bash
trtllm-build \
 --checkpoint_dir /mnt/ai-data/models/quantized/Mistral-7B-v0.3-int4-awq \
 --output_dir /mnt/ai-data/models/trtllm-engine/Mistral-7B-v0.3-int4-awq/ \
 --gemm_plugin float16 \
 --max_batch_size 1 \
 --max_input_len 4096 \
 --max_seq_len 6144 \
 --max_num_tokens 6144 \
 --context_fmha enable \
 --paged_kv_cache enable \
 --remove_input_padding enable \
 --use_paged_context_fmha enable \
 --tokens_per_block 64 \
 --use_fused_mlp enable
```

### Argument Breakdown:

  * `--checkpoint_dir`: The input directory containing the quantized model from Step 2.
  * `--output_dir`: The destination for the compiled TensorRT engine files.
  * `--gemm_plugin float16`: Specifies the data type for GEMM (General Matrix Multiply) operations.
  * `--max_batch_size`, `--max_input_len`, `--max_seq_len`, `--max_num_tokens`: These define the engine's static operational limits. Requests exceeding these values will fail. `max_seq_len` is the total sequence length (input + output), while `max_num_tokens` allows for dynamic shapes up to this limit.
  * `--context_fmha`, `--use_paged_context_fmha`: Enables **Flash Attention** and **Paged Context Flash Attention**, which are highly efficient attention mechanisms that drastically reduce memory usage and improve throughput for long sequences.
  * `--paged_kv_cache`: Enables a more efficient memory management scheme for the KV cache, reducing internal fragmentation.
  * `--remove_input_padding`: An optimization that processes tokens without padding, preventing wasted computation.
  * `--tokens_per_block`: The number of tokens managed by each block in the paged KV cache.
  * `--use_fused_mlp`: Fuses the multiple operations within the model's MLP (Multi-Layer Perceptron) blocks into a single, faster CUDA kernel.

-----

## Step 4: Build the Docker Image

This `Dockerfile` creates a self-contained, optimized runtime environment for executing the model. It cleverly avoids bundling the large model/engine files into the image itself, keeping the image lightweight and reusable.

```bash
docker build -t mistral-runner-optimized -f Dockerfile.gpu .
```

### Dockerfile Analysis:

  * `FROM nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3`: It uses NVIDIA's official Triton server image as a base. This is a best practice, as it comes pre-loaded with the correct versions of CUDA, cuDNN, TensorRT, and other essential drivers and libraries.
  * `ENV ...`: These environment variables are set to tune performance. For example, `CUDA_MODULE_LOADING=LAZY` defers the loading of CUDA kernels until they are needed, and `TLLM_ENABLE_XQA=1` enables a highly optimized kernel for models with Grouped/Multi-Query Attention.
  * `RUN ...`: This block performs setup inside the image. It uses `git` to clone a specific version (`v0.20.0`) of the TensorRT-LLM repository, copies only the necessary example scripts (like `run.py`) into the `/app` directory, and then uninstalls git to keep the final image clean.
  * **Decoupled Design:** The key insight here is that the Docker image **does not contain the model or the engine**. It only contains the *code* required to run them. The engine and tokenizer will be mounted as volumes at runtime.

-----

## Step 5: Run Inference with Docker

Finally, execute the `docker run` command to start the container and perform inference. This command brings together the Docker image, the compiled engine, and the model's tokenizer.

```bash
docker run --rm --gpus all \
 -v /mnt/ai-data/models/trtllm-engine/Mistral-7B-v0.3-int4-awq:/app/trt_engine:ro \
 -v /mnt/ai-data/models/hf/Mistral-7B-v0.3:/app/tokenizer:ro \
 mistral-runner-optimized \
 python3 /app/run.py \
 --engine_dir=/app/trt_engine \
 --tokenizer_dir=/app/tokenizer \
 --input_text="Hello World!" \
 --max_output_len=150 \
 --temperature=0.7
```

### Argument Breakdown:

  * `--rm`: Automatically removes the container when it finishes, keeping your system clean.
  * `--gpus all`: Grants the container access to all available GPUs on the host machine.
  * `-v ... :ro`: These are the volume mounts.
      * The first `-v` maps your locally built **TensorRT engine** into the container's `/app/trt_engine` directory in read-only (`ro`) mode.
      * The second `-v` maps the original model's **tokenizer** files into `/app/tokenizer`. The runner needs this to correctly process the input text.
  * `mistral-runner-optimized`: The name of the Docker image you built.
  * `python3 /app/run.py ...`: The command executed inside the container. It runs the inference script, pointing it to the mounted engine and tokenizer directories and providing the prompt and generation parameters.