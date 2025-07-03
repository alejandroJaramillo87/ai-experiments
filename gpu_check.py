import torch
import time
import argparse
import subprocess
import sys


def get_gpu_temperature(gpu_index=0):
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i", str(gpu_index)
        ], encoding="utf-8")
        return int(result.strip())
    except Exception as e:
        print(f"Error fetching temperature: {e}")
        return -1


def get_gpu_utilization(gpu_index=0):
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i", str(gpu_index)
        ], encoding="utf-8")
        return int(result.strip())
    except Exception as e:
        print(f"Error fetching utilization: {e}")
        return -1


def check_cuda(device_index=0):
    print("✅ CUDA Available:", torch.cuda.is_available())
    print("🧠 CUDA Device Count:", torch.cuda.device_count())
    print("🚀 Using Device Index:", device_index)
    print("🎯 Device Name:", torch.cuda.get_device_name(device_index))
    print("🧱 CUDA Version:", torch.version.cuda)
    print("🔥 Built With CUDA:", torch.backends.cuda.is_built())


def tensor_benchmark(size=10000, dtype="float32", runs=10, warmup=3, device_index=0, temp_limit=85):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }

    if dtype.lower() == "float32":
        torch.set_float32_matmul_precision('high')

    dtype_selected = dtype_map.get(dtype.lower(), torch.float32)
    device = torch.device(f"cuda:{device_index}")

    print(f"\n🧪 Benchmark: {runs} matmuls | dtype={dtype} | size={size}x{size} | GPU={device_index}")
    
    # Allocate large tensors once to minimize overhead
    try:
        a = torch.rand(size, size, dtype=dtype_selected, device=device)
        b = torch.rand(size, size, dtype=dtype_selected, device=device)
    except RuntimeError as e:
        print(f"❌ Allocation failed: {e}")
        sys.exit(1)

    # Warm-up
    for _ in range(warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark loop
    for i in range(runs):
        temp = get_gpu_temperature(device_index)
        if temp >= temp_limit:
            print(f"🛑 Temp {temp}°C exceeds safe limit of {temp_limit}°C. Aborting.")
            break

        start = time.time()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()

        util = get_gpu_utilization(device_index)
        print(f"✅ Run {i+1:02d}: {end - start:.4f}s | Temp: {temp}°C | Util: {util}%")

        time.sleep(0.25)  # Less idle time between runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Benchmarking Script")
    parser.add_argument("--size", type=int, default=10000, help="Matrix size (N for NxN)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed matmul runs")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up iterations")
    parser.add_argument("--device", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--temp-limit", type=int, default=85, help="Max temp in Celsius before aborting")

    args = parser.parse_args()

    check_cuda(args.device)
    tensor_benchmark(
        size=args.size,
        dtype=args.dtype,
        runs=args.runs,
        warmup=args.warmup,
        device_index=args.device,
        temp_limit=args.temp_limit
    )
