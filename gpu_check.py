import torch
import time

def check_cuda():
    print("✅ CUDA Available:", torch.cuda.is_available())
    print("🧠 CUDA Device Count:", torch.cuda.device_count())
    print("🚀 Current Device Index:", torch.cuda.current_device())
    print("🎯 Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("🧱 CUDA Version:", torch.version.cuda)
    print("🔥 PyTorch Built With CUDA:", torch.backends.cuda.is_built())

def tensor_test():
    print("\nRunning tensor computation on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a = torch.rand(10000, 10000, device=device)
    b = torch.rand(10000, 10000, device=device)

    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # wait for GPU to finish computation
    end = time.time()

    print(f"✅ Matrix multiplication completed in {end - start:.3f} seconds on {device}")

if __name__ == "__main__":
    check_cuda()
    tensor_test()
