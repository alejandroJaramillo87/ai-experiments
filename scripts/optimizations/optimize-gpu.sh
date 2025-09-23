#!/bin/bash
set -e

echo "Starting GPU optimizations for RTX 5090 AI inference..."

# Enable persistence mode
echo "Enabling NVIDIA persistence mode..."
nvidia-smi -pm 1

# Set maximum power limit (600W for RTX 5090)
echo "Setting power limit to 600W..."
nvidia-smi -pl 600

# Clock locking disabled - causes performance degradation on RTX 5090
# echo "Locking GPU clocks to 2400-2550 MHz..."
# nvidia-smi -lgc 2400,2550

# Memory clock locking disabled - causes performance degradation on RTX 5090
# echo "Locking memory clocks to 3002 MHz..."
# nvidia-smi -lmc 3002

# Compute mode set to DEFAULT for optimal performance
echo "Setting compute mode to DEFAULT..."
nvidia-smi -c DEFAULT

# Set GPU IRQ affinity to cores 24-31
echo "Setting GPU IRQ affinity..."
GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" | awk '{print $1}' | tr -d ':')
if [ -n "$GPU_IRQS" ]; then
    for irq in $GPU_IRQS; do
        echo f0000000 > /proc/irq/$irq/smp_affinity 2>/dev/null || true
    done
    echo "GPU IRQs pinned to cores 24-31"
else
    echo "No GPU IRQs found to pin"
fi

# Create CUDA cache directory if it doesn't exist
if [ ! -d /tmp/cuda_cache ]; then
    mkdir -p /tmp/cuda_cache
    chmod 1777 /tmp/cuda_cache
    echo "Created CUDA cache directory"
fi

echo "GPU optimization complete!"
echo ""
echo "=== Current GPU Settings ==="
nvidia-smi --query-gpu=persistence_mode,power.limit,clocks.gr,clocks.mem,compute_mode --format=csv
echo ""
echo "=== PCIe Status ==="
nvidia-smi -q | grep -A 4 "GPU Link Info"