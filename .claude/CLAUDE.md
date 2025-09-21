# AI Infrastructure Base Repository

## Repository Purpose

This repository serves as the **base infrastructure configuration** for an RTX 5090 AI workstation. It is designed to be consumed by other AI projects via **git submodules**, providing a consistent hardware-optimized environment across all projects.

### Key Architecture Decision
- **Base repo pattern**: This repo contains infrastructure; project repos include it as a submodule
- **Hardware-specific optimization**: Configurations tuned for RTX 5090 + 32-core CPU
- **Multi-service AI inference**: Multiple containerized AI services with different specializations

## Infrastructure Services

### GPU Services
- **llama-gpu** (port 8004): Primary GPU inference service using RTX 5090
  - CUDA-optimized for RTX 5090 Blackwell architecture
  - Container: `llama-gpu`
  - Health endpoint: `http://localhost:8004/v1/health`
  
- **vllm-gpu** (port 8005): High-performance GPU inference with vLLM
  - Optimized for large model serving and batching
  - Container: `vllm-gpu` 
  - Health endpoint: `http://localhost:8006/health`

### CPU Service (Latency Optimized)
- **llama-cpu** (port 8001): 12 dedicated cores (0-11), 96GB RAM
  - SMT disabled for exclusive L1/L2 cache access
  - Optimized for single-model inference latency
  - 20-30% faster response times vs multi-container setup
- Health endpoint: `http://localhost:8001/v1/health`

### UI Service
- **open-webui** (port 3000): Web interface for model interaction
  - Access: `http://localhost:3000`
  - Pre-configured to connect to GPU services

## Resource Configuration

### Hardware Specifications
- **GPU**: RTX 5090 (32GB VRAM) - assigned to GPU services
- **CPU**: 32 cores total, 12 dedicated to LLM (cores 0-11), 4 reserved for system (cores 12-15)
- **Memory**: 128GB+ total system RAM
- **Storage**: Models stored in `/mnt/ai-data/models/`

### Network Topology
- **Internal network**: `ai-network` (172.20.0.0/16)
- **External access**: Services exposed on localhost with specific ports
- **Security**: Read-only containers, no-new-privileges, minimal capabilities

### Model Storage
- **Host path**: `/mnt/ai-data/models/`
- **Container mount**: `/app/models:ro` (read-only)
- **Shared**: All services access the same model directory

## Development Environment

### Python Environment (Poetry)
- **Python**: 3.12 (managed via pyenv)
- **CUDA**: 12.8 optimized PyTorch dependencies
- **Key packages**: transformers, torch, accelerate, deepspeed, bitsandbytes
- **MLOps**: wandb, mlflow, tensorboard
- **Quantization**: tensorrt-llm, nemo-toolkit

### Container Runtime
- **Docker Compose**: Multi-service orchestration
- **NVIDIA Container Runtime**: GPU access for containers
- **Resource limits**: Memory and CPU constraints per service
- **Health checks**: Automated service monitoring

## Usage Patterns

### For Project Repositories (Consumers)
When this repo is included as a git submodule in your AI projects:

1. **Service Discovery**: Your project can assume these services are available:
   ```python
   # GPU inference (primary)
   GPU_API_BASE = "http://localhost:8004/v1"
   
   # High-performance GPU inference  
   VLLM_API_BASE = "http://localhost:8005/v1"
   
   # Latency-optimized CPU inference
   CPU_API_BASE = "http://localhost:8001/v1"
   ```

2. **Model Loading**: Models should be placed in `/mnt/ai-data/models/`
3. **Testing**: Use `make status` to verify services are running
4. **Performance**: GPU services for heavy workloads, CPU service for low-latency interactive chat

### For Infrastructure Maintenance
- **Service management**: Use Makefile commands (`make up`, `make gpu-up`, etc.)
- **Monitoring**: Check `make logs` and `make status`
- **Updates**: Modify this repo, other projects inherit changes via git submodule update

## Common Workflows

### Starting Services
```bash
make up              # Start all services
make gpu-up          # Start only GPU services  
make cpu-up          # Start latency-optimized CPU service
```

### Monitoring
```bash
make status          # Service status overview
make logs            # View all logs
make logs-gpu        # GPU service logs only
```

### Model Management
```bash
make shell-gpu       # Shell access to GPU container
make update-models   # Refresh model cache
```

### Development
```bash
make install         # Install Poetry dependencies
make shell          # Poetry shell environment
```

## Security Configuration

### Container Security
- **Read-only root filesystem**: Prevents runtime modifications
- **No new privileges**: Blocks privilege escalation
- **Minimal capabilities**: Drops all unnecessary Linux capabilities
- **Non-root user**: Services run as non-privileged users where possible

### Network Security
- **Internal network**: Services communicate via private Docker network
- **Localhost binding**: External ports only accessible from host
- **No external dependencies**: All required images can be built locally

## Integration with Claude Code

### Context for AI Projects
When Claude Code works in projects that include this as a submodule, it will understand:
- Available inference endpoints and their capabilities
- Hardware constraints and optimization opportunities  
- Model storage conventions and paths
- Service health monitoring and debugging approaches

### Infrastructure Updates
Changes to this base repo (hardware upgrades, service configurations, etc.) propagate to all consuming projects through git submodule updates.

## File Structure
```
.
├── .claude/CLAUDE.md         # This configuration file
├── docker/                   # Container definitions
├── docker-compose.yaml       # Service orchestration
├── docs/                     # Infrastructure documentation
│   └── optimizations/        # System performance tuning
│       ├── README.md         # Optimization overview and status
│       ├── bios-optimizations.md    # BIOS/firmware settings
│       ├── os-optimizations.md      # OS-level configuration
│       └── hugepages-setup.md       # Huge pages management
├── scripts/                  # Automation and maintenance scripts
│   └── optimizations/        # Performance management tools
│       └── manage-hugepages-models.sh  # Dynamic model loading
├── pyproject.toml           # Python environment definition
├── Makefile                 # Common commands
└── README.md                # Project overview
```

## System Optimizations

The following performance optimizations have been implemented:

### BIOS Level
- CPU C-states disabled for consistent inference latency
- Precision Boost Overdrive enabled for maximum throughput
- DDR5-6000 EXPO profiles with optimized timings
- **GPU Performance**: PCIe Gen5, Resizable BAR, Above 4G Decoding enabled

### OS Level
- Swap disabled to prevent model data paging
- CPU governor set to performance mode
- Memory locking enabled for containers
- CPU pinning via Docker cpuset (cores 0-7, 8-15, 16-23)
- Huge pages (2MB) configured for model loading
- THP (Transparent Huge Pages) disabled

### GPU Optimizations (RTX 5090)
**Three-tier optimization strategy for maximum GPU performance:**

#### 1. BIOS Configuration (Foundation)
- PCIe Gen5 x16 for 128 GB/s bidirectional bandwidth
- Resizable BAR and Above 4G Decoding for direct VRAM access
- See `docs/optimizations/bios/bios-optimizations.md`

#### 2. OS-Level GPU Settings (System)
- **NVIDIA driver persistence mode**: Reduces CUDA initialization by 200ms
- **Clock locking**: GPU @ 2400-2550 MHz, Memory @ 3002 MHz for consistent performance
- **Power management**: 600W power limit, exclusive compute mode
- **IRQ affinity**: GPU interrupts pinned to cores 24-31
- **Kernel parameters**: Optimized for Blackwell architecture
- See `docs/optimizations/os/os-optimizations.md#gpu-optimizations`

#### 3. Container Runtime (Application)
- **CUDA memory pools**: Reduced allocation overhead by 30%
- **Tensor cores**: 5th gen optimization for 40% faster GEMM ops
- **FP8 support**: Up to 2x throughput for compatible models
- **CUDA graphs**: 20% reduction in kernel launch overhead
- **L2 cache**: Full 96MB utilization for Blackwell
- See `docs/optimizations/gpu/gpu-optimizations.md`

### Performance Verification
- Baseline benchmark: `scripts/benchmark.py --label "baseline"`
- Post-optimization: `scripts/benchmark.py --label "optimized"`
- Expected improvements:
  - PCIe Gen5 vs Gen1: 32x bandwidth increase
  - Overall inference: 40-100% throughput gain
  - Latency reduction: 200-500ms first token time

See `docs/optimizations/README.md` for verification commands and full details.

## Hardware Upgrade Path
This configuration is optimized for RTX 5090 but designed to be adaptable:
- **GPU upgrades**: Modify CUDA versions and memory limits in docker-compose.yaml
- **CPU scaling**: Adjust core allocation (currently 12 cores for LLM, 4 for system)
- **Memory expansion**: Update service memory limits
- **Multi-GPU**: Add additional GPU services with different device assignments

## Documentation Standards

### Linux Philosophy
All documentation in this repository follows Linux/Unix conventions:
- **Plain text focus**: Clear, concise technical writing without decorative elements
- **No emojis**: Professional documentation uses standard ASCII characters only
- **Command-line oriented**: Examples use shell commands and standard Unix tools
- **Minimalist approach**: Include only essential information, avoid redundancy
- **Man page style**: Structure similar to traditional Unix manual pages when appropriate