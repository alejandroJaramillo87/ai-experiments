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

### CPU Services (Load Balanced)
- **llama-cpu-0** (port 8001): CPU cores 0-7, 32GB RAM
- **llama-cpu-1** (port 8002): CPU cores 8-15, 32GB RAM  
- **llama-cpu-2** (port 8003): CPU cores 16-23, 32GB RAM
- Health endpoints: `http://localhost:800[1-3]/v1/health`

### UI Service
- **open-webui** (port 3000): Web interface for model interaction
  - Access: `http://localhost:3000`
  - Pre-configured to connect to GPU services

## Resource Configuration

### Hardware Specifications
- **GPU**: RTX 5090 (32GB VRAM) - assigned to GPU services
- **CPU**: 32 cores total, segmented across CPU services (8 cores each)
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
   
   # Load-balanced CPU inference
   CPU_APIS = [
       "http://localhost:8001/v1",
       "http://localhost:8002/v1", 
       "http://localhost:8003/v1"
   ]
   ```

2. **Model Loading**: Models should be placed in `/mnt/ai-data/models/`
3. **Testing**: Use `make status` to verify services are running
4. **Performance**: GPU services for heavy workloads, CPU services for lighter tasks

### For Infrastructure Maintenance
- **Service management**: Use Makefile commands (`make up`, `make gpu-up`, etc.)
- **Monitoring**: Check `make logs` and `make status`
- **Updates**: Modify this repo, other projects inherit changes via git submodule update

## Common Workflows

### Starting Services
```bash
make up              # Start all services
make gpu-up          # Start only GPU services  
make cpu-up          # Start only CPU services
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
├── scripts/                  # Automation and maintenance scripts  
├── pyproject.toml           # Python environment definition
├── Makefile                 # Common commands
└── README.md                # Project overview
```

## Hardware Upgrade Path
This configuration is optimized for RTX 5090 but designed to be adaptable:
- **GPU upgrades**: Modify CUDA versions and memory limits in docker-compose.yaml
- **CPU scaling**: Adjust cpuset configurations and add more CPU services
- **Memory expansion**: Update service memory limits
- **Multi-GPU**: Add additional GPU services with different device assignments