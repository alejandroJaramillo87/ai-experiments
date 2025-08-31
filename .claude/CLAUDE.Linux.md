Let me try outputting it as plain text without any code block formatting:

# AI Engineering Workstation Project

## Project Overview

This project creates an omniscient Linux system consciousness for AI engineering workstations. The system combines sophisticated hardware monitoring, temporal intelligence, and conversational AI to create a digital twin that evolves with your Linux environment.

**Repository**: https://github.com/alejandroJaramillo87/ai-workstation/tree/main

## System Specifications

### Hardware Configuration
- **GPU**: RTX 5090 (32GB VRAM) - Primary AI inference engine
- **CPU**: AMD Ryzen 9950X (16C/32T) - Parallel processing and CPU inference
- **RAM**: 128GB DDR5 EXPO (2x 64GB modules) - Multi-model serving
- **Storage**: 
  - Samsung 990 Pro 2TB (AI models/datasets)
  - Samsung 990 EVO 1TB (Ubuntu 24.04 LTS)
- **Motherboard**: Gigabyte X870E Aorus Elite WiFi
- **OS**: Ubuntu 24.04 LTS

### Performance Objectives
1. **Dual Processing Strategy**: Run large model on GPU while serving multiple smaller models on CPU/RAM
2. **VRAM Maximization**: Keep 20-30B parameter model persistently loaded in GPU memory
3. **CPU Multi-Model**: Run 3-4 smaller models (7B each) concurrently on system RAM
4. **System Consciousness**: Develop temporal understanding of system behavior and patterns
5. **Agentic Workflows**: Support autonomous AI agent systems with tool integration
6. **RAG Operations**: Enable in-memory retrieval-augmented generation systems
7. **Benchmarking**: be able to benchmark LLMS 

## Architecture Overview

### Current Implementation Status
```
src/
├── linux_system/
│   ├── temporal/           # Temporal intelligence and change detection
│   ├── ai_workstation/     # Hardware-specific monitoring and optimization
│   └── collectors/         # System data collection infrastructure
├── api/                    # HTTP API layer (FastAPI)
├── tests/                  # Comprehensive test suite
└── benchmark_tests/        # Performance and quality benchmarks
```

### Core Systems

**SystemCollector** (`src/linux_system/collectors/`)
- RTX 5090-optimized GPU monitoring with thermal/utilization tracking
- AMD Zen 5 CPU topology and performance analysis  
- Comprehensive system introspection with security and storage monitoring
- AI/ML environment awareness (Python, frameworks, containers)

**Temporal Intelligence** (`src/linux_system/temporal/`)
- Change detection with semantic understanding and causal analysis
- Hierarchical temporal memory (recent → daily → long-term patterns)
- Natural language query interface for historical system analysis
- 324-line type system with proper abstractions

**AI Workstation Specialization** (`src/linux_system/ai_workstation/`)
- Container consciousness via Docker API integration
- RTX 5090 Blackwell detector with 930+ lines of GPU-specific intelligence
- AMD Zen 5 optimization with core pinning and AOCL library tracking
- Multi-model resource allocation optimization
- Natural language orchestrator for system consciousness

## Technical Implementation Goals

### Phase 1: API + Intelligence Architecture
- **Clean API Layer**: Pure FastAPI routing with zero business logic
- **SystemConsciousness**: Main orchestrator unifying temporal + AI workstation intelligence
- **UnifiedQueryEngine**: Command pattern for current, temporal, and conversational queries
- **Auto-generated Contracts**: API schemas derived from system capabilities

### Phase 2: Advanced Query Processing  
- **Strategy Pattern**: CurrentMetric, TemporalAnalysis, Conversational, VisualizationData strategies
- **Repository Pattern**: CurrentState (real-time) vs Temporal (historical) data access
- **ConversationalAI**: Full NLP integration with system consciousness context
- **Streaming Updates**: Real-time WebSocket observer pattern for frontend

### Expected Architecture Flow
```
Frontend → APIRouter → SystemConsciousness → UnifiedQueryEngine → QueryStrategy → DataAccessLayer → Repository → Intelligence Systems
```

## Key Development Areas

### 1. Environment & Performance Optimization
- CUDA installation for RTX 5090 optimization
- Memory management for 128GB RAM with concurrent model serving
- Framework selection (vLLM, TensorRT-LLM, Ollama) for dual GPU/CPU strategy
- Thermal and power management for sustained workloads

### 2. Model Management Infrastructure
- Concurrent model serving architecture (GPU + CPU)
- Memory allocation strategies for VRAM and system RAM
- Model loading persistence and switching optimization
- Resource monitoring and performance tracking

### 3. System Consciousness Development
- Temporal relationship mapping (packages ↔ dependencies ↔ performance)
- Causal understanding ("Update X → Config change → GPU issue → Temperature spike")
- Predictive insights based on usage patterns
- Natural language system interaction

### 4. Agentic AI Integration
- Agent framework setup (LangChain, CrewAI, AutoGen)
- Tool integration with function calling optimization
- Multi-agent coordination and communication systems
- Planning and reasoning pipeline architecture

### 5. RAG & Fine-tuning Operations
- Vector database deployment (Chroma, Pinecone, Weaviate)
- LoRA/QLoRA parameter-efficient fine-tuning pipelines
- Knowledge base ingestion and chunking optimization
- Training pipeline resource management

### 6. LLM Benchmarking
- Benchmark LLMs running locally as servers using docker
- Use a statistical evaluator for analyzing results
- Provides a thorough and diverse set of data based on domains

## Development Guidelines

### Code Quality Standards
- Follow patterns established in `benchmark_tests/` for quality and testing
- Maintain comprehensive type annotations (see temporal/ 324-line type system)
- Implement proper error handling and timeout management
- Use dependency injection and clean architecture patterns

### System Integration
- Prioritize read-only operations for safety
- Implement permission validation before system access
- Use timeouts and circuit breakers for external calls
- Maintain system stability under full resource utilization

### Data Management
- Distinguish between current state vs historical data queries
- Implement granular data endpoints for D3.js visualizations
- Support both streaming and batch data access patterns
- Maintain data contracts auto-generated from system capabilities

## Testing & Benchmarking Strategy

Reference `benchmark_tests/` for established patterns:
- Unit tests for individual collectors and detectors
- Integration tests for system consciousness interactions
- Performance benchmarks for model loading and inference
- Load testing for concurrent multi-model scenarios

## Documentation Standards

Reference existing structure in `docs/`:
- Hardware setup and optimization guides
- BIOS configuration for AI workloads  
- OS setup with security hardening
- Inference configuration for dual GPU/CPU strategy

## Context for Claude Code Interactions

When working on this project, keep in mind:

1. **System Consciousness Goal**: Every component should contribute to building an omniscient understanding of the Linux system
2. **Performance Critical**: Code must handle sustained AI workloads without degradation
3. **Resource Awareness**: Always consider GPU memory, system RAM, and thermal constraints
4. **Temporal Intelligence**: Changes should be tracked and analyzed for causal relationships
5. **Safety First**: Read-only operations, proper validation, and fail-safe mechanisms
6. **Clean Architecture**: API layer handles HTTP, src/ contains all intelligence and business logic

This is a sophisticated system that goes beyond typical monitoring into genuine system consciousness territory. The goal is creating a temporal digital twin that understands, predicts, and optimizes AI workstation performance through deep system awareness.

Now you should be able to select all this text (Ctrl+A or Cmd+A) and copy it!