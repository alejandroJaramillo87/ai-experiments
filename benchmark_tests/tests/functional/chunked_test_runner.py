#!/usr/bin/env python3
"""
Intelligent Test Chunking System

Handles large-scale test execution with:
- Smart test batching for 26k+ test suites
- Progress indicators and timeout management  
- Memory usage monitoring and cleanup
- Backend-aware concurrency (llama.cpp vs vLLM)
- Graceful degradation and error recovery

Designed to prevent timeout issues while maintaining comprehensive coverage.
"""

import time
import logging
import psutil
import threading
from typing import List, Dict, Any, Optional, Union, Generator
from dataclasses import dataclass
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for test chunking"""
    chunk_size: int = 10          # Tests per chunk
    max_concurrent: int = 1       # Concurrent chunks (1 for llama.cpp, 4+ for vLLM)
    timeout_per_test: int = 30    # Seconds per individual test
    chunk_timeout: int = 600      # Seconds per chunk (10 minutes)
    memory_limit_mb: int = 8000   # Memory limit before cleanup
    progress_interval: int = 5    # Progress updates every N tests

@dataclass 
class ChunkResult:
    """Result of processing a test chunk"""
    chunk_id: int
    tests_processed: int
    tests_passed: int
    tests_failed: int
    processing_time: float
    memory_usage_mb: float
    errors: List[str]
    results: List[Dict[str, Any]]

class BackendDetector:
    """Detect whether we're running llama.cpp (sequential) or vLLM (concurrent)"""
    
    @staticmethod
    def detect_backend() -> str:
        """Detect backend type from Docker logs"""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "compose", "logs", "llama-gpu"], 
                capture_output=True, text=True, timeout=10
            )
            
            logs = result.stdout.lower()
            if any(keyword in logs for keyword in ['llama.cpp', 'llama-server', 'gguf', 'llamacpp']):
                return 'llama.cpp'
            elif any(keyword in logs for keyword in ['vllm', 'ray', 'asyncio']):
                return 'vllm'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    @staticmethod
    def get_optimal_concurrency() -> int:
        """Get optimal concurrency based on backend"""
        backend = BackendDetector.detect_backend()
        if backend == 'llama.cpp':
            return 1  # Sequential only
        elif backend == 'vllm':
            return min(4, psutil.cpu_count() // 2)  # Conservative parallel
        else:
            return 1  # Safe default

class ResourceMonitor:
    """Monitor system resources during test execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.monitoring = False
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.peak_memory = 0
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                time.sleep(1)  # Check every second
            except Exception:
                break
                
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def cleanup_if_needed(self, limit_mb: int) -> bool:
        """Force garbage collection if memory exceeds limit"""
        if self.get_current_memory_mb() > limit_mb:
            import gc
            gc.collect()
            time.sleep(0.5)  # Allow cleanup to take effect
            return True
        return False

class ChunkedTestRunner:
    """Main chunked test execution engine"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.resource_monitor = ResourceMonitor()
        
        # Auto-detect optimal concurrency
        self.config.max_concurrent = BackendDetector.get_optimal_concurrency()
        backend = BackendDetector.detect_backend()
        
        logger.info(f"ChunkedTestRunner initialized")
        logger.info(f"Backend detected: {backend}")
        logger.info(f"Max concurrent chunks: {self.config.max_concurrent}")
        logger.info(f"Chunk size: {self.config.chunk_size}")
    
    def chunk_test_list(self, test_list: List[Dict[str, Any]]) -> Generator[List[Dict[str, Any]], None, None]:
        """Split test list into manageable chunks"""
        for i in range(0, len(test_list), self.config.chunk_size):
            yield test_list[i:i + self.config.chunk_size]
    
    def execute_test_chunk(self, chunk_id: int, test_chunk: List[Dict[str, Any]], 
                          base_args: List[str]) -> ChunkResult:
        """Execute a single chunk of tests"""
        logger.info(f"Processing chunk {chunk_id} ({len(test_chunk)} tests)")
        
        start_time = time.time()
        start_memory = self.resource_monitor.get_current_memory_mb()
        
        passed = 0
        failed = 0
        errors = []
        results = []
        
        for test_idx, test_data in enumerate(test_chunk):
            try:
                # Memory cleanup check
                self.resource_monitor.cleanup_if_needed(self.config.memory_limit_mb)
                
                # Execute individual test  
                test_result = self._execute_single_test(test_data, base_args)
                
                if test_result.get('success', False):
                    passed += 1
                else:
                    failed += 1
                    
                results.append(test_result)
                
                # Progress logging
                if (test_idx + 1) % self.config.progress_interval == 0:
                    logger.info(f"Chunk {chunk_id}: {test_idx + 1}/{len(test_chunk)} tests processed")
                    
            except Exception as e:
                error_msg = f"Test {test_data.get('id', 'unknown')} failed: {str(e)}"
                errors.append(error_msg)
                failed += 1
                logger.error(error_msg)
        
        end_time = time.time()
        processing_time = end_time - start_time
        memory_usage = self.resource_monitor.get_current_memory_mb() - start_memory
        
        logger.info(f"Chunk {chunk_id} completed: {passed} passed, {failed} failed, "
                   f"{processing_time:.1f}s, {memory_usage:.1f}MB")
        
        return ChunkResult(
            chunk_id=chunk_id,
            tests_processed=len(test_chunk),
            tests_passed=passed,
            tests_failed=failed,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            errors=errors,
            results=results
        )
    
    def _execute_single_test(self, test_data: Dict[str, Any], base_args: List[str]) -> Dict[str, Any]:
        """Execute a single test with timeout protection"""
        import subprocess
        
        # Build test-specific command
        test_args = base_args.copy()
        test_args.extend([
            "--test-id", test_data.get('id', 'unknown')
        ])
        
        try:
            # Execute with timeout
            result = subprocess.run(
                ["python", "benchmark_runner.py"] + test_args,
                timeout=self.config.timeout_per_test,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            return {
                'test_id': test_data.get('id'),
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': self.config.timeout_per_test,  # Approximate
                'exit_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'test_id': test_data.get('id'),
                'success': False,
                'error': 'Test timed out',
                'execution_time': self.config.timeout_per_test
            }
        except Exception as e:
            return {
                'test_id': test_data.get('id'),
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def execute_chunked_tests(self, test_list: List[Dict[str, Any]], 
                            base_args: List[str]) -> Dict[str, Any]:
        """Execute full test suite using intelligent chunking"""
        logger.info(f"Starting chunked execution of {len(test_list)} tests")
        logger.info(f"Configuration: {self.config.chunk_size} tests per chunk, "
                   f"{self.config.max_concurrent} concurrent chunks")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            chunks = list(self.chunk_test_list(test_list))
            total_chunks = len(chunks)
            
            logger.info(f"Split into {total_chunks} chunks")
            
            all_results = []
            total_passed = 0
            total_failed = 0
            all_errors = []
            
            if self.config.max_concurrent == 1:
                # Sequential processing for llama.cpp
                for chunk_id, chunk in enumerate(chunks):
                    chunk_result = self.execute_test_chunk(chunk_id, chunk, base_args)
                    all_results.append(chunk_result)
                    total_passed += chunk_result.tests_passed
                    total_failed += chunk_result.tests_failed
                    all_errors.extend(chunk_result.errors)
                    
            else:
                # Concurrent processing for vLLM
                with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
                    future_to_chunk = {
                        executor.submit(self.execute_test_chunk, chunk_id, chunk, base_args): chunk_id
                        for chunk_id, chunk in enumerate(chunks)
                    }
                    
                    for future in as_completed(future_to_chunk):
                        chunk_result = future.result()
                        all_results.append(chunk_result)
                        total_passed += chunk_result.tests_passed
                        total_failed += chunk_result.tests_failed
                        all_errors.extend(chunk_result.errors)
            
            return {
                'summary': {
                    'total_tests': len(test_list),
                    'total_passed': total_passed,
                    'total_failed': total_failed,
                    'total_chunks': total_chunks,
                    'success_rate': total_passed / len(test_list) if test_list else 0,
                    'peak_memory_mb': self.resource_monitor.peak_memory
                },
                'chunk_results': all_results,
                'errors': all_errors
            }
            
        finally:
            self.resource_monitor.stop_monitoring()

def create_quick_test_runner(chunk_size: int = 5, timeout: int = 15) -> ChunkedTestRunner:
    """Create a test runner optimized for quick execution"""
    config = ChunkConfig(
        chunk_size=chunk_size,
        timeout_per_test=timeout,
        chunk_timeout=timeout * chunk_size + 60,
        memory_limit_mb=4000,
        progress_interval=2
    )
    return ChunkedTestRunner(config)

def create_comprehensive_test_runner(chunk_size: int = 25, timeout: int = 60) -> ChunkedTestRunner:
    """Create a test runner optimized for comprehensive testing"""  
    config = ChunkConfig(
        chunk_size=chunk_size,
        timeout_per_test=timeout,
        chunk_timeout=timeout * chunk_size + 300,
        memory_limit_mb=12000,
        progress_interval=5
    )
    return ChunkedTestRunner(config)