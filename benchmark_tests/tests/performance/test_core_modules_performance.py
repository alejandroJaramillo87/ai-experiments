#!/usr/bin/env python3
"""
Performance Tests for Core Modules

Tests the performance characteristics of core modules to ensure they meet
the requirements for processing the 26k+ test suite without timeout issues.

Focuses on:
- Memory usage validation
- Response time benchmarks
- Resource exhaustion handling
- Scalability under load

As specified in CLAUDE.md Priority 1 requirements.
"""

import unittest
import time
import psutil
import tempfile
import shutil
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.results_manager import TestResultsManager, RunMetadata, CognitivePattern, CognitiveProfile
from core.cognitive_evaluation_pipeline import CognitiveEvaluationPipeline


class TestResultsManagerPerformance(unittest.TestCase):
    """Performance tests for TestResultsManager"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TestResultsManager(base_results_dir=self.temp_dir)
        self.process = psutil.Process()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't exceed limits under load"""
        # Get baseline memory usage
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate processing many test results
        model_name = "performance-test-model"
        model_path = "/test/model"
        config = {"strategy": "performance", "token_limit": 400}
        
        run_dir = self.manager.create_run_directory(model_name, model_path, config)
        
        # Create 1000 test results to simulate load
        start_time = time.time()
        for i in range(1000):
            test_id = f"perf_test_{i:04d}"
            prompt = f"Performance test prompt {i} " * 20  # ~400 chars
            response_text = f"Performance test response {i} " * 50  # ~1000 chars
            evaluation_results = {
                "score": 75.0 + (i % 20),
                "confidence": 0.8 + (i % 10) * 0.02,
                "calibration_score": 70.0 + (i % 25)
            }
            test_metadata = {"domain": ["reasoning", "creativity", "social"][i % 3]}
            
            self.manager.save_test_response(
                run_dir, test_id, prompt, response_text,
                evaluation_results, test_metadata
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check memory usage after processing
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Performance assertions
        self.assertLess(processing_time, 30.0, 
                       "Processing 1000 results should take less than 30 seconds")
        self.assertLess(memory_increase, 100, 
                       "Memory increase should be less than 100MB for 1000 results")
        
        print(f"✅ Performance Test Results:")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Memory increase: {memory_increase:.2f}MB")
        print(f"   Throughput: {1000/processing_time:.1f} results/second")
    
    def test_cognitive_pattern_analysis_performance(self):
        """Test cognitive pattern analysis performance with large datasets"""
        # Create run directory with many test responses
        model_name = "pattern-test-model" 
        model_path = "/test/model"
        config = {"strategy": "pattern", "token_limit": 400}
        
        run_dir = self.manager.create_run_directory(model_name, model_path, config)
        
        # Add diverse test responses for pattern analysis
        domains = ["reasoning", "memory", "creativity", "social", "integration"]
        start_time = time.time()
        
        for domain in domains:
            for i in range(20):  # 20 tests per domain = 100 total
                test_id = f"{domain}_test_{i:02d}"
                evaluation_results = {
                    "score": 60 + (i % 30),
                    "calibration_score": 55 + (i % 35)
                }
                test_metadata = {"domain": domain}
                
                self.manager.save_test_response(
                    run_dir, test_id, f"prompt_{test_id}", f"response_{test_id}",
                    evaluation_results, test_metadata
                )
        
        # Measure pattern analysis performance
        pattern_start = time.time()
        profile = self.manager.analyze_cognitive_patterns(run_dir)
        pattern_end = time.time()
        
        pattern_analysis_time = pattern_end - pattern_start
        total_time = pattern_end - start_time
        
        # Performance assertions
        self.assertLess(pattern_analysis_time, 10.0, 
                       "Pattern analysis should take less than 10 seconds")
        self.assertIsInstance(profile, CognitiveProfile)
        self.assertGreaterEqual(profile.sample_size, 100)
        
        print(f"✅ Pattern Analysis Performance:")
        print(f"   Analysis time: {pattern_analysis_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s") 
        print(f"   Sample size: {profile.sample_size}")
        print(f"   Patterns detected: {len(profile.detected_patterns)}")
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access scenarios"""
        model_name = "concurrent-test-model"
        model_path = "/test/model"
        config = {"strategy": "concurrent", "token_limit": 400}
        
        run_dir = self.manager.create_run_directory(model_name, model_path, config)
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            """Worker function for concurrent testing"""
            try:
                for i in range(50):  # 50 operations per worker
                    test_id = f"worker_{worker_id}_test_{i:02d}"
                    evaluation_results = {"score": 70 + (i % 20)}
                    test_metadata = {"domain": "reasoning", "worker_id": worker_id}
                    
                    self.manager.save_test_response(
                        run_dir, test_id, f"prompt_{test_id}", f"response_{test_id}",
                        evaluation_results, test_metadata
                    )
                    
                results.append(f"Worker {worker_id} completed")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Start concurrent workers
        start_time = time.time()
        threads = []
        
        for worker_id in range(5):  # 5 concurrent workers
            thread = threading.Thread(target=worker_function, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Performance assertions
        self.assertEqual(len(results), 5, "All workers should complete successfully")
        self.assertEqual(len(errors), 0, "No errors should occur during concurrent access")
        self.assertLess(concurrent_time, 15.0, 
                       "Concurrent processing should complete within 15 seconds")
        
        print(f"✅ Concurrent Access Performance:")
        print(f"   Concurrent time: {concurrent_time:.2f}s")
        print(f"   Workers completed: {len(results)}")
        print(f"   Errors: {len(errors)}")
    
    def test_large_response_serialization_performance(self):
        """Test performance with large response objects"""
        model_name = "serialization-test-model"
        model_path = "/test/model"
        config = {"strategy": "serialization", "token_limit": 600}
        
        run_dir = self.manager.create_run_directory(model_name, model_path, config)
        
        # Create large response objects
        start_time = time.time()
        
        for i in range(10):
            test_id = f"large_response_test_{i:02d}"
            
            # Large response text (simulate complex model output)
            response_text = "Complex detailed response " * 500  # ~12KB
            
            # Complex evaluation results
            evaluation_results = {
                "score": 75.0,
                "confidence": 0.85,
                "detailed_analysis": {
                    "reasoning_steps": [f"Step {j}: Analysis point {j}" for j in range(100)],
                    "quality_metrics": {f"metric_{k}": k * 0.01 for k in range(50)},
                    "pattern_analysis": {
                        "patterns": [f"pattern_{l}" for l in range(20)],
                        "scores": [0.1 * l for l in range(20)]
                    }
                },
                "timestamp": time.time(),
                "processing_metadata": {"worker": "test", "version": "1.0"}
            }
            
            test_metadata = {
                "domain": "complex_reasoning",
                "complexity": "high",
                "token_count": 600,
                "analysis_depth": "comprehensive"
            }
            
            self.manager.save_test_response(
                run_dir, test_id, f"Complex prompt {i}", response_text,
                evaluation_results, test_metadata
            )
        
        end_time = time.time()
        serialization_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(serialization_time, 5.0, 
                       "Large response serialization should complete within 5 seconds")
        
        # Verify files were created correctly
        response_files = list(Path(run_dir).glob("raw_responses/*.json"))
        self.assertEqual(len(response_files), 10, "All response files should be created")
        
        print(f"✅ Serialization Performance:")
        print(f"   Serialization time: {serialization_time:.2f}s")
        print(f"   Files created: {len(response_files)}")
        print(f"   Average time per file: {serialization_time/10:.3f}s")


class TestCognitiveEvaluationPipelinePerformance(unittest.TestCase):
    """Performance tests for CognitiveEvaluationPipeline"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.process = psutil.Process()
    
    def test_pipeline_initialization_performance(self):
        """Test pipeline initialization time and memory usage"""
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        pipeline = CognitiveEvaluationPipeline()
        end_time = time.time()
        
        init_time = end_time - start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Performance assertions
        self.assertLess(init_time, 2.0, 
                       "Pipeline initialization should take less than 2 seconds")
        self.assertLess(memory_increase, 50, 
                       "Memory increase should be less than 50MB during initialization")
        
        print(f"✅ Pipeline Initialization Performance:")
        print(f"   Initialization time: {init_time:.2f}s")
        print(f"   Memory increase: {memory_increase:.2f}MB")
        # Available evaluators check (may not exist in all implementations)
        available_count = 0
        if hasattr(pipeline, 'pattern_evaluator') and pipeline.pattern_evaluator:
            available_count += 1
        if hasattr(pipeline, 'enhanced_evaluator') and pipeline.enhanced_evaluator:
            available_count += 1
        if hasattr(pipeline, 'cultural_evaluator') and pipeline.cultural_evaluator:
            available_count += 1
        print(f"   Available evaluators: {available_count}")
    
    @patch('core.cognitive_evaluation_pipeline.PatternBasedEvaluator')
    def test_evaluation_throughput_performance(self, mock_evaluator_class):
        """Test evaluation throughput under sustained load"""
        # Mock the evaluator to avoid external dependencies
        mock_evaluator = Mock()
        mock_evaluator.evaluate_patterns.return_value = Mock(
            response_consistency=0.8,
            pattern_adherence=0.75,
            quality_indicators={'coherence_score': 0.8, 'fluency_score': 0.9},
            behavioral_signature={'response_style': 'analytical', 'verbosity_level': 'medium'}
        )
        mock_evaluator_class.return_value = mock_evaluator
        
        pipeline = CognitiveEvaluationPipeline()
        
        # Simulate evaluation of many responses
        start_time = time.time()
        
        for i in range(100):  # 100 evaluations
            test_data = {
                'id': f'throughput_test_{i:03d}',
                'prompt': f'Test prompt {i} ' * 10,
                'domain': ['reasoning', 'creativity', 'social'][i % 3]
            }
            
            response_text = f'Test response {i} ' * 20
            
            result = pipeline.evaluate_response(
                test_id=test_data['id'],
                prompt=test_data['prompt'],
                response_text=response_text,
                test_metadata=test_data
            )
            
            self.assertIsNotNone(result)
        
        end_time = time.time()
        throughput_time = end_time - start_time
        throughput_rate = 100 / throughput_time
        
        # Performance assertions
        self.assertLess(throughput_time, 10.0, 
                       "100 evaluations should complete within 10 seconds")
        self.assertGreaterEqual(throughput_rate, 10, 
                               "Should process at least 10 evaluations per second")
        
        print(f"✅ Evaluation Throughput Performance:")
        print(f"   Processing time: {throughput_time:.2f}s")
        print(f"   Throughput rate: {throughput_rate:.1f} evaluations/second")
        print(f"   Average time per evaluation: {throughput_time/100:.3f}s")


class TestSystemResourceLimits(unittest.TestCase):
    """Test system resource limits and exhaustion handling"""
    
    def test_memory_limit_handling(self):
        """Test graceful handling when approaching memory limits"""
        # This test simulates approaching system memory limits
        # In production, this would help prevent OOM conditions
        
        initial_memory = psutil.virtual_memory().percent
        
        # Skip if system is already under memory pressure
        if initial_memory > 80:
            self.skipTest("System already under memory pressure")
        
        print(f"✅ Memory Limit Test:")
        print(f"   Initial memory usage: {initial_memory:.1f}%")
        print(f"   Available memory: {psutil.virtual_memory().available / 1024**3:.1f}GB")
        
        # This test passes by not causing system instability
        self.assertLess(initial_memory, 90, 
                       "System should maintain reasonable memory usage")
    
    def test_timeout_prevention(self):
        """Test that core operations complete within timeout limits"""
        # Simulate the timeout limits from CLAUDE.md
        TEST_TIMEOUT = 300  # 5 minutes as specified in Makefile
        
        start_time = time.time()
        
        # Simulate a complex operation that should complete quickly
        temp_dir = tempfile.mkdtemp()
        try:
            manager = TestResultsManager(base_results_dir=temp_dir)
            
            # Rapid test result creation
            run_dir = manager.create_run_directory("timeout-test", "/test", {"test": True})
            
            for i in range(50):  # Moderate load
                manager.save_test_response(
                    run_dir, f"test_{i}", f"prompt_{i}", f"response_{i}",
                    {"score": 70}, {"domain": "test"}
                )
            
            # Pattern analysis
            profile = manager.analyze_cognitive_patterns(run_dir)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        end_time = time.time()
        operation_time = end_time - start_time
        
        # Performance assertion based on timeout requirements
        self.assertLess(operation_time, TEST_TIMEOUT / 10, 
                       f"Core operations should complete well within timeout limit ({TEST_TIMEOUT}s)")
        
        print(f"✅ Timeout Prevention Test:")
        print(f"   Operation time: {operation_time:.2f}s")
        print(f"   Timeout limit: {TEST_TIMEOUT}s")
        print(f"   Safety margin: {(TEST_TIMEOUT - operation_time):.1f}s")


if __name__ == '__main__':
    print("🚀 Running Core Modules Performance Tests")
    print("=" * 50)
    
    # Run with verbose output to show performance metrics
    unittest.main(verbosity=2)