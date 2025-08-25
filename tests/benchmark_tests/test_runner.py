#!/usr/bin/env python3
"""
TestRunner - Flexible Test Execution Engine

A modular test execution system that separates test definitions from execution logic,
supporting both sequential and concurrent test execution against local LLM APIs.

Author: Claude Code
Version: 1.0.0
"""

import json
import time
import logging
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for single test execution result"""
    test_id: str
    test_name: str
    success: bool
    response_text: str
    execution_time: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    error_message: Optional[str]
    timestamp: str
    api_response: Dict[str, Any]


@dataclass
class ExecutionProgress:
    """Container for execution progress information"""
    total_tests: int
    completed_tests: int
    successful_tests: int
    failed_tests: int
    current_test: Optional[str]
    estimated_remaining_time: float
    average_execution_time: float


@dataclass
class APIConfiguration:
    """Container for API configuration"""
    endpoint: str
    model: str
    headers: Dict[str, str]
    timeout: int
    retry_attempts: int
    retry_delay: float
    api_type: str  # "completions" or "chat"


class TestRunner:
    """
    Flexible test execution engine for LLM benchmarking
    
    Handles loading test definitions from JSON, executing tests against APIs,
    and collecting results with comprehensive error handling and progress tracking.
    """
    
    def __init__(self, config_path: Optional[str] = None, api_endpoint: Optional[str] = None):
        """
        Initialize TestRunner with optional configuration
        
        Args:
            config_path: Path to configuration JSON file
            api_endpoint: API endpoint URL (overrides config/metadata)
        """
        self.tests = {}
        self.test_metadata = {}
        self.categories = {}
        self.api_config = None
        self.config = self._load_config(config_path)
        self.execution_progress = None
        self._start_times = []
        self._api_endpoint_override = api_endpoint
        
        logger.info("TestRunner initialized")
    
    def load_test_suite(self, suite_path: str) -> bool:
        """
        Load test suite from JSON file
        
        Args:
            suite_path: Path to test suite JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(suite_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if 'tests' not in data:
                logger.error(f"Invalid test suite format: missing 'tests' key")
                return False
            
            # Load tests into registry
            for test in data['tests']:
                if 'id' not in test:
                    logger.warning(f"Test missing 'id' field, skipping")
                    continue
                self.tests[test['id']] = test
            
            logger.info(f"Loaded {len(self.tests)} tests from {suite_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Test suite file not found: {suite_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test suite file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading test suite: {e}")
            return False
    
    def load_test_metadata(self, metadata_path: str) -> bool:
        """
        Load suite metadata and API configuration
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.test_metadata = json.load(f)
            
            # Extract API configuration (can be overridden by constructor arg)
            if 'api_config' in self.test_metadata:
                api_data = self.test_metadata['api_config']
                endpoint = self._api_endpoint_override or api_data.get('endpoint', 'http://127.0.0.1:8004/v1/completions')
                
                self.api_config = APIConfiguration(
                    endpoint=endpoint,
                    model=api_data.get('model', 'default-model'),
                    headers=api_data.get('headers', {'Content-Type': 'application/json'}),
                    timeout=api_data.get('timeout', 600),
                    retry_attempts=3,
                    retry_delay=1.0,
                    api_type=self._detect_api_type(endpoint)
                )
            
            logger.info(f"Loaded test metadata from {metadata_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def load_categories(self, categories_path: str) -> bool:
        """
        Load category definitions
        
        Args:
            categories_path: Path to categories JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(categories_path, 'r', encoding='utf-8') as f:
                self.categories = json.load(f)
            
            logger.info(f"Loaded categories from {categories_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Categories file not found: {categories_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return False
    
    def configure_api(self, endpoint: str, model: str, headers: Dict = None) -> None:
        """
        Configure API connection settings
        
        Args:
            endpoint: Full API endpoint URL (e.g., 'http://127.0.0.1:8004/v1/completions')
            model: Model name
            headers: HTTP headers dictionary
        """
        self.api_config = APIConfiguration(
            endpoint=endpoint,
            model=model,
            headers=headers or {'Content-Type': 'application/json'},
            timeout=600,
            retry_attempts=3,
            retry_delay=1.0,
            api_type=self._detect_api_type(endpoint)
        )
        logger.info(f"API configured: {endpoint} ({self.api_config.api_type})")
    
    def execute_single_test(self, test_id: str) -> TestResult:
        """
        Execute a single test case
        
        Args:
            test_id: ID of test to execute
            
        Returns:
            TestResult: Result of test execution
        """
        if test_id not in self.tests:
            error_msg = f"Test {test_id} not found"
            logger.error(error_msg)
            return self._create_error_result(test_id, error_msg)
        
        if not self.api_config:
            error_msg = "API not configured"
            logger.error(error_msg)
            return self._create_error_result(test_id, error_msg)
        
        test_case = self.tests[test_id]
        logger.info(f"Executing test: {test_case.get('name', test_id)}")
        
        # Build API request
        success, api_response, error_msg = self._make_api_request(test_case)
        
        if success:
            return self._create_success_result(test_id, test_case, api_response)
        else:
            return self._create_error_result(test_id, error_msg, api_response)
    
    def get_test_ids_by_category(self, category: str) -> List[str]:
        """
        Get list of test IDs for a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of test IDs in the category
        """
        if not self.categories or 'categories' not in self.categories:
            # Fallback: filter by category field in test data
            return [test_id for test_id, test in self.tests.items() 
                   if test.get('category') == category]
        
        category_data = self.categories['categories'].get(category, {})
        return category_data.get('test_ids', [])
    
    def save_results(self, results: List[TestResult], output_dir: str = None) -> bool:
        """
        Save test results to files (compatible with existing format)
        
        Args:
            results: List of TestResult objects
            output_dir: Output directory (defaults to test_results)
            
        Returns:
            bool: True if saved successfully
        """
        if output_dir is None:
            output_dir = "test_results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            for result in results:
                # Create filename compatible with existing format
                filename_safe_name = result.test_name.lower().replace(":", "").replace(" ", "_")
                output_path = os.path.join(output_dir, f"{filename_safe_name}_completion.txt")
                
                # Save in same format as monolithic version
                with open(output_path, "w", encoding="utf-8") as f:
                    test_case = self.tests.get(result.test_id, {})
                    prompt = test_case.get('prompt', 'N/A')
                    
                    f.write(f"PROMPT:\n{'-'*20}\n{prompt}\n\n")
                    f.write(f"COMPLETION:\n{'-'*20}\n{result.response_text}\n\n")
                    f.write(f"METRICS:\n{'-'*20}\n")
                    f.write(f"Duration: {result.execution_time:.2f}s\n")
                    f.write(f"Prompt Tokens: {result.prompt_tokens}\n")
                    f.write(f"Completion Tokens: {result.completion_tokens}\n")
                    f.write(f"Tokens per Second: {result.tokens_per_second:.2f} T/s\n")
                    
                    if not result.success and result.error_message:
                        f.write(f"Error: {result.error_message}\n")
                
                # Also save JSON version for programmatic access
                json_path = os.path.join(output_dir, f"{filename_safe_name}_result.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(results)} results to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def execute_sequential(self, test_ids: List[str] = None, delay: float = None) -> List[TestResult]:
        """
        Execute tests sequentially with delays between tests
        
        Args:
            test_ids: List of test IDs to execute (defaults to all tests)
            delay: Delay in seconds between tests (defaults to config setting)
            
        Returns:
            List of TestResult objects
        """
        if test_ids is None:
            test_ids = list(self.tests.keys())
        
        if delay is None:
            delay = self.config['execution_defaults'].get('sequential_delay', 1.0)
        
        results = []
        total_tests = len(test_ids)
        
        # Initialize progress tracking
        self.execution_progress = ExecutionProgress(
            total_tests=total_tests,
            completed_tests=0,
            successful_tests=0,
            failed_tests=0,
            current_test=None,
            estimated_remaining_time=0.0,
            average_execution_time=0.0
        )
        
        logger.info(f"Starting sequential execution of {total_tests} tests (delay: {delay}s)")
        
        for i, test_id in enumerate(test_ids):
            # Update progress
            self.execution_progress.current_test = test_id
            
            # Execute test
            start_time = time.time()
            result = self.execute_single_test(test_id)
            end_time = time.time()
            
            results.append(result)
            self._start_times.append(end_time - start_time)
            
            # Update progress metrics
            self.execution_progress.completed_tests = i + 1
            if result.success:
                self.execution_progress.successful_tests += 1
            else:
                self.execution_progress.failed_tests += 1
            
            # Calculate timing estimates
            if self._start_times:
                self.execution_progress.average_execution_time = sum(self._start_times) / len(self._start_times)
                remaining_tests = total_tests - (i + 1)
                self.execution_progress.estimated_remaining_time = (
                    remaining_tests * (self.execution_progress.average_execution_time + delay)
                )
            
            # Log progress
            progress_pct = ((i + 1) / total_tests) * 100
            logger.info(f"Progress: {i + 1}/{total_tests} ({progress_pct:.1f}%) - "
                       f"Success rate: {self.execution_progress.successful_tests}/{i + 1}")
            
            # Add delay between tests (except after the last test)
            if i < len(test_ids) - 1 and delay > 0:
                time.sleep(delay)
        
        # Final progress update
        self.execution_progress.current_test = None
        
        success_rate = (self.execution_progress.successful_tests / total_tests) * 100
        logger.info(f"Sequential execution completed: {self.execution_progress.successful_tests}/{total_tests} "
                   f"successful ({success_rate:.1f}%)")
        
        return results
    
    def execute_concurrent(self, test_ids: List[str] = None, workers: int = 3) -> List[TestResult]:
        """
        Execute tests concurrently using thread pool
        
        Args:
            test_ids: List of test IDs to execute (defaults to all tests)
            workers: Number of concurrent worker threads
            
        Returns:
            List of TestResult objects
        """
        if test_ids is None:
            test_ids = list(self.tests.keys())
        
        results = []
        total_tests = len(test_ids)
        completed_count = 0
        successful_count = 0
        failed_count = 0
        
        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        
        # Initialize progress tracking
        self.execution_progress = ExecutionProgress(
            total_tests=total_tests,
            completed_tests=0,
            successful_tests=0,
            failed_tests=0,
            current_test=None,
            estimated_remaining_time=0.0,
            average_execution_time=0.0
        )
        
        logger.info(f"Starting concurrent execution of {total_tests} tests with {workers} workers")
        
        def update_progress(result: TestResult):
            """Thread-safe progress update"""
            nonlocal completed_count, successful_count, failed_count
            
            with progress_lock:
                completed_count += 1
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Update progress object
                self.execution_progress.completed_tests = completed_count
                self.execution_progress.successful_tests = successful_count
                self.execution_progress.failed_tests = failed_count
                
                # Calculate timing estimates
                if self._start_times:
                    self.execution_progress.average_execution_time = sum(self._start_times) / len(self._start_times)
                    remaining_tests = total_tests - completed_count
                    self.execution_progress.estimated_remaining_time = (
                        remaining_tests * self.execution_progress.average_execution_time / workers
                    )
                
                # Log progress
                progress_pct = (completed_count / total_tests) * 100
                logger.info(f"Progress: {completed_count}/{total_tests} ({progress_pct:.1f}%) - "
                           f"Success rate: {successful_count}/{completed_count}")
        
        # Execute tests concurrently
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_test = {
                executor.submit(self.execute_single_test, test_id): test_id 
                for test_id in test_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_id = future_to_test[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    self._start_times.append(result.execution_time)
                    update_progress(result)
                    
                except Exception as e:
                    # Create error result if thread execution fails
                    error_result = self._create_error_result(test_id, f"Thread execution error: {e}")
                    results.append(error_result)
                    update_progress(error_result)
        
        # Final progress update
        self.execution_progress.current_test = None
        
        success_rate = (successful_count / total_tests) * 100
        logger.info(f"Concurrent execution completed: {successful_count}/{total_tests} "
                   f"successful ({success_rate:.1f}%)")
        
        return results
    
    def execute_category(self, category: str, sequential: bool = True, **kwargs) -> List[TestResult]:
        """
        Execute all tests in a specific category
        
        Args:
            category: Category name
            sequential: Whether to run sequentially (True) or concurrently (False)
            **kwargs: Additional arguments for execution method
            
        Returns:
            List of TestResult objects
        """
        test_ids = self.get_test_ids_by_category(category)
        
        if not test_ids:
            logger.warning(f"No tests found for category: {category}")
            return []
        
        logger.info(f"Executing {len(test_ids)} tests from category: {category}")
        
        if sequential:
            return self.execute_sequential(test_ids, **kwargs)
        else:
            # Use concurrent execution
            workers = kwargs.get('workers', 3)
            return self.execute_concurrent(test_ids, workers)
    
    def get_progress(self) -> ExecutionProgress:
        """Get current execution progress information"""
        return self.execution_progress or ExecutionProgress(0, 0, 0, 0, None, 0, 0)
    
    # Private helper methods
    
    def _detect_api_type(self, endpoint: str) -> str:
        """
        Detect API type from endpoint URL
        
        Args:
            endpoint: API endpoint URL
            
        Returns:
            'chat' or 'completions'
        """
        if "/chat/completions" in endpoint:
            return "chat"
        elif "/completions" in endpoint:
            return "completions"
        else:
            # Default based on common patterns
            return "completions"
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "api_defaults": {
                "timeout": 600,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "execution_defaults": {
                "sequential_delay": 1.0,
                "progress_reporting": True,
                "save_raw_responses": True
            },
            "output_settings": {
                "results_directory": "test_results",
                "include_metadata": True,
                "pretty_print": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}. Using defaults.")
        
        return default_config
    
    def _make_api_request(self, test_case: Dict) -> Tuple[bool, Dict, str]:
        """
        Make API request with retry logic
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        # Build request payload based on API type
        if self.api_config.api_type == "chat":
            payload = self._build_chat_payload(test_case)
        else:
            payload = self._build_completions_payload(test_case)
        
        # Attempt request with retries
        for attempt in range(self.api_config.retry_attempts):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_config.endpoint,
                    headers=self.api_config.headers,
                    json=payload,
                    timeout=self.api_config.timeout
                )
                end_time = time.time()
                
                response.raise_for_status()
                response_data = response.json()
                
                # Add timing information
                response_data['_execution_time'] = end_time - start_time
                
                return True, response_data, ""
                
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout (attempt {attempt + 1})"
                logger.warning(error_msg)
                if attempt < self.api_config.retry_attempts - 1:
                    time.sleep(self.api_config.retry_delay * (attempt + 1))
                    continue
                return False, {}, error_msg
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                logger.warning(error_msg)
                if attempt < self.api_config.retry_attempts - 1:
                    time.sleep(self.api_config.retry_delay * (attempt + 1))
                    continue
                return False, {}, error_msg
            
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(error_msg)
                return False, {}, error_msg
        
        return False, {}, "Max retries exceeded"
    
    def _build_completions_payload(self, test_case: Dict) -> Dict:
        """Build payload for completions API"""
        return {
            "model": self.api_config.model,
            "prompt": test_case.get('prompt', ''),
            **test_case.get('parameters', {})
        }
    
    def _build_chat_payload(self, test_case: Dict) -> Dict:
        """Build payload for chat API"""
        # Convert test case to chat format
        messages = []
        
        # Check if test case already has messages format (from instruct tests)
        if 'messages' in test_case:
            messages = test_case['messages']
        else:
            # Convert prompt-based test to chat format (from base model tests)
            messages = [{"role": "user", "content": test_case.get('prompt', '')}]
        
        # Convert parameters for chat API
        params = self._convert_params_for_chat(test_case.get('parameters', {}))
        
        return {
            "model": self.api_config.model,
            "messages": messages,
            **params
        }
    
    def _convert_params_for_chat(self, params: Dict) -> Dict:
        """Convert completion parameters to chat parameters"""
        chat_params = params.copy()
        
        # Remove parameters that don't exist in chat API
        chat_params.pop('exclude', None)  # instruct-specific parameter
        chat_params.pop('effort', None)   # instruct-specific parameter
        
        return chat_params
    
    def _create_success_result(self, test_id: str, test_case: Dict, api_response: Dict) -> TestResult:
        """Create TestResult for successful execution"""
        execution_time = api_response.get('_execution_time', 0.0)
        
        # Extract response text based on API type
        if self.api_config.api_type == "chat":
            # Chat API: response in choices[0].message.content
            completion_text = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            # Completions API: response in choices[0].text
            completion_text = api_response.get("choices", [{}])[0].get("text", "")
        
        # Extract token usage
        usage = api_response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        
        tokens_per_second = 0.0
        if completion_tokens > 0 and execution_time > 0:
            tokens_per_second = completion_tokens / execution_time
        
        return TestResult(
            test_id=test_id,
            test_name=test_case.get('name', test_id),
            success=True,
            response_text=completion_text,
            execution_time=execution_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
            error_message=None,
            timestamp=datetime.now().isoformat(),
            api_response=api_response
        )
    
    def _create_error_result(self, test_id: str, error_message: str, api_response: Dict = None) -> TestResult:
        """Create TestResult for failed execution"""
        test_case = self.tests.get(test_id, {})
        
        return TestResult(
            test_id=test_id,
            test_name=test_case.get('name', test_id),
            success=False,
            response_text="",
            execution_time=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            tokens_per_second=0.0,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            api_response=api_response or {}
        )


# Convenience functions for quick usage

def load_and_configure_runner(test_definitions_dir: str = "test_definitions", 
                             api_endpoint: str = None,
                             test_type: str = "base") -> TestRunner:
    """
    Load and configure a TestRunner with default paths
    
    Args:
        test_definitions_dir: Directory containing test definition files
        api_endpoint: Optional API endpoint URL to override defaults
        test_type: Type of tests to load ("base" or "instruct")
        
    Returns:
        Configured TestRunner instance
    """
    runner = TestRunner(api_endpoint=api_endpoint)
    
    # Get the directory where test_runner.py is located (benchmark_tests/)
    runner_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine paths based on test type
    if test_type == "instruct":
        # Load instruct model tests from instruct-models directory
        base_dir = os.path.join(runner_dir, "instruct-models")
        suite_filename = "instruct_tests_complete.json"
    else:
        # Load base model tests from base_models directory  
        base_dir = os.path.join(runner_dir, "base_models")
        suite_filename = "reasoning_tests_complete.json"
    
    # Build full paths
    suite_path = os.path.join(base_dir, test_definitions_dir, suite_filename)
    metadata_path = os.path.join(base_dir, test_definitions_dir, "test_suite_metadata.json")
    categories_path = os.path.join(base_dir, test_definitions_dir, "categories.json")
    
    # Load test files
    runner.load_test_suite(suite_path)
    runner.load_test_metadata(metadata_path)
    runner.load_categories(categories_path)
    
    return runner


if __name__ == "__main__":
    """Command-line interface for TestRunner"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="TestRunner - Flexible LLM Test Execution Engine")
    parser.add_argument("--endpoint", "-e", 
                       default="http://127.0.0.1:8005/v1/completions",
                       help="API endpoint URL (default: %(default)s)")
    parser.add_argument("--model", "-m", 
                       default="your-base-model-name",
                       help="Model name (default: %(default)s)")
    parser.add_argument("--mode", 
                       choices=["single", "sequential", "concurrent", "category"],
                       default="single",
                       help="Execution mode (default: %(default)s)")
    parser.add_argument("--test-id", "-t",
                       help="Specific test ID to run (for single mode)")
    parser.add_argument("--category", "-c",
                       help="Category to run (for category mode)")
    parser.add_argument("--workers", "-w", 
                       type=int, default=3,
                       help="Number of concurrent workers (default: %(default)s)")
    parser.add_argument("--delay", "-d", 
                       type=float, default=1.0,
                       help="Delay between sequential tests in seconds (default: %(default)s)")
    parser.add_argument("--output-dir", "-o",
                       default="test_results",
                       help="Output directory for results (default: %(default)s)")
    parser.add_argument("--test-definitions", 
                       default="test_definitions",
                       help="Path to test definitions directory (default: %(default)s)")
    parser.add_argument("--test-type",
                       choices=["base", "instruct"],
                       default="base", 
                       help="Type of tests to run (default: %(default)s)")
    parser.add_argument("--list-categories", action="store_true",
                       help="List available test categories and exit")
    parser.add_argument("--list-tests", action="store_true", 
                       help="List available tests and exit")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running tests")
    
    args = parser.parse_args()
    
    print("TestRunner - Test Execution Engine")
    print("=" * 50)
    print(f"API Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Test Type: {args.test_type}")
    print(f"Mode: {args.mode}")
    
    # Load and configure runner
    try:
        runner = load_and_configure_runner(test_definitions_dir=args.test_definitions, 
                                         api_endpoint=args.endpoint,
                                         test_type=args.test_type)
        runner.configure_api(args.endpoint, args.model)
        
        print(f"Loaded {len(runner.tests)} tests from {args.test_definitions}")
        
    except Exception as e:
        print(f"❌ Error loading test suite: {e}")
        sys.exit(1)
    
    # Handle list commands
    if args.list_categories:
        print("\nAvailable Categories:")
        if runner.categories and 'categories' in runner.categories:
            for category, info in runner.categories['categories'].items():
                test_count = len(info.get('test_ids', []))
                print(f"  {category}: {test_count} tests - {info.get('description', 'No description')}")
        else:
            # Fallback: group by category field in tests
            categories = {}
            for test in runner.tests.values():
                cat = test.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            for category, count in categories.items():
                print(f"  {category}: {count} tests")
        sys.exit(0)
    
    if args.list_tests:
        print("\nAvailable Tests:")
        for test_id, test in runner.tests.items():
            category = test.get('category', 'unknown')
            print(f"  {test_id}: {test.get('name', 'No name')} [{category}]")
        sys.exit(0)
    
    # Determine what to execute
    test_ids_to_run = []
    execution_description = ""
    
    if args.mode == "single":
        if args.test_id:
            if args.test_id in runner.tests:
                test_ids_to_run = [args.test_id]
                execution_description = f"single test: {args.test_id}"
            else:
                print(f"❌ Test ID '{args.test_id}' not found")
                sys.exit(1)
        else:
            # Default to first test
            test_ids_to_run = [list(runner.tests.keys())[0]]
            execution_description = f"single test (first): {test_ids_to_run[0]}"
    
    elif args.mode == "category":
        if not args.category:
            print("❌ Category mode requires --category parameter")
            sys.exit(1)
        test_ids_to_run = runner.get_test_ids_by_category(args.category)
        if not test_ids_to_run:
            print(f"❌ No tests found for category: {args.category}")
            sys.exit(1)
        execution_description = f"category '{args.category}': {len(test_ids_to_run)} tests"
    
    elif args.mode in ["sequential", "concurrent"]:
        test_ids_to_run = list(runner.tests.keys())
        execution_description = f"all {len(test_ids_to_run)} tests ({args.mode})"
    
    print(f"\nPlanned execution: {execution_description}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for i, test_id in enumerate(test_ids_to_run[:5]):  # Show first 5
            test_name = runner.tests[test_id].get('name', 'No name')
            print(f"  {i+1}. {test_id}: {test_name}")
        if len(test_ids_to_run) > 5:
            print(f"  ... and {len(test_ids_to_run) - 5} more tests")
        print(f"\nExecution parameters:")
        print(f"  Workers (concurrent): {args.workers}")
        print(f"  Delay (sequential): {args.delay}s")
        print(f"  Output directory: {args.output_dir}")
        sys.exit(0)
    
    # Execute tests
    print(f"\n🚀 Starting execution...")
    start_time = time.time()
    
    try:
        if args.mode == "single":
            results = [runner.execute_single_test(test_ids_to_run[0])]
        elif args.mode == "sequential":
            results = runner.execute_sequential(test_ids_to_run, delay=args.delay)
        elif args.mode == "concurrent":
            results = runner.execute_concurrent(test_ids_to_run, workers=args.workers)
        elif args.mode == "category":
            # Use concurrent for categories by default, but allow sequential
            concurrent_mode = args.workers > 1
            results = runner.execute_category(args.category, sequential=not concurrent_mode, 
                                            workers=args.workers, delay=args.delay)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save results
        success_count = sum(1 for r in results if r.success)
        runner.save_results(results, args.output_dir)
        
        # Summary
        print(f"\n{'='*50}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        print(f"Success rate: {(success_count/len(results)*100):.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per test: {total_time/len(results):.1f}s")
        
        if results and results[0].success:
            avg_tokens_per_sec = sum(r.tokens_per_second for r in results if r.success) / success_count
            print(f"Average tokens/sec: {avg_tokens_per_sec:.1f} T/s")
        
        print(f"Results saved to: {args.output_dir}/")
        
    except KeyboardInterrupt:
        print("\n❌ Execution cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)