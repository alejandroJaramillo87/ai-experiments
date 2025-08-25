# TestRunner Class Interface Design

## Core Architecture

The TestRunner is designed as a flexible, configurable engine that separates test execution from test definitions and evaluation. It handles loading JSON test suites, executing them against local LLM APIs, and collecting raw results.

## Class Structure

```python
class TestRunner:
    """
    Flexible test execution engine for LLM benchmarking
    
    Responsibilities:
    - Load test definitions from JSON files
    - Execute tests sequentially or concurrently
    - Handle API communication and error recovery
    - Collect and save raw results
    - Provide progress tracking and logging
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize TestRunner with configuration"""
        
    def load_test_suite(self, suite_path: str) -> bool:
        """Load test suite from JSON file"""
        
    def load_test_metadata(self, metadata_path: str) -> bool:
        """Load suite metadata and API configuration"""
        
    def configure_api(self, endpoint: str, model: str, headers: Dict = None) -> None:
        """Configure API connection settings"""
        
    def execute_single_test(self, test_id: str) -> TestResult:
        """Execute a single test case"""
        
    def execute_sequential(self, test_ids: List[str] = None) -> List[TestResult]:
        """Execute tests sequentially with delays"""
        
    def execute_concurrent(self, test_ids: List[str] = None, workers: int = 3) -> List[TestResult]:
        """Execute tests concurrently with thread/async pools"""
        
    def execute_category(self, category: str) -> List[TestResult]:
        """Execute all tests in a specific category"""
        
    def save_results(self, results: List[TestResult], output_dir: str = None) -> bool:
        """Save raw test results to files"""
        
    def get_progress(self) -> ExecutionProgress:
        """Get current execution progress information"""
        
    def cancel_execution(self) -> bool:
        """Cancel running execution gracefully"""
```

## Data Classes

```python
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
    api_response: Dict[str, Any]  # Raw API response
    
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
```

## Configuration System

### TestRunner Configuration File (test_runner_config.json)
```json
{
  "api_defaults": {
    "timeout": 600,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "retry_backoff": 1.5
  },
  "execution_defaults": {
    "sequential_delay": 1.0,
    "concurrent_workers": 3,
    "progress_reporting": true,
    "save_raw_responses": true
  },
  "output_settings": {
    "results_directory": "test_results",
    "filename_template": "{test_id}_{timestamp}.json",
    "include_metadata": true,
    "pretty_print": true
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "test_runner.log"
  }
}
```

## Key Methods Implementation Strategy

### 1. Test Loading
```python
def load_test_suite(self, suite_path: str) -> bool:
    """
    Load and validate test suite JSON
    - Parse JSON structure
    - Validate required fields
    - Build internal test registry
    - Load category mappings
    """
```

### 2. API Communication
```python
def _make_api_request(self, test_case: Dict) -> Tuple[bool, Dict, str]:
    """
    Handle single API request with retries
    - Build request payload
    - Make HTTP request
    - Handle timeout and errors
    - Implement retry logic with backoff
    """
```

### 3. Concurrent Execution
```python
def execute_concurrent(self, test_ids: List[str] = None, workers: int = 3):
    """
    Use ThreadPoolExecutor for concurrent execution
    - Manage worker threads
    - Handle progress tracking across threads
    - Collect results thread-safely
    - Implement graceful cancellation
    """
```

### 4. Progress Tracking
```python
def _update_progress(self, test_result: TestResult):
    """
    Thread-safe progress updates
    - Update completion counters
    - Calculate timing estimates
    - Trigger progress callbacks
    - Log execution milestones
    """
```

## Integration Points

### 1. Test Definition Interface
- Loads from `test_definitions/reasoning_tests_complete.json`
- Uses `test_definitions/test_suite_metadata.json` for API config
- Supports filtering by category via `test_definitions/categories.json`

### 2. Result Output Interface
- Saves results compatible with existing ReasoningEvaluator
- Maintains same file structure as monolithic version
- Provides both individual and batch result formats

### 3. Configuration Interface
- Supports runtime configuration overrides
- Inherits defaults from suite metadata
- Allows per-test parameter customization

## Error Handling Strategy

### 1. Network Errors
- Automatic retries with exponential backoff
- Timeout handling with graceful degradation
- Connection error recovery

### 2. API Errors
- HTTP status code handling
- Malformed response recovery
- Rate limiting respect

### 3. Data Errors
- JSON parsing error recovery
- Missing field validation
- Type checking and conversion

## Usage Examples

### Basic Sequential Execution
```python
runner = TestRunner()
runner.load_test_suite("test_definitions/reasoning_tests_complete.json")
runner.load_test_metadata("test_definitions/test_suite_metadata.json")

results = runner.execute_sequential()
runner.save_results(results)
```

### Category-Specific Concurrent Execution  
```python
runner = TestRunner()
runner.load_test_suite("test_definitions/reasoning_tests_complete.json")

# Execute only mathematical reasoning tests
results = runner.execute_category("mathematical_reasoning")
runner.save_results(results, "math_results/")
```

### Custom Configuration
```python
runner = TestRunner("custom_config.json")
runner.configure_api(
    endpoint="http://localhost:8004/v1/completions",
    model="my-custom-model",
    headers={"Authorization": "Bearer token"}
)

results = runner.execute_concurrent(workers=5)
```

This design provides clean separation of concerns, flexibility for different execution patterns, and maintains compatibility with the existing system.