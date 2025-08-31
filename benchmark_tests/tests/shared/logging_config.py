"""
Standardized Logging Configuration for Tests

Provides consistent logging setup across all test suites with appropriate
levels and formatting for different test contexts.

Author: Claude Code  
Version: 1.0.0
"""

import logging
import sys
from typing import Optional


def configure_test_logging(
    level: str = "INFO",
    format_style: str = "simple",
    capture_warnings: bool = True
) -> logging.Logger:
    """
    Configure standardized logging for test environments
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_style: 'simple' or 'detailed'
        capture_warnings: Whether to capture warnings in logs
        
    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create formatter
    if format_style == "detailed":
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
    else:  # simple
        formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Capture warnings if requested
    if capture_warnings:
        logging.captureWarnings(True)
    
    # Configure specific loggers for quieter output during tests
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING) 
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return root_logger


def get_test_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for specific test module
    
    Args:
        name: Logger name (usually __name__)
        level: Optional specific level for this logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    return logger


def silence_noisy_loggers():
    """Silence loggers that are too verbose during testing"""
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'transformers.tokenization_utils',
        'transformers.configuration_utils',
        'transformers.modeling_utils',
        'sentence_transformers',
        'torch',
        'tensorflow'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def configure_calibration_logging() -> logging.Logger:
    """Specific logging configuration for calibration tests"""
    logger = configure_test_logging(
        level="INFO",
        format_style="detailed",
        capture_warnings=True
    )
    
    # Calibration-specific logger adjustments
    logging.getLogger('evaluator.advanced').setLevel(logging.WARNING)
    logging.getLogger('calibration_validator').setLevel(logging.INFO)
    
    return logger


def configure_functional_logging() -> logging.Logger:
    """Specific logging configuration for functional tests"""
    logger = configure_test_logging(
        level="WARNING",  # Less verbose for functional tests
        format_style="simple",
        capture_warnings=True
    )
    
    # Keep important test information visible
    logging.getLogger('tests.functional').setLevel(logging.INFO)
    
    return logger


def configure_debug_logging() -> logging.Logger:
    """Maximum verbosity logging for debugging"""
    logger = configure_test_logging(
        level="DEBUG",
        format_style="detailed", 
        capture_warnings=True
    )
    
    # Enable debug for key components
    logging.getLogger('evaluator').setLevel(logging.DEBUG)
    logging.getLogger('benchmark_runner').setLevel(logging.DEBUG)
    
    return logger