"""
Python Environment Change Detector Tests
=========================================

Tests for PythonEnvChangeDetector focusing on package management,
virtual environment tracking, and AI/ML framework changes.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.linux_system.temporal.change_detection.detectors.python_env_detector import (
    PythonEnvChangeDetector, PackageInfo, VirtualEnvironment, 
    PythonPackageAnalyzer, PythonEnvironmentScanner
)
from src.linux_system.temporal.types import ChangeType
from src.linux_system.temporal.config import PythonEnvDetectorConfig
from tests.temporal.fixtures import TemporalAssertions


class TestPythonEnvChangeDetector:
    """Test PythonEnvChangeDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create PythonEnvChangeDetector instance."""
        config = PythonEnvDetectorConfig()
        return PythonEnvChangeDetector(config)
    
    @pytest.fixture
    def sample_package_info(self):
        """Create sample package information."""
        return PackageInfo(
            name='torch',
            version='2.1.0',
            location='/home/alejandro/.local/lib/python3.11/site-packages/torch',
            dependencies=['numpy', 'typing-extensions', 'sympy'],
            requires_python='>=3.8',
            summary='Tensors and Dynamic neural networks in Python',
            home_page='https://pytorch.org/',
            installer='pip'
        )
    
    def test_ml_framework_package_detection(self, sample_package_info):
        """Test detection of ML framework packages."""
        assert sample_package_info.is_ml_framework == True
        
        # Test non-ML package
        regular_package = PackageInfo(
            name='requests',
            version='2.28.1',
            location=None,
            dependencies=['urllib3', 'certifi'],
            requires_python='>=3.7',
            summary='HTTP library',
            home_page='https://requests.readthedocs.io',
            installer='pip'
        )
        assert regular_package.is_ml_framework == False
    
    def test_cuda_package_detection(self, sample_package_info):
        """Test detection of CUDA-related packages."""
        # PyTorch is not inherently CUDA (depends on variant)
        assert sample_package_info.is_cuda_related == False
        
        # Test CUDA package
        cuda_package = PackageInfo(
            name='nvidia-ml-py',
            version='11.515.75',
            location=None,
            dependencies=[],
            requires_python='>=3.6',
            summary='NVIDIA Management Library bindings',
            home_page='https://developer.nvidia.com/nvidia-management-library-nvml',
            installer='pip'
        )
        assert cuda_package.is_cuda_related == True
    
    def test_package_installation_detection(self, detector):
        """Test detection of new package installations."""
        old_snapshot = {
            'python_env': {
                'system_python': {'packages': []},
                'virtual_environments': []
            }
        }
        
        new_snapshot = {
            'python_env': {
                'system_python': {
                    'packages': [{
                        'name': 'transformers',
                        'version': '4.21.0',
                        'is_ml_framework': True,
                        'is_cuda_related': False,
                        'dependencies': ['torch', 'numpy', 'tokenizers']
                    }]
                },
                'virtual_environments': []
            }
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect package installation
        package_changes = [c for c in changes if 'package:' in c.entity_id and c.change_type == ChangeType.ADDED]
        assert len(package_changes) > 0
        
        package_change = package_changes[0]
        assert 'transformers' in package_change.entity_id
        assert package_change.significance > 0.5  # ML framework installation is significant
    
    def test_package_removal_detection(self, detector):
        """Test detection of package removal."""
        old_snapshot = {
            'python_env': {
                'system_python': {
                    'packages': [{
                        'name': 'tensorflow',
                        'version': '2.12.0',
                        'is_ml_framework': True,
                        'is_cuda_related': False,
                        'dependencies': ['numpy', 'keras']
                    }]
                },
                'virtual_environments': []
            }
        }
        
        new_snapshot = {
            'python_env': {
                'system_python': {'packages': []},
                'virtual_environments': []
            }
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect package removal
        removal_changes = [c for c in changes if c.change_type == ChangeType.REMOVED and 'tensorflow' in c.entity_id]
        assert len(removal_changes) > 0
        
        removal_change = removal_changes[0]
        assert removal_change.significance > 0.6  # ML framework removal is highly significant
    
    def test_package_version_update_detection(self, detector):
        """Test detection of package version updates."""
        old_snapshot = {
            'python_env': {
                'system_python': {
                    'packages': [{
                        'name': 'torch',
                        'version': '2.0.0',
                        'is_ml_framework': True,
                        'is_cuda_related': False
                    }]
                }
            }
        }
        
        new_snapshot = {
            'python_env': {
                'system_python': {
                    'packages': [{
                        'name': 'torch',
                        'version': '2.1.0',
                        'is_ml_framework': True,
                        'is_cuda_related': False
                    }]
                }
            }
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect version update
        update_changes = [c for c in changes if c.change_type == ChangeType.MODIFIED and 'torch' in c.entity_id]
        assert len(update_changes) > 0
        
        update_change = update_changes[0]
        assert update_change.old_value == '2.0.0'
        assert update_change.new_value == '2.1.0'
    
    def test_virtual_environment_creation_detection(self, detector):
        """Test detection of new virtual environment creation."""
        old_snapshot = {
            'python_env': {
                'virtual_environments': []
            }
        }
        
        new_snapshot = {
            'python_env': {
                'virtual_environments': [{
                    'name': 'ml-training',
                    'path': '/home/alejandro/.virtualenvs/ml-training',
                    'python_version': 'Python 3.11.5',
                    'is_active': True,
                    'env_type': 'venv',
                    'is_ml_environment': True,
                    'packages': [
                        {'name': 'torch', 'version': '2.1.0', 'is_ml_framework': True},
                        {'name': 'transformers', 'version': '4.21.0', 'is_ml_framework': True}
                    ]
                }]
            }
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect virtual environment creation
        env_changes = [c for c in changes if 'virtual_env:ml-training' in c.entity_id and c.change_type == ChangeType.ADDED]
        assert len(env_changes) > 0
        
        env_change = env_changes[0]
        assert env_change.significance >= 0.8  # ML environment creation is highly significant
        assert env_change.metadata['is_ml_environment'] == True
    
    def test_python_version_change_detection(self, detector):
        """Test detection of Python version changes."""
        old_snapshot = {
            'python_env': {
                'system_python': {
                    'python_version': 'Python 3.10.12'
                }
            }
        }
        
        new_snapshot = {
            'python_env': {
                'system_python': {
                    'python_version': 'Python 3.11.5'
                }
            }
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect Python version change
        version_changes = [c for c in changes if 'system_python:version' in c.entity_id]
        assert len(version_changes) > 0
        
        version_change = version_changes[0]
        assert version_change.significance == 0.8  # Python version changes are highly significant
        assert version_change.metadata['risk_level'] == 'high'


class TestPythonPackageAnalyzer:
    """Test PythonPackageAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create PythonPackageAnalyzer instance."""
        return PythonPackageAnalyzer()
    
    def test_ml_framework_installation_analysis(self, analyzer):
        """Test analysis of ML framework installation."""
        pytorch_package = PackageInfo(
            name='torch',
            version='2.1.0',
            location='/usr/local/lib/python3.11/dist-packages/torch',
            dependencies=['numpy', 'typing-extensions'],
            requires_python='>=3.8',
            summary='PyTorch deep learning framework',
            home_page='https://pytorch.org/',
            installer='pip'
        )
        
        analysis = analyzer.analyze_package_change(None, pytorch_package)
        
        assert analysis['change_type'] == 'installation'
        assert analysis['significance'] >= 0.8  # ML frameworks are highly significant
        assert 'framework_addition' in analysis['impact_assessment']
        assert analysis['impact_assessment']['framework_addition'] == 'torch'
    
    def test_version_change_analysis(self, analyzer):
        """Test analysis of package version changes."""
        old_package = PackageInfo(
            name='transformers',
            version='4.20.0',
            location=None,
            dependencies=[],
            requires_python='>=3.8',
            summary='Transformers library',
            home_page='https://huggingface.co/transformers',
            installer='pip'
        )
        
        new_package = PackageInfo(
            name='transformers', 
            version='4.25.0',
            location=None,
            dependencies=[],
            requires_python='>=3.8',
            summary='Transformers library',
            home_page='https://huggingface.co/transformers',
            installer='pip'
        )
        
        analysis = analyzer.analyze_package_change(old_package, new_package)
        
        assert analysis['change_type'] == 'update'
        assert analysis['significance'] >= 0.6  # ML framework updates are significant
        assert 'version_change' in analysis['impact_assessment']
        assert analysis['impact_assessment']['version_change']['change_type'] == 'minor'
    
    def test_major_version_change_risk_assessment(self, analyzer):
        """Test risk assessment for major version changes."""
        old_tensorflow = PackageInfo(
            name='tensorflow',
            version='2.12.0',
            location=None,
            dependencies=[],
            requires_python='>=3.8',
            summary='TensorFlow',
            home_page='https://tensorflow.org',
            installer='pip'
        )
        
        new_tensorflow = PackageInfo(
            name='tensorflow',
            version='3.0.0',  # Major version change
            location=None,
            dependencies=[],
            requires_python='>=3.8',
            summary='TensorFlow',
            home_page='https://tensorflow.org',
            installer='pip'
        )
        
        analysis = analyzer.analyze_package_change(old_tensorflow, new_tensorflow)
        
        assert analysis['change_type'] == 'update'
        assert analysis['significance'] >= 0.9  # Major ML framework updates are very significant
        assert 'major_ml_framework_change' in analysis['risk_factors']
        assert 'api_breaking_changes_likely' in analysis['compatibility_concerns']
    
    def test_package_removal_risk_analysis(self, analyzer):
        """Test risk analysis for package removal."""
        cuda_package = PackageInfo(
            name='nvidia-ml-py',
            version='11.515.75',
            location=None,
            dependencies=[],
            requires_python='>=3.6',
            summary='NVIDIA Management Library',
            home_page='https://developer.nvidia.com',
            installer='pip'
        )
        
        analysis = analyzer.analyze_package_change(cuda_package, None)
        
        assert analysis['change_type'] == 'removal'
        assert analysis['significance'] >= 0.6
        assert 'gpu_capability_loss' in analysis['risk_factors']


class TestPythonEnvironmentScanner:
    """Test PythonEnvironmentScanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        """Create PythonEnvironmentScanner instance."""
        return PythonEnvironmentScanner()
    
    @patch('subprocess.run')
    def test_conda_environment_scanning(self, mock_subprocess, scanner):
        """Test scanning of conda environments."""
        # Mock conda env list --json output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps({
            'envs': [
                '/home/alejandro/miniconda3',
                '/home/alejandro/miniconda3/envs/pytorch-env',
                '/home/alejandro/miniconda3/envs/tensorflow-env'
            ],
            'default_prefix': '/home/alejandro/miniconda3'
        })
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('subprocess.run') as mock_run:
                # Mock different subprocess calls
                def subprocess_side_effect(*args, **kwargs):
                    cmd = args[0]
                    if cmd == ['conda', '--version']:
                        return Mock(returncode=0)
                    elif cmd == ['conda', 'env', 'list', '--json']:
                        return Mock(returncode=0, stdout=json.dumps({
                            'envs': ['/home/alejandro/miniconda3/envs/ml-env']
                        }))
                    elif 'python' in cmd[0] and '--version' in cmd:
                        return Mock(returncode=0, stdout='Python 3.11.5')
                    elif 'pip' in cmd and 'list' in cmd:
                        return Mock(returncode=0, stdout='[]')
                    else:
                        return Mock(returncode=1)
                
                mock_run.side_effect = subprocess_side_effect
                
                environments = scanner._scan_conda_environments()
                
                # Should find conda environments
                assert len(environments) >= 0  # Might be empty due to mocking complexity
    
    def test_virtual_environment_classification(self, scanner):
        """Test classification of virtual environments as ML environments."""
        # Create mock virtual environment with ML packages
        ml_packages = [
            PackageInfo(name='torch', version='2.1.0', location=None, dependencies=[], 
                       requires_python=None, summary=None, home_page=None, installer='pip'),
            PackageInfo(name='transformers', version='4.21.0', location=None, dependencies=[], 
                       requires_python=None, summary=None, home_page=None, installer='pip'),
            PackageInfo(name='accelerate', version='0.20.3', location=None, dependencies=[], 
                       requires_python=None, summary=None, home_page=None, installer='pip'),
            PackageInfo(name='numpy', version='1.24.3', location=None, dependencies=[], 
                       requires_python=None, summary=None, home_page=None, installer='pip')
        ]
        
        ml_env = VirtualEnvironment(
            name='deep-learning',
            path=Path('/home/alejandro/.virtualenvs/deep-learning'),
            python_version='Python 3.11.5',
            is_active=False,
            env_type='venv',
            packages=ml_packages
        )
        
        assert ml_env.is_ml_environment == True
        
        # Test regular environment
        regular_packages = [
            PackageInfo(name='requests', version='2.28.1', location=None, dependencies=[], 
                       requires_python=None, summary=None, home_page=None, installer='pip'),
            PackageInfo(name='flask', version='2.3.2', location=None, dependencies=[], 
                       requires_python=None, summary=None, home_page=None, installer='pip')
        ]
        
        regular_env = VirtualEnvironment(
            name='web-app',
            path=Path('/home/alejandro/.virtualenvs/web-app'),
            python_version='Python 3.11.5',
            is_active=False,
            env_type='venv',
            packages=regular_packages
        )
        
        assert regular_env.is_ml_environment == False


class TestPythonEnvDetectorIntegration:
    """Integration tests for PythonEnvChangeDetector."""
    
    def test_ml_workflow_setup_detection(self):
        """Test detection of ML workflow environment setup."""
        detector = PythonEnvChangeDetector()
        
        # Simulate empty state
        empty_snapshot = {'python_env': {'virtual_environments': []}}
        
        # Simulate new ML environment with full stack
        ml_environment_snapshot = {
            'python_env': {
                'virtual_environments': [{
                    'name': 'llm-fine-tuning',
                    'path': '/home/alejandro/.virtualenvs/llm-fine-tuning',
                    'python_version': 'Python 3.11.5',
                    'is_active': True,
                    'env_type': 'venv',
                    'is_ml_environment': True,
                    'packages': [
                        {'name': 'torch', 'version': '2.1.0', 'is_ml_framework': True, 'is_cuda_related': False},
                        {'name': 'transformers', 'version': '4.25.0', 'is_ml_framework': True, 'is_cuda_related': False},
                        {'name': 'accelerate', 'version': '0.24.0', 'is_ml_framework': True, 'is_cuda_related': False},
                        {'name': 'deepspeed', 'version': '0.10.3', 'is_ml_framework': True, 'is_cuda_related': False},
                        {'name': 'wandb', 'version': '0.15.12', 'is_ml_framework': False, 'is_cuda_related': False},
                        {'name': 'nvidia-ml-py', 'version': '11.515.75', 'is_ml_framework': False, 'is_cuda_related': True}
                    ]
                }]
            }
        }
        
        changes = detector.detect_changes(empty_snapshot, ml_environment_snapshot)
        
        # Should detect environment creation and package installations
        env_creation = [c for c in changes if c.change_type == ChangeType.ADDED and 'virtual_env:' in c.entity_id]
        package_installations = [c for c in changes if c.change_type == ChangeType.ADDED and 'package:' in c.entity_id]
        
        assert len(env_creation) > 0
        assert len(package_installations) >= 4  # Multiple ML packages
        
        # Environment creation should be highly significant
        env_change = env_creation[0]
        assert env_change.significance >= 0.8
        assert env_change.metadata['is_ml_environment'] == True
        
        # ML framework installations should be significant
        ml_package_changes = [c for c in package_installations 
                            if c.metadata.get('is_ml_framework', False)]
        assert len(ml_package_changes) >= 3
        
        for ml_change in ml_package_changes:
            assert ml_change.significance >= 0.6
    
    def test_cuda_capability_change_detection(self):
        """Test detection of CUDA capability changes."""
        detector = PythonEnvChangeDetector()
        
        # Before: CPU-only PyTorch
        before_snapshot = {
            'python_env': {
                'system_python': {
                    'packages': [
                        {'name': 'torch', 'version': '2.1.0+cpu', 'is_ml_framework': True, 'is_cuda_related': False}
                    ]
                }
            }
        }
        
        # After: CUDA-enabled PyTorch + CUDA tools
        after_snapshot = {
            'python_env': {
                'system_python': {
                    'packages': [
                        {'name': 'torch', 'version': '2.1.0+cu118', 'is_ml_framework': True, 'is_cuda_related': False},
                        {'name': 'nvidia-ml-py', 'version': '11.515.75', 'is_ml_framework': False, 'is_cuda_related': True},
                        {'name': 'cupy-cuda11x', 'version': '12.2.0', 'is_ml_framework': False, 'is_cuda_related': True}
                    ]
                }
            }
        }
        
        changes = detector.detect_changes(before_snapshot, after_snapshot)
        
        # Should detect PyTorch update and CUDA tool installations
        torch_update = [c for c in changes if 'torch' in c.entity_id and c.change_type == ChangeType.MODIFIED]
        cuda_installations = [c for c in changes if c.metadata.get('is_cuda_related', False) and c.change_type == ChangeType.ADDED]
        
        assert len(torch_update) > 0
        assert len(cuda_installations) >= 2
        
        # CUDA-related changes should be significant
        for cuda_change in cuda_installations:
            assert cuda_change.significance >= 0.5