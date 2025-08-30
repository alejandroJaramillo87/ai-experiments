"""
Python Environment Change Detector
==================================

Tracks changes in Python environments, packages, and AI/ML frameworks.
Specialized for detecting significant changes in the Python ecosystem
that could affect AI workstation capabilities.
"""

import re
import subprocess
import json
import pkg_resources
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..base_detector import BaseChangeDetector
from ...types import SystemChange, ChangeType
from ...config import PythonEnvDetectorConfig


@dataclass
class PackageInfo:
    """Comprehensive package information."""
    name: str
    version: str
    location: Optional[str]
    dependencies: List[str]
    requires_python: Optional[str]
    summary: Optional[str]
    home_page: Optional[str]
    installer: Optional[str]  # pip, conda, poetry, etc.
    
    @property
    def is_ml_framework(self) -> bool:
        """Check if this is a major ML/AI framework."""
        ml_frameworks = {
            'torch', 'pytorch', 'tensorflow', 'transformers', 'accelerate',
            'deepspeed', 'lightning', 'keras', 'scikit-learn', 'numpy',
            'pandas', 'matplotlib', 'seaborn', 'plotly', 'jupyter',
            'huggingface-hub', 'datasets', 'tokenizers', 'optimum'
        }
        return self.name.lower() in ml_frameworks
    
    @property  
    def is_cuda_related(self) -> bool:
        """Check if this package is CUDA/GPU related."""
        cuda_keywords = ['cuda', 'gpu', 'nvidia', 'cupy', 'cudnn', 'nccl']
        return any(keyword in self.name.lower() for keyword in cuda_keywords)


@dataclass
class VirtualEnvironment:
    """Virtual environment information."""
    name: str
    path: Path
    python_version: str
    is_active: bool
    env_type: str  # venv, conda, poetry, pipenv
    packages: List[PackageInfo]
    
    @property
    def is_ml_environment(self) -> bool:
        """Check if this is primarily an ML/AI environment."""
        ml_package_count = sum(1 for pkg in self.packages if pkg.is_ml_framework)
        return ml_package_count >= 3  # 3+ ML packages suggests ML environment


class PythonEnvironmentScanner:
    """Scans and analyzes Python environments."""
    
    def __init__(self):
        self.conda_available = self._check_conda_availability()
        self.poetry_available = self._check_poetry_availability()
    
    def scan_system_python(self) -> Dict[str, Any]:
        """Scan system Python installation."""
        try:
            import sys
            import site
            
            return {
                'python_version': sys.version,
                'python_executable': sys.executable,
                'python_path': sys.path,
                'site_packages': site.getsitepackages(),
                'user_site': site.getusersitepackages() if hasattr(site, 'getusersitepackages') else None,
                'packages': self._get_installed_packages()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def scan_virtual_environments(self) -> List[VirtualEnvironment]:
        """Scan for virtual environments."""
        environments = []
        
        # Scan conda environments
        if self.conda_available:
            environments.extend(self._scan_conda_environments())
        
        # Scan venv environments (common locations)
        environments.extend(self._scan_venv_environments())
        
        # Scan poetry environments
        if self.poetry_available:
            environments.extend(self._scan_poetry_environments())
        
        return environments
    
    def _get_installed_packages(self, python_exec: str = None) -> List[PackageInfo]:
        """Get list of installed packages."""
        packages = []
        
        try:
            if python_exec:
                # Use specific Python executable
                result = subprocess.run([
                    python_exec, '-m', 'pip', 'list', '--format=json'
                ], capture_output=True, text=True, timeout=30)
            else:
                # Use current Python
                result = subprocess.run([
                    'pip', 'list', '--format=json'
                ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pip_packages = json.loads(result.stdout)
                
                for pkg_data in pip_packages:
                    try:
                        # Get detailed package information
                        pkg_info = self._get_package_details(pkg_data['name'], python_exec)
                        packages.append(pkg_info)
                    except Exception:
                        # Fallback to basic info
                        packages.append(PackageInfo(
                            name=pkg_data['name'],
                            version=pkg_data['version'],
                            location=None,
                            dependencies=[],
                            requires_python=None,
                            summary=None,
                            home_page=None,
                            installer='pip'
                        ))
        
        except Exception as e:
            # Log error but continue
            pass
        
        return packages
    
    def _get_package_details(self, package_name: str, python_exec: str = None) -> PackageInfo:
        """Get detailed package information."""
        try:
            if python_exec:
                result = subprocess.run([
                    python_exec, '-m', 'pip', 'show', package_name
                ], capture_output=True, text=True, timeout=15)
            else:
                result = subprocess.run([
                    'pip', 'show', package_name
                ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                details = self._parse_pip_show_output(result.stdout)
                return PackageInfo(
                    name=details.get('Name', package_name),
                    version=details.get('Version', 'unknown'),
                    location=details.get('Location'),
                    dependencies=self._parse_requires(details.get('Requires', '')),
                    requires_python=details.get('Requires-Python'),
                    summary=details.get('Summary'),
                    home_page=details.get('Home-page'),
                    installer='pip'
                )
        except Exception:
            pass
        
        # Fallback for minimal info
        return PackageInfo(
            name=package_name,
            version='unknown',
            location=None,
            dependencies=[],
            requires_python=None,
            summary=None,
            home_page=None,
            installer='pip'
        )
    
    def _parse_pip_show_output(self, output: str) -> Dict[str, str]:
        """Parse pip show output into dictionary."""
        details = {}
        for line in output.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                details[key.strip()] = value.strip()
        return details
    
    def _parse_requires(self, requires_str: str) -> List[str]:
        """Parse requirements string into list."""
        if not requires_str:
            return []
        return [req.strip() for req in requires_str.split(',') if req.strip()]
    
    def _check_conda_availability(self) -> bool:
        """Check if conda is available."""
        try:
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_poetry_availability(self) -> bool:
        """Check if poetry is available."""
        try:
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _scan_conda_environments(self) -> List[VirtualEnvironment]:
        """Scan conda environments."""
        environments = []
        
        try:
            result = subprocess.run(['conda', 'env', 'list', '--json'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                env_data = json.loads(result.stdout)
                
                for env_path in env_data.get('envs', []):
                    env_path_obj = Path(env_path)
                    env_name = env_path_obj.name
                    
                    # Get Python executable for this environment
                    python_exec = env_path_obj / 'bin' / 'python'
                    if not python_exec.exists():
                        python_exec = env_path_obj / 'python.exe'  # Windows
                    
                    if python_exec.exists():
                        # Get Python version
                        try:
                            version_result = subprocess.run([
                                str(python_exec), '--version'
                            ], capture_output=True, text=True, timeout=10)
                            python_version = version_result.stdout.strip()
                        except Exception:
                            python_version = 'unknown'
                        
                        # Get packages in this environment
                        packages = self._get_installed_packages(str(python_exec))
                        
                        environments.append(VirtualEnvironment(
                            name=env_name,
                            path=env_path_obj,
                            python_version=python_version,
                            is_active=env_path == env_data.get('default_prefix'),
                            env_type='conda',
                            packages=packages
                        ))
        
        except Exception as e:
            # Log error but continue
            pass
        
        return environments
    
    def _scan_venv_environments(self) -> List[VirtualEnvironment]:
        """Scan for venv/virtualenv environments."""
        environments = []
        
        # Common locations for virtual environments
        common_venv_locations = [
            Path.home() / '.virtualenvs',
            Path.home() / 'venvs',
            Path.home() / '.local' / 'share' / 'virtualenvs',
            Path.cwd() / 'venv',
            Path.cwd() / '.venv'
        ]
        
        for location in common_venv_locations:
            if location.exists() and location.is_dir():
                try:
                    for env_dir in location.iterdir():
                        if env_dir.is_dir():
                            python_exec = env_dir / 'bin' / 'python'
                            if not python_exec.exists():
                                python_exec = env_dir / 'Scripts' / 'python.exe'  # Windows
                            
                            if python_exec.exists():
                                # This looks like a virtual environment
                                try:
                                    version_result = subprocess.run([
                                        str(python_exec), '--version'
                                    ], capture_output=True, text=True, timeout=10)
                                    python_version = version_result.stdout.strip()
                                    
                                    packages = self._get_installed_packages(str(python_exec))
                                    
                                    environments.append(VirtualEnvironment(
                                        name=env_dir.name,
                                        path=env_dir,
                                        python_version=python_version,
                                        is_active=False,  # Would need more complex detection
                                        env_type='venv',
                                        packages=packages
                                    ))
                                except Exception:
                                    continue
                except Exception:
                    continue
        
        return environments
    
    def _scan_poetry_environments(self) -> List[VirtualEnvironment]:
        """Scan poetry environments."""
        environments = []
        
        try:
            result = subprocess.run(['poetry', 'env', 'list', '--full-path'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            env_name = parts[0]
                            env_path = Path(parts[1])
                            
                            python_exec = env_path / 'bin' / 'python'
                            if not python_exec.exists():
                                python_exec = env_path / 'Scripts' / 'python.exe'
                            
                            if python_exec.exists():
                                try:
                                    version_result = subprocess.run([
                                        str(python_exec), '--version'
                                    ], capture_output=True, text=True, timeout=10)
                                    python_version = version_result.stdout.strip()
                                    
                                    packages = self._get_installed_packages(str(python_exec))
                                    
                                    environments.append(VirtualEnvironment(
                                        name=env_name,
                                        path=env_path,
                                        python_version=python_version,
                                        is_active='(Activated)' in line,
                                        env_type='poetry',
                                        packages=packages
                                    ))
                                except Exception:
                                    continue
        
        except Exception as e:
            pass
        
        return environments


class PythonPackageAnalyzer:
    """Analyzes Python packages for significance and relationships."""
    
    def __init__(self):
        self.ml_frameworks = {
            'torch', 'pytorch', 'tensorflow', 'transformers', 'accelerate',
            'deepspeed', 'lightning', 'keras', 'scikit-learn', 'huggingface-hub'
        }
        
        self.development_tools = {
            'jupyter', 'ipython', 'notebook', 'jupyterlab', 'black', 'flake8',
            'mypy', 'pytest', 'coverage', 'pre-commit'
        }
        
        self.data_tools = {
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
            'bokeh', 'altair', 'dask', 'polars'
        }
    
    def analyze_package_change(self, old_package: Optional[PackageInfo], 
                             new_package: Optional[PackageInfo]) -> Dict[str, Any]:
        """Analyze significance of package change."""
        analysis = {
            'change_type': 'unknown',
            'significance': 0.0,
            'impact_assessment': {},
            'risk_factors': [],
            'compatibility_concerns': []
        }
        
        if old_package is None and new_package is not None:
            # Package installation
            analysis['change_type'] = 'installation'
            analysis.update(self._analyze_installation(new_package))
            
        elif old_package is not None and new_package is None:
            # Package removal
            analysis['change_type'] = 'removal'
            analysis.update(self._analyze_removal(old_package))
            
        elif old_package is not None and new_package is not None:
            # Package update
            analysis['change_type'] = 'update'
            analysis.update(self._analyze_update(old_package, new_package))
        
        return analysis
    
    def _analyze_installation(self, package: PackageInfo) -> Dict[str, Any]:
        """Analyze package installation significance."""
        significance = 0.3  # Base significance
        impact = {}
        risk_factors = []
        
        # Major ML framework installation is highly significant
        if package.name.lower() in self.ml_frameworks:
            significance += 0.5
            impact['framework_addition'] = package.name
            
            if package.is_cuda_related:
                significance += 0.2
                impact['gpu_capability_added'] = True
        
        # Development tool installation
        if package.name.lower() in self.development_tools:
            significance += 0.2
            impact['development_capability_added'] = package.name
        
        # Large packages (indicators of major functionality)
        if package.location and Path(package.location).exists():
            try:
                # Rough estimation of package size impact
                size_mb = sum(f.stat().st_size for f in Path(package.location).rglob('*') 
                            if f.is_file()) / 1024 / 1024
                if size_mb > 500:  # Large package (>500MB)
                    significance += 0.2
                    impact['large_package_mb'] = size_mb
            except Exception:
                pass
        
        # Dependency impact
        if len(package.dependencies) > 10:
            significance += 0.1
            risk_factors.append('high_dependency_count')
        
        return {
            'significance': min(1.0, significance),
            'impact_assessment': impact,
            'risk_factors': risk_factors
        }
    
    def _analyze_removal(self, package: PackageInfo) -> Dict[str, Any]:
        """Analyze package removal significance."""
        significance = 0.3  # Base significance
        impact = {}
        risk_factors = []
        
        # Removing major ML framework is highly significant
        if package.name.lower() in self.ml_frameworks:
            significance += 0.6
            impact['framework_removal'] = package.name
            risk_factors.append('ml_capability_loss')
            
            if package.is_cuda_related:
                significance += 0.3
                risk_factors.append('gpu_capability_loss')
        
        # Removing development tools
        if package.name.lower() in self.development_tools:
            significance += 0.3
            impact['development_tool_removed'] = package.name
        
        # Dependencies could affect other packages
        if len(package.dependencies) > 5:
            risk_factors.append('dependency_breakage_risk')
            significance += 0.2
        
        return {
            'significance': min(1.0, significance),
            'impact_assessment': impact,
            'risk_factors': risk_factors
        }
    
    def _analyze_update(self, old_package: PackageInfo, 
                       new_package: PackageInfo) -> Dict[str, Any]:
        """Analyze package update significance."""
        significance = 0.2  # Base significance for updates
        impact = {}
        risk_factors = []
        compatibility_concerns = []
        
        # Version change analysis
        version_change = self._analyze_version_change(old_package.version, new_package.version)
        significance += version_change['significance_delta']
        impact['version_change'] = version_change
        
        # Major ML framework updates are significant
        if new_package.name.lower() in self.ml_frameworks:
            significance += 0.4
            impact['framework_update'] = new_package.name
            
            # Major version changes in ML frameworks are risky
            if version_change['change_type'] == 'major':
                risk_factors.append('major_ml_framework_change')
                compatibility_concerns.append('api_breaking_changes_likely')
                significance += 0.3
        
        # CUDA-related package updates
        if new_package.is_cuda_related:
            significance += 0.2
            if version_change['change_type'] == 'major':
                risk_factors.append('cuda_compatibility_risk')
        
        return {
            'significance': min(1.0, significance),
            'impact_assessment': impact,
            'risk_factors': risk_factors,
            'compatibility_concerns': compatibility_concerns
        }
    
    def _analyze_version_change(self, old_version: str, new_version: str) -> Dict[str, Any]:
        """Analyze semantic version change."""
        try:
            old_parts = self._parse_version(old_version)
            new_parts = self._parse_version(new_version)
            
            if old_parts[0] != new_parts[0]:
                return {
                    'change_type': 'major',
                    'significance_delta': 0.4,
                    'description': f'Major version change: {old_version} → {new_version}'
                }
            elif old_parts[1] != new_parts[1]:
                return {
                    'change_type': 'minor',
                    'significance_delta': 0.2,
                    'description': f'Minor version change: {old_version} → {new_version}'
                }
            elif len(old_parts) > 2 and len(new_parts) > 2 and old_parts[2] != new_parts[2]:
                return {
                    'change_type': 'patch',
                    'significance_delta': 0.1,
                    'description': f'Patch version change: {old_version} → {new_version}'
                }
            else:
                return {
                    'change_type': 'unknown',
                    'significance_delta': 0.1,
                    'description': f'Version change: {old_version} → {new_version}'
                }
        except Exception:
            return {
                'change_type': 'unknown',
                'significance_delta': 0.1,
                'description': f'Version change: {old_version} → {new_version}'
            }
    
    def _parse_version(self, version: str) -> List[int]:
        """Parse version string into numeric components."""
        # Remove common prefixes and suffixes
        clean_version = re.sub(r'^v?([0-9])', r'\1', version)
        clean_version = re.sub(r'([0-9]+).*$', r'\1', clean_version)
        
        # Extract numeric parts
        parts = []
        for part in clean_version.split('.'):
            try:
                parts.append(int(re.match(r'(\d+)', part).group(1)))
            except (AttributeError, ValueError):
                break
        
        return parts


class PythonEnvChangeDetector(BaseChangeDetector):
    """Detects changes in Python environments and packages."""
    
    def __init__(self, config: PythonEnvDetectorConfig = None):
        super().__init__()
        self.config = config or PythonEnvDetectorConfig()
        self.scanner = PythonEnvironmentScanner()
        self.analyzer = PythonPackageAnalyzer()
        self.previous_system_state = None
        self.previous_environments = {}
    
    def detect_changes(self, old_snapshot: Dict[str, Any], 
                      new_snapshot: Dict[str, Any]) -> List[SystemChange]:
        """Detect Python environment and package changes."""
        changes = []
        
        try:
            # Extract Python environment data
            old_python_data = old_snapshot.get('python_env', {})
            new_python_data = new_snapshot.get('python_env', {})
            
            # If no Python data in snapshots, scan current state
            if not new_python_data:
                new_python_data = self._scan_current_python_state()
            
            # Detect system Python changes
            changes.extend(self._detect_system_python_changes(
                old_python_data.get('system_python', {}),
                new_python_data.get('system_python', {})
            ))
            
            # Detect virtual environment changes
            changes.extend(self._detect_virtual_env_changes(
                old_python_data.get('virtual_environments', []),
                new_python_data.get('virtual_environments', [])
            ))
            
            # Detect package changes across all environments
            changes.extend(self._detect_package_changes(old_python_data, new_python_data))
            
        except Exception as e:
            self._log_error(f"Error detecting Python environment changes: {e}")
        
        return changes
    
    def _scan_current_python_state(self) -> Dict[str, Any]:
        """Scan current Python environment state."""
        try:
            system_python = self.scanner.scan_system_python()
            virtual_envs = self.scanner.scan_virtual_environments()
            
            return {
                'system_python': system_python,
                'virtual_environments': [self._serialize_virtual_env(env) for env in virtual_envs],
                'scan_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _serialize_virtual_env(self, env: VirtualEnvironment) -> Dict[str, Any]:
        """Serialize virtual environment to dict."""
        return {
            'name': env.name,
            'path': str(env.path),
            'python_version': env.python_version,
            'is_active': env.is_active,
            'env_type': env.env_type,
            'is_ml_environment': env.is_ml_environment,
            'packages': [self._serialize_package(pkg) for pkg in env.packages]
        }
    
    def _serialize_package(self, pkg: PackageInfo) -> Dict[str, Any]:
        """Serialize package info to dict."""
        return {
            'name': pkg.name,
            'version': pkg.version,
            'location': pkg.location,
            'dependencies': pkg.dependencies,
            'requires_python': pkg.requires_python,
            'summary': pkg.summary,
            'installer': pkg.installer,
            'is_ml_framework': pkg.is_ml_framework,
            'is_cuda_related': pkg.is_cuda_related
        }
    
    def _detect_system_python_changes(self, old_system: Dict[str, Any], 
                                    new_system: Dict[str, Any]) -> List[SystemChange]:
        """Detect changes in system Python installation."""
        changes = []
        
        if not old_system or not new_system:
            return changes
        
        # Check Python version change
        old_version = old_system.get('python_version', '')
        new_version = new_system.get('python_version', '')
        
        if old_version and new_version and old_version != new_version:
            change = SystemChange(
                category="python_env",
                change_type=ChangeType.MODIFIED,
                entity_id="system_python:version",
                old_value=old_version,
                new_value=new_version,
                significance=0.8,  # Python version changes are significant
                metadata={
                    'change_type': 'python_version_change',
                    'risk_level': 'high',  # Python upgrades can break environments
                    'compatibility_impact': 'system_wide'
                }
            )
            changes.append(change)
        
        # Check for new site-packages locations
        old_paths = set(old_system.get('python_path', []))
        new_paths = set(new_system.get('python_path', []))
        
        added_paths = new_paths - old_paths
        removed_paths = old_paths - new_paths
        
        if added_paths:
            change = SystemChange(
                category="python_env",
                change_type=ChangeType.ADDED,
                entity_id="system_python:paths",
                old_value=None,
                new_value=list(added_paths),
                significance=0.5,
                metadata={
                    'change_type': 'python_path_addition',
                    'added_paths': list(added_paths)
                }
            )
            changes.append(change)
        
        if removed_paths:
            change = SystemChange(
                category="python_env",
                change_type=ChangeType.REMOVED,
                entity_id="system_python:paths",
                old_value=list(removed_paths),
                new_value=None,
                significance=0.6,
                metadata={
                    'change_type': 'python_path_removal',
                    'removed_paths': list(removed_paths),
                    'risk_level': 'medium'
                }
            )
            changes.append(change)
        
        return changes
    
    def _detect_virtual_env_changes(self, old_envs: List[Dict[str, Any]], 
                                   new_envs: List[Dict[str, Any]]) -> List[SystemChange]:
        """Detect virtual environment changes."""
        changes = []
        
        # Create lookup maps
        old_env_map = {env['name']: env for env in old_envs}
        new_env_map = {env['name']: env for env in new_envs}
        
        # Detect new environments
        for env_name, env_data in new_env_map.items():
            if env_name not in old_env_map:
                significance = 0.6
                if env_data.get('is_ml_environment', False):
                    significance = 0.8  # ML environments are more significant
                
                change = SystemChange(
                    category="python_env",
                    change_type=ChangeType.ADDED,
                    entity_id=f"virtual_env:{env_name}",
                    old_value=None,
                    new_value={
                        'name': env_name,
                        'path': env_data['path'],
                        'python_version': env_data['python_version'],
                        'env_type': env_data['env_type'],
                        'is_ml_environment': env_data.get('is_ml_environment', False)
                    },
                    significance=significance,
                    metadata={
                        'change_type': 'virtual_env_creation',
                        'env_type': env_data['env_type'],
                        'package_count': len(env_data.get('packages', [])),
                        'is_ml_environment': env_data.get('is_ml_environment', False)
                    }
                )
                changes.append(change)
        
        # Detect removed environments
        for env_name, env_data in old_env_map.items():
            if env_name not in new_env_map:
                significance = 0.5
                if env_data.get('is_ml_environment', False):
                    significance = 0.7
                
                change = SystemChange(
                    category="python_env",
                    change_type=ChangeType.REMOVED,
                    entity_id=f"virtual_env:{env_name}",
                    old_value={
                        'name': env_name,
                        'path': env_data['path'],
                        'env_type': env_data['env_type']
                    },
                    new_value=None,
                    significance=significance,
                    metadata={
                        'change_type': 'virtual_env_removal',
                        'env_type': env_data['env_type'],
                        'was_ml_environment': env_data.get('is_ml_environment', False)
                    }
                )
                changes.append(change)
        
        # Detect environment modifications (activation status, etc.)
        for env_name in set(old_env_map.keys()) & set(new_env_map.keys()):
            old_env = old_env_map[env_name]
            new_env = new_env_map[env_name]
            
            # Check activation status change
            if old_env.get('is_active') != new_env.get('is_active'):
                change = SystemChange(
                    category="python_env",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"virtual_env:{env_name}:activation",
                    old_value=old_env.get('is_active'),
                    new_value=new_env.get('is_active'),
                    significance=0.4,
                    metadata={
                        'change_type': 'env_activation_change',
                        'env_name': env_name,
                        'activated': new_env.get('is_active', False)
                    }
                )
                changes.append(change)
        
        return changes
    
    def _detect_package_changes(self, old_python_data: Dict[str, Any], 
                               new_python_data: Dict[str, Any]) -> List[SystemChange]:
        """Detect package changes across all Python environments."""
        changes = []
        
        # Process system Python packages
        old_system_packages = self._extract_packages(old_python_data.get('system_python', {}))
        new_system_packages = self._extract_packages(new_python_data.get('system_python', {}))
        
        changes.extend(self._compare_package_sets(
            old_system_packages, new_system_packages, 'system_python'
        ))
        
        # Process virtual environment packages
        old_envs = {env['name']: env for env in old_python_data.get('virtual_environments', [])}
        new_envs = {env['name']: env for env in new_python_data.get('virtual_environments', [])}
        
        for env_name in set(old_envs.keys()) | set(new_envs.keys()):
            old_packages = self._extract_packages(old_envs.get(env_name, {}))
            new_packages = self._extract_packages(new_envs.get(env_name, {}))
            
            changes.extend(self._compare_package_sets(
                old_packages, new_packages, f'virtual_env:{env_name}'
            ))
        
        return changes
    
    def _extract_packages(self, env_data: Dict[str, Any]) -> Dict[str, PackageInfo]:
        """Extract packages from environment data."""
        packages = {}
        
        for pkg_data in env_data.get('packages', []):
            pkg_info = PackageInfo(
                name=pkg_data.get('name', ''),
                version=pkg_data.get('version', ''),
                location=pkg_data.get('location'),
                dependencies=pkg_data.get('dependencies', []),
                requires_python=pkg_data.get('requires_python'),
                summary=pkg_data.get('summary'),
                home_page=pkg_data.get('home_page'),
                installer=pkg_data.get('installer', 'unknown')
            )
            packages[pkg_info.name] = pkg_info
        
        return packages
    
    def _compare_package_sets(self, old_packages: Dict[str, PackageInfo], 
                             new_packages: Dict[str, PackageInfo], 
                             environment_id: str) -> List[SystemChange]:
        """Compare two sets of packages and detect changes."""
        changes = []
        
        all_package_names = set(old_packages.keys()) | set(new_packages.keys())
        
        for package_name in all_package_names:
            old_package = old_packages.get(package_name)
            new_package = new_packages.get(package_name)
            
            # Analyze the change
            analysis = self.analyzer.analyze_package_change(old_package, new_package)
            
            if analysis['significance'] > self.config.min_significance_threshold:
                if analysis['change_type'] == 'installation':
                    change_type = ChangeType.ADDED
                    old_value = None
                    new_value = {
                        'name': new_package.name,
                        'version': new_package.version,
                        'is_ml_framework': new_package.is_ml_framework,
                        'is_cuda_related': new_package.is_cuda_related
                    }
                elif analysis['change_type'] == 'removal':
                    change_type = ChangeType.REMOVED
                    old_value = {
                        'name': old_package.name,
                        'version': old_package.version,
                        'is_ml_framework': old_package.is_ml_framework
                    }
                    new_value = None
                else:  # update
                    change_type = ChangeType.MODIFIED
                    old_value = old_package.version
                    new_value = new_package.version
                
                change = SystemChange(
                    category="python_env",
                    change_type=change_type,
                    entity_id=f"package:{environment_id}:{package_name}",
                    old_value=old_value,
                    new_value=new_value,
                    significance=analysis['significance'],
                    metadata={
                        'environment': environment_id,
                        'package_name': package_name,
                        'change_analysis': analysis,
                        'is_ml_framework': new_package.is_ml_framework if new_package else (old_package.is_ml_framework if old_package else False),
                        'is_cuda_related': new_package.is_cuda_related if new_package else (old_package.is_cuda_related if old_package else False)
                    }
                )
                changes.append(change)
        
        return changes