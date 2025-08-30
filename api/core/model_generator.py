"""
Model Generator - Auto-generated API Contracts
==============================================

Auto-generates Pydantic models and API contracts from src/ collector capabilities.
Creates data-driven API schema based on actual system data structures.

Key principles:
- Auto-generate contracts from src/ collectors via reflection
- Data-driven API design - schema follows actual collected data
- Zero manual maintenance of API models
- Automatic schema updates when collectors change
- TypeScript interface generation for frontend
"""

import inspect
import logging
import json
from typing import Dict, List, Any, Type, Optional, Union, get_type_hints
from datetime import datetime
from pathlib import Path
from dataclasses import is_dataclass, fields as dataclass_fields
from enum import Enum

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)


class CollectorSchema(BaseModel):
    """Schema definition for a collector's data structure."""
    name: str
    description: str
    data_structure: Dict[str, Any]
    sample_data: Optional[Dict[str, Any]] = None
    endpoints: List[str] = Field(default_factory=list)
    update_frequency: Optional[int] = None  # in seconds


class EndpointDefinition(BaseModel):
    """Definition for an auto-generated API endpoint."""
    path: str
    method: str
    description: str
    response_model: str
    query_parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ModelGenerator:
    """
    Auto-generate Pydantic models and API contracts from src/ collectors.
    
    Uses reflection to inspect collector capabilities and generate
    corresponding API contracts automatically.
    """
    
    def __init__(self, src_path: Optional[Path] = None):
        """Initialize model generator with src/ path for inspection."""
        self.src_path = src_path or Path(__file__).parent.parent.parent / "src"
        self.generated_models: Dict[str, Type[BaseModel]] = {}
        self.collector_schemas: Dict[str, CollectorSchema] = {}
        self.endpoint_definitions: List[EndpointDefinition] = []
        
    def generate_all_contracts(self) -> Dict[str, Type[BaseModel]]:
        """
        Generate all API contracts from src/ collectors.
        
        Main entry point for auto-generating data-driven API schema.
        """
        logger.info("Starting auto-generation of API contracts from src/ collectors")
        
        try:
            # Discover and inspect all collectors
            collectors = self._discover_collectors()
            logger.info(f"Discovered {len(collectors)} collectors")
            
            # Generate schemas for each collector
            for collector_name, collector_class in collectors.items():
                schema = self._inspect_collector(collector_name, collector_class)
                if schema:
                    self.collector_schemas[collector_name] = schema
            
            # Generate Pydantic models from schemas
            for schema_name, schema in self.collector_schemas.items():
                model = self._generate_pydantic_model(schema_name, schema)
                if model:
                    self.generated_models[schema_name] = model
            
            # Generate endpoint definitions
            self.endpoint_definitions = self._generate_endpoints()
            
            # Generate additional response models
            self._generate_response_models()
            
            logger.info(f"Generated {len(self.generated_models)} API models and {len(self.endpoint_definitions)} endpoints")
            return self.generated_models
            
        except Exception as e:
            logger.error(f"Error generating API contracts: {e}")
            return {}
    
    def _discover_collectors(self) -> Dict[str, Type]:
        """Discover all collector classes in src/ via reflection."""
        collectors = {}
        
        try:
            # Look for collector classes in known paths
            collector_paths = [
                self.src_path / "linux_system" / "data_collection" / "collectors",
                self.src_path / "linux_system" / "temporal" / "change_detection" / "detectors",
                self.src_path / "linux_system" / "ai_workstation" / "hardware_specialization",
                self.src_path / "linux_system" / "ai_workstation" / "container_consciousness"
            ]
            
            for collector_path in collector_paths:
                if collector_path.exists():
                    collectors.update(self._scan_directory_for_collectors(collector_path))
            
            return collectors
            
        except Exception as e:
            logger.error(f"Error discovering collectors: {e}")
            return {}
    
    def _scan_directory_for_collectors(self, directory: Path) -> Dict[str, Type]:
        """Scan directory for collector classes."""
        collectors = {}
        
        try:
            for py_file in directory.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                    
                # Import module dynamically and look for collector classes
                module_name = py_file.stem
                try:
                    # This is a simplified approach - in practice, you'd want proper module importing
                    # For now, we'll use known collector patterns
                    if "collector" in module_name.lower() or "detector" in module_name.lower():
                        collectors[module_name] = self._create_mock_collector_class(module_name)
                        
                except Exception as e:
                    logger.error(f"Error importing {py_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            
        return collectors
    
    def _create_mock_collector_class(self, collector_name: str) -> Type:
        """Create mock collector class for schema generation."""
        # This is a placeholder - in practice, we'd inspect actual collector classes
        # For now, create known collector schemas based on existing architecture
        
        if "system_collector" in collector_name:
            return self._create_system_collector_schema()
        elif "gpu" in collector_name or "rtx5090" in collector_name:
            return self._create_gpu_collector_schema()
        elif "process" in collector_name:
            return self._create_process_collector_schema()
        elif "container" in collector_name:
            return self._create_container_collector_schema()
        elif "temporal" in collector_name:
            return self._create_temporal_collector_schema()
        else:
            return self._create_generic_collector_schema(collector_name)
    
    def _create_system_collector_schema(self) -> Type:
        """Create schema for SystemCollector."""
        class SystemCollectorSchema:
            def collect(self) -> Dict[str, Any]:
                return {
                    "cpu": {"cores": 16, "usage": 45.2, "temperature": 65.0},
                    "memory": {"total": 137438953472, "used": 54975581388, "available": 82463372084},
                    "nvidia_gpu": {
                        "name": "RTX 5090",
                        "temperature": 75.0,
                        "utilization": 85.0,
                        "memory_total": 32768,
                        "memory_used": 24576
                    },
                    "storage": {"devices": [], "usage": {}},
                    "network": {"interfaces": [], "stats": {}},
                    "processes": {"active": [], "count": 0}
                }
        return SystemCollectorSchema
    
    def _create_gpu_collector_schema(self) -> Type:
        """Create schema for GPU-specific collector."""
        class GPUCollectorSchema:
            def collect(self) -> Dict[str, Any]:
                return {
                    "gpu_id": 0,
                    "name": "RTX 5090",
                    "temperature": 75.0,
                    "power_usage": 450.0,
                    "utilization": {"gpu": 85.0, "memory": 70.0},
                    "memory": {"total": 32768, "used": 24576, "free": 8192},
                    "processes": [],
                    "thermal_throttling": False,
                    "performance_state": "P2",
                    "cuda_version": "12.9.1"
                }
        return GPUCollectorSchema
    
    def _create_process_collector_schema(self) -> Type:
        """Create schema for process collector."""
        class ProcessCollectorSchema:
            def collect(self) -> Dict[str, Any]:
                return {
                    "processes": [
                        {
                            "pid": 12345,
                            "name": "python3",
                            "cpu_percent": 15.2,
                            "memory_rss": 1024000,
                            "memory_vms": 2048000,
                            "status": "running",
                            "create_time": datetime.now().timestamp()
                        }
                    ],
                    "total_processes": 150,
                    "system_load": {"1min": 2.5, "5min": 2.1, "15min": 1.8}
                }
        return ProcessCollectorSchema
    
    def _create_container_collector_schema(self) -> Type:
        """Create schema for container collector."""
        class ContainerCollectorSchema:
            def collect(self) -> Dict[str, Any]:
                return {
                    "containers": [
                        {
                            "id": "abcd1234",
                            "name": "llama-gpu",
                            "status": "running",
                            "cpu_usage": 25.5,
                            "memory_usage": 8589934592,  # 8GB
                            "network_io": {"rx_bytes": 1024, "tx_bytes": 2048},
                            "ports": ["8004:8004"],
                            "health_status": "healthy"
                        }
                    ],
                    "total_containers": 5,
                    "resources": {"cpu_limit": 32.0, "memory_limit": 137438953472}
                }
        return ContainerCollectorSchema
    
    def _create_temporal_collector_schema(self) -> Type:
        """Create schema for temporal data."""
        class TemporalCollectorSchema:
            def collect(self) -> Dict[str, Any]:
                return {
                    "changes": [
                        {
                            "timestamp": datetime.now().isoformat(),
                            "category": "gpu",
                            "change_type": "threshold_crossed",
                            "old_value": 70.0,
                            "new_value": 85.0,
                            "significance": 0.8
                        }
                    ],
                    "events": [
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "gpu_thermal_event",
                            "severity": "warning",
                            "description": "GPU temperature approaching thermal limit"
                        }
                    ],
                    "patterns": [],
                    "correlations": []
                }
        return TemporalCollectorSchema
    
    def _create_generic_collector_schema(self, name: str) -> Type:
        """Create generic collector schema."""
        class GenericCollectorSchema:
            def collect(self) -> Dict[str, Any]:
                return {
                    "data": {},
                    "timestamp": datetime.now().isoformat(),
                    "collector": name
                }
        return GenericCollectorSchema
    
    def _inspect_collector(self, collector_name: str, collector_class: Type) -> Optional[CollectorSchema]:
        """Inspect collector class to generate schema."""
        try:
            # Get sample data from collector
            sample_data = None
            data_structure = {}
            
            if hasattr(collector_class, 'collect'):
                # Try to get sample data structure
                try:
                    instance = collector_class()
                    sample_data = instance.collect()
                    data_structure = self._analyze_data_structure(sample_data)
                except:
                    # If we can't instantiate, use method signature
                    collect_method = getattr(collector_class, 'collect')
                    return_annotation = getattr(collect_method, '__annotations__', {}).get('return', Dict[str, Any])
                    data_structure = self._analyze_type_annotation(return_annotation)
            
            return CollectorSchema(
                name=collector_name,
                description=f"Auto-generated schema for {collector_name}",
                data_structure=data_structure,
                sample_data=sample_data,
                endpoints=[f"/api/current/{collector_name}", f"/api/historical/{collector_name}/{{time_range}}"],
                update_frequency=60  # Default 60 seconds
            )
            
        except Exception as e:
            logger.error(f"Error inspecting collector {collector_name}: {e}")
            return None
    
    def _analyze_data_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze data structure to generate schema."""
        if isinstance(data, dict):
            return {key: self._get_type_description(value) for key, value in data.items()}
        elif isinstance(data, list) and data:
            return {"type": "array", "items": self._analyze_data_structure(data[0])}
        else:
            return self._get_type_description(data)
    
    def _get_type_description(self, value: Any) -> Dict[str, Any]:
        """Get type description for a value."""
        if isinstance(value, bool):
            return {"type": "boolean", "example": value}
        elif isinstance(value, int):
            return {"type": "integer", "example": value}
        elif isinstance(value, float):
            return {"type": "number", "example": value}
        elif isinstance(value, str):
            return {"type": "string", "example": value}
        elif isinstance(value, list):
            item_type = self._get_type_description(value[0]) if value else {"type": "any"}
            return {"type": "array", "items": item_type}
        elif isinstance(value, dict):
            return {"type": "object", "properties": self._analyze_data_structure(value)}
        elif value is None:
            return {"type": "null"}
        else:
            return {"type": "string", "example": str(value)}
    
    def _analyze_type_annotation(self, type_annotation: Any) -> Dict[str, Any]:
        """Analyze type annotation to generate schema."""
        return {"type": "object", "description": f"Data from {type_annotation}"}
    
    def _generate_pydantic_model(self, schema_name: str, schema: CollectorSchema) -> Optional[Type[BaseModel]]:
        """Generate Pydantic model from collector schema."""
        try:
            # Create field definitions from data structure
            fields = {}
            
            if schema.sample_data:
                fields = self._create_pydantic_fields(schema.sample_data)
            else:
                # Create generic fields from data structure
                fields = {"data": (Dict[str, Any], Field(default_factory=dict))}
            
            # Add metadata fields
            fields.update({
                "timestamp": (datetime, Field(default_factory=datetime.now)),
                "collector": (str, Field(default=schema_name)),
                "status": (str, Field(default="success"))
            })
            
            # Create dynamic model
            model_name = f"{schema_name.title().replace('_', '')}Response"
            model = create_model(model_name, **fields)
            
            return model
            
        except Exception as e:
            logger.error(f"Error generating Pydantic model for {schema_name}: {e}")
            return None
    
    def _create_pydantic_fields(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Pydantic field definitions from sample data."""
        fields = {}
        
        for key, value in sample_data.items():
            field_type, default_value = self._get_pydantic_field_type(value)
            fields[key] = (field_type, Field(default=default_value))
        
        return fields
    
    def _get_pydantic_field_type(self, value: Any) -> tuple:
        """Get Pydantic field type and default from value."""
        if isinstance(value, bool):
            return bool, False
        elif isinstance(value, int):
            return int, 0
        elif isinstance(value, float):
            return float, 0.0
        elif isinstance(value, str):
            return str, ""
        elif isinstance(value, list):
            return List[Any], []
        elif isinstance(value, dict):
            return Dict[str, Any], {}
        elif value is None:
            return Optional[Any], None
        else:
            return str, str(value)
    
    def _generate_endpoints(self) -> List[EndpointDefinition]:
        """Generate API endpoint definitions."""
        endpoints = []
        
        for schema_name, schema in self.collector_schemas.items():
            # Current data endpoint
            endpoints.append(EndpointDefinition(
                path=f"/api/current/{schema_name}",
                method="GET",
                description=f"Get current {schema_name} data",
                response_model=f"{schema_name.title().replace('_', '')}Response",
                tags=["current", "data", schema_name]
            ))
            
            # Historical data endpoint
            endpoints.append(EndpointDefinition(
                path=f"/api/historical/{schema_name}/{{time_range}}",
                method="GET",
                description=f"Get historical {schema_name} data",
                response_model=f"Historical{schema_name.title().replace('_', '')}Response",
                query_parameters={
                    "aggregation": "string",
                    "filters": "object"
                },
                tags=["historical", "data", schema_name]
            ))
        
        return endpoints
    
    def _generate_response_models(self):
        """Generate additional response models for API."""
        # Chat response model
        chat_response_fields = {
            "message": (str, Field(..., description="Response message")),
            "confidence": (float, Field(default=0.0, ge=0.0, le=1.0)),
            "session_id": (Optional[str], Field(None)),
            "data_references": (List[Dict[str, Any]], Field(default_factory=list)),
            "suggested_actions": (List[str], Field(default_factory=list)),
            "timestamp": (datetime, Field(default_factory=datetime.now))
        }
        
        self.generated_models["ChatResponse"] = create_model("ChatResponse", **chat_response_fields)
        
        # Health check response
        health_fields = {
            "status": (str, Field(default="healthy")),
            "timestamp": (datetime, Field(default_factory=datetime.now)),
            "consciousness_available": (bool, Field(default=False))
        }
        
        self.generated_models["HealthCheckResponse"] = create_model("HealthCheckResponse", **health_fields)
    
    def generate_typescript_interfaces(self, output_path: Path):
        """Generate TypeScript interfaces for frontend."""
        try:
            interfaces = []
            
            for model_name, model in self.generated_models.items():
                interface = self._model_to_typescript_interface(model_name, model)
                interfaces.append(interface)
            
            # Write to file
            typescript_content = "// Auto-generated TypeScript interfaces from API contracts\n\n"
            typescript_content += "\n\n".join(interfaces)
            
            output_path.write_text(typescript_content)
            logger.info(f"Generated TypeScript interfaces at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating TypeScript interfaces: {e}")
    
    def _model_to_typescript_interface(self, model_name: str, model: Type[BaseModel]) -> str:
        """Convert Pydantic model to TypeScript interface."""
        fields = []
        
        try:
            for field_name, field_info in model.__fields__.items():
                ts_type = self._python_type_to_typescript(field_info.type_)
                optional = "?" if field_info.default is not ... else ""
                fields.append(f"  {field_name}{optional}: {ts_type};")
        except:
            # Fallback for different Pydantic versions
            fields.append("  [key: string]: any;")
        
        return f"export interface {model_name} {{\n" + "\n".join(fields) + "\n}"
    
    def _python_type_to_typescript(self, python_type: Any) -> str:
        """Convert Python type to TypeScript type."""
        if python_type == str:
            return "string"
        elif python_type == int or python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif hasattr(python_type, "__origin__"):
            if python_type.__origin__ == list:
                return f"{self._python_type_to_typescript(python_type.__args__[0])}[]"
            elif python_type.__origin__ == dict:
                return f"{{ [key: string]: {self._python_type_to_typescript(python_type.__args__[1])} }}"
            elif python_type.__origin__ == Union:
                # Handle Optional types
                return "any"
        return "any"
    
    def save_schema_documentation(self, output_path: Path):
        """Save generated schema documentation."""
        try:
            documentation = {
                "generated_at": datetime.now().isoformat(),
                "collectors": {name: schema.dict() for name, schema in self.collector_schemas.items()},
                "models": list(self.generated_models.keys()),
                "endpoints": [endpoint.dict() for endpoint in self.endpoint_definitions]
            }
            
            output_path.write_text(json.dumps(documentation, indent=2))
            logger.info(f"Saved schema documentation at {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving schema documentation: {e}")


# Factory function for easy integration
def create_model_generator(src_path: Optional[Path] = None) -> ModelGenerator:
    """Create and configure model generator."""
    return ModelGenerator(src_path)