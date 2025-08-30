"""
Mock Snapshot Factory
======================

Factory for creating realistic SystemCollector snapshots for testing.
Generates comprehensive mock data that mimics real system behavior,
including RTX 5090 GPU data, process information, and system metrics.
"""

import json
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MockScenarioConfig:
    """Configuration for mock scenario generation."""
    gpu_temp_celsius: float = 75.0
    gpu_memory_used_mb: int = 8192  # 8GB of 32GB used
    gpu_utilization_percent: int = 45
    gpu_power_watts: int = 180
    system_memory_used_percent: float = 65.0
    cpu_usage_percent: float = 25.0
    active_processes: int = 150
    python_packages_count: int = 200


class MockNvidiaGPUData:
    """Factory for NVIDIA RTX 5090 GPU mock data."""
    
    @staticmethod
    def create_basic_metrics(config: MockScenarioConfig) -> str:
        """Create nvidia-smi basic metrics CSV output."""
        timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        
        return f"""timestamp, name, pci.bus_id, driver_version, pstate, pcie.link.gen.max, pcie.link.gen.current, temperature.gpu, fan.speed, utilization.gpu, utilization.memory, memory.total, memory.free, memory.used, power.draw, power.limit
{timestamp}, NVIDIA GeForce RTX 5090, 00000000:01:00.0, 535.98, P2, 4, 4, {config.gpu_temp_celsius}, 75, {config.gpu_utilization_percent}, 85, 32768, {32768 - config.gpu_memory_used_mb}, {config.gpu_memory_used_mb}, {config.gpu_power_watts:.1f}, 600.00"""
    
    @staticmethod
    def create_detailed_xml(config: MockScenarioConfig) -> str:
        """Create nvidia-smi XML output."""
        return f"""<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
<nvidia_smi_log>
	<timestamp>Wed Nov 15 10:30:45 2023</timestamp>
	<driver_version>535.98</driver_version>
	<cuda_version>12.2</cuda_version>
	<attached_gpus>1</attached_gpus>
	<gpu id="00000000:01:00.0">
		<product_name>NVIDIA GeForce RTX 5090</product_name>
		<product_brand>GeForce</product_brand>
		<product_architecture>Ada Lovelace</product_architecture>
		<display_mode>Enabled</display_mode>
		<display_active>Disabled</display_active>
		<persistence_mode>Disabled</persistence_mode>
		<addressing_mode>None</addressing_mode>
		<mig_mode>
			<current_mig>N/A</current_mig>
			<pending_mig>N/A</pending_mig>
		</mig_mode>
		<mig_devices>
			None
		</mig_devices>
		<accounting_mode>Disabled</accounting_mode>
		<accounting_mode_buffer_size>4000</accounting_mode_buffer_size>
		<driver_model>
			<current_dm>WDDM</current_dm>
			<pending_dm>WDDM</pending_dm>
		</driver_model>
		<serial>1234567890123</serial>
		<uuid>GPU-a1b2c3d4-e5f6-7890-1234-567890abcdef</uuid>
		<minor_number>0</minor_number>
		<vbios_version>95.02.5C.00.01</vbios_version>
		<multigpu_board>No</multigpu_board>
		<board_id>0x100</board_id>
		<board_part_number>900-1G137-2530-000</board_part_number>
		<gpu_part_number>2684-200-A1</gpu_part_number>
		<gpu_module_id>1</gpu_module_id>
		<inforom_version>
			<img_version>G137.0200.00.02</img_version>
			<oem_object>2.0</oem_object>
			<ecc_object>N/A</ecc_object>
			<pwr_object>N/A</pwr_object>
		</inforom_version>
		<gpu_operation_mode>
			<current_gom>N/A</current_gom>
			<pending_gom>N/A</pending_gom>
		</gpu_operation_mode>
		<gpu_virtualization_mode>
			<virtualization_mode>None</virtualization_mode>
		</gpu_virtualization_mode>
		<ibmnpu>
			<relaxed_ordering_mode>N/A</relaxed_ordering_mode>
		</ibmnpu>
		<pci>
			<pci_bus>01</pci_bus>
			<pci_device>00</pci_device>
			<pci_domain>0000</pci_domain>
			<pci_device_id>268410DE</pci_device_id>
			<pci_bus_id>00000000:01:00.0</pci_bus_id>
			<pci_sub_system_id>408A10DE</pci_sub_system_id>
			<pci_gpu_link_info>
				<pcie_gen>
					<max_link_gen>4</max_link_gen>
					<current_link_gen>4</current_link_gen>
				</pcie_gen>
				<link_widths>
					<max_link_width>16x</max_link_width>
					<current_link_width>16x</current_link_width>
				</link_widths>
			</pci_gpu_link_info>
			<pci_bridge_chip>
				<bridge_chip_type>N/A</bridge_chip_type>
				<bridge_chip_fw>N/A</bridge_chip_fw>
			</pci_bridge_chip>
			<replay_counter>0</replay_counter>
			<replay_rollover_counter>0</replay_rollover_counter>
			<tx_util>0 KB/s</tx_util>
			<rx_util>0 KB/s</rx_util>
		</pci>
		<fan_speed>75 %</fan_speed>
		<performance_state>P2</performance_state>
		<clocks_throttle_reasons>
			<clocks_throttle_reason_gpu_idle>Not Active</clocks_throttle_reason_gpu_idle>
			<clocks_throttle_reason_applications_clocks_setting>Not Active</clocks_throttle_reason_applications_clocks_setting>
			<clocks_throttle_reason_sw_power_cap>Not Active</clocks_throttle_reason_sw_power_cap>
			<clocks_throttle_reason_hw_slowdown>Not Active</clocks_throttle_reason_hw_slowdown>
			<clocks_throttle_reason_hw_thermal_slowdown>Not Active</clocks_throttle_reason_hw_thermal_slowdown>
			<clocks_throttle_reason_hw_power_brake_slowdown>Not Active</clocks_throttle_reason_hw_power_brake_slowdown>
			<clocks_throttle_reason_sync_boost>Not Active</clocks_throttle_reason_sync_boost>
			<clocks_throttle_reason_sw_thermal_slowdown>Not Active</clocks_throttle_reason_sw_thermal_slowdown>
			<clocks_throttle_reason_display_clocks_setting>Not Active</clocks_throttle_reason_display_clocks_setting>
		</clocks_throttle_reasons>
		<memory_usage>
			<total>{32768} MiB</total>
			<reserved>158 MiB</reserved>
			<used>{config.gpu_memory_used_mb} MiB</used>
			<free>{32768 - config.gpu_memory_used_mb} MiB</free>
		</memory_usage>
		<compute_mode>Default</compute_mode>
		<utilization>
			<gpu_util>{config.gpu_utilization_percent} %</gpu_util>
			<memory_util>85 %</memory_util>
			<encoder_util>0 %</encoder_util>
			<decoder_util>0 %</decoder_util>
			<jpeg_util>N/A %</jpeg_util>
			<ofa_util>N/A %</ofa_util>
		</utilization>
		<encoder_stats>
			<session_count>0</session_count>
			<average_fps>0</average_fps>
			<average_latency>0</average_latency>
		</encoder_stats>
		<fbc_stats>
			<session_count>0</session_count>
			<average_fps>0</average_fps>
			<average_latency>0</average_latency>
		</fbc_stats>
		<ecc_mode>
			<current_ecc>N/A</current_ecc>
			<pending_ecc>N/A</pending_ecc>
		</ecc_mode>
		<ecc_errors>
			<volatile>
				<single_bit>
					<device_memory>N/A</device_memory>
					<register_file>N/A</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>N/A</l2_cache>
					<texture_memory>N/A</texture_memory>
					<texture_shm>N/A</texture_shm>
					<cbu>N/A</cbu>
					<total>N/A</total>
				</single_bit>
				<double_bit>
					<device_memory>N/A</device_memory>
					<register_file>N/A</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>N/A</l2_cache>
					<texture_memory>N/A</texture_memory>
					<texture_shm>N/A</texture_shm>
					<cbu>N/A</cbu>
					<total>N/A</total>
				</double_bit>
			</volatile>
			<aggregate>
				<single_bit>
					<device_memory>N/A</device_memory>
					<register_file>N/A</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>N/A</l2_cache>
					<texture_memory>N/A</texture_memory>
					<texture_shm>N/A</texture_shm>
					<cbu>N/A</cbu>
					<total>N/A</total>
				</single_bit>
				<double_bit>
					<device_memory>N/A</device_memory>
					<register_file>N/A</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>N/A</l2_cache>
					<texture_memory>N/A</texture_memory>
					<texture_shm>N/A</texture_shm>
					<cbu>N/A</cbu>
					<total>N/A</total>
				</double_bit>
			</aggregate>
		</ecc_errors>
		<retired_pages>
			<multiple_single_bit_retirement>
				<retired_count>N/A</retired_count>
				<retired_pagelist>N/A</retired_pagelist>
			</multiple_single_bit_retirement>
			<double_bit_retirement>
				<retired_count>N/A</retired_count>
				<retired_pagelist>N/A</retired_pagelist>
			</double_bit_retirement>
			<pending_blacklist>N/A</pending_blacklist>
			<pending_retirement>N/A</pending_retirement>
		</retired_pages>
		<remapped_rows>N/A</remapped_rows>
		<temperature>
			<gpu_temp>{config.gpu_temp_celsius} C</gpu_temp>
			<gpu_temp_max_threshold>95 C</gpu_temp_max_threshold>
			<gpu_temp_slow_threshold>92 C</gpu_temp_slow_threshold>
			<gpu_temp_max_gpu_threshold>90 C</gpu_temp_max_gpu_threshold>
			<gpu_target_temperature>83 C</gpu_target_temperature>
			<memory_temp>N/A</memory_temp>
			<memory_temp_max_threshold>N/A</memory_temp_max_threshold>
		</temperature>
		<supported_gpu_target_temp>
			<gpu_target_temp_min>65 C</gpu_target_temp_min>
			<gpu_target_temp_max>95 C</gpu_target_temp_max>
		</supported_gpu_target_temp>
		<power_readings>
			<power_state>P2</power_state>
			<power_management>Supported</power_management>
			<power_draw>{config.gpu_power_watts:.2f} W</power_draw>
			<power_limit>600.00 W</power_limit>
			<default_power_limit>600.00 W</default_power_limit>
			<enforced_power_limit>600.00 W</enforced_power_limit>
			<min_power_limit>150.00 W</min_power_limit>
			<max_power_limit>600.00 W</max_power_limit>
		</power_readings>
		<clocks>
			<graphics_clock>1920 MHz</graphics_clock>
			<sm_clock>1920 MHz</sm_clock>
			<mem_clock>1313 MHz</mem_clock>
			<video_clock>1710 MHz</video_clock>
		</clocks>
		<applications_clocks>
			<graphics_clock>1755 MHz</graphics_clock>
			<mem_clock>1313 MHz</mem_clock>
		</applications_clocks>
		<default_applications_clocks>
			<graphics_clock>1755 MHz</graphics_clock>
			<mem_clock>1313 MHz</mem_clock>
		</default_applications_clocks>
		<deferred_clocks>
			<mem_clock>N/A</mem_clock>
		</deferred_clocks>
		<max_clocks>
			<graphics_clock>2520 MHz</graphics_clock>
			<sm_clock>2520 MHz</sm_clock>
			<mem_clock>1313 MHz</mem_clock>
			<video_clock>1950 MHz</video_clock>
		</max_clocks>
		<max_customer_boost_clocks>
			<graphics_clock>2520 MHz</graphics_clock>
		</max_customer_boost_clocks>
		<clock_policy>
			<auto_boost>N/A</auto_boost>
			<auto_boost_default>N/A</auto_boost_default>
		</clock_policy>
		<voltage>
			<graphics_volt>N/A</graphics_volt>
		</voltage>
		<fabric>
			<state>N/A</state>
			<status>N/A</status>
		</fabric>
		<supported_clocks>
			<supported_mem_clock>
				<value>1313 MHz</value>
				<supported_graphics_clock>2520 MHz</supported_graphics_clock>
				<supported_graphics_clock>2505 MHz</supported_graphics_clock>
				<supported_graphics_clock>2490 MHz</supported_graphics_clock>
				<supported_graphics_clock>2475 MHz</supported_graphics_clock>
				<supported_graphics_clock>2460 MHz</supported_graphics_clock>
				<supported_graphics_clock>2445 MHz</supported_graphics_clock>
				<supported_graphics_clock>2430 MHz</supported_graphics_clock>
				<supported_graphics_clock>2415 MHz</supported_graphics_clock>
				<supported_graphics_clock>2400 MHz</supported_graphics_clock>
				<supported_graphics_clock>2385 MHz</supported_graphics_clock>
				<supported_graphics_clock>2370 MHz</supported_graphics_clock>
				<supported_graphics_clock>2355 MHz</supported_graphics_clock>
				<supported_graphics_clock>2340 MHz</supported_graphics_clock>
				<supported_graphics_clock>2325 MHz</supported_graphics_clock>
				<supported_graphics_clock>2310 MHz</supported_graphics_clock>
				<supported_graphics_clock>2295 MHz</supported_graphics_clock>
				<supported_graphics_clock>2280 MHz</supported_graphics_clock>
				<supported_graphics_clock>2265 MHz</supported_graphics_clock>
				<supported_graphics_clock>2250 MHz</supported_graphics_clock>
				<supported_graphics_clock>2235 MHz</supported_graphics_clock>
				<supported_graphics_clock>2220 MHz</supported_graphics_clock>
				<supported_graphics_clock>2205 MHz</supported_graphics_clock>
				<supported_graphics_clock>2190 MHz</supported_graphics_clock>
				<supported_graphics_clock>2175 MHz</supported_graphics_clock>
				<supported_graphics_clock>2160 MHz</supported_graphics_clock>
				<supported_graphics_clock>2145 MHz</supported_graphics_clock>
				<supported_graphics_clock>2130 MHz</supported_graphics_clock>
				<supported_graphics_clock>2115 MHz</supported_graphics_clock>
				<supported_graphics_clock>2100 MHz</supported_graphics_clock>
				<supported_graphics_clock>2085 MHz</supported_graphics_clock>
				<supported_graphics_clock>2070 MHz</supported_graphics_clock>
				<supported_graphics_clock>2055 MHz</supported_graphics_clock>
				<supported_graphics_clock>2040 MHz</supported_graphics_clock>
				<supported_graphics_clock>2025 MHz</supported_graphics_clock>
				<supported_graphics_clock>2010 MHz</supported_graphics_clock>
				<supported_graphics_clock>1995 MHz</supported_graphics_clock>
				<supported_graphics_clock>1980 MHz</supported_graphics_clock>
				<supported_graphics_clock>1965 MHz</supported_graphics_clock>
				<supported_graphics_clock>1950 MHz</supported_graphics_clock>
				<supported_graphics_clock>1935 MHz</supported_graphics_clock>
				<supported_graphics_clock>1920 MHz</supported_graphics_clock>
				<supported_graphics_clock>1905 MHz</supported_graphics_clock>
				<supported_graphics_clock>1890 MHz</supported_graphics_clock>
				<supported_graphics_clock>1875 MHz</supported_graphics_clock>
				<supported_graphics_clock>1860 MHz</supported_graphics_clock>
				<supported_graphics_clock>1845 MHz</supported_graphics_clock>
				<supported_graphics_clock>1830 MHz</supported_graphics_clock>
				<supported_graphics_clock>1815 MHz</supported_graphics_clock>
				<supported_graphics_clock>1800 MHz</supported_graphics_clock>
				<supported_graphics_clock>1785 MHz</supported_graphics_clock>
				<supported_graphics_clock>1770 MHz</supported_graphics_clock>
				<supported_graphics_clock>1755 MHz</supported_graphics_clock>
				<supported_graphics_clock>1740 MHz</supported_graphics_clock>
				<supported_graphics_clock>1725 MHz</supported_graphics_clock>
				<supported_graphics_clock>1710 MHz</supported_graphics_clock>
				<supported_graphics_clock>1695 MHz</supported_graphics_clock>
				<supported_graphics_clock>1680 MHz</supported_graphics_clock>
				<supported_graphics_clock>1665 MHz</supported_graphics_clock>
				<supported_graphics_clock>1650 MHz</supported_graphics_clock>
				<supported_graphics_clock>1635 MHz</supported_graphics_clock>
				<supported_graphics_clock>1620 MHz</supported_graphics_clock>
				<supported_graphics_clock>1605 MHz</supported_graphics_clock>
				<supported_graphics_clock>1590 MHz</supported_graphics_clock>
				<supported_graphics_clock>1575 MHz</supported_graphics_clock>
				<supported_graphics_clock>1560 MHz</supported_graphics_clock>
				<supported_graphics_clock>1545 MHz</supported_graphics_clock>
				<supported_graphics_clock>1530 MHz</supported_graphics_clock>
				<supported_graphics_clock>1515 MHz</supported_graphics_clock>
				<supported_graphics_clock>1500 MHz</supported_graphics_clock>
				<supported_graphics_clock>1485 MHz</supported_graphics_clock>
				<supported_graphics_clock>1470 MHz</supported_graphics_clock>
				<supported_graphics_clock>1455 MHz</supported_graphics_clock>
				<supported_graphics_clock>1440 MHz</supported_graphics_clock>
				<supported_graphics_clock>1425 MHz</supported_graphics_clock>
				<supported_graphics_clock>1410 MHz</supported_graphics_clock>
				<supported_graphics_clock>1395 MHz</supported_graphics_clock>
				<supported_graphics_clock>1380 MHz</supported_graphics_clock>
				<supported_graphics_clock>1365 MHz</supported_graphics_clock>
				<supported_graphics_clock>1350 MHz</supported_graphics_clock>
				<supported_graphics_clock>1335 MHz</supported_graphics_clock>
				<supported_graphics_clock>1320 MHz</supported_graphics_clock>
				<supported_graphics_clock>1305 MHz</supported_graphics_clock>
				<supported_graphics_clock>1290 MHz</supported_graphics_clock>
				<supported_graphics_clock>1275 MHz</supported_graphics_clock>
				<supported_graphics_clock>1260 MHz</supported_graphics_clock>
				<supported_graphics_clock>1245 MHz</supported_graphics_clock>
				<supported_graphics_clock>1230 MHz</supported_graphics_clock>
				<supported_graphics_clock>1215 MHz</supported_graphics_clock>
				<supported_graphics_clock>1200 MHz</supported_graphics_clock>
				<supported_graphics_clock>1185 MHz</supported_graphics_clock>
				<supported_graphics_clock>1170 MHz</supported_graphics_clock>
				<supported_graphics_clock>1155 MHz</supported_graphics_clock>
				<supported_graphics_clock>1140 MHz</supported_graphics_clock>
				<supported_graphics_clock>1125 MHz</supported_graphics_clock>
				<supported_graphics_clock>1110 MHz</supported_graphics_clock>
				<supported_graphics_clock>1095 MHz</supported_graphics_clock>
				<supported_graphics_clock>1080 MHz</supported_graphics_clock>
				<supported_graphics_clock>1065 MHz</supported_graphics_clock>
				<supported_graphics_clock>1050 MHz</supported_graphics_clock>
				<supported_graphics_clock>1035 MHz</supported_graphics_clock>
				<supported_graphics_clock>1020 MHz</supported_graphics_clock>
				<supported_graphics_clock>1005 MHz</supported_graphics_clock>
				<supported_graphics_clock>990 MHz</supported_graphics_clock>
				<supported_graphics_clock>975 MHz</supported_graphics_clock>
				<supported_graphics_clock>960 MHz</supported_graphics_clock>
				<supported_graphics_clock>945 MHz</supported_graphics_clock>
				<supported_graphics_clock>930 MHz</supported_graphics_clock>
				<supported_graphics_clock>915 MHz</supported_graphics_clock>
				<supported_graphics_clock>900 MHz</supported_graphics_clock>
				<supported_graphics_clock>885 MHz</supported_graphics_clock>
				<supported_graphics_clock>870 MHz</supported_graphics_clock>
				<supported_graphics_clock>855 MHz</supported_graphics_clock>
				<supported_graphics_clock>840 MHz</supported_graphics_clock>
				<supported_graphics_clock>825 MHz</supported_graphics_clock>
				<supported_graphics_clock>810 MHz</supported_graphics_clock>
				<supported_graphics_clock>795 MHz</supported_graphics_clock>
				<supported_graphics_clock>780 MHz</supported_graphics_clock>
				<supported_graphics_clock>765 MHz</supported_graphics_clock>
				<supported_graphics_clock>750 MHz</supported_graphics_clock>
				<supported_graphics_clock>735 MHz</supported_graphics_clock>
				<supported_graphics_clock>720 MHz</supported_graphics_clock>
				<supported_graphics_clock>705 MHz</supported_graphics_clock>
				<supported_graphics_clock>690 MHz</supported_graphics_clock>
				<supported_graphics_clock>675 MHz</supported_graphics_clock>
				<supported_graphics_clock>660 MHz</supported_graphics_clock>
				<supported_graphics_clock>645 MHz</supported_graphics_clock>
				<supported_graphics_clock>630 MHz</supported_graphics_clock>
				<supported_graphics_clock>615 MHz</supported_graphics_clock>
				<supported_graphics_clock>600 MHz</supported_graphics_clock>
				<supported_graphics_clock>585 MHz</supported_graphics_clock>
				<supported_graphics_clock>570 MHz</supported_graphics_clock>
				<supported_graphics_clock>555 MHz</supported_graphics_clock>
				<supported_graphics_clock>540 MHz</supported_graphics_clock>
				<supported_graphics_clock>525 MHz</supported_graphics_clock>
				<supported_graphics_clock>510 MHz</supported_graphics_clock>
				<supported_graphics_clock>495 MHz</supported_graphics_clock>
				<supported_graphics_clock>480 MHz</supported_graphics_clock>
				<supported_graphics_clock>465 MHz</supported_graphics_clock>
				<supported_graphics_clock>450 MHz</supported_graphics_clock>
				<supported_graphics_clock>435 MHz</supported_graphics_clock>
				<supported_graphics_clock>420 MHz</supported_graphics_clock>
				<supported_graphics_clock>405 MHz</supported_graphics_clock>
				<supported_graphics_clock>390 MHz</supported_graphics_clock>
				<supported_graphics_clock>375 MHz</supported_graphics_clock>
				<supported_graphics_clock>360 MHz</supported_graphics_clock>
				<supported_graphics_clock>345 MHz</supported_graphics_clock>
				<supported_graphics_clock>330 MHz</supported_graphics_clock>
				<supported_graphics_clock>315 MHz</supported_graphics_clock>
				<supported_graphics_clock>300 MHz</supported_graphics_clock>
				<supported_graphics_clock>285 MHz</supported_graphics_clock>
				<supported_graphics_clock>270 MHz</supported_graphics_clock>
				<supported_graphics_clock>255 MHz</supported_graphics_clock>
				<supported_graphics_clock>240 MHz</supported_graphics_clock>
				<supported_graphics_clock>225 MHz</supported_graphics_clock>
				<supported_graphics_clock>210 MHz</supported_graphics_clock>
			</supported_mem_clock>
		</supported_clocks>
		<processes>
		</processes>
		<accounted_processes>
		</accounted_processes>
	</gpu>
</nvidia_smi_log>"""
    
    @staticmethod
    def create_gpu_processes(pids: List[int] = None, process_names: List[str] = None) -> str:
        """Create GPU processes CSV output."""
        if pids is None:
            pids = [12345, 23456]
        if process_names is None:
            process_names = ["python", "python"]
        
        header = "pid, process_name, gpu_uuid, gpu_name, used_memory"
        processes = []
        
        for i, pid in enumerate(pids):
            name = process_names[i] if i < len(process_names) else "unknown"
            memory = 2048 + (i * 1024)  # Varying memory usage
            processes.append(f"{pid}, {name}, GPU-a1b2c3d4-e5f6-7890-1234-567890abcdef, NVIDIA GeForce RTX 5090, {memory} MiB")
        
        return header + "\n" + "\n".join(processes)
    
    @staticmethod
    def with_temperature(temp_celsius: float) -> Dict[str, Any]:
        """Create GPU data with specific temperature."""
        config = MockScenarioConfig(gpu_temp_celsius=temp_celsius)
        return {
            'basic_metrics': MockNvidiaGPUData.create_basic_metrics(config),
            'detailed_xml': MockNvidiaGPUData.create_detailed_xml(config),
            'compute_processes': MockNvidiaGPUData.create_gpu_processes(),
            'graphics_processes': "",
            'clock_speeds': "clocks.current.graphics, clocks.current.sm, clocks.current.memory, clocks.current.video\n1920, 1920, 1313, 1710",
            'topology': "GPU0\tX\nCPU Affinity\tNUMA Node(s)\t0-15\nGPU0\t X"
        }
    
    @staticmethod
    def with_memory_usage(memory_used_mb: int) -> Dict[str, Any]:
        """Create GPU data with specific memory usage."""
        config = MockScenarioConfig(gpu_memory_used_mb=memory_used_mb)
        return {
            'basic_metrics': MockNvidiaGPUData.create_basic_metrics(config),
            'detailed_xml': MockNvidiaGPUData.create_detailed_xml(config),
            'compute_processes': MockNvidiaGPUData.create_gpu_processes(),
            'graphics_processes': "",
            'clock_speeds': "clocks.current.graphics, clocks.current.sm, clocks.current.memory, clocks.current.video\n1920, 1920, 1313, 1710"
        }
    
    @staticmethod
    def with_processes(pids: List[int], process_names: List[str] = None) -> Dict[str, Any]:
        """Create GPU data with specific processes."""
        config = MockScenarioConfig()
        return {
            'basic_metrics': MockNvidiaGPUData.create_basic_metrics(config),
            'detailed_xml': MockNvidiaGPUData.create_detailed_xml(config),
            'compute_processes': MockNvidiaGPUData.create_gpu_processes(pids, process_names),
            'graphics_processes': "",
            'clock_speeds': "clocks.current.graphics, clocks.current.sm, clocks.current.memory, clocks.current.video\n1920, 1920, 1313, 1710"
        }


class MockSnapshotFactory:
    """
    Factory for creating comprehensive SystemCollector snapshots.
    
    Generates realistic mock data that matches the structure and content
    of actual SystemCollector output for thorough testing.
    """
    
    @staticmethod
    def create_baseline_snapshot(config: MockScenarioConfig = None) -> Dict[str, Any]:
        """Create a baseline system snapshot with realistic data."""
        if config is None:
            config = MockScenarioConfig()
        
        timestamp = datetime.now()
        
        return {
            'metadata': {
                'collector': 'system',
                'timestamp': timestamp.isoformat(),
                'collection_count': 1,
                'data_hash': 'abc123def456',
                'collection_duration_ms': 1250
            },
            'data': {
                'cpu': MockSnapshotFactory._create_cpu_data(config),
                'memory': MockSnapshotFactory._create_memory_data(config),
                'nvidia_gpu': MockSnapshotFactory._create_nvidia_gpu_data(config),
                'cuda': MockSnapshotFactory._create_cuda_data(),
                'storage': MockSnapshotFactory._create_storage_data(),
                'security': MockSnapshotFactory._create_security_data(),
                'python_env': MockSnapshotFactory._create_python_env_data(config),
                'hardware': MockSnapshotFactory._create_hardware_data(),
                'kernel': MockSnapshotFactory._create_kernel_data(),
                'network': MockSnapshotFactory._create_network_data(),
                'processes': MockSnapshotFactory._create_processes_data(config),
                'performance': MockSnapshotFactory._create_performance_data(),
                'docker_nvidia': MockSnapshotFactory._create_docker_nvidia_data()
            }
        }
    
    @staticmethod
    def create_thermal_event_scenario() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create before/after snapshots for GPU thermal event."""
        base_config = MockScenarioConfig(gpu_temp_celsius=75.0)
        thermal_config = MockScenarioConfig(gpu_temp_celsius=84.0)  # Above warning threshold
        
        before_snapshot = MockSnapshotFactory.create_baseline_snapshot(base_config)
        after_snapshot = MockSnapshotFactory.create_baseline_snapshot(thermal_config)
        
        # Update timestamps
        before_time = datetime.now() - timedelta(minutes=1)
        after_time = datetime.now()
        
        before_snapshot['metadata']['timestamp'] = before_time.isoformat()
        after_snapshot['metadata']['timestamp'] = after_time.isoformat()
        
        return before_snapshot, after_snapshot
    
    @staticmethod
    def create_process_spawn_scenario() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create before/after snapshots for process spawning."""
        base_config = MockScenarioConfig()
        
        before_snapshot = MockSnapshotFactory.create_baseline_snapshot(base_config)
        after_snapshot = deepcopy(before_snapshot)
        
        # Add new GPU process in after snapshot
        after_snapshot['data']['nvidia_gpu'] = MockNvidiaGPUData.with_processes(
            [12345, 23456, 34567],  # Added process 34567
            ["python", "python", "pytorch_training"]
        )
        
        # Update timestamps
        after_time = datetime.now()
        before_time = after_time - timedelta(seconds=30)
        
        before_snapshot['metadata']['timestamp'] = before_time.isoformat()
        after_snapshot['metadata']['timestamp'] = after_time.isoformat()
        
        return before_snapshot, after_snapshot
    
    @staticmethod
    def create_memory_pressure_scenario() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create before/after snapshots for GPU memory pressure."""
        normal_config = MockScenarioConfig(gpu_memory_used_mb=8192)  # 8GB used
        pressure_config = MockScenarioConfig(gpu_memory_used_mb=31000)  # 31GB used (high pressure)
        
        before_snapshot = MockSnapshotFactory.create_baseline_snapshot(normal_config)
        after_snapshot = MockSnapshotFactory.create_baseline_snapshot(pressure_config)
        
        # Update timestamps
        before_time = datetime.now() - timedelta(minutes=2)
        after_time = datetime.now()
        
        before_snapshot['metadata']['timestamp'] = before_time.isoformat()
        after_snapshot['metadata']['timestamp'] = after_time.isoformat()
        
        return before_snapshot, after_snapshot
    
    @staticmethod
    def create_python_package_installation_scenario() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create before/after snapshots for Python package installation."""
        base_config = MockScenarioConfig()
        
        before_snapshot = MockSnapshotFactory.create_baseline_snapshot(base_config)
        after_snapshot = deepcopy(before_snapshot)
        
        # Add new package to after snapshot
        before_packages = before_snapshot['data']['python_env']['packages']['pip3_list']
        after_packages = before_packages + "\ntensorflow-gpu==2.14.0"
        after_snapshot['data']['python_env']['packages']['pip3_list'] = after_packages
        
        # Update timestamps
        before_time = datetime.now() - timedelta(minutes=5)
        after_time = datetime.now()
        
        before_snapshot['metadata']['timestamp'] = before_time.isoformat()
        after_snapshot['metadata']['timestamp'] = after_time.isoformat()
        
        return before_snapshot, after_snapshot
    
    # Private helper methods for creating specific data sections
    
    @staticmethod
    def _create_cpu_data(config: MockScenarioConfig) -> Dict[str, Any]:
        """Create mock CPU data."""
        return {
            'cpuinfo': {
                'processors': [
                    {
                        'processor': '0',
                        'vendor_id': 'AuthenticAMD',
                        'cpu_family': '25',
                        'model': '116',
                        'model_name': 'AMD Ryzen 9 9950X 16-Core Processor',
                        'stepping': '1',
                        'microcode': '0xa601206',
                        'cpu_MHz': '4200.000',
                        'cache_size': '1024 KB'
                    }
                ],
                'count': 32  # 16 cores, 32 threads
            },
            'usage': {
                'cpu': {
                    'user': int(config.cpu_usage_percent * 100),
                    'nice': 50,
                    'system': int(config.cpu_usage_percent * 50),
                    'idle': int((100 - config.cpu_usage_percent) * 100),
                    'iowait': 100,
                    'irq': 10,
                    'softirq': 20
                }
            },
            'frequency': {'current': {'cpu0': '4200000', 'cpu1': '4200000'}},
            'thermal': {'thermal_zone0': '45000', 'thermal_zone1': '42000'},
            'numa': {
                'topology': 'available: 2 nodes (0-1)\nnode 0 cpus: 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23\nnode 0 size: 65536 MB\nnode 0 free: 32768 MB',
                'hardware': 'available: 2 nodes (0-1)\nnode distances:\nnode   0   1\n  0:  10  32\n  1:  32  10'
            },
            'loadavg': ['2.15', '2.08', '1.95'],
            'lscpu': 'Architecture:        x86_64\nCPU op-mode(s):      32-bit, 64-bit\nByte Order:          Little Endian\nCPU(s):              32'
        }
    
    @staticmethod
    def _create_memory_data(config: MockScenarioConfig) -> Dict[str, Any]:
        """Create mock memory data."""
        total_memory_kb = 128 * 1024 * 1024  # 128GB in KB
        used_memory_kb = int(total_memory_kb * (config.system_memory_used_percent / 100))
        free_memory_kb = total_memory_kb - used_memory_kb
        
        return {
            'meminfo': {
                'MemTotal': f'{total_memory_kb} kB',
                'MemFree': f'{free_memory_kb} kB',
                'MemUsed': f'{used_memory_kb} kB',
                'MemAvailable': f'{free_memory_kb + 8192000} kB',  # Include buffers/cache
                'Buffers': '2048000 kB',
                'Cached': '6144000 kB',
                'SwapTotal': '33554432 kB',  # 32GB swap
                'SwapFree': '33554432 kB',
                'SwapUsed': '0 kB'
            },
            'vmstat': {
                'pgpgin': '1234567',
                'pgpgout': '7654321',
                'pswpin': '0',
                'pswpout': '0',
                'pgfault': '9876543',
                'pgmajfault': '12345'
            },
            'swaps': 'Filename\t\t\t\tType\t\tSize\tUsed\tPriority\n/dev/nvme0n1p3                  partition\t33554432\t0\t-2',
            'numa_maps': {},
            'free': '               total        used        free      shared  buff/cache   available\nMem:       131072000    65536000    32768000     1024000    32768000    98304000\nSwap:       33554432           0    33554432'
        }
    
    @staticmethod
    def _create_nvidia_gpu_data(config: MockScenarioConfig) -> Dict[str, Any]:
        """Create mock NVIDIA GPU data."""
        return {
            'basic_metrics': MockNvidiaGPUData.create_basic_metrics(config),
            'detailed_xml': MockNvidiaGPUData.create_detailed_xml(config),
            'compute_processes': MockNvidiaGPUData.create_gpu_processes(),
            'graphics_processes': "",
            'clock_speeds': "clocks.current.graphics, clocks.current.sm, clocks.current.memory, clocks.current.video\n1920, 1920, 1313, 1710",
            'topology': "GPU0\tX\nCPU Affinity\tNUMA Node(s)\t0-15\nGPU0\t X",
            'utilization_samples': f"utilization.gpu [%], utilization.memory [%], memory.used [MiB], temperature.gpu, power.draw [W]\n{config.gpu_utilization_percent}, 85, {config.gpu_memory_used_mb}, {config.gpu_temp_celsius}, {config.gpu_power_watts}"
        }
    
    @staticmethod
    def _create_cuda_data() -> Dict[str, Any]:
        """Create mock CUDA environment data."""
        return {
            'nvcc_version': 'Cuda compilation tools, release 12.2, V12.2.140\nBuilt on Tue_Aug_15_22:02:13_PDT_2023',
            'runtime_version': 'driver_version, cuda_version\n535.98, 12.2',
            'compute_capability': 'compute_cap\n8.9',
            'environment': {
                'cuda_home': '12.2',
                'library_paths': {
                    '/usr/local/cuda/lib64': ['libcudart.so.12', 'libcublas.so.12', 'libcurand.so.10'],
                    '/usr/lib/x86_64-linux-gnu': ['libcuda.so.1', 'libnvidia-ml.so.1']
                },
                'environment_vars': {
                    'CUDA_HOME': '/usr/local/cuda',
                    'CUDA_ROOT': '/usr/local/cuda',
                    'LD_LIBRARY_PATH': '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu',
                    'PATH': '/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
                }
            }
        }
    
    @staticmethod
    def _create_storage_data() -> Dict[str, Any]:
        """Create mock storage data."""
        return {
            'hardware': {
                'block_devices_detailed': 'NAME    SIZE TYPE FSTYPE MOUNTPOINT                   UUID                                 MODEL          SERIAL       ROTA DISC-GRAN DISC-MAX WSAME\nnvme0n1 2T   disk                                                               Samsung SSD 990 PRO 2TB S7CKN123456      0        0B       2T     0B',
                'nvme_devices': 'Node             SN                   Model                                    Namespace Usage                      Format           FW Rev  \n/dev/nvme0n1     S7CKN123456         Samsung SSD 990 PRO 2TB                 1          2.00  TB /   2.00  TB    512   B +  0 B   4B1Q',
                'scsi_devices': '',
                'device_uuids': '/dev/nvme0n1p1: UUID="1234-5678" TYPE="vfat" PARTUUID="abcd-efgh"\n/dev/nvme0n1p2: UUID="12345678-1234-1234-1234-123456789abc" TYPE="ext4" PARTUUID="efgh-ijkl"',
                'partition_tables': 'Model: Samsung SSD 990 PRO 2TB (nvme)\nDisk /dev/nvme0n1: 2000GB\nSector size (logical/physical): 512B/512B\nPartition Table: gpt'
            },
            'performance': {
                'iostat_extended': 'Device            r/s     w/s     rkB/s     wkB/s   rrqm/s   wrqm/s  %rrqm  %wrqm     await r_await w_await  svctm  %util\nnvme0n1         45.67   23.45   1234.56   987.65     0.12     5.67   0.26  19.45     2.34    1.89    3.45   0.98   5.67',
                'diskstats': '   8       0 nvme0n1 123456 1234 9876543 45678 23456 2345 876543 12345 0 5678 58023',
                'io_schedulers': {'nvme0n1': '[none] mq-deadline kyber bfq'},
                'nvme_performance': {'/dev/nvme0n1_id': 'NVME Identify Controller:\nvid       : 0x144d\nssvid     : 0x144d'},
                'storage_temperatures': {'nvme0n1': 'Temperature:                        45 Celsius'},
                'io_pressure': 'some avg10=0.12 avg60=0.08 avg300=0.05 total=1234567\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0'
            },
            'filesystem': {
                'inode_usage': 'Filesystem      Inodes   IUsed   IFree IUse% Mounted on\n/dev/nvme0n1p2 30965760 1234567 29731193    4% /',
                'mount_details': 'TARGET SOURCE    FSTYPE OPTIONS\n/      /dev/nvme0n1p2 ext4   rw,relatime,errors=remount-ro',
                'filesystem_info': {},
                'directory_usage': {
                    '/': '85G',
                    '/home': '250G',
                    '/tmp': '2.5G',
                    '/var/lib/docker': '15G'
                }
            },
            'health': {
                'smart_data': {
                    'nvme0n1': 'smartctl 7.3 2022-02-28 r5338 [x86_64-linux-6.2.0-39-generic] (local build)\n=== START OF SMART DATA SECTION ===\nSMART overall-health self-assessment test result: PASSED\nSMART/Health Information (NVMe Log 0x02)\nCritical Warning:                   0x00\nTemperature:                        45 Celsius\nAvailable Spare:                    100%\nAvailable Spare Threshold:          10%\nPercentage Used:                    2%\nData Units Read:                    12,345,678 [6.31 TB]\nData Units Written:                 8,765,432 [4.48 TB]'
                }
            }
        }
    
    @staticmethod
    def _create_security_data() -> Dict[str, Any]:
        """Create mock security data."""
        return {
            'users': {
                'current_users': 'alejandro   pts/0        2023-11-15 10:30 (192.168.1.100)',
                'login_history': 'alejandro   pts/0        192.168.1.100    Wed Nov 15 10:30 - 11:45  (01:15)',
                'last_logins': 'alejandro            pts/0     192.168.1.100    Wed Nov 15 10:30:00 +0000 2023',
                'user_accounts': 'root:x:0:0:root:/root:/bin/bash\nalejandro:x:1000:1000:Alejandro:/home/alejandro:/bin/bash',
                'groups': 'root:x:0:\nalejandro:x:1000:\nsudo:x:27:alejandro',
                'current_user_id': 'uid=1000(alejandro) gid=1000(alejandro) groups=1000(alejandro),27(sudo),44(video)'
            },
            'network': {
                'active_connections': 'LISTEN      0       128             0.0.0.0:22             0.0.0.0:*       users:(("sshd",pid=1234,fd=3))',
                'connection_stats': 'Total: 123\nTCP: 45\nUDP: 78',
                'listening_ports': 'tcp   LISTEN 0      128        0.0.0.0:22       0.0.0.0:*'
            },
            'firewall': {
                'ufw_status': 'Status: active\n\nTo                         Action      From\n--                         ------      ----\n22/tcp                     ALLOW       Anywhere'
            }
        }
    
    @staticmethod
    def _create_python_env_data(config: MockScenarioConfig) -> Dict[str, Any]:
        """Create mock Python environment data."""
        return {
            'installations': {
                'python3': {
                    'path': '/usr/bin/python3',
                    'version': 'Python 3.11.6',
                    'sys_info': 'Path: /usr/bin/python3\nVersion: 3.11.6 (main, Oct  8 2023, 05:06:43) [GCC 13.2.0] on linux\nPlatform: linux'
                }
            },
            'packages': {
                'pip3_list': 'Package         Version\n--------------- ---------\nnumpy           1.24.3\npandas          2.0.3\ntorch           2.1.0+cu121\ntorchvision     0.16.0+cu121\ntransformers    4.35.0\naccelerate      0.24.1',
                'pip3_outdated': 'Package     Version Latest Type\n----------- ------- ------ -----\nnumpy       1.24.3  1.25.2 wheel'
            },
            'virtual_envs': {
                'virtual_environments': {
                    '/home/alejandro/venv/ml': 'torch==2.1.0+cu121\ntorchvision==0.16.0+cu121\ntransformers==4.35.0'
                }
            },
            'conda': {
                'conda_info': 'conda version : 23.7.4\nconda-build version : 3.26.1\npython version : 3.11.5.final.0',
                'environments': 'base                     /opt/conda\nml-env                   /opt/conda/envs/ml-env'
            },
            'system_packages': {
                'dpkg_python': 'ii  python3           3.11.2-1+b1     amd64        interactive high-level object-oriented language',
                'site_packages': "['/usr/lib/python311.zip', '/usr/lib/python3.11', '/usr/local/lib/python3.11/dist-packages']"
            },
            'ai_frameworks': {
                'torch': '2.1.0+cu121',
                'tensorflow': 'Not installed',
                'transformers': '4.35.0',
                'accelerate': '0.24.1',
                'numpy': '1.24.3',
                'pandas': '2.0.3',
                'torch_cuda': 'CUDA available: True\nCUDA devices: 1'
            }
        }
    
    @staticmethod
    def _create_hardware_data() -> Dict[str, Any]:
        """Create mock hardware data."""
        return {
            'pci_devices': '00:00.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex\n01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 5090] (rev a1)',
            'pci_tree': '-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex\n           +-01.0-[01]----00.0  NVIDIA Corporation AD102 [GeForce RTX 5090]',
            'usb_devices': 'Bus 001 Device 002: ID 8087:0024 Intel Corp. Integrated Rate Matching Hub\nBus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub',
            'sensors': 'k10temp-pci-00c3\nAdapter: PCI adapter\nTctl:         +45.0°C\n\nnvidia-gpu-0\nAdapter: NVIDIA adapter\ntemp1:        +75.0°C',
            'system_info': 'Handle 0x0001, DMI type 1, 27 bytes\nSystem Information\n\tManufacturer: ASUS\n\tProduct Name: Custom Build\n\tVersion: 1.0\n\tSerial Number: 123456789'
        }
    
    @staticmethod
    def _create_kernel_data() -> Dict[str, Any]:
        """Create mock kernel data."""
        return {
            'version': 'Linux workstation 6.2.0-39-generic #40~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov  2 18:01:13 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux',
            'uptime': '123456.78 987654.32',
            'cmdline': 'BOOT_IMAGE=/vmlinuz-6.2.0-39-generic root=UUID=12345678-1234-1234-1234-123456789abc ro quiet splash',
            'nvidia_modules': 'nvidia_drm             69632  8\nnvidia_modeset        1142784  10 nvidia_drm\nnvidia              39387136  545 nvidia_modeset\ndrm_kms_helper        311296  1 nvidia_drm'
        }
    
    @staticmethod
    def _create_network_data() -> Dict[str, Any]:
        """Create mock network data."""
        return {
            'interfaces': '1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000\n    inet 127.0.0.1/8 scope host lo\n2: enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000\n    inet 192.168.1.100/24 brd 192.168.1.255 scope global dynamic enp5s0',
            'routing': 'default via 192.168.1.1 dev enp5s0 proto dhcp metric 100\n192.168.1.0/24 dev enp5s0 proto kernel scope link src 192.168.1.100 metric 100',
            'connections': 'tcp   LISTEN 0      128        0.0.0.0:22       0.0.0.0:*\ntcp   LISTEN 0      5      127.0.0.1:631      0.0.0.0:*',
            'statistics': 'Inter-|   Receive                                                |  Transmit\n face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n    lo:  123456      789    0    0    0     0          0         0   123456      789    0    0    0     0       0          0\nenp5s0: 9876543210 87654321    0    0    0     0          0    123456 1234567890 12345678    0    0    0     0       0          0'
        }
    
    @staticmethod
    def _create_processes_data(config: MockScenarioConfig) -> Dict[str, Any]:
        """Create mock process data."""
        return {
            'ps_cpu': f'USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nroot           1  0.0  0.1 168348 13984 ?        Ss   Nov14   0:02 /sbin/init\nalejandro   1234 15.2  2.5 1234567 89012 pts/0    S+   10:30   1:23 python train.py\nalejandro   2345  5.1  1.2  654321 34567 ?        S    10:25   0:45 /usr/bin/python3',
            'ps_memory': f'USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nalejandro   1234 15.2  2.5 1234567 89012 pts/0    S+   10:30   1:23 python train.py\nalejandro   2345  5.1  1.2  654321 34567 ?        S    10:25   0:45 /usr/bin/python3\nroot           1  0.0  0.1 168348 13984 ?        Ss   Nov14   0:02 /sbin/init',
            'process_tree': f'    1 ?        Ss     0:02 /sbin/init\n 1234 pts/0    S+     1:23  \\_ python train.py\n 2345 ?        S      0:45  \\_ /usr/bin/python3'
        }
    
    @staticmethod
    def _create_performance_data() -> Dict[str, Any]:
        """Create mock performance data."""
        return {
            'iostat': 'Linux 6.2.0-39-generic (workstation) \t11/15/2023 \t_x86_64_\t(32 CPU)\n\nDevice            tps    kB_read/s    kB_wrtn/s    kB_dscd/s    kB_read    kB_wrtn    kB_dscd\nnvme0n1         45.67      1234.56       987.65         0.00   12345678    9876543          0',
            'vmstat': 'procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st\n 2  0      0 32768000 2048000 6144000    0    0   123   98 1234 5678 25  5 69  1  0'
        }
    
    @staticmethod
    def _create_docker_nvidia_data() -> Dict[str, Any]:
        """Create mock Docker NVIDIA data."""
        return {
            'docker_info': 'Client:\n Context:    default\n Debug Mode: false\n\nServer:\n Containers: 5\n  Running: 2\n  Paused: 0\n  Stopped: 3\n Images: 15\n Server Version: 24.0.7\n Storage Driver: overlay2\n  Backing Filesystem: extfs\n Logging Driver: json-file\n Cgroup Driver: systemd\n Cgroup Version: 2\n Plugins:\n  Volume: local\n  Network: bridge host ipvlan macvlan null overlay\n  Log: awslogs fluentd gcplogs gelf journald json-file local logentries splunk syslog\n Swarm: inactive\n Runtimes: io.containerd.runc.v2 nvidia runc\n Default Runtime: runc\n Init Binary: docker-init\n containerd version: 061aa650c09e0bf6e6c9823e36a9a59682c99ad7\n runc version: v1.1.9-0-gccaecfc\n init version: de40ad0\n Security Options:\n  apparmor\n  seccomp\n   Profile: builtin\n  cgroupns\n Kernel Version: 6.2.0-39-generic\n Operating System: Ubuntu 22.04.3 LTS\n OSType: linux\n Architecture: x86_64\n CPUs: 32\n Total Memory: 125GiB\n Name: workstation\n ID: 1234:5678:9abc:def0:1111:2222:3333:4444\n Docker Root Dir: /var/lib/docker\n Debug Mode: false\n Registry: https://index.docker.io/v1/\n Labels:\n Experimental: false\n Insecure Registries:\n  127.0.0.0/8\n Live Restore Enabled: false',
            'nvidia_ctk_version': 'NVIDIA Container Toolkit version 1.14.3',
            'gpu_containers': 'CONTAINER ID   IMAGE                    NAMES               STATUS\nabc123def456   pytorch/pytorch:latest   ml-training         Up 2 hours',
            'compose_services': '[]'  # Empty JSON array
        }