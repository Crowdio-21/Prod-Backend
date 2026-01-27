"""
Energy Tracker Module

Tracks and simulates energy consumption for edge computing workers.
Supports both real battery monitoring and simulated energy models.

Energy Models:
- Linear model: E = base + (cpu_factor * cpu_usage) + (network_factor * bytes_transferred)
- Battery drain model for mobile devices
- Plugged-in vs battery mode considerations
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json


class PowerState(Enum):
    """Power state of a device"""
    PLUGGED_IN = "plugged_in"
    BATTERY = "battery"
    LOW_BATTERY = "low_battery"  # Below threshold
    CRITICAL_BATTERY = "critical_battery"  # Below critical threshold
    UNKNOWN = "unknown"


class EnergyMode(Enum):
    """Energy consumption mode for simulation"""
    IDLE = "idle"
    LIGHT_WORK = "light_work"
    MODERATE_WORK = "moderate_work"
    HEAVY_WORK = "heavy_work"


@dataclass
class EnergyProfile:
    """
    Energy consumption profile for a device type.
    All power values in milliwatts (mW).
    """
    device_type: str
    
    # Base power consumption (mW)
    idle_power: float = 500.0  # When idle
    cpu_power_per_percent: float = 50.0  # Additional power per % CPU usage
    network_power_active: float = 300.0  # When network is active
    network_power_per_kb: float = 0.5  # Per KB transferred
    gpu_power_per_percent: float = 100.0  # Per % GPU usage (if applicable)
    display_power: float = 200.0  # Display power (mobile)
    
    # Battery capacity (mWh)
    battery_capacity: float = 50000.0  # Default 50Wh
    
    # Thresholds (percentage)
    low_battery_threshold: float = 20.0
    critical_battery_threshold: float = 10.0
    
    # Task energy estimates (mWh per task)
    avg_task_energy: float = 5.0
    
    @classmethod
    def mobile_device(cls) -> "EnergyProfile":
        """Profile for mobile/Android device"""
        return cls(
            device_type="mobile",
            idle_power=200.0,
            cpu_power_per_percent=30.0,
            network_power_active=400.0,
            network_power_per_kb=0.8,
            battery_capacity=15000.0,  # ~15Wh typical smartphone
            display_power=500.0,
            avg_task_energy=2.0,
        )
    
    @classmethod
    def laptop_device(cls) -> "EnergyProfile":
        """Profile for laptop device"""
        return cls(
            device_type="laptop",
            idle_power=3000.0,
            cpu_power_per_percent=100.0,
            network_power_active=200.0,
            network_power_per_kb=0.3,
            battery_capacity=60000.0,  # ~60Wh typical laptop
            display_power=2000.0,
            avg_task_energy=10.0,
        )
    
    @classmethod
    def desktop_device(cls) -> "EnergyProfile":
        """Profile for desktop device (always plugged in)"""
        return cls(
            device_type="desktop",
            idle_power=50000.0,  # ~50W idle
            cpu_power_per_percent=500.0,
            network_power_active=100.0,
            network_power_per_kb=0.1,
            battery_capacity=float('inf'),  # Always plugged in
            avg_task_energy=20.0,
        )
    
    @classmethod
    def raspberry_pi(cls) -> "EnergyProfile":
        """Profile for Raspberry Pi or similar edge device"""
        return cls(
            device_type="edge_device",
            idle_power=2000.0,  # ~2W
            cpu_power_per_percent=20.0,
            network_power_active=100.0,
            network_power_per_kb=0.2,
            battery_capacity=20000.0,  # If battery powered
            avg_task_energy=3.0,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceEnergyState:
    """Current energy state of a device"""
    worker_id: str
    profile: EnergyProfile
    
    # Current state
    power_state: PowerState = PowerState.UNKNOWN
    battery_level: float = 100.0  # Percentage 0-100
    is_charging: bool = False
    
    # Consumption tracking
    total_energy_consumed: float = 0.0  # mWh consumed since tracking started
    energy_consumed_current_task: float = 0.0  # mWh for current task
    
    # Task tracking
    tasks_completed: int = 0
    total_task_energy: float = 0.0  # Total energy spent on tasks
    
    # Timestamps
    tracking_started: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Real-time metrics
    current_power_draw: float = 0.0  # Current instantaneous power (mW)
    avg_power_draw: float = 0.0  # Average power draw (mW)
    
    def update_power_state(self) -> None:
        """Update power state based on battery level and charging status"""
        if self.is_charging:
            self.power_state = PowerState.PLUGGED_IN
        elif self.battery_level <= self.profile.critical_battery_threshold:
            self.power_state = PowerState.CRITICAL_BATTERY
        elif self.battery_level <= self.profile.low_battery_threshold:
            self.power_state = PowerState.LOW_BATTERY
        else:
            self.power_state = PowerState.BATTERY
    
    def can_accept_tasks(self, min_battery: float = 10.0) -> bool:
        """Check if device has enough energy to accept tasks"""
        if self.is_charging:
            return True
        return self.battery_level > min_battery
    
    def estimate_remaining_tasks(self) -> int:
        """Estimate how many more tasks can be completed"""
        if self.is_charging:
            return float('inf')
        
        remaining_energy = (self.battery_level / 100.0) * self.profile.battery_capacity
        reserved_energy = (self.profile.critical_battery_threshold / 100.0) * self.profile.battery_capacity
        available_energy = remaining_energy - reserved_energy
        
        if available_energy <= 0 or self.profile.avg_task_energy <= 0:
            return 0
        
        return int(available_energy / self.profile.avg_task_energy)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "device_type": self.profile.device_type,
            "power_state": self.power_state.value,
            "battery_level": self.battery_level,
            "is_charging": self.is_charging,
            "total_energy_consumed": self.total_energy_consumed,
            "tasks_completed": self.tasks_completed,
            "avg_task_energy": self.total_task_energy / self.tasks_completed if self.tasks_completed > 0 else 0,
            "current_power_draw": self.current_power_draw,
            "can_accept_tasks": self.can_accept_tasks(),
            "estimated_remaining_tasks": self.estimate_remaining_tasks(),
        }


class EnergyTracker:
    """
    Tracks energy consumption across all workers.
    
    Supports:
    - Real battery level tracking (from device reports)
    - Simulated energy consumption models
    - Energy-aware task scheduling recommendations
    - Energy efficiency analysis
    """
    
    def __init__(self, simulation_mode: bool = False):
        """
        Initialize energy tracker.
        
        Args:
            simulation_mode: If True, simulate energy consumption based on models
        """
        self.simulation_mode = simulation_mode
        
        # Device states: worker_id -> DeviceEnergyState
        self._device_states: Dict[str, DeviceEnergyState] = {}
        
        # Energy profiles: device_type -> EnergyProfile
        self._profiles: Dict[str, EnergyProfile] = {
            "mobile": EnergyProfile.mobile_device(),
            "laptop": EnergyProfile.laptop_device(),
            "desktop": EnergyProfile.desktop_device(),
            "edge_device": EnergyProfile.raspberry_pi(),
            "PC": EnergyProfile.desktop_device(),  # Default for PC workers
            "Android": EnergyProfile.mobile_device(),
        }
        
        # Energy consumption history for analysis
        self._consumption_history: List[Dict[str, Any]] = []
        
        # Lock for thread-safe access
        self._lock = asyncio.Lock()
        
        # Simulation parameters
        self._simulation_speed: float = 1.0  # Speed multiplier for simulation
    
    # ==================== Device Registration ====================
    
    async def register_device(
        self,
        worker_id: str,
        device_type: str = "PC",
        battery_level: float = 100.0,
        is_charging: bool = True,
        custom_profile: Optional[EnergyProfile] = None
    ) -> DeviceEnergyState:
        """
        Register a device for energy tracking.
        
        Args:
            worker_id: Unique worker identifier
            device_type: Type of device (mobile, laptop, desktop, edge_device)
            battery_level: Initial battery level (0-100)
            is_charging: Whether device is currently charging
            custom_profile: Optional custom energy profile
        
        Returns:
            DeviceEnergyState for the registered device
        """
        async with self._lock:
            profile = custom_profile or self._profiles.get(device_type, EnergyProfile.desktop_device())
            
            state = DeviceEnergyState(
                worker_id=worker_id,
                profile=profile,
                battery_level=battery_level,
                is_charging=is_charging,
            )
            state.update_power_state()
            
            self._device_states[worker_id] = state
            print(f"EnergyTracker: Registered {device_type} device {worker_id}")
            
            return state
    
    async def unregister_device(self, worker_id: str) -> Optional[DeviceEnergyState]:
        """Unregister a device"""
        async with self._lock:
            return self._device_states.pop(worker_id, None)
    
    # ==================== State Updates ====================
    
    async def update_battery_status(
        self,
        worker_id: str,
        battery_level: float,
        is_charging: bool
    ) -> Optional[DeviceEnergyState]:
        """
        Update device battery status from real device report.
        
        Args:
            worker_id: Worker identifier
            battery_level: Current battery level (0-100)
            is_charging: Whether device is charging
        
        Returns:
            Updated DeviceEnergyState
        """
        async with self._lock:
            if worker_id not in self._device_states:
                return None
            
            state = self._device_states[worker_id]
            
            # Calculate energy consumed since last update
            old_level = state.battery_level
            if not is_charging and battery_level < old_level:
                consumed = ((old_level - battery_level) / 100.0) * state.profile.battery_capacity
                state.total_energy_consumed += consumed
            
            state.battery_level = battery_level
            state.is_charging = is_charging
            state.last_update = time.time()
            state.update_power_state()
            
            return state
    
    async def record_task_energy(
        self,
        worker_id: str,
        task_id: str,
        execution_time: float,
        cpu_usage: float = 50.0,
        bytes_transferred: int = 0
    ) -> float:
        """
        Record energy consumption for a completed task.
        
        Args:
            worker_id: Worker that executed the task
            task_id: Task identifier
            execution_time: Execution time in seconds
            cpu_usage: Average CPU usage during task (0-100)
            bytes_transferred: Total bytes sent/received
        
        Returns:
            Estimated energy consumed (mWh)
        """
        async with self._lock:
            if worker_id not in self._device_states:
                return 0.0
            
            state = self._device_states[worker_id]
            profile = state.profile
            
            # Calculate energy consumption
            # Power = idle + cpu_contribution + network_contribution
            hours = execution_time / 3600.0
            
            power_draw = (
                profile.idle_power +
                (profile.cpu_power_per_percent * cpu_usage) +
                (profile.network_power_active if bytes_transferred > 0 else 0) +
                (profile.network_power_per_kb * bytes_transferred / 1024.0)
            )
            
            energy_consumed = power_draw * hours  # mWh
            
            # Update state
            state.total_energy_consumed += energy_consumed
            state.total_task_energy += energy_consumed
            state.tasks_completed += 1
            state.current_power_draw = power_draw
            
            # Update battery level (if on battery)
            if not state.is_charging:
                battery_drain = (energy_consumed / profile.battery_capacity) * 100.0
                state.battery_level = max(0, state.battery_level - battery_drain)
                state.update_power_state()
            
            # Record in history
            self._consumption_history.append({
                "timestamp": time.time(),
                "worker_id": worker_id,
                "task_id": task_id,
                "energy_consumed": energy_consumed,
                "power_draw": power_draw,
                "execution_time": execution_time,
                "cpu_usage": cpu_usage,
                "bytes_transferred": bytes_transferred,
            })
            
            return energy_consumed
    
    # ==================== Simulation Mode ====================
    
    async def simulate_battery_drain(
        self,
        worker_id: str,
        duration_seconds: float,
        work_mode: EnergyMode = EnergyMode.MODERATE_WORK
    ) -> float:
        """
        Simulate battery drain over time.
        
        Args:
            worker_id: Worker identifier
            duration_seconds: Simulated time duration
            work_mode: Type of work being performed
        
        Returns:
            Energy consumed during simulation (mWh)
        """
        async with self._lock:
            if worker_id not in self._device_states:
                return 0.0
            
            state = self._device_states[worker_id]
            profile = state.profile
            
            # Determine power draw based on work mode
            mode_multipliers = {
                EnergyMode.IDLE: 1.0,
                EnergyMode.LIGHT_WORK: 1.5,
                EnergyMode.MODERATE_WORK: 2.5,
                EnergyMode.HEAVY_WORK: 4.0,
            }
            
            multiplier = mode_multipliers.get(work_mode, 2.0)
            power_draw = profile.idle_power * multiplier
            
            hours = duration_seconds / 3600.0
            energy_consumed = power_draw * hours
            
            state.total_energy_consumed += energy_consumed
            state.current_power_draw = power_draw
            
            if not state.is_charging:
                battery_drain = (energy_consumed / profile.battery_capacity) * 100.0
                state.battery_level = max(0, state.battery_level - battery_drain)
                state.update_power_state()
            
            return energy_consumed
    
    # ==================== Energy-Aware Scheduling ====================
    
    async def get_available_workers_by_energy(
        self,
        min_battery: float = 10.0,
        prefer_charging: bool = True
    ) -> List[str]:
        """
        Get list of workers that have sufficient energy for tasks.
        
        Args:
            min_battery: Minimum battery level required
            prefer_charging: Whether to prioritize charging devices
        
        Returns:
            List of worker IDs sorted by energy availability
        """
        async with self._lock:
            available = []
            
            for worker_id, state in self._device_states.items():
                if state.can_accept_tasks(min_battery):
                    score = 0
                    if state.is_charging:
                        score = 1000  # High priority for charging devices
                    else:
                        score = state.battery_level
                    available.append((worker_id, score))
            
            # Sort by score (higher is better)
            available.sort(key=lambda x: x[1], reverse=True)
            
            return [w[0] for w in available]
    
    async def recommend_task_assignment(
        self,
        worker_ids: List[str],
        task_complexity: float = 1.0
    ) -> Optional[str]:
        """
        Recommend best worker for a task based on energy considerations.
        
        Args:
            worker_ids: List of available worker IDs
            task_complexity: Relative task complexity (1.0 = average)
        
        Returns:
            Recommended worker ID or None if no suitable worker
        """
        async with self._lock:
            best_worker = None
            best_score = -1
            
            for worker_id in worker_ids:
                if worker_id not in self._device_states:
                    continue
                
                state = self._device_states[worker_id]
                
                # Skip if insufficient energy
                estimated_energy = state.profile.avg_task_energy * task_complexity
                if not state.is_charging:
                    remaining_capacity = (state.battery_level / 100.0) * state.profile.battery_capacity
                    if remaining_capacity < estimated_energy * 2:  # Need 2x buffer
                        continue
                
                # Score: prefer charging devices, then by battery level
                if state.is_charging:
                    score = 1000
                else:
                    score = state.battery_level
                
                if score > best_score:
                    best_score = score
                    best_worker = worker_id
            
            return best_worker
    
    # ==================== Metrics and Analysis ====================
    
    async def get_device_state(self, worker_id: str) -> Optional[DeviceEnergyState]:
        """Get current energy state for a device"""
        async with self._lock:
            return self._device_states.get(worker_id)
    
    async def get_all_device_states(self) -> Dict[str, DeviceEnergyState]:
        """Get all device states"""
        async with self._lock:
            return dict(self._device_states)
    
    async def get_energy_summary(self) -> Dict[str, Any]:
        """Get summary of energy consumption across all devices"""
        async with self._lock:
            if not self._device_states:
                return {"total_devices": 0}
            
            total_energy = sum(s.total_energy_consumed for s in self._device_states.values())
            total_tasks = sum(s.tasks_completed for s in self._device_states.values())
            
            charging_count = sum(1 for s in self._device_states.values() if s.is_charging)
            low_battery_count = sum(
                1 for s in self._device_states.values()
                if s.power_state in [PowerState.LOW_BATTERY, PowerState.CRITICAL_BATTERY]
            )
            
            avg_battery = sum(s.battery_level for s in self._device_states.values()) / len(self._device_states)
            
            return {
                "total_devices": len(self._device_states),
                "devices_charging": charging_count,
                "devices_low_battery": low_battery_count,
                "average_battery_level": avg_battery,
                "total_energy_consumed_mwh": total_energy,
                "total_tasks_completed": total_tasks,
                "avg_energy_per_task_mwh": total_energy / total_tasks if total_tasks > 0 else 0,
                "device_states": {
                    worker_id: state.to_dict()
                    for worker_id, state in self._device_states.items()
                }
            }
    
    async def export_consumption_history(self, filepath: str) -> None:
        """Export energy consumption history to file"""
        async with self._lock:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "summary": await self.get_energy_summary(),
                "consumption_history": self._consumption_history,
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"EnergyTracker: Exported consumption history to {filepath}")
    
    async def reset(self) -> None:
        """Reset all energy tracking data"""
        async with self._lock:
            for state in self._device_states.values():
                state.total_energy_consumed = 0.0
                state.total_task_energy = 0.0
                state.tasks_completed = 0
                state.tracking_started = time.time()
            self._consumption_history.clear()
        print("EnergyTracker: Reset all tracking data")
