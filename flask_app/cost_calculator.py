"""
Cost Calculator and Energy Profiler
====================================

Calculates computational cost, energy consumption, and electricity costs
for image processing operations.

Features:
- GPU/CPU power profiling
- Energy consumption calculation (Watt-hours, Joules)
- Electricity cost estimation
- Memory power tracking
- Optimization recommendations
- Historical cost tracking
"""

import time
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnergyProfile:
    """Hardware energy consumption profile"""
    gpu_active_power: float = 250.0  # Watts (typical for NVIDIA RTX 3080)
    gpu_idle_power: float = 30.0     # Watts
    cpu_active_power: float = 65.0   # Watts (typical for modern CPU)
    cpu_idle_power: float = 15.0     # Watts
    memory_power: float = 3.0        # Watts per GB
    electricity_rate: float = 0.12   # USD per kWh (US average)
    compute_cost_per_hour: float = 0.50  # Additional compute cost (cloud)


@dataclass
class CostMetrics:
    """Cost and energy metrics for a single operation"""
    operation_id: str
    timestamp: str

    # Timing
    total_time_seconds: float
    gpu_time_seconds: float
    cpu_time_seconds: float

    # Memory
    memory_peak_gb: float
    memory_average_gb: float

    # Energy
    gpu_energy_wh: float
    cpu_energy_wh: float
    memory_energy_wh: float
    total_energy_wh: float
    total_energy_kwh: float
    total_energy_joules: float

    # Cost
    energy_cost_usd: float
    compute_cost_usd: float
    total_cost_usd: float

    # Breakdown
    cost_breakdown_percentage: Dict[str, float]
    energy_breakdown_percentage: Dict[str, float]

    # Metadata
    image_size_pixels: int
    cost_per_megapixel: float


class CostCalculator:
    """
    Calculate energy consumption and cost for image processing operations.

    Tracks GPU/CPU time, memory usage, and converts to energy consumption
    and monetary cost.
    """

    def __init__(self, profile: Optional[EnergyProfile] = None, storage_path: Optional[Path] = None):
        """
        Initialize cost calculator.

        Args:
            profile: Energy profile for hardware (uses defaults if None)
            storage_path: Path to store cost history
        """
        self.profile = profile or EnergyProfile()
        self.storage_path = storage_path or Path("data/cost_history.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Cost history
        self.history: List[CostMetrics] = []

        # Current operation tracking
        self._current_operation: Optional[Dict] = None

        # Load history
        self._load_history()

    def start_operation(self, operation_id: str):
        """Start tracking an operation"""
        self._current_operation = {
            'operation_id': operation_id,
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().used / (1024**3),  # GB
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'memory_samples': []
        }
        logger.debug(f"Started tracking operation: {operation_id}")

    def record_gpu_time(self, seconds: float):
        """Record GPU processing time"""
        if self._current_operation:
            self._current_operation['gpu_time'] += seconds

    def record_cpu_time(self, seconds: float):
        """Record CPU processing time"""
        if self._current_operation:
            self._current_operation['cpu_time'] += seconds

    def sample_memory(self):
        """Sample current memory usage"""
        if self._current_operation:
            memory_gb = psutil.virtual_memory().used / (1024**3)
            self._current_operation['memory_samples'].append(memory_gb)

    def end_operation(self, image_size_pixels: int) -> CostMetrics:
        """
        End tracking and calculate costs.

        Args:
            image_size_pixels: Number of pixels processed

        Returns:
            CostMetrics with detailed cost breakdown
        """
        if not self._current_operation:
            raise RuntimeError("No operation in progress")

        # Calculate timing
        total_time = time.time() - self._current_operation['start_time']
        gpu_time = self._current_operation['gpu_time']
        cpu_time = self._current_operation['cpu_time']

        # Calculate memory
        memory_samples = self._current_operation['memory_samples']
        if memory_samples:
            memory_peak = max(memory_samples)
            memory_average = sum(memory_samples) / len(memory_samples)
        else:
            current_memory = psutil.virtual_memory().used / (1024**3)
            memory_peak = current_memory
            memory_average = current_memory

        # Calculate costs
        cost_data = self.calculate_energy_cost(
            gpu_time_seconds=gpu_time,
            cpu_time_seconds=cpu_time,
            memory_gb=memory_average,
            total_time_seconds=total_time
        )

        # Create metrics
        metrics = CostMetrics(
            operation_id=self._current_operation['operation_id'],
            timestamp=datetime.now().isoformat(),
            total_time_seconds=total_time,
            gpu_time_seconds=gpu_time,
            cpu_time_seconds=cpu_time,
            memory_peak_gb=memory_peak,
            memory_average_gb=memory_average,
            gpu_energy_wh=cost_data['energy']['gpu_wh'],
            cpu_energy_wh=cost_data['energy']['cpu_wh'],
            memory_energy_wh=cost_data['energy']['memory_wh'],
            total_energy_wh=cost_data['energy']['total_wh'],
            total_energy_kwh=cost_data['energy']['total_kwh'],
            total_energy_joules=cost_data['energy']['total_joules'],
            energy_cost_usd=cost_data['cost']['energy_usd'],
            compute_cost_usd=cost_data['cost']['compute_usd'],
            total_cost_usd=cost_data['cost']['total_usd'],
            cost_breakdown_percentage=cost_data['cost']['breakdown_percentage'],
            energy_breakdown_percentage=cost_data['energy']['breakdown_percentage'],
            image_size_pixels=image_size_pixels,
            cost_per_megapixel=cost_data['cost']['total_usd'] / (image_size_pixels / 1_000_000) if image_size_pixels > 0 else 0.0
        )

        # Store in history
        self.history.append(metrics)

        # Auto-save
        if len(self.history) % 10 == 0:
            self._save_history()

        # Clear current operation
        self._current_operation = None

        logger.info(f"Operation {metrics.operation_id}: ${metrics.total_cost_usd:.4f}, {metrics.total_energy_wh:.2f}Wh")

        return metrics

    def calculate_energy_cost(
        self,
        gpu_time_seconds: float,
        cpu_time_seconds: float,
        memory_gb: float,
        total_time_seconds: Optional[float] = None
    ) -> Dict:
        """
        Calculate energy consumption and cost.

        Args:
            gpu_time_seconds: GPU processing time
            cpu_time_seconds: CPU processing time
            memory_gb: Average memory usage in GB
            total_time_seconds: Total elapsed time (for memory calculation)

        Returns:
            Dictionary with energy and cost breakdown
        """
        # Energy calculations (Watt-hours)
        gpu_energy_wh = self.profile.gpu_active_power * (gpu_time_seconds / 3600)
        cpu_energy_wh = self.profile.cpu_active_power * (cpu_time_seconds / 3600)

        # Memory energy (based on total time or max of GPU/CPU time)
        memory_time = total_time_seconds if total_time_seconds else max(gpu_time_seconds, cpu_time_seconds)
        memory_energy_wh = self.profile.memory_power * (memory_time / 3600) * memory_gb

        total_energy_wh = gpu_energy_wh + cpu_energy_wh + memory_energy_wh
        total_energy_kwh = total_energy_wh / 1000.0
        total_energy_joules = total_energy_wh * 3600  # 1 Wh = 3600 J

        # Cost calculations (USD)
        energy_cost = total_energy_kwh * self.profile.electricity_rate
        compute_cost = (gpu_time_seconds / 3600) * self.profile.compute_cost_per_hour
        total_cost = energy_cost + compute_cost

        # Breakdown percentages
        if total_energy_wh > 0:
            energy_breakdown = {
                'gpu': (gpu_energy_wh / total_energy_wh) * 100,
                'cpu': (cpu_energy_wh / total_energy_wh) * 100,
                'memory': (memory_energy_wh / total_energy_wh) * 100
            }
        else:
            energy_breakdown = {'gpu': 0, 'cpu': 0, 'memory': 0}

        if total_cost > 0:
            cost_breakdown = {
                'energy': (energy_cost / total_cost) * 100,
                'compute': (compute_cost / total_cost) * 100
            }
        else:
            cost_breakdown = {'energy': 0, 'compute': 0}

        return {
            'energy': {
                'gpu_wh': gpu_energy_wh,
                'cpu_wh': cpu_energy_wh,
                'memory_wh': memory_energy_wh,
                'total_wh': total_energy_wh,
                'total_kwh': total_energy_kwh,
                'total_joules': total_energy_joules,
                'breakdown_percentage': energy_breakdown
            },
            'cost': {
                'energy_usd': energy_cost,
                'compute_usd': compute_cost,
                'total_usd': total_cost,
                'breakdown_percentage': cost_breakdown
            }
        }

    def estimate_cost(
        self,
        image_size_pixels: int,
        processing_mode: str = 'balanced'
    ) -> Dict:
        """
        Estimate cost for an operation before executing.

        Args:
            image_size_pixels: Number of pixels to process
            processing_mode: 'eco', 'balanced', or 'max_quality'

        Returns:
            Estimated cost and time
        """
        # Estimate processing time based on mode and size
        megapixels = image_size_pixels / 1_000_000

        # Base time estimates (seconds per megapixel)
        time_estimates = {
            'eco': {
                'gpu_time_per_mp': 0.05,
                'cpu_time_per_mp': 0.1,
                'workers': 2
            },
            'balanced': {
                'gpu_time_per_mp': 0.15,
                'cpu_time_per_mp': 0.2,
                'workers': 4
            },
            'max_quality': {
                'gpu_time_per_mp': 0.3,
                'cpu_time_per_mp': 0.4,
                'workers': 5
            }
        }

        mode_config = time_estimates.get(processing_mode, time_estimates['balanced'])

        gpu_time = megapixels * mode_config['gpu_time_per_mp']
        cpu_time = megapixels * mode_config['cpu_time_per_mp']
        memory_gb = min(16.0, megapixels * 0.5)  # Estimate memory

        cost_data = self.calculate_energy_cost(
            gpu_time_seconds=gpu_time,
            cpu_time_seconds=cpu_time,
            memory_gb=memory_gb
        )

        return {
            'mode': processing_mode,
            'estimated_time_seconds': max(gpu_time, cpu_time),
            'estimated_cost_usd': cost_data['cost']['total_usd'],
            'estimated_energy_wh': cost_data['energy']['total_wh'],
            'workers': mode_config['workers'],
            'cost_breakdown': cost_data['cost'],
            'energy_breakdown': cost_data['energy']
        }

    def get_optimization_recommendations(self, metrics: CostMetrics) -> List[str]:
        """
        Generate optimization recommendations based on metrics.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check GPU utilization
        gpu_percentage = (metrics.gpu_energy_wh / metrics.total_energy_wh) * 100 if metrics.total_energy_wh > 0 else 0
        if gpu_percentage < 30:
            recommendations.append(
                f"Low GPU utilization ({gpu_percentage:.1f}%). Consider GPU-accelerated workers."
            )

        # Check cost efficiency
        cost_per_mp = metrics.cost_per_megapixel
        if cost_per_mp > 0.01:
            recommendations.append(
                f"High cost per megapixel (${cost_per_mp:.4f}). Consider 'Eco' mode or downsampling."
            )

        # Check memory usage
        if metrics.memory_peak_gb > 12.0:
            recommendations.append(
                f"High memory usage ({metrics.memory_peak_gb:.1f}GB). Consider batch processing smaller regions."
            )

        # Check processing time
        if metrics.total_time_seconds > 10.0:
            recommendations.append(
                f"Long processing time ({metrics.total_time_seconds:.1f}s). Consider preview mode for iteration."
            )

        # Energy efficiency
        if metrics.total_energy_wh > 50:
            recommendations.append(
                f"High energy consumption ({metrics.total_energy_wh:.1f}Wh). Consider reducing workers or using ROI selection."
            )

        return recommendations

    def get_statistics(self, last_n: Optional[int] = None) -> Dict:
        """
        Get aggregate statistics from history.

        Args:
            last_n: Only include last N operations (None = all)

        Returns:
            Statistics dictionary
        """
        if not self.history:
            return {
                'total_operations': 0,
                'total_cost_usd': 0.0,
                'total_energy_kwh': 0.0,
                'average_cost_usd': 0.0,
                'average_time_seconds': 0.0
            }

        history_slice = self.history[-last_n:] if last_n else self.history

        total_operations = len(history_slice)
        total_cost = sum(m.total_cost_usd for m in history_slice)
        total_energy = sum(m.total_energy_kwh for m in history_slice)
        total_time = sum(m.total_time_seconds for m in history_slice)

        return {
            'total_operations': total_operations,
            'total_cost_usd': total_cost,
            'total_energy_kwh': total_energy,
            'average_cost_usd': total_cost / total_operations,
            'average_energy_wh': (total_energy * 1000) / total_operations,
            'average_time_seconds': total_time / total_operations,
            'cost_per_hour': (total_cost / (total_time / 3600)) if total_time > 0 else 0.0,
            'images_per_dollar': total_operations / total_cost if total_cost > 0 else 0.0
        }

    def _save_history(self):
        """Save cost history to storage"""
        try:
            data = {
                'profile': asdict(self.profile),
                'history': [asdict(m) for m in self.history[-1000:]],  # Keep last 1000
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved cost history to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save cost history: {e}")

    def _load_history(self):
        """Load cost history from storage"""
        if not self.storage_path.exists():
            logger.info("No cost history found. Starting fresh.")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load profile if available
            if 'profile' in data:
                self.profile = EnergyProfile(**data['profile'])

            # Load history
            self.history = [CostMetrics(**m) for m in data.get('history', [])]

            logger.info(f"Loaded {len(self.history)} cost records from history")
        except Exception as e:
            logger.error(f"Failed to load cost history: {e}")
