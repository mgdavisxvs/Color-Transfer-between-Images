"""
Performance Tracking and Learning System for TSM
=================================================

This module implements the "Performance Review System" and "Probabilistic Weight Learning"
components of TSM. It tracks worker performance over time and learns which workers
excel at specific tasks, RAL colors, or image types.

Key Features:
- Worker performance history tracking
- RAL-specific performance learning
- Image-type specialization learning
- Dynamic weight adjustment
- Anomaly detection for unreliable workers
- Persistent storage of learned weights
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkerPerformanceRecord:
    """Individual performance record for a worker execution"""
    worker_id: str
    timestamp: str
    delta_e_mean: float
    delta_e_std: float
    delta_e_max: float
    delta_e_percentile_95: float
    processing_time: float
    target_ral_code: Optional[str]
    image_type: Optional[str]
    complexity: float
    success: bool


@dataclass
class WorkerStatistics:
    """Aggregated statistics for a worker"""
    worker_id: str
    total_executions: int
    successful_executions: int
    average_delta_e: float
    average_processing_time: float
    reliability_score: float  # 0-1 (based on success rate and consistency)
    current_weight: float
    specialties_performance: Dict[str, float]  # RAL ranges or image types
    recent_trend: str  # "improving", "stable", "declining"


class PerformanceTracker:
    """
    Tracks and learns from worker performance over time.

    This is the "Performance Review System" that maintains worker statistics
    and adjusts weights based on historical performance.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize performance tracker.

        Args:
            storage_path: Path to store performance data (JSON file)
        """
        self.storage_path = storage_path or Path("data/tsm_performance.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory performance records
        self.records: List[WorkerPerformanceRecord] = []

        # Worker statistics cache
        self.worker_stats: Dict[str, WorkerStatistics] = {}

        # Current weights for each worker
        self.worker_weights: Dict[str, float] = {}

        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'delta_e_max': 100.0,  # Extremely high delta E
            'processing_time_std': 3.0,  # Processing time > 3 std deviations
            'success_rate_min': 0.5  # Below 50% success rate
        }

        # Load existing data
        self._load_from_storage()

    def record_performance(
        self,
        worker_id: str,
        delta_e_stats: Dict[str, float],
        processing_time: float,
        target_ral_code: Optional[str] = None,
        image_type: Optional[str] = None,
        complexity: float = 0.5,
        success: bool = True
    ):
        """
        Record a worker's performance for a single execution.

        Args:
            worker_id: ID of the worker
            delta_e_stats: Dictionary with mean, std, max, percentile_95
            processing_time: Time taken in seconds
            target_ral_code: Target RAL color code
            image_type: Type of image (e.g., "textured", "flat_color")
            complexity: Image complexity score (0-1)
            success: Whether execution was successful
        """
        record = WorkerPerformanceRecord(
            worker_id=worker_id,
            timestamp=datetime.now().isoformat(),
            delta_e_mean=delta_e_stats.get('mean', 0.0),
            delta_e_std=delta_e_stats.get('std', 0.0),
            delta_e_max=delta_e_stats.get('max', 0.0),
            delta_e_percentile_95=delta_e_stats.get('percentile_95', 0.0),
            processing_time=processing_time,
            target_ral_code=target_ral_code,
            image_type=image_type,
            complexity=complexity,
            success=success
        )

        self.records.append(record)

        # Update worker statistics
        self._update_worker_statistics(worker_id)

        # Auto-save periodically (every 10 records)
        if len(self.records) % 10 == 0:
            self._save_to_storage()

        logger.info(f"Recorded performance for {worker_id}: ΔE={delta_e_stats.get('mean', 0):.2f}, time={processing_time:.2f}s")

    def get_worker_weights(self, context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get current weights for all workers, optionally contextualized.

        Args:
            context: Optional context (target_ral_code, image_type, complexity)

        Returns:
            Dictionary mapping worker_id to weight (0-1)
        """
        if not context:
            # Return base weights
            return self.worker_weights.copy()

        # Contextualized weights based on specialties
        contextualized_weights = {}

        for worker_id, base_weight in self.worker_weights.items():
            stats = self.worker_stats.get(worker_id)
            if not stats:
                contextualized_weights[worker_id] = base_weight
                continue

            # Adjust weight based on specialty performance
            specialty_bonus = 0.0

            # Check RAL code specialty
            target_ral = context.get('target_ral_code')
            if target_ral:
                ral_range = self._get_ral_range(target_ral)
                specialty_performance = stats.specialties_performance.get(ral_range, 0.0)
                if specialty_performance > 0:
                    specialty_bonus += 0.2 * specialty_performance

            # Check image type specialty
            image_type = context.get('image_type')
            if image_type:
                specialty_performance = stats.specialties_performance.get(image_type, 0.0)
                if specialty_performance > 0:
                    specialty_bonus += 0.2 * specialty_performance

            # Apply bonus (capped)
            contextualized_weight = min(1.0, base_weight + specialty_bonus)
            contextualized_weights[worker_id] = contextualized_weight

        # Normalize weights to sum to 1.0
        total_weight = sum(contextualized_weights.values())
        if total_weight > 0:
            contextualized_weights = {
                k: v / total_weight for k, v in contextualized_weights.items()
            }

        return contextualized_weights

    def detect_anomaly(self, worker_id: str, delta_e_stats: Dict, processing_time: float) -> Tuple[bool, str]:
        """
        Detect if a worker's performance is anomalous.

        Returns:
            (is_anomaly, reason)
        """
        # Check for extremely high delta E
        if delta_e_stats.get('max', 0) > self.anomaly_thresholds['delta_e_max']:
            return True, f"Extremely high ΔE max: {delta_e_stats['max']:.2f}"

        # Check processing time against historical average
        stats = self.worker_stats.get(worker_id)
        if stats and stats.total_executions > 5:
            # Calculate processing time statistics from recent records
            recent_records = [
                r for r in self.records[-100:]
                if r.worker_id == worker_id and r.success
            ]

            if len(recent_records) >= 5:
                recent_times = [r.processing_time for r in recent_records]
                mean_time = np.mean(recent_times)
                std_time = np.std(recent_times)

                if std_time > 0 and processing_time > mean_time + self.anomaly_thresholds['processing_time_std'] * std_time:
                    return True, f"Processing time too slow: {processing_time:.2f}s (avg: {mean_time:.2f}s)"

        # Check success rate
        if stats and stats.reliability_score < self.anomaly_thresholds['success_rate_min']:
            return True, f"Low reliability score: {stats.reliability_score:.2f}"

        return False, ""

    def get_worker_statistics(self, worker_id: str) -> Optional[WorkerStatistics]:
        """Get aggregated statistics for a specific worker"""
        return self.worker_stats.get(worker_id)

    def get_all_statistics(self) -> Dict[str, WorkerStatistics]:
        """Get statistics for all workers"""
        return self.worker_stats.copy()

    def get_best_workers(self, n: int = 3, context: Optional[Dict] = None) -> List[str]:
        """
        Get the top N best performing workers, optionally for a specific context.

        Args:
            n: Number of workers to return
            context: Optional context for contextualized selection

        Returns:
            List of worker IDs, ordered by performance
        """
        weights = self.get_worker_weights(context)

        # Sort by weight
        sorted_workers = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        return [worker_id for worker_id, _ in sorted_workers[:n]]

    def _update_worker_statistics(self, worker_id: str):
        """Update aggregated statistics for a worker"""
        # Get all records for this worker
        worker_records = [r for r in self.records if r.worker_id == worker_id]

        if not worker_records:
            return

        # Calculate aggregated metrics
        total_executions = len(worker_records)
        successful_executions = sum(1 for r in worker_records if r.success)
        success_rate = successful_executions / total_executions if total_executions > 0 else 0

        successful_records = [r for r in worker_records if r.success]

        if successful_records:
            avg_delta_e = np.mean([r.delta_e_mean for r in successful_records])
            avg_time = np.mean([r.processing_time for r in successful_records])

            # Calculate reliability score (considers success rate and consistency)
            delta_e_consistency = 1.0 / (1.0 + np.std([r.delta_e_mean for r in successful_records]))
            reliability_score = 0.6 * success_rate + 0.4 * delta_e_consistency
        else:
            avg_delta_e = float('inf')
            avg_time = 0.0
            reliability_score = 0.0

        # Calculate specialty performance
        specialties_performance = self._calculate_specialty_performance(successful_records)

        # Determine trend (last 10 vs previous 10)
        recent_trend = self._calculate_trend(successful_records)

        # Calculate weight based on reliability and performance
        # Lower delta E = better, higher reliability = better
        if avg_delta_e < float('inf'):
            # Normalize delta E to 0-1 (assuming 50 is very bad, 0 is perfect)
            normalized_delta_e = max(0, 1.0 - (avg_delta_e / 50.0))
            weight = 0.6 * normalized_delta_e + 0.4 * reliability_score
        else:
            weight = 0.0

        # Create or update statistics
        self.worker_stats[worker_id] = WorkerStatistics(
            worker_id=worker_id,
            total_executions=total_executions,
            successful_executions=successful_executions,
            average_delta_e=avg_delta_e,
            average_processing_time=avg_time,
            reliability_score=reliability_score,
            current_weight=weight,
            specialties_performance=specialties_performance,
            recent_trend=recent_trend
        )

        # Update weight in weights dict
        self.worker_weights[worker_id] = weight

    def _calculate_specialty_performance(self, records: List[WorkerPerformanceRecord]) -> Dict[str, float]:
        """Calculate performance for different specialties (RAL ranges, image types)"""
        specialties = {}

        # Group by RAL range
        ral_groups = {}
        for record in records:
            if record.target_ral_code:
                ral_range = self._get_ral_range(record.target_ral_code)
                if ral_range not in ral_groups:
                    ral_groups[ral_range] = []
                ral_groups[ral_range].append(record)

        # Calculate average performance for each RAL range
        for ral_range, group_records in ral_groups.items():
            if group_records:
                avg_delta_e = np.mean([r.delta_e_mean for r in group_records])
                # Convert to performance score (lower delta E = higher score)
                performance = max(0, 1.0 - (avg_delta_e / 50.0))
                specialties[ral_range] = performance

        # Group by image type
        type_groups = {}
        for record in records:
            if record.image_type:
                if record.image_type not in type_groups:
                    type_groups[record.image_type] = []
                type_groups[record.image_type].append(record)

        # Calculate average performance for each image type
        for img_type, group_records in type_groups.items():
            if group_records:
                avg_delta_e = np.mean([r.delta_e_mean for r in group_records])
                performance = max(0, 1.0 - (avg_delta_e / 50.0))
                specialties[img_type] = performance

        return specialties

    def _calculate_trend(self, records: List[WorkerPerformanceRecord]) -> str:
        """Calculate performance trend: improving, stable, or declining"""
        if len(records) < 10:
            return "stable"

        # Compare last 10 records to previous 10
        recent = records[-10:]
        previous = records[-20:-10] if len(records) >= 20 else records[:-10]

        recent_avg = np.mean([r.delta_e_mean for r in recent])
        previous_avg = np.mean([r.delta_e_mean for r in previous])

        # Lower delta E is better (improving)
        improvement = previous_avg - recent_avg

        if improvement > 2.0:
            return "improving"
        elif improvement < -2.0:
            return "declining"
        else:
            return "stable"

    def _get_ral_range(self, ral_code: str) -> str:
        """Extract RAL range from code (e.g., RAL_3020 -> RAL_3000-3999)"""
        try:
            # Extract number from code
            number = int(''.join(filter(str.isdigit, ral_code)))
            # Get thousands place
            range_base = (number // 1000) * 1000
            return f"RAL_{range_base}-{range_base + 999}"
        except:
            return "unknown"

    def _save_to_storage(self):
        """Save performance data to persistent storage"""
        try:
            data = {
                'records': [asdict(r) for r in self.records[-1000:]],  # Keep last 1000
                'worker_stats': {
                    k: asdict(v) for k, v in self.worker_stats.items()
                },
                'worker_weights': self.worker_weights,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved performance data to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def _load_from_storage(self):
        """Load performance data from persistent storage"""
        if not self.storage_path.exists():
            logger.info("No existing performance data found. Starting fresh.")
            # Initialize default weights (equal for all workers)
            self.worker_weights = {
                "worker_reinhard": 0.2,
                "worker_linear": 0.2,
                "worker_histogram": 0.2,
                "worker_lab_specific": 0.2,
                "worker_region": 0.2
            }
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Reconstruct records
            self.records = [
                WorkerPerformanceRecord(**r) for r in data.get('records', [])
            ]

            # Reconstruct statistics
            self.worker_stats = {
                k: WorkerStatistics(**v) for k, v in data.get('worker_stats', {}).items()
            }

            # Load weights
            self.worker_weights = data.get('worker_weights', {})

            logger.info(f"Loaded {len(self.records)} performance records from storage")

        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            # Initialize defaults
            self.worker_weights = {
                "worker_reinhard": 0.2,
                "worker_linear": 0.2,
                "worker_histogram": 0.2,
                "worker_lab_specific": 0.2,
                "worker_region": 0.2
            }

    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        if not self.worker_stats:
            return "No performance data available yet."

        report_lines = [
            "TSM Performance Report",
            "=" * 50,
            f"Total Records: {len(self.records)}",
            f"Workers Tracked: {len(self.worker_stats)}",
            "",
            "Worker Rankings (by weight):",
            "-" * 50
        ]

        # Sort by weight
        sorted_workers = sorted(
            self.worker_stats.items(),
            key=lambda x: x[1].current_weight,
            reverse=True
        )

        for i, (worker_id, stats) in enumerate(sorted_workers, 1):
            report_lines.extend([
                f"{i}. {worker_id}",
                f"   Weight: {stats.current_weight:.3f}",
                f"   Avg ΔE: {stats.average_delta_e:.2f}",
                f"   Reliability: {stats.reliability_score:.2f}",
                f"   Executions: {stats.successful_executions}/{stats.total_executions}",
                f"   Trend: {stats.recent_trend}",
                f"   Avg Time: {stats.average_processing_time:.2f}s",
                ""
            ])

        return "\n".join(report_lines)
