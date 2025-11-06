"""
Tom Sawyer Method (TSM) Ensemble Orchestrator
==============================================

This module implements the "Master Conductor" of TSM - the orchestrator that:
1. Analyzes image complexity
2. Selects appropriate workers
3. Coordinates parallel execution
4. Aggregates results using weighted voting
5. Tracks performance and learns over time
6. Handles anomalies and resilience

This is the core integration point that brings all TSM components together.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from transfer_algorithms import (
    BaseTransferWorker,
    TransferResult,
    WorkerFactory
)
from complexity_analyzer import ImageComplexityAnalyzer, ComplexityReport
from performance_tracker import PerformanceTracker
from color_utils import rgb_to_lab, delta_e_ciede2000

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Final result from TSM ensemble processing"""
    best_result_rgb: np.ndarray
    best_worker_id: str
    ensemble_rgb: Optional[np.ndarray]  # Weighted blend if applicable
    all_results: List[TransferResult]
    complexity_report: ComplexityReport
    qc_report: Dict
    execution_summary: Dict
    processing_time_total: float


class AggregationOracle:
    """
    The "Aggregation Oracle" that combines results from multiple workers
    using weighted voting and quality metrics.
    """

    def __init__(self, performance_tracker: PerformanceTracker):
        self.performance_tracker = performance_tracker

    def aggregate(
        self,
        results: List[TransferResult],
        target_rgb: np.ndarray,
        context: Optional[Dict] = None
    ) -> Tuple[np.ndarray, str, Dict]:
        """
        Aggregate results from multiple workers using weighted voting.

        Args:
            results: List of TransferResult from each worker
            target_rgb: Target color for QC evaluation
            context: Context for contextualized weights

        Returns:
            (best_result_rgb, best_worker_id, qc_scores)
        """
        if not results:
            raise ValueError("No results to aggregate")

        # Get worker weights (contextualized if context provided)
        worker_weights = self.performance_tracker.get_worker_weights(context)

        # Evaluate quality of each result
        qc_scores = {}
        weighted_scores = {}

        for result in results:
            # Calculate Delta E for quality assessment
            result_lab = rgb_to_lab(result.result_rgb)
            target_lab = rgb_to_lab(target_rgb)

            # Calculate per-pixel delta E
            delta_e_map = np.zeros((result_lab.shape[0], result_lab.shape[1]))
            for i in range(result_lab.shape[0]):
                for j in range(result_lab.shape[1]):
                    delta_e_map[i, j] = delta_e_ciede2000(
                        result_lab[i, j],
                        target_lab.reshape(-1, 3)[0]  # Single target color
                    )

            # Calculate statistics
            qc_score = {
                'mean': float(np.mean(delta_e_map)),
                'std': float(np.std(delta_e_map)),
                'max': float(np.max(delta_e_map)),
                'percentile_95': float(np.percentile(delta_e_map, 95)),
                'processing_time': result.processing_time
            }

            qc_scores[result.worker_id] = qc_score

            # Calculate weighted score (lower delta E = better)
            # Invert delta E so higher is better
            quality_score = max(0, 50.0 - qc_score['mean']) / 50.0

            # Get worker weight
            worker_weight = worker_weights.get(result.worker_id, 0.2)

            # Combined weighted score
            weighted_score = 0.7 * quality_score + 0.3 * worker_weight

            weighted_scores[result.worker_id] = weighted_score

            logger.info(
                f"Worker {result.worker_id}: ΔE={qc_score['mean']:.2f}, "
                f"quality={quality_score:.3f}, weight={worker_weight:.3f}, "
                f"weighted_score={weighted_score:.3f}"
            )

        # Select best result based on weighted scores
        best_worker_id = max(weighted_scores, key=weighted_scores.get)
        best_result = next(r for r in results if r.worker_id == best_worker_id)

        logger.info(f"Selected best worker: {best_worker_id} (score: {weighted_scores[best_worker_id]:.3f})")

        return best_result.result_rgb, best_worker_id, qc_scores

    def create_ensemble_blend(
        self,
        results: List[TransferResult],
        worker_weights: Dict[str, float],
        top_n: int = 3
    ) -> np.ndarray:
        """
        Create weighted blend of top N results (experimental).

        This is an alternative aggregation strategy that blends multiple
        results instead of selecting one winner.

        Args:
            results: All worker results
            worker_weights: Weight for each worker
            top_n: Number of top results to blend

        Returns:
            Blended RGB image
        """
        # Sort results by weight
        sorted_results = sorted(
            results,
            key=lambda r: worker_weights.get(r.worker_id, 0),
            reverse=True
        )[:top_n]

        if not sorted_results:
            return results[0].result_rgb

        # Normalize weights for top N
        top_weights = [worker_weights.get(r.worker_id, 0) for r in sorted_results]
        total_weight = sum(top_weights)

        if total_weight == 0:
            # Equal weights if all are zero
            top_weights = [1.0 / len(sorted_results)] * len(sorted_results)
        else:
            top_weights = [w / total_weight for w in top_weights]

        # Weighted blend
        blended = np.zeros_like(sorted_results[0].result_rgb, dtype=np.float32)

        for result, weight in zip(sorted_results, top_weights):
            blended += result.result_rgb.astype(np.float32) * weight

        blended = np.clip(blended, 0, 255).astype(np.uint8)

        logger.info(f"Created ensemble blend from top {len(sorted_results)} workers")

        return blended


class TSMOrchestrator:
    """
    Main Tom Sawyer Method Orchestrator.

    This is the "Project Manager" that coordinates the entire TSM system:
    - Adaptive worker selection based on complexity
    - Parallel execution of workers
    - Result aggregation
    - Performance tracking and learning
    - Anomaly detection and resilience
    """

    def __init__(
        self,
        performance_tracker: Optional[PerformanceTracker] = None,
        max_workers: int = 5
    ):
        """
        Initialize TSM orchestrator.

        Args:
            performance_tracker: Optional existing tracker (for persistence)
            max_workers: Maximum number of parallel workers
        """
        self.complexity_analyzer = ImageComplexityAnalyzer()
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.aggregation_oracle = AggregationOracle(self.performance_tracker)
        self.max_workers = max_workers

        logger.info("TSM Orchestrator initialized")

    def process(
        self,
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        target_ral_code: Optional[str] = None,
        mode: str = "adaptive",
        use_ensemble_blend: bool = False
    ) -> EnsembleResult:
        """
        Process color transfer using Tom Sawyer Method.

        Args:
            source_rgb: Source image RGB
            target_rgb: Target color RGB
            target_ral_code: Optional RAL code for context
            mode: "adaptive" (auto-select workers), "all" (use all), or "best" (use top 3)
            use_ensemble_blend: Whether to create weighted blend (vs single best)

        Returns:
            EnsembleResult with best result and metadata
        """
        start_time = time.time()

        # Step 1: Analyze image complexity
        logger.info("Step 1: Analyzing image complexity...")
        complexity_report = self.complexity_analyzer.analyze(source_rgb)

        logger.info(f"Complexity: {complexity_report.overall_complexity:.2f}")
        logger.info(f"Image type: {complexity_report.image_characteristics['type']}")
        logger.info(f"Recommended workers: {len(complexity_report.recommended_workers)}")

        # Step 2: Select workers based on mode
        logger.info("Step 2: Selecting workers...")
        selected_worker_ids = self._select_workers(mode, complexity_report)

        logger.info(f"Selected {len(selected_worker_ids)} workers: {selected_worker_ids}")

        # Step 3: Execute workers in parallel
        logger.info("Step 3: Executing workers in parallel...")
        results = self._execute_workers_parallel(
            selected_worker_ids,
            source_rgb,
            target_rgb
        )

        logger.info(f"Completed {len(results)} worker executions")

        # Step 4: Anomaly detection - filter out bad results
        logger.info("Step 4: Detecting anomalies...")
        filtered_results = self._filter_anomalies(results, target_rgb)

        if len(filtered_results) < len(results):
            logger.warning(f"Filtered out {len(results) - len(filtered_results)} anomalous results")

        if not filtered_results:
            logger.error("All results were anomalous! Using original results.")
            filtered_results = results

        # Step 5: Aggregate results using weighted voting
        logger.info("Step 5: Aggregating results...")
        context = {
            'target_ral_code': target_ral_code,
            'image_type': complexity_report.image_characteristics['type'],
            'complexity': complexity_report.overall_complexity
        }

        best_result_rgb, best_worker_id, qc_scores = self.aggregation_oracle.aggregate(
            filtered_results,
            target_rgb,
            context
        )

        # Optional: Create ensemble blend
        ensemble_rgb = None
        if use_ensemble_blend and len(filtered_results) > 1:
            logger.info("Creating ensemble blend...")
            worker_weights = self.performance_tracker.get_worker_weights(context)
            ensemble_rgb = self.aggregation_oracle.create_ensemble_blend(
                filtered_results,
                worker_weights,
                top_n=min(3, len(filtered_results))
            )

        # Step 6: Record performance for learning
        logger.info("Step 6: Recording performance metrics...")
        for result in filtered_results:
            qc = qc_scores.get(result.worker_id, {})
            self.performance_tracker.record_performance(
                worker_id=result.worker_id,
                delta_e_stats=qc,
                processing_time=result.processing_time,
                target_ral_code=target_ral_code,
                image_type=complexity_report.image_characteristics['type'],
                complexity=complexity_report.overall_complexity,
                success=True
            )

        # Calculate total processing time
        processing_time_total = time.time() - start_time

        # Create execution summary
        execution_summary = {
            'total_workers_executed': len(results),
            'workers_after_anomaly_filter': len(filtered_results),
            'best_worker': best_worker_id,
            'mode': mode,
            'parallel_execution': True,
            'ensemble_blend_created': ensemble_rgb is not None,
            'processing_time_total': processing_time_total,
            'processing_times_per_worker': {
                r.worker_id: r.processing_time for r in results
            }
        }

        logger.info(f"TSM processing complete in {processing_time_total:.2f}s")

        # Compile final QC report
        final_qc = qc_scores.get(best_worker_id, {})
        final_qc['best_worker'] = best_worker_id
        final_qc['all_workers_scores'] = qc_scores

        return EnsembleResult(
            best_result_rgb=best_result_rgb,
            best_worker_id=best_worker_id,
            ensemble_rgb=ensemble_rgb,
            all_results=filtered_results,
            complexity_report=complexity_report,
            qc_report=final_qc,
            execution_summary=execution_summary,
            processing_time_total=processing_time_total
        )

    def _select_workers(self, mode: str, complexity_report: ComplexityReport) -> List[str]:
        """Select which workers to execute based on mode and complexity"""
        if mode == "all":
            # Use all available workers
            return [
                "worker_reinhard",
                "worker_linear",
                "worker_histogram",
                "worker_lab_specific",
                "worker_region"
            ]
        elif mode == "best":
            # Use top 3 performing workers from history
            return self.performance_tracker.get_best_workers(n=3)
        elif mode == "adaptive":
            # Use complexity-based recommendations
            return complexity_report.recommended_workers
        else:
            # Default to adaptive
            return complexity_report.recommended_workers

    def _execute_workers_parallel(
        self,
        worker_ids: List[str],
        source_rgb: np.ndarray,
        target_rgb: np.ndarray
    ) -> List[TransferResult]:
        """Execute multiple workers in parallel using ThreadPoolExecutor"""
        results = []

        # Create worker instances
        workers = []
        for worker_id in worker_ids:
            worker = WorkerFactory.create_worker(worker_id)
            if worker:
                workers.append(worker)

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_worker = {
                executor.submit(worker.transfer, source_rgb, target_rgb): worker
                for worker in workers
            }

            for future in as_completed(future_to_worker):
                worker = future_to_worker[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Worker {worker.worker_id} completed in {result.processing_time:.2f}s")
                except Exception as e:
                    logger.error(f"Worker {worker.worker_id} failed: {e}")

        return results

    def _filter_anomalies(
        self,
        results: List[TransferResult],
        target_rgb: np.ndarray
    ) -> List[TransferResult]:
        """Filter out anomalous results using anomaly detection"""
        filtered = []

        for result in results:
            # Calculate quick QC metrics for anomaly detection
            result_lab = rgb_to_lab(result.result_rgb)
            target_lab = rgb_to_lab(target_rgb)

            # Sample a few pixels for quick check (avoid full computation)
            h, w = result_lab.shape[:2]
            sample_indices = [
                (h // 4, w // 4),
                (h // 2, w // 2),
                (3 * h // 4, 3 * w // 4)
            ]

            delta_es = []
            for i, j in sample_indices:
                de = delta_e_ciede2000(result_lab[i, j], target_lab.reshape(-1, 3)[0])
                delta_es.append(de)

            avg_delta_e = np.mean(delta_es)
            max_delta_e = np.max(delta_es)

            delta_e_stats = {
                'mean': avg_delta_e,
                'max': max_delta_e
            }

            # Check for anomaly
            is_anomaly, reason = self.performance_tracker.detect_anomaly(
                result.worker_id,
                delta_e_stats,
                result.processing_time
            )

            if is_anomaly:
                logger.warning(f"Anomaly detected in {result.worker_id}: {reason}")
            else:
                filtered.append(result)

        return filtered if filtered else results  # Return all if all are anomalous

    def get_status_report(self) -> str:
        """Generate comprehensive status report"""
        report_lines = [
            "=" * 60,
            "TSM Orchestrator Status Report",
            "=" * 60,
            "",
            "Configuration:",
            f"  Max parallel workers: {self.max_workers}",
            f"  Performance tracking: {'Enabled' if self.performance_tracker else 'Disabled'}",
            "",
        ]

        # Add performance report
        if self.performance_tracker:
            report_lines.append(self.performance_tracker.get_performance_report())

        return "\n".join(report_lines)
