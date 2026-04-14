"""
Batch Processing Module
~~~~~~~~~~~~~~~~~~~~~~

Supports batch processing of multiple IQ files with parameter sweeping,
progress display, and resume capability.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner
from ChannelSimulation.viewer.signal_proc.business.export_utils import ExportUtils
except ImportError:
    from pipeline_runner import PipelineRunner
    from export_utils import ExportUtils


@dataclass
class BatchTask:
    """Batch processing task"""
    input_file: str
    output_dir: str
    config: Dict[str, Any]
    seed_offset: int = 0
    status: str = "pending"  # pending / running / completed / failed
    error: str = ""
    elapsed: float = 0.0


@dataclass
class BatchResult:
    """Batch processing result"""
    task: BatchTask
    metrics: Dict[str, float]
    output_iq: str
    elapsed: float


class BatchProcessor:
    """Batch processor"""

    def __init__(self):
        self._tasks: List[BatchTask] = []
        self._results: List[BatchResult] = []
        self._progress_callback: Optional[callable] = None
        self._stop_requested = False

    def set_progress_callback(self, callback: callable) -> None:
        """Set progress callback function

        Args:
            callback: Callback function with signature (current, total, message)
        """
        self._progress_callback = callback

    def stop(self) -> None:
        """Request to stop processing"""
        self._stop_requested = True

    def add_task(
        self,
        input_file: str,
        output_dir: str,
        config: Dict[str, Any],
        seed_offset: int = 0,
    ) -> None:
        """Add processing task"""
        task = BatchTask(
            input_file=input_file,
            output_dir=output_dir,
            config=config,
            seed_offset=seed_offset,
        )
        self._tasks.append(task)

    def add_parameter_sweep(
        self,
        input_files: List[str],
        output_dir: str,
        base_config: Dict[str, Any],
        sweep_params: Dict[str, List[Any]],
    ) -> None:
        """Add parameter sweep task

        Args:
            input_files: List of input files
            output_dir: Output directory
            base_config: Base configuration
            sweep_params: Sweep parameters {param_name: [values]}
        """
        # Generate parameter combinations
        import itertools

        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())

        for values in itertools.product(*param_values):
            config = self._deep_copy_config(base_config)
            for name, value in zip(param_names, values):
                # Update config (assuming parameters under stage_configs)
                parts = name.split(".")
                if len(parts) == 2:
                    stage, param = parts
                    if stage in config.get("stage_configs", {}):
                        config["stage_configs"][stage]["params"][param] = value

            # Create task for each input file
            for input_file in input_files:
                output_name = Path(input_file).stem
                # Add parameter suffix
                param_suffix = "_".join(f"{n}{v}" for n, v in zip(param_names, values))
                task_output_dir = Path(output_dir) / f"{output_name}_{param_suffix}"

                self.add_task(
                    input_file=input_file,
                    output_dir=str(task_output_dir),
                    config=config,
                    seed_offset=0,
                )

    def run(self) -> List[BatchResult]:
        """Execute batch processing

        Returns:
            List of processing results
        """
        self._results.clear()
        self._stop_requested = False
        total = len(self._tasks)

        for i, task in enumerate(self._tasks):
            if self._stop_requested:
                task.status = "stopped"
                continue

            # Report progress
            if self._progress_callback:
                self._progress_callback(i, total, f"Processing {task.input_file}")

            # Update status
            task.status = "running"
            start_time = time.perf_counter()

            try:
                result = self._process_task(task)
                task.status = "completed"
                task.elapsed = time.perf_counter() - start_time
                self._results.append(result)

            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                task.elapsed = time.perf_counter() - start_time

        # Final progress report
        if self._progress_callback:
            self._progress_callback(total, total, "Completed")

        return self._results

    def _process_task(self, task: BatchTask) -> BatchResult:
        """Process a single task"""
        # Create output directory
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create pipeline
        runner = PipelineRunner(fs=task.config.get("fs", 100_000_000))
        runner.set_config(task.config)

        # Load file
        runner.load_iq_file(task.input_file)

        # Run pipeline
        results = runner.run()

        # Export results
        output_iq = output_dir / "processed.iq"
        ExportUtils.export_iq(results["final"], output_iq)

        # Export spectrum
        from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumCalculator
        calc = SpectrumCalculator(fs=task.config.get("fs", 100_000_000))
        data = calc.calculate_all(results["original"], results["final"])

        # TODO: Export spectrum PNG

        # Calculate metrics
        from ChannelSimulation.viewer.signal_proc.metrics_calculator import MetricsCalculator
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all(
            results["original"],
            results["final"],
            task.config.get("fs", 100_000_000),
        )

        # Save metadata
        meta = {
            "input_file": task.input_file,
            "output_iq": str(output_iq),
            "config": task.config,
            "metrics": metrics,
        }
        with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return BatchResult(
            task=task,
            metrics=metrics,
            output_iq=str(output_iq),
            elapsed=task.elapsed,
        )

    def _deep_copy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy configuration"""
        import copy
        return copy.deepcopy(config)

    def get_summary_csv(self, results: List[BatchResult]) -> str:
        """Generate summary CSV

        Args:
            results: List of processing results

        Returns:
            CSV content
        """
        lines = [
            "filename,param_combination,processing_time(s),EVM(%),crest_factor(dB),ACLR(dBc),power(dBFS),SNR(dB),status,error",
        ]

        for result in results:
            task = result.task
            metrics = result.metrics

            param_str = json.dumps(task.config.get("stage_configs", {}), ensure_ascii=False)
            param_str = param_str.replace(",", ";")

            line = [
                Path(task.input_file).name,
                param_str[:100],  # truncate
                f"{result.elapsed:.2f}",
                f"{metrics.get('evm', 0):.2f}",
                f"{metrics.get('crest_factor', 0):.2f}",
                f"{metrics.get('aclr_main', 0):.2f}",
                f"{metrics.get('power_dbfs', 0):.2f}",
                f"{metrics.get('snr', 0):.2f}",
                task.status,
                task.error[:50] if task.error else "",
            ]
            lines.append(",".join(line))

        return "\n".join(lines)

    def export_summary(
        self,
        results: List[BatchResult],
        output_path: str,
    ) -> None:
        """Export summary CSV

        Args:
            results: List of processing results
            output_path: Output file path
        """
        csv_content = self.get_summary_csv(results)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

    def get_completed_count(self) -> int:
        """Get number of completed tasks"""
        return sum(1 for t in self._tasks if t.status == "completed")

    def get_failed_count(self) -> int:
        """Get number of failed tasks"""
        return sum(1 for t in self._tasks if t.status == "failed")


class BatchProcessorDialog:
    """Batch processing dialog

    Used to configure and execute batch processing in the UI.
    """

    def __init__(self, parent=None):
        self._parent = parent
        self._processor = BatchProcessor()

    def exec(self) -> None:
        """Show dialog"""
        # TODO: Implement full dialog UI
        pass
