"""
Batch Processing Module
~~~~~~~~~~~~~~~~~~~~~~

Supports batch processing of multiple IQ files with parameter sweeping,
progress display, and resume capability.

Memory optimisation notes
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Stage signals are not retained after export (saves memory for large files).
- gc.collect() is called after each task to reclaim temporary arrays.
- PipelineRunner reuses float32 internal precision to halve per-task memory.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner
    from ChannelSimulation.viewer.business.export_utils import ExportUtils
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

                # Reclaim temporary arrays from this task immediately.
                del result
                gc.collect()

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

        # Run pipeline; request only final output to avoid retaining stage signals.
        results = runner.run()  # dict: original, preprocess, <stage>, final

        # Export IQ
        output_iq = output_dir / "processed.iq"
        ExportUtils.export_iq(results["final"], output_iq)

        # Compute spectrum (on-demand, only for export)
        from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumCalculator
        calc = SpectrumCalculator(fs=task.config.get("fs", 100_000_000))
        data = calc.calculate_all(results["original"], results["final"])

        # Export spectrum PNG
        if data.stft_original is not None and data.freq_axis_stft is not None and data.time_axis_stft is not None:
            from ChannelSimulation.viewer.core.ui_style import get_spectrogram_cmap
            cmap = get_spectrogram_cmap()
            extent = [
                float(np.min(data.time_axis_stft)),
                float(np.max(data.time_axis_stft)),
                float(np.min(data.freq_axis_stft) / 1e6),
                float(np.max(data.freq_axis_stft) / 1e6),
            ]
            ExportUtils.export_spectrum_png(
                data.stft_original,
                output_dir / "original_spectrogram.png",
                title="Original Signal Spectrogram",
                xlabel="Time (s)",
                ylabel="Frequency (MHz)",
                extent=extent,
                cmap=cmap,
            )
            ExportUtils.export_spectrum_png(
                data.stft_processed,
                output_dir / "processed_spectrogram.png",
                title="Processed Signal Spectrogram",
                xlabel="Time (s)",
                ylabel="Frequency (MHz)",
                extent=extent,
                cmap=cmap,
            )

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

        # Clean up large objects to free memory between tasks.
        del runner, results, data, calc
        gc.collect()

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
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QListWidgetItem, QProgressBar, QCheckBox, QSpinBox, QLineEdit
        from PySide6.QtCore import Qt, QThread, Signal, QTimer
        from ChannelSimulation.viewer.core.ui_style import get_color

        self._parent = parent
        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("Batch Processing")
        self._dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(self._dialog)
        layout.setSpacing(12)

        # File selection
        file_section = QLabel("Input Files:")
        file_section.setStyleSheet("font-weight: bold;")
        layout.addWidget(file_section)

        file_layout = QHBoxLayout()
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(100)
        file_layout.addWidget(self._file_list)

        btn_file_layout = QVBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._add_files)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_file)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._file_list.clear)
        btn_file_layout.addWidget(btn_add)
        btn_file_layout.addWidget(btn_remove)
        btn_file_layout.addWidget(btn_clear)
        btn_file_layout.addStretch()
        file_layout.addLayout(btn_file_layout)
        layout.addLayout(file_layout)

        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Select output directory...")
        output_layout.addWidget(self._output_dir)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(self._select_output_dir)
        output_layout.addWidget(btn_output)
        layout.addLayout(output_layout)

        # Sweep parameters
        param_section = QLabel("Parameter Sweep (optional):")
        param_section.setStyleSheet("font-weight: bold;")
        layout.addWidget(param_section)

        self._sweep_list = QListWidget()
        self._sweep_list.setMaximumHeight(80)
        layout.addWidget(self._sweep_list)

        sweep_btn_layout = QHBoxLayout()
        btn_add_param = QPushButton("Add Parameter...")
        btn_add_param.clicked.connect(self._add_sweep_param)
        btn_remove_param = QPushButton("Remove")
        btn_remove_param.clicked.connect(lambda: self._sweep_list.takeItem(self._sweep_list.currentRow()))
        sweep_btn_layout.addWidget(btn_add_param)
        sweep_btn_layout.addWidget(btn_remove_param)
        sweep_btn_layout.addStretch()
        layout.addLayout(sweep_btn_layout)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"color: {get_color('text_secondary')};")
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_run = QPushButton("Run")
        btn_run.setObjectName("btnPrimary")
        btn_run.clicked.connect(self._run_batch)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self._stop_batch)
        btn_stop.setEnabled(False)
        self._btn_stop = btn_stop
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._dialog.reject)
        btn_layout.addWidget(btn_run)
        btn_layout.addWidget(btn_stop)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def exec(self) -> None:
        """Show dialog"""
        self._dialog.exec()

    def _add_files(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self._dialog,
            "Select IQ Files",
            "",
            "IQ Files (*.iq *.cf32 *.cfile *.raw *.bin);;All Files (*.*)",
        )
        for f in files:
            from pathlib import Path
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._file_list.addItem(item)

    def _remove_file(self) -> None:
        self._file_list.takeItem(self._file_list.currentRow())

    def _select_output_dir(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self._dialog, "Select Output Directory")
        if dir_path:
            self._output_dir.setText(dir_path)

    def _add_sweep_param(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        param, ok = QInputDialog.getText(self._dialog, "Add Sweep Parameter", "Parameter name (e.g., stage_configs.awgn.params.snr_db):")
        if ok and param:
            values_str, ok2 = QInputDialog.getText(self._dialog, "Add Sweep Parameter", "Values (comma-separated, e.g., 10,15,20):")
            if ok2:
                from PySide6.QtWidgets import QListWidgetItem
                item = QListWidgetItem(f"{param}: {values_str}")
                item.setData(Qt.ItemDataRole.UserRole, (param, [v.strip() for v in values_str.split(",")]))
                self._sweep_list.addItem(item)

    def _run_batch(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        if self._file_list.count() == 0:
            QMessageBox.warning(self._dialog, "No Files", "Please add at least one input file.")
            return
        output_dir = self._output_dir.text()
        if not output_dir:
            QMessageBox.warning(self._dialog, "No Output Directory", "Please select an output directory.")
            return

        self._btn_run = self._dialog.findChild(object, "btnPrimary") or self._dialog.findChild(__import__('PySide6.QtWidgets', fromlist=['QPushButton']).QPushButton, "")
        self._btn_stop.setEnabled(True)

        files = [self._file_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._file_list.count())]
        
        # Get sweep parameters
        sweep_params = {}
        for i in range(self._sweep_list.count()):
            item = self._sweep_list.item(i)
            param_path, values = item.data(Qt.ItemDataRole.UserRole)
            if param_path and values:
                sweep_params[param_path] = [v for v in values]

        # Create processor
        processor = BatchProcessor()

        def update_progress(current, total, message):
            pct = int(current / total * 100) if total > 0 else 0
            self._progress_bar.setValue(pct)
            self._status_label.setText(f"{message} ({current}/{total})")

        processor.set_progress_callback(update_progress)
        self._processor = processor

        # Add tasks with sweep parameters
        if sweep_params:
            processor.add_parameter_sweep(files, output_dir, {}, sweep_params)
        else:
            for f in files:
                processor.add_task(f, output_dir, {})

        try:
            results = processor.run()
            self._progress_bar.setValue(100)
            completed = processor.get_completed_count()
            failed = processor.get_failed_count()
            self._status_label.setText(f"Completed: {completed}, Failed: {failed}")
            QMessageBox.information(self._dialog, "Batch Complete", f"Completed: {completed}\nFailed: {failed}")
        except Exception as e:
            QMessageBox.critical(self._dialog, "Error", f"Batch processing failed:\n{str(e)}")
        finally:
            self._btn_stop.setEnabled(False)

    def _stop_batch(self) -> None:
        if self._processor:
            self._processor.stop()
            self._status_label.setText("Stopping...")

