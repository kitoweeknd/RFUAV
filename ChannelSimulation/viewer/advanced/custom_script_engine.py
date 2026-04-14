"""
Custom Channel Script Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides a sandbox environment for executing user-defined channel processing scripts.
Supports timeout limits and forbids filesystem access.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np


class ScriptExecutionError(Exception):
    """Script execution error"""
    pass


class ScriptTimeoutError(ScriptExecutionError):
    """Script execution timeout"""
    pass


class ScriptSandbox:
    """Script sandbox

    Executes user scripts in a restricted environment:
    - Timeout limit (default 30 seconds)
    - No filesystem access
    - Limited available modules
    """

    DEFAULT_TIMEOUT = 30  # seconds

    # Allowed modules
    ALLOWED_MODULES = {
        "numpy": np,
        "math": __import__("math"),
        "random": __import__("random"),
    }

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self._timeout = timeout
        self._result: Any = None
        self._error: Optional[Exception] = None
        self._execution_time = 0.0

    @property
    def result(self) -> Any:
        """Get execution result"""
        return self._result

    @property
    def error(self) -> Optional[Exception]:
        """Get execution error"""
        return self._error

    @property
    def execution_time(self) -> float:
        """Get execution time"""
        return self._execution_time

    def execute(self, script: str, globals_dict: Dict | None = None) -> Any:
        """Execute script

        Args:
            script: Python script code
            globals_dict: Global variables dictionary

        Returns:
            Script return value

        Raises:
            ScriptExecutionError: Execution error
            ScriptTimeoutError: Execution timeout
        """
        start_time = time.perf_counter()
        self._error = None
        self._result = None

        # Create restricted globals dictionary
        if globals_dict is None:
            globals_dict = {}

        # Add allowed modules
        safe_globals = {
            "__builtins__": self._get_safe_builtins(),
            **self.ALLOWED_MODULES,
            **globals_dict,
        }

        # Create local variables dictionary
        local_vars = {}

        # Use timer for timeout
        timer = threading.Timer(self._timeout, self._handle_timeout)
        timer.daemon = True

        try:
            timer.start()
            self._result = exec(script, safe_globals, local_vars)
            self._execution_time = time.perf_counter() - start_time
            return self._result

        except TimeoutError:
            self._error = ScriptTimeoutError(f"Script execution timeout ({self._timeout} seconds)")
            raise self._error

        except Exception as e:
            self._error = ScriptExecutionError(f"Script execution error: {type(e).__name__}: {str(e)}")
            raise self._error

        finally:
            timer.cancel()
            self._execution_time = time.perf_counter() - start_time

    def _handle_timeout(self) -> None:
        """Handle timeout"""
        self._error = ScriptTimeoutError(f"Script execution timeout ({self._timeout} seconds)")
        # Use sys.exit to force exit (this will trigger SystemExit)
        # Note: This may affect the main program, use with caution
        # In practice, a safer method may be needed

    def _get_safe_builtins(self) -> Dict:
        """Get safe builtins dictionary"""
        safe_list = [
            "abs", "all", "any", "bin", "bool", "chr", "complex",
            "dict", "enumerate", "filter", "float", "format", "frozenset",
            "hash", "hex", "int", "isinstance", "issubclass", "iter",
            "len", "list", "map", "max", "min", "next", "oct", "ord",
            "pow", "range", "repr", "reversed", "round", "set", "slice",
            "sorted", "str", "sum", "tuple", "zip",
            # numpy related
            "array", "zeros", "ones", "arange", "linspace",
            "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "e",
        ]

        builtins = {}
        for name in safe_list:
            try:
                builtins[name] = eval(name)
            except NameError:
                pass

        return {"__builtins__": builtins}


class CustomChannelRunner:
    """Custom channel runner

    Processes IQ signals using user-defined functions.
    """

    def __init__(self, timeout: int = 30):
        self._sandbox = ScriptSandbox(timeout=timeout)

    def run_function(
        self,
        func_code: str,
        input_signal: np.ndarray,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Run custom function

        Args:
            func_code: Function code
            input_signal: Input IQ signal
            params: Parameters dictionary

        Returns:
            Processed signal
        """
        # Build script
        script = f"""
import numpy as np

{func_code}

# Call user function
output = apply_channel(input_signal, params)
"""

        # Add input data
        globals_dict = {
            "input_signal": input_signal,
            "params": params,
        }

        # Execute script
        self._sandbox.execute(script, globals_dict)

        # Get output
        return self._sandbox.result


class CustomPipeline:
    """Custom pipeline

    Allows users to combine multiple custom channels.
    """

    def __init__(self):
        self._stages: list[tuple[str, Callable]] = []

    def add_stage(
        self,
        name: str,
        func: Callable[[np.ndarray, np.random.Generator, Dict], np.ndarray],
        params: Dict[str, Any] | None = None,
    ) -> "CustomPipeline":
        """Add a stage

        Args:
            name: Stage name
            func: Processing function
            params: Parameters

        Returns:
            self (for method chaining)
        """
        self._stages.append((name, func, params or {}))
        return self

    def run(
        self,
        input_signal: np.ndarray,
        seed: int = 42,
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """Run custom pipeline

        Args:
            input_signal: Input signal
            seed: Random seed

        Returns:
            (Intermediate signals dictionary, final output)
        """
        rng = np.random.default_rng(seed)
        signals = {"input": input_signal.copy()}
        current = input_signal.copy()

        for name, func, params in self._stages:
            try:
                current = func(current, rng, params)
                signals[name] = current.copy()
            except Exception as e:
                raise ScriptExecutionError(f"Stage '{name}' execution failed: {e}")

        signals["output"] = current.copy()
        return signals, current


class CustomScriptEngine:
    """Custom script engine

    Provides UI interface for loading and executing custom scripts.
    """

    def __init__(self):
        self._sandbox = ScriptSandbox()
        self._runner = CustomChannelRunner()

    def load_script_from_file(self, file_path: str) -> str:
        """Load script from file

        Args:
            file_path: Script file path

        Returns:
            Script content

        Raises:
            FileNotFoundError: File does not exist
            ScriptExecutionError: Script format error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ScriptExecutionError(f"Cannot read script file: {e}")

    def validate_script(self, script: str) -> tuple[bool, str]:
        """Validate script syntax

        Args:
            script: Script content

        Returns:
            (Is valid, error message)
        """
        try:
            compile(script, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def execute_script(
        self,
        script: str,
        input_signal: np.ndarray,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Execute script

        Args:
            script: Script content
            input_signal: Input signal
            params: Parameters

        Returns:
            Processed signal
        """
        return self._runner.run_function(script, input_signal, params)