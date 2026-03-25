import ast
import enum
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple


# ===========================================================================
# NEW — Runtime wrapper generators (updated)
# ===========================================================================

def generate_controller_wrapper() -> str:
    """
    Generate the check_control() wrapper that must be prepended to code
    instrumented with ControllerInstrumenter.
    """
    return '''\
# CROWDio runtime marker: controller_wrapper_v1
__CROWDIO_CONTROLLER_WRAPPER__ = True

import threading
import time

class TaskKilledException(Exception):
    """Raised by check_control() when the task has been killed."""

pause_event = threading.Event()
kill_event  = threading.Event()

def check_control():
    """
    Cooperative checkpoint injected by ControllerInstrumenter.
    - If kill_event is set  → raises TaskKilledException.
    - If pause_event is set → blocks via Event.wait() until cleared,
      then re-checks kill (handles kill-during-pause).
    """
    if kill_event.is_set():
        raise TaskKilledException("Task was killed")
    if pause_event.is_set():
        while pause_event.is_set():
            pause_event.wait(timeout=0.05)
            if kill_event.is_set():
                raise TaskKilledException("Task was killed during pause")

def setup_control():
    """Reset both events before starting a new task run."""
    pause_event.clear()
    kill_event.clear()
'''

def generate_manual_checkpoint_wrapper() -> str:
    """
    Generate wrapper code that exposes a ``checkpoint`` (ManualCheckpointUtils)
    instance to instrumented scripts.
    """
    return '''\
# CROWDio runtime marker: manual_checkpoint_wrapper_v1
__CROWDIO_MANUAL_CHECKPOINT_WRAPPER__ = True

import threading
import time

class TaskKilledException(Exception):
    """Raised when the task has been killed via checkpoint.kill()."""

class _ManualCheckpointUtils:
    """Cooperative checkpoint helper exposed as `checkpoint` to task scripts."""
    def __init__(self):
        self.pause_event = threading.Event()
        self.kill_event  = threading.Event()
        self._lock  = threading.Lock()
        self._state = {}
    def pause(self):
        self.pause_event.set()
    def resume(self):
        self.pause_event.clear()
    def kill(self):
        self.kill_event.set()
        self.pause_event.clear()   # wake up any waiting thread
    def reset(self):
        self.pause_event.clear()
        self.kill_event.clear()
        with self._lock:
            self._state = {}
    def check(self):
        """Cooperative checkpoint: raise if killed, block if paused."""
        if self.kill_event.is_set():
            raise TaskKilledException("Task was killed")
        if self.pause_event.is_set():
            while self.pause_event.is_set():
                self.pause_event.wait(timeout=0.05)
                if self.kill_event.is_set():
                    raise TaskKilledException("Task was killed during pause")
    def save(self, state_dict):
        """Persist checkpoint state (dict of variable names → values)."""
        with self._lock:
            self._state.update(state_dict)
    def get_state(self):
        """Return a snapshot of the latest saved state."""
        with self._lock:
            return dict(self._state)
    def clear_state(self):
        with self._lock:
            self._state = {}

# Singleton exposed to task scripts
checkpoint = _ManualCheckpointUtils()
'''

def generate_runtime_wrappers(
    include_checkpoint: bool = True,
    include_task_control: bool = True,
    include_controller: bool = False,
    include_manual_checkpoint: bool = False,
) -> str:
    """
    Build runtime wrappers in dependency-safe order.
    Args:
        include_checkpoint:        Include legacy _ckpt_update / _MobileCheckpointState.
        include_task_control:      Include legacy _TaskControl (thread-local flags).
        include_controller:        Include EventTaskControl check_control() wrapper.
        include_manual_checkpoint: Include ManualCheckpointUtils `checkpoint` singleton.
    Returns:
        Combined wrapper code string.
    """
    parts = []
    if include_checkpoint:
        parts.append(generate_mobile_checkpoint_wrapper())
    if include_task_control:
        parts.append(generate_task_control_wrapper())
    if include_controller:
        parts.append(generate_controller_wrapper())
    if include_manual_checkpoint:
        parts.append(generate_manual_checkpoint_wrapper())
    return "\n".join(parts)

# ===========================================================================
# NEW pipeline functions for controller/manual checkpointing
# ===========================================================================

def instrument_with_controller(
    func_code: str,
) -> Tuple[str, int, int]:
    """
    Instrument function code with check_control() calls (ControllerInstrumenter).
    This is the preferred alternative to instrument_for_task_control().
    - Injects a single ``check_control()`` call at the top of every loop body.
    - check_control() raises TaskKilledException (propagates through try/finally)
      and blocks via threading.Event.wait() for pause — no busy-loop.
    - Idempotent: safe to call multiple times.
    Args:
        func_code: Original function source code.
    Returns:
        Tuple of (instrumented_code, num_functions_instrumented, num_loops_instrumented).
    """
    try:
        tree = ast.parse(func_code)
        instrumenter = ControllerInstrumenter()
        tree = instrumenter.visit(tree)
        ast.fix_missing_locations(tree)
        return (
            ast.unparse(tree),
            instrumenter.functions_instrumented,
            instrumenter.loops_instrumented,
        )
    except Exception as e:
        logger.warning("[CodeInstrumenter] Controller instrumentation failed: %s", e)
        return func_code, 0, 0

def prepare_code_with_controller(
    func_code: str,
    include_checkpoint: bool = False,
    checkpoint_state_vars: Optional[List[str]] = None,
) -> str:
    """
    Full pipeline: inject check_control() calls and prepend EventTaskControl wrapper.
    Optionally also injects _ckpt_update() calls for checkpoint state capture.
    Args:
        func_code:              Original function source code.
        include_checkpoint:     If True, also run CheckpointInstrumenter and
                                prepend the mobile checkpoint wrapper.
        checkpoint_state_vars:  Variable names to capture (required when
                                include_checkpoint=True).
    Returns:
        Code with controller wrapper (and optionally checkpoint wrapper)
        prepended, and check_control() calls injected.
    """
    if "__CROWDIO_CONTROLLER_WRAPPER__ = True" in func_code:
        return func_code
    instrumented, num_funcs, num_loops = instrument_with_controller(func_code)
    logger.debug(
        "[CodeInstrumenter] Controller: instrumented %s functions, %s loops",
        num_funcs,
        num_loops,
    )
    if include_checkpoint and checkpoint_state_vars:
        try:
            tree = ast.parse(instrumented)
            ckpt_instrumenter = CheckpointInstrumenter(checkpoint_state_vars)
            tree = ckpt_instrumenter.visit(tree)
            ast.fix_missing_locations(tree)
            instrumented = ast.unparse(tree)
            logger.debug(
                "[CodeInstrumenter] Checkpoint: instrumented %s loops",
                ckpt_instrumenter.loops_instrumented,
            )
        except Exception as e:
            logger.warning("[CodeInstrumenter] Checkpoint pass failed: %s", e)
    wrapper = generate_runtime_wrappers(
        include_checkpoint=include_checkpoint,
        include_task_control=False,
        include_controller=True,
    )
    return wrapper + "\n" + instrumented

def prepare_code_manual_checkpoint(
    func_code: str,
) -> str:
    """
    Prepend the manual checkpoint wrapper to a script that already contains
    explicit ``checkpoint.check()`` / ``checkpoint.save()`` calls.
    No AST transformation is performed — this is a pure prepend for scripts
    authored with cooperative checks already in place.
    Args:
        func_code: Script source code with checkpoint.check() calls.
    Returns:
        Wrapper code + original script, ready for exec() on the worker.
    """
    if "__CROWDIO_MANUAL_CHECKPOINT_WRAPPER__ = True" in func_code:
        return func_code
    wrapper = generate_manual_checkpoint_wrapper()
    return wrapper + "\n" + func_code
######################################################################
# NEW ① — Manual Checkpointing Utils (SDK helpers)
######################################################################

class TaskKilledException(Exception):
    """
    Raised by check_control() / ManualCheckpointUtils.check() when the task
    has been killed. Propagates naturally through try/finally blocks so
    resources are cleaned up correctly — unlike a bare `return 'killed'`.
    """


class ManualCheckpointUtils:
    """
    SDK helpers for cooperative, author-placed checkpoints.

    Why use this instead of AST injection?
    - No fragility from decorators, generators, async, or complex try/finally.
    - Script authors call check() exactly where it is safe to pause/resume.
    - Works with both the legacy _TaskControl globals and the new
      EventTaskControl (threading.Event) model — detected automatically.

    Usage (script author writes):
        from sdk import checkpoint   # ManualCheckpointUtils instance
        def my_task(data):
            results = []
            for i, item in enumerate(data):
                checkpoint.check()                        # cooperative point
                checkpoint.save({'i': i, 'results': results})
                results.append(process(item))
            return results

    The framework prepends the necessary wrapper (generate_manual_checkpoint_wrapper)
    so `checkpoint` is available without any import in the script itself.
    """

    _lock: threading.Lock
    _state: Dict[str, Any]
    _use_events: bool
    _pause_event: Optional[threading.Event]
    _kill_event: Optional[threading.Event]
    _paused_flag: Optional[bool]
    _killed_flag: Optional[bool]

    def __init__(
        self,
        pause_event: Optional[threading.Event] = None,
        kill_event: Optional[threading.Event] = None,
    ):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}

        if pause_event is not None and kill_event is not None:
            self._use_events = True
            self._pause_event = pause_event
            self._kill_event = kill_event
        else:
            self._use_events = False
            self._pause_event = None
            self._kill_event = None
            self._paused_flag = False
            self._killed_flag = False

    def pause(self) -> None:
        if self._use_events:
            self._pause_event.set()
        else:
            self._paused_flag = True

    def resume(self) -> None:
        if self._use_events:
            self._pause_event.clear()
        else:
            self._paused_flag = False

    def kill(self) -> None:
        if self._use_events:
            self._kill_event.set()
        else:
            self._killed_flag = True

    def reset(self) -> None:
        if self._use_events:
            self._pause_event.clear()
            self._kill_event.clear()
        else:
            self._paused_flag = False
            self._killed_flag = False
        with self._lock:
            self._state = {}

    def check(self) -> None:
        if self._is_killed():
            raise TaskKilledException("Task was killed")
        if self._is_paused():
            self._wait_for_resume()
        if self._is_killed():
            raise TaskKilledException("Task was killed during pause")

    def _is_killed(self) -> bool:
        if self._use_events:
            return self._kill_event.is_set()
        return bool(self._killed_flag)

    def _is_paused(self) -> bool:
        if self._use_events:
            return self._pause_event.is_set()
        return bool(self._paused_flag)

    def _wait_for_resume(self) -> None:
        if self._use_events:
            while self._pause_event.is_set():
                self._pause_event.wait(timeout=0.05)
        else:
            while self._paused_flag:
                time.sleep(0.05)

    def save(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(state)

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def clear_state(self) -> None:
        with self._lock:
            self._state = {}

######################################################################
# NEW ② — ControllerInstrumenter  (AST injector for check_control())
######################################################################

class ControllerInstrumenter(ast.NodeTransformer):
    """
    AST transformer that injects a single ``check_control()`` call at the
    start of every loop body.

    Why prefer this over TaskControlInstrumenter?
    - One injected call instead of two (kill-if + pause-while).
    - ``check_control()`` raises TaskKilledException, which propagates
      through try/finally cleanly — unlike a bare ``return 'killed'``.
    - Idempotent: skips loops that already contain a check_control() call.
    - Respects the generated pause-wait loops so it never double-injects.

    The check_control() function itself is provided by the runtime wrapper
    (generate_controller_wrapper / generate_runtime_wrappers).
    """
    def __init__(self) -> None:
        super().__init__()
        self.loops_instrumented = 0
        self.functions_instrumented = 0

    @staticmethod
    def _has_check_control(stmts: List[ast.stmt]) -> bool:
        for stmt in stmts:
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == 'check_control'
            ):
                return True
        return False

    @staticmethod
    def _is_generated_pause_loop(node: ast.While) -> bool:
        test = node.test
        if not (isinstance(test, ast.Call) and isinstance(test.func, ast.Attribute)):
            return False
        attr: ast.Attribute = test.func
        return (
            isinstance(attr.value, ast.Name)
            and attr.value.id in ('pause_event', '_TaskControl')
            and attr.attr in ('is_set', 'is_paused')
        )

    @staticmethod
    def _create_check_control_call() -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='check_control', ctx=ast.Load()),
                args=[],
                keywords=[]
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        self.functions_instrumented += 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        self.functions_instrumented += 1
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        if not self._has_check_control(node.body):
            node.body.insert(0, self._create_check_control_call())
            self.loops_instrumented += 1
        return node

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> ast.While:
        if self._is_generated_pause_loop(node):
            return node
        self.generic_visit(node)
        if not self._has_check_control(node.body):
            node.body.insert(0, self._create_check_control_call())
            self.loops_instrumented += 1
        return node

######################################################################
# NEW ③ — EventTaskControl  (threading.Event-based signal model)
######################################################################

class EventTaskControl:
    """
    Thread-safe pause / kill control using threading.Event.
    """
    def __init__(self) -> None:
        self.pause_event: threading.Event = threading.Event()
        self.kill_event: threading.Event = threading.Event()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def kill(self) -> None:
        self.kill_event.set()
        self.pause_event.clear()

    def reset(self) -> None:
        self.pause_event.clear()
        self.kill_event.clear()

    def check(self) -> None:
        if self.kill_event.is_set():
            raise TaskKilledException("Task was killed")
        if self.pause_event.is_set():
            while self.pause_event.is_set():
                self.pause_event.wait(timeout=0.05)
                if self.kill_event.is_set():
                    raise TaskKilledException("Task was killed during pause")

    def make_check_control(self):
        def check_control():
            self.check()
        return check_control
"""
AST-based code instrumentation for mobile workers

This module provides code transformation for workers that don't support
sys.settrace() (e.g., Chaquopy on Android). It uses AST manipulation to
inject explicit checkpoint calls into function code.

Key Features:
- CheckpointInstrumenter: Injects _ckpt_update() calls at strategic points
- ResumeInstrumenter: Transforms code to resume from checkpoint state
- Mobile checkpoint wrapper: Provides explicit state capture for mobile

This is TRANSPARENT to developers - they write pure logic, and the framework
automatically instruments code for mobile workers.
"""

import ast
import enum
import logging
from typing import List, Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Optional, Tuple


class WorkerType(enum.Enum):
    """Types of workers in the system"""
    PC_PYTHON = "pc_python"           # Standard Python with sys.settrace() support
    ANDROID_CHAQUOPY = "android_chaquopy"  # Chaquopy - no settrace
    ANDROID_KOTLIN = "android_kotlin"      # Kotlin-based worker
    UNKNOWN = "unknown"


def detect_worker_capabilities(
    platform: str = None,
    runtime: str = None,
    capabilities: Dict[str, bool] = None
) -> Dict[str, Any]:
    """
    Detect worker capabilities based on platform info
    
    Args:
        platform: Platform identifier (e.g., "android", "windows", "linux")
        runtime: Runtime identifier (e.g., "chaquopy", "cpython")
        capabilities: Optional pre-defined capabilities dict
        
    Returns:
        Dictionary of capabilities including:
        - supports_settrace: Whether sys.settrace() works
        - supports_frame_introspection: Whether frame locals work
        - worker_type: WorkerType enum value
    """
    result = {
        "supports_settrace": True,
        "supports_frame_introspection": True,
        "worker_type": WorkerType.PC_PYTHON,
        "needs_instrumentation": False
    }
    
    # Check explicit capabilities first
    if capabilities:
        if capabilities.get("supports_settrace") is False:
            result["supports_settrace"] = False
            result["needs_instrumentation"] = True
        if capabilities.get("supports_frame_introspection") is False:
            result["supports_frame_introspection"] = False
            result["needs_instrumentation"] = True
    
    # Detect based on platform/runtime
    if platform and platform.lower() == "android":
        if runtime and "chaquopy" in runtime.lower():
            result["worker_type"] = WorkerType.ANDROID_CHAQUOPY
            result["supports_settrace"] = False
            result["supports_frame_introspection"] = False
            result["needs_instrumentation"] = True
        elif runtime and "kotlin" in runtime.lower():
            result["worker_type"] = WorkerType.ANDROID_KOTLIN
            result["supports_settrace"] = False
            result["supports_frame_introspection"] = False
            result["needs_instrumentation"] = True
    
    return result


class CheckpointInstrumenter(ast.NodeTransformer):
    """
    AST transformer that injects checkpoint update calls into code
    
    Transforms:
    - Loop bodies: Injects _ckpt_update() at end of each iteration
    - Key assignments: Captures state after checkpoint variables are updated
    
    Example transformation:
    
    Before:
        for i in range(n):
            total += compute(data[i])
            progress_percent = (i + 1) / n * 100
    
    After:
        for i in range(n):
            total += compute(data[i])
            progress_percent = (i + 1) / n * 100
            _ckpt_update({'i': i, 'total': total, 'progress_percent': progress_percent})
    """
    
    def __init__(self, checkpoint_state_vars: List[str]):
        """
        Initialize instrumenter
        
        Args:
            checkpoint_state_vars: List of variable names to capture
        """
        super().__init__()
        self.checkpoint_state_vars = checkpoint_state_vars
        self.loops_instrumented = 0
        self.assignments_instrumented = 0

    @staticmethod
    def _is_pause_wait_test(test: ast.expr) -> bool:
        """Return True for task-control pause wait guards."""
        if isinstance(test, ast.Name):
            return test.id == 'paused'

        if isinstance(test, ast.Attribute):
            return (
                isinstance(test.value, ast.Name)
                and test.value.id == '_TaskControl'
                and test.attr == 'paused'
            )

        if isinstance(test, ast.Call):
            if isinstance(test.func, ast.Attribute):
                return (
                    isinstance(test.func.value, ast.Name)
                    and test.func.value.id == '_TaskControl'
                    and test.func.attr == 'is_paused'
                )

        return False
    
    def _create_ckpt_update_call(self) -> ast.Try:
        """
        Create AST for: try: _ckpt_update({var1: var1, ...}); except NameError: pass
        """
        keys = [ast.Constant(value=var) for var in self.checkpoint_state_vars]
        values = [ast.Name(id=var, ctx=ast.Load()) for var in self.checkpoint_state_vars]
        state_dict = ast.Dict(keys=keys, values=values)
        call = ast.Call(
            func=ast.Name(id='_ckpt_update', ctx=ast.Load()),
            args=[state_dict],
            keywords=[]
        )
        return ast.Try(
            body=[ast.Expr(value=call)],
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='NameError', ctx=ast.Load()),
                name=None,
                body=[ast.Pass()]
            )],
            orelse=[],
            finalbody=[]
        )
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Inject checkpoint call at end of for loop body, skip if already present."""
        self.generic_visit(node)
        # Idempotency: skip if _ckpt_update already present
        if not any(self._is_ckpt_update_call(stmt) for stmt in node.body):
            ckpt_call = self._create_ckpt_update_call()
            node.body.append(ckpt_call)
            self.loops_instrumented += 1
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Inject checkpoint call at end of while loop body, skip if already present."""
        if self._is_pause_wait_test(node.test):
            return node
        self.generic_visit(node)
        if not any(self._is_ckpt_update_call(stmt) for stmt in node.body):
            ckpt_call = self._create_ckpt_update_call()
            node.body.append(ckpt_call)
            self.loops_instrumented += 1
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Instrument assignments to checkpointed variables with checkpoint call."""
        self.generic_visit(node)
        # Only instrument if assignment is to a tracked variable
        if any(isinstance(t, ast.Name) and t.id in self.checkpoint_state_vars for t in node.targets):
            self.assignments_instrumented += 1
            return [node, self._create_ckpt_update_call()]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        return self.visit_For(node)

    def _is_ckpt_update_call(self, node: ast.stmt) -> bool:
        # Detects if a statement is a _ckpt_update call
        if isinstance(node, ast.Try):
            for stmt in node.body:
                if self._is_ckpt_update_call(stmt):
                    return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            return isinstance(func, ast.Name) and func.id == '_ckpt_update'
        return False


class TaskControlInstrumenter(ast.NodeTransformer):
    """
    AST transformer that injects pause/kill control checks into task code
    
        For Android workers where the UI needs to suspend, kill, or resume a task,
        this transformer injects:
    - Kill check (`if killed: return "killed"`) and pause wait
      (`while paused: time.sleep(0.1)`) at the start of loop bodies
    
    Example transformation:
    
    Before:
        def example_task(n):
            result = 0
            for i in range(n):
                result += i * i
            return result
    
    After:
        def example_task(n):
            result = 0
            for i in range(n):
                if _TaskControl.is_killed():
                    return 'killed'
                while _TaskControl.is_paused():
                    time.sleep(0.1)
                result += i * i
            return result
    """
    
    def __init__(self, checkpoint_state_vars: Optional[List[str]] = None):
        super().__init__()
        self.checkpoint_state_vars = checkpoint_state_vars or []
        self.functions_instrumented = 0
        self.loops_instrumented = 0
    
    def _create_kill_check(self) -> ast.If:
        """Create AST for: if _TaskControl.is_killed(): return 'killed'"""
        body = [ast.Return(value=ast.Constant(value='killed'))]
        return ast.If(
            test=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='_TaskControl', ctx=ast.Load()),
                    attr='is_killed',
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
            body=body,
            orelse=[]
        )
    
    def _create_pause_wait(self) -> ast.While:
        """Create AST for: while _TaskControl.is_paused(): time.sleep(0.1)"""
        return ast.While(
            test=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='_TaskControl', ctx=ast.Load()),
                    attr='is_paused',
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
            body=[
                ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='time', ctx=ast.Load()),
                        attr='sleep',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=0.1)],
                    keywords=[]
                ))
            ],
            orelse=[]
        )
    
    def _create_control_checks(self) -> List[ast.stmt]:
        """Create the kill check + pause wait pair"""
        return [self._create_kill_check(), self._create_pause_wait()]
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function instrumentation count"""
        # First, visit children (to instrument nested loops)
        self.generic_visit(node)
        self.functions_instrumented += 1

        return node
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Inject kill check and pause wait at start of for loop body"""
        # First, visit children
        self.generic_visit(node)
        
        # Insert control checks at the start of the loop body
        control_stmts = self._create_control_checks()
        for i, stmt in enumerate(control_stmts):
            node.body.insert(i, stmt)
        self.loops_instrumented += 1
        
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Inject kill check and pause wait at start of while loop body"""
        # Skip the `while paused:` loops we ourselves inject
        if (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and isinstance(node.test.func.value, ast.Name)
            and node.test.func.value.id == '_TaskControl'
            and node.test.func.attr == 'is_paused'
        ):
            return node
        
        # First, visit children
        self.generic_visit(node)
        
        # Insert control checks at the start of the loop body
        control_stmts = self._create_control_checks()
        for i, stmt in enumerate(control_stmts):
            node.body.insert(i, stmt)
        self.loops_instrumented += 1
        
        return node


class ResumeInstrumenter(ast.NodeTransformer):
    """
    AST transformer for checkpoint resume
    
    Transforms code to:
    1. Replace initial variable assignments with checkpoint values
    2. Adjust loop ranges to start from checkpoint position
    """
    
    def __init__(self, checkpoint_state: Dict[str, Any], checkpoint_state_vars: List[str]):
        """
        Initialize resume instrumenter
        
        Args:
            checkpoint_state: Dictionary with saved checkpoint state
            checkpoint_state_vars: List of variables to restore
        """
        super().__init__()
        self.checkpoint_state = checkpoint_state
        self.checkpoint_state_vars = checkpoint_state_vars
        self.modified_vars = set()
        self.found_main_loop = False
        self.in_function = False
        self.func_name = None
    
    def _create_ast_value(self, value: Any) -> Optional[ast.expr]:
        """Create an AST node for a Python value (no double call, correct None check)"""
        if isinstance(value, (int, float)):
            return ast.Constant(value=value)
        elif isinstance(value, str):
            return ast.Constant(value=value)
        elif isinstance(value, bool):
            return ast.Constant(value=value)
        elif value is None:
            return ast.Constant(value=None)
        elif isinstance(value, list):
            elts = []
            for v in value:
                node = self._create_ast_value(v)
                if node is not None:
                    elts.append(node)
            return ast.List(elts=elts, ctx=ast.Load())
        elif isinstance(value, dict):
            keys = []
            vals = []
            for k, v in value.items():
                key_node = ast.Constant(value=k)
                val_node = self._create_ast_value(v)
                if val_node is not None:
                    keys.append(key_node)
                    vals.append(val_node)
            return ast.Dict(keys=keys, values=vals)
        elif isinstance(value, tuple):
            elts = []
            for v in value:
                node = self._create_ast_value(v)
                if node is not None:
                    elts.append(node)
            return ast.Tuple(elts=elts, ctx=ast.Load())
        return None
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function entry"""
        self.in_function = True
        self.func_name = node.name
        self.generic_visit(node)
        self.in_function = False
        return node
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Replace initial variable assignments with checkpoint values"""
        if not self.in_function:
            return node
        # Check if this is a simple assignment to a checkpointed variable
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if var_name in self.checkpoint_state and var_name not in self.modified_vars:
                value = self.checkpoint_state[var_name]
                new_value = self._create_ast_value(value)
                if new_value is not None:
                    node.value = new_value
                    self.modified_vars.add(var_name)
        return node
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Adjust for loop range to start from checkpointed loop index (not percent progress)"""
        if not self.in_function:
            return node
        # Only handle simple for i in range(n) or range(start, n)
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
            if loop_var in self.checkpoint_state:
                resume_from = self.checkpoint_state[loop_var]
                args = node.iter.args if isinstance(node.iter, ast.Call) else []
                if len(args) == 1:
                    # range(n) -> range(resume_from, n)
                    node.iter.args = [ast.Constant(value=resume_from), args[0]]
                elif len(args) >= 2:
                    # range(start, end) -> range(max(start, resume_from), end)
                    orig_start, end_arg = args[0], args[1]
                    start_expr = ast.Call(
                            func=ast.Name(id='max', ctx=ast.Load()),
                            args=[
                            orig_start,
                            ast.Constant(value=resume_from)
                        ],
                        keywords=[]
                    )
                    node.iter.args = [start_expr, args[1]] + args[2:]
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        return self.visit_For(node)


def instrument_for_mobile(
    func_code: str,
    checkpoint_state_vars: List[str]
) -> Tuple[str, int]:
    """
    Instrument function code for mobile workers without sys.settrace()
    
    Injects explicit _ckpt_update() calls at strategic points so the worker
    can capture state without frame introspection.
    
    Args:
        func_code: Original function source code
        checkpoint_state_vars: List of variable names to capture
        
    Returns:
        Tuple of (instrumented_code, num_loops_instrumented)
    """
    try:
        # Parse the code
        tree = ast.parse(func_code)
        
        # Apply instrumentation
        instrumenter = CheckpointInstrumenter(checkpoint_state_vars)
        tree = instrumenter.visit(tree)
        
        # Fix missing line numbers
        ast.fix_missing_locations(tree)
        
        # Convert back to code
        instrumented_code = ast.unparse(tree)
        
        return instrumented_code, instrumenter.loops_instrumented
        
    except Exception as e:
        logger.warning("[CodeInstrumenter] Instrumentation failed: %s", e)
        return func_code, 0


def prepare_code_for_mobile_resume(
    func_code: str,
    checkpoint_state: Dict[str, Any],
    checkpoint_state_vars: List[str]
) -> str:
    """
    Transform function code to resume from checkpoint on mobile worker
    
    Applies:
    1. Variable injection (replace initial values with checkpoint)
    2. Loop adjustment (start from checkpoint position)
    3. Instrumentation (inject _ckpt_update calls)
    
    Args:
        func_code: Original function source code
        checkpoint_state: Saved checkpoint state
        checkpoint_state_vars: List of variable names
        
    Returns:
        Transformed code ready for resumed execution
    """
    try:
        # Parse the code
        tree = ast.parse(func_code)
        
        # Apply resume transformation
        resume_instrumenter = ResumeInstrumenter(checkpoint_state, checkpoint_state_vars)
        tree = resume_instrumenter.visit(tree)
        
        # Apply checkpoint instrumentation
        checkpoint_instrumenter = CheckpointInstrumenter(checkpoint_state_vars)
        tree = checkpoint_instrumenter.visit(tree)
        
        # Fix missing line numbers
        ast.fix_missing_locations(tree)
        
        # Convert back to code
        return ast.unparse(tree)
        
    except Exception as e:
        logger.warning("[CodeInstrumenter] Resume transformation failed: %s", e)
        return func_code


def generate_mobile_checkpoint_wrapper() -> str:
    """
    Generate the checkpoint wrapper code that must be prepended to instrumented functions
    
    This provides:
    - _MobileCheckpointState: Thread-safe state container
    - _ckpt_update(): Function called by instrumented code
    - _ckpt_get_state(): Function for checkpoint handler to read state
    
    Returns:
        Python code string to prepend to instrumented functions
    """
    return '''
# Mobile checkpoint wrapper - provides explicit state capture
import threading

class _MobileCheckpointState:
    """Thread-safe checkpoint state container for mobile workers"""
    _lock = threading.Lock()
    _state = {}
    
    @classmethod
    def update(cls, state_dict):
        with cls._lock:
            cls._state.update(state_dict)
    
    @classmethod
    def get(cls):
        with cls._lock:
            return dict(cls._state)
    
    @classmethod
    def reset(cls):
        with cls._lock:
            cls._state = {}

def _ckpt_update(state_dict):
    """Called by instrumented code to update checkpoint state"""
    _MobileCheckpointState.update(state_dict)

def _ckpt_get_state():
    """Called by checkpoint handler to read current state"""
    return _MobileCheckpointState.get()
'''


def generate_task_control_wrapper() -> str:
    """
    Generate the task control wrapper code that must be prepended to instrumented functions
    
    This provides:
    - _TaskControl (thread-local paused/killed flags)
    - pause(), resume(), kill() control functions
    
    These flags are checked by the control checks injected by TaskControlInstrumenter.
    
    Returns:
        Python code string to prepend to instrumented functions
    """
    return '''# CROWDio runtime marker: task_control_wrapper_v1
__CROWDIO_TASK_CONTROL_WRAPPER__ = True

import time
import threading

class _TaskControl:
    """Thread-local task control flags for pause/kill."""
    _local = threading.local()

    @classmethod
    def _ensure_state(cls):
        if not hasattr(cls._local, 'paused'):
            cls._local.paused = False
        if not hasattr(cls._local, 'killed'):
            cls._local.killed = False

    @classmethod
    def is_paused(cls):
        cls._ensure_state()
        return cls._local.paused

    @classmethod
    def is_killed(cls):
        cls._ensure_state()
        return cls._local.killed

    @classmethod
    def set_paused(cls, value):
        cls._ensure_state()
        cls._local.paused = bool(value)

    @classmethod
    def set_killed(cls, value):
        cls._ensure_state()
        cls._local.killed = bool(value)


def pause():
    _TaskControl.set_paused(True)


def resume():
    _TaskControl.set_paused(False)


def kill():
    _TaskControl.set_killed(True)
'''


def generate_runtime_wrappers(
    include_checkpoint: bool = True,
    include_task_control: bool = True,
) -> str:
    """Build runtime wrappers in dependency-safe order."""
    parts = []
    if include_checkpoint:
        parts.append(generate_mobile_checkpoint_wrapper())
    if include_task_control:
        parts.append(generate_task_control_wrapper())
    return "\n".join(parts)


def instrument_for_task_control(
    func_code: str,
    checkpoint_state_vars: Optional[List[str]] = None
) -> Tuple[str, int, int]:
    """
    Instrument function code with pause/kill control checks
    
    Injects:
    - `if _TaskControl.is_killed(): return 'killed'` in loops
    - `while _TaskControl.is_paused(): time.sleep(0.1)` in loops
    
    Args:
        func_code: Original function source code
        checkpoint_state_vars: Optional list of checkpoint variable names for
            saving state on kill/pause
        
    Returns:
        Tuple of (instrumented_code, num_functions_instrumented, num_loops_instrumented)
    """
    try:
        tree = ast.parse(func_code)
        
        instrumenter = TaskControlInstrumenter(checkpoint_state_vars=checkpoint_state_vars)
        tree = instrumenter.visit(tree)
        
        ast.fix_missing_locations(tree)
        
        instrumented_code = ast.unparse(tree)
        
        return instrumented_code, instrumenter.functions_instrumented, instrumenter.loops_instrumented
        
    except Exception as e:
        logger.warning("[CodeInstrumenter] Task control instrumentation failed: %s", e)
        return func_code, 0, 0


def prepare_code_with_task_control(
    func_code: str,
    checkpoint_state_vars: Optional[List[str]] = None
) -> str:
    """
    Full pipeline: instrument code with task control checks and prepend wrapper
    
    Args:
        func_code: Original function source code
        checkpoint_state_vars: Optional list of checkpoint variable names
        
    Returns:
        Code with control wrapper prepended and control checks injected
    """
    if "__CROWDIO_TASK_CONTROL_WRAPPER__ = True" in func_code:
        return func_code

    instrumented, num_funcs, num_loops = instrument_for_task_control(
        func_code, checkpoint_state_vars=checkpoint_state_vars
    )
    logger.debug(
        "[CodeInstrumenter] Task control: instrumented %s functions, %s loops",
        num_funcs,
        num_loops,
    )

    wrapper = generate_runtime_wrappers(include_checkpoint=True, include_task_control=True)
    return wrapper + "\n" + instrumented
