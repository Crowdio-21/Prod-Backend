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
    
    def _create_ckpt_update_call(self) -> ast.Expr:
        """
        Create AST for: _ckpt_update({var1: var1, var2: var2, ...})
        """
        # Create dict with checkpoint variables
        keys = [ast.Constant(value=var) for var in self.checkpoint_state_vars]
        values = [ast.Name(id=var, ctx=ast.Load()) for var in self.checkpoint_state_vars]
        
        state_dict = ast.Dict(keys=keys, values=values)
        
        # Create function call: _ckpt_update(state_dict)
        call = ast.Call(
            func=ast.Name(id='_ckpt_update', ctx=ast.Load()),
            args=[state_dict],
            keywords=[]
        )
        
        return ast.Expr(value=call)
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Inject checkpoint call at end of for loop body"""
        # First, visit children
        self.generic_visit(node)
        
        # Add checkpoint call at end of loop body
        ckpt_call = self._create_ckpt_update_call()
        node.body.append(ckpt_call)
        self.loops_instrumented += 1
        
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Inject checkpoint call at end of while loop body"""
        # First, visit children
        self.generic_visit(node)
        
        # Add checkpoint call at end of loop body
        ckpt_call = self._create_ckpt_update_call()
        node.body.append(ckpt_call)
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
        """Create an AST node for a Python value"""
        if isinstance(value, (int, float)):
            return ast.Constant(value=value)
        elif isinstance(value, str):
            return ast.Constant(value=value)
        elif isinstance(value, bool):
            return ast.Constant(value=value)
        elif value is None:
            return ast.Constant(value=None)
        elif isinstance(value, list):
            return ast.List(
                elts=[self._create_ast_value(v) for v in value if self._create_ast_value(v)],
                ctx=ast.Load()
            )
        elif isinstance(value, dict):
            return ast.Dict(
                keys=[ast.Constant(value=k) for k in value.keys()],
                values=[self._create_ast_value(v) for v in value.values()]
            )
        elif isinstance(value, tuple):
            return ast.Tuple(
                elts=[self._create_ast_value(v) for v in value if self._create_ast_value(v)],
                ctx=ast.Load()
            )
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
                # Replace with checkpoint value
                value = self.checkpoint_state[var_name]
                new_value = self._create_ast_value(value)
                if new_value:
                    node.value = new_value
                    self.modified_vars.add(var_name)
        
        return node
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Adjust for loop range to start from checkpoint position"""
        if not self.in_function or self.found_main_loop:
            return node
        
        # Check if this is a for loop with range()
        if isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Name) and func.id == 'range':
                # Get progress from checkpoint
                progress = self.checkpoint_state.get("progress_percent", 0)
                
                if progress > 0:
                    args = node.iter.args
                    
                    if len(args) == 1:
                        # range(n) -> range(_resume_start, n)
                        end_arg = args[0]
                        start_expr = ast.Call(
                            func=ast.Name(id='int', ctx=ast.Load()),
                            args=[ast.BinOp(
                                left=ast.BinOp(
                                    left=ast.Constant(value=progress),
                                    op=ast.Mult(),
                                    right=end_arg
                                ),
                                op=ast.Div(),
                                right=ast.Constant(value=100)
                            )],
                            keywords=[]
                        )
                        node.iter.args = [start_expr, end_arg]
                        self.found_main_loop = True
                    
                    elif len(args) >= 2:
                        # range(start, end) -> range(max(start, calculated), end)
                        orig_start, end_arg = args[0], args[1]
                        start_expr = ast.Call(
                            func=ast.Name(id='max', ctx=ast.Load()),
                            args=[
                                orig_start,
                                ast.Call(
                                    func=ast.Name(id='int', ctx=ast.Load()),
                                    args=[ast.BinOp(
                                        left=ast.BinOp(
                                            left=ast.Constant(value=progress),
                                            op=ast.Mult(),
                                            right=end_arg
                                        ),
                                        op=ast.Div(),
                                        right=ast.Constant(value=100)
                                    )],
                                    keywords=[]
                                )
                            ],
                            keywords=[]
                        )
                        node.iter.args = [start_expr, args[1]] + args[2:]
                        self.found_main_loop = True
        
        self.generic_visit(node)
        return node


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
        print(f"[CodeInstrumenter] Instrumentation failed: {e}")
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
        print(f"[CodeInstrumenter] Resume transformation failed: {e}")
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
