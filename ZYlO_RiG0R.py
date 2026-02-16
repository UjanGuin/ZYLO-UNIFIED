#!/usr/bin/env python3
from zhipuai import ZhipuAI
from openai import OpenAI
import os
import re
import json
import time
import math
import shlex
import base64
import io
import sys
import uuid
import signal
import atexit
import logging
import tempfile
import traceback
import subprocess
from typing import Optional, Tuple, Dict, Any

try:
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import display
except ImportError:
    InteractiveShell = None

from flask import Flask, request, jsonify, Response, send_from_directory, Blueprint, stream_with_context

# ... (API Keys and Config)
# -------------------------
# IPython Execution Engine
# -------------------------
GLOBAL_EXECUTORS = {}

class IPythonExecutor:
    def __init__(self):
        if InteractiveShell is None:
            self.shell = None
            logger.error("IPython not installed. Incremental execution disabled.")
        else:
            # Create a dedicated shell instance for this executor
            self.shell = InteractiveShell()
            # Disable automatic display of the last expression
            self.shell.ast_node_interactivity = "last_expr"
            
        self.execution_history = []
        self.namespace = self.shell.user_ns if self.shell else {}
        
        # Pre-import common research tools
        if self.shell:
            pre_imports = [
                "import math, sys, json, os",
                "import numpy as np",
                "import sympy as sp",
                "from sympy import symbols, simplify, Eq, solve, diff, integrate, limit, expand, factor",
                "from sympy.abc import x, y, z, a, b, c, n",
                "import matplotlib.pyplot as plt",
                "plt.switch_backend('agg')",
                "def is_squarefree(n): return all(v == 1 for v in sp.factorint(n).values())",
                "import sympy; sympy.is_squarefree = is_squarefree; sys.modules['sympy'].is_squarefree = is_squarefree; sp.is_squarefree = is_squarefree"
            ]
            for cmd in pre_imports:
                self.shell.run_cell(cmd)
        
    def execute(self, code: str, mode: str = "exec", timeout: int = 30):
        if not self.shell:
            return {"stdout": "", "stderr": "IPython not available", "success": False, "figures": []}

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        
        try:
            if mode == "eval":
                result_val = self.shell.ev(code)
                output = repr(result_val)
                success = True
                error = ""
            else:
                # run_cell returns an ExecutionResult
                result = self.shell.run_cell(code)
                output = stdout_capture.getvalue()
                error = stderr_capture.getvalue()
                
                # Capture the result value (the Out[...] part) if it exists
                if hasattr(result, 'result') and result.result is not None:
                    if output: output += "\n"
                    output += f"Out: {repr(result.result)}"
                
                success = result.success if hasattr(result, 'success') else True
                if not success and hasattr(result, 'error_in_exec') and result.error_in_exec:
                    error += f"\n{result.error_in_exec}"
                if not success and hasattr(result, 'error_before_exec') and result.error_before_exec:
                    error += f"\n{result.error_before_exec}"
            
            # Capture figures
            figures = []
            try:
                # Check if there are active figures
                if "plt" in self.namespace:
                    plt = self.namespace["plt"]
                    for i in plt.get_fignums():
                        fig = plt.figure(i)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        img_str = base64.b64encode(buf.read()).decode('utf-8')
                        figures.append(img_str)
                        plt.close(fig)
            except Exception as fe:
                logger.error(f"Figure capture error: {fe}")

            res_dict = {
                "stdout": output,
                "stderr": error,
                "success": success,
                "figures": figures
            }
            
            # Store in history
            self.execution_history.append({
                "code": code,
                "output": output,
                "error": error,
                "success": success,
                "figures": figures
            })
            
            return res_dict
            
        except Exception as e:
            return {"stdout": stdout_capture.getvalue(), "stderr": str(e), "success": False, "figures": []}
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
    def get_namespace_snapshot(self):
        # Return current variables (excluding builtins and common imports)
        excluded = ['In', 'Out', 'get_ipython', 'exit', 'quit', 'np', 'plt', 'symbols', 'math', 'sys', 'json', 'os']
        return {k: str(v) for k, v in self.namespace.items() 
                if not k.startswith('_') and k not in excluded}

def get_executor(session_id: str) -> IPythonExecutor:
    if session_id not in GLOBAL_EXECUTORS:
        GLOBAL_EXECUTORS[session_id] = IPythonExecutor()
    return GLOBAL_EXECUTORS[session_id]

# Replace Cerebras import with a safe import guard
try:
    from cerebras.cloud.sdk import Cerebras
except Exception:
    Cerebras = None

# -------------------------
# Configuration (env-first)
# -------------------------
CEREBRAS_KEYS = [
    os.getenv("CEREBRAS_API_KEY", "paste_your_api_key_here"),
    "paste_your_api_key_here"
]
MODEL_NAME = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
# NVIDIA Expert Mode Config
NVIDIA_API_KEY = "paste_your_api_key_here"
EXPERT_MODEL = "z-ai/glm5"

GLM_API_KEY = os.getenv("ZHIPU_API_KEY", "paste_your_api_key_here")
GLM_MODEL = "glm-4.7"
PORT = int(os.getenv("PORT", "5005"))
DATA_DIR = os.getenv("OSS_SERVER_DATA", "./oss_server_data")
SESSION_DIR = os.path.join(DATA_DIR, "sessions")
LOGFILE = os.path.join(DATA_DIR, "server.log")

glm_client = ZhipuAI(api_key=GLM_API_KEY)
expert_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# Research-Grade Phase Temperatures
# ---------------------------------
# 1. Planning: High entropy to explore solution space (avoid CoT loops)
TEMP_PLANNING = 0.65
# 2. Execution: Moderate constraint for tool interaction
TEMP_EXECUTION = 0.0
# 3. Exposition: Strict determinism for final proofs & numbers
TEMP_EXPOSITION = 0.03
# ---------------------------------
# GLM-4.7 Confidence Calibration
# ---------------------------------
CONFIDENCE_BY_VERDICT = {
    "correct": 0.97,     # Independently verified
    "uncertain": 0.65,   # Logical ambiguity / incomplete proof
    "incorrect": 0.25    # Explicit mathematical error
}

if not any(CEREBRAS_KEYS):
    raise RuntimeError("Set CEREBRAS_API_KEY in environment before running this server.")

# Ensure directories exist
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("oss_server")

# -------------------------
# Create Cerebras clients
# -------------------------
if Cerebras is None:
    raise RuntimeError("Cerebras SDK not available. pip install cerebras_cloud_sdk")
cerebras_clients = [Cerebras(api_key=k) for k in CEREBRAS_KEYS]

# -------------------------
# Flask Blueprint
# -------------------------
rigor_bp = Blueprint('rigor', __name__, url_prefix='/rigor', static_folder='static')
# app = Flask(__name__, static_folder="static")

# -------------------------
# System Prompt: Standard
# -------------------------

SYSTEM_PROMPT = r"""
You are GPT-OSS-120B, a Ph.D.-level research assistant in mathematics and physics.
You are a computational research assistant. When solving problems:

1. NEVER generate all code upfront.
2. ALWAYS execute code immediately after writing it.
3. USE the execution output to inform your next step.
4. CONTINUE iterating until the problem is fully solved.
5. Each code block should be small (5-15 lines) and focused.

====================
MODES OF OPERATION
====================

You operate in TWO mutually exclusive modes.

--------------------------------
MODE A — TOOL MODE (MANDATORY)
--------------------------------
Trigger TOOL MODE PREFERENTIALLY when verified computation is required:
- numerical computation
- algebraic manipulation
- calculus evaluation
- statistics
- physics calculations
- data verification

In TOOL MODE:
1. You MAY provide a brief explanation of your plan in natural language before the JSON block.
2. Use a SINGLE JSON block for the tool call.
3. Variables PERSIST. Use previously defined variables.
4. Write SMALL blocks (5-15 lines).
5. Execute IMMEDIATELY after writing.

The valid tool response format is:

{
  "tool": "ipython",
  "code": "print(1+1)",
  "mode": "exec",
  "continue": true
}

Rules:
- "mode": "exec" (default) runs code. "eval" evaluates expression. "display" for plots.
- "continue": set to true if you need more steps. False if this is the final calculation.
- Execute IMMEDIATELY after writing.

--------------------------------
MODE B — DIRECT ANSWER MODE
--------------------------------
Trigger this mode ONLY if the problem is:
- purely theoretical
- proof-based
- conceptual
- contains no calculations requiring verification

In DIRECT ANSWER MODE:

FORMATTING RULES (STRICT):
- Use Markdown for headings, paragraphs, and lists.
- Use LaTeX ONLY for mathematical expressions.
- Inline math: \( ... \)
- Display math: \[ ... \]

ABSOLUTE PROHIBITIONS:
- NEVER place text inside \( \) or \[ \].
- NEVER use \text{}, \textbf{}, \mathrm{} for prose.
- NEVER use aligned, array, cases, enumerate, or itemize.
- NEVER produce a full LaTeX document.

--------------------------------
FINAL OUTPUT RULE
--------------------------------
- TOOL MODE → JSON ONLY
- DIRECT ANSWER MODE → Markdown + LaTeX (math only)

Violating these rules is a critical failure.
""".strip()

# -------------------------
# System Prompt: Expert (NVIDIA GLM-5)
# -------------------------
SYSTEM_PROMPT_EXPERT = r"""
You are GLM-5, an advanced reasoning engine developed by Z-AI and served via NVIDIA. You are a Ph.D.-level expert in mathematics, physics, and computer science.

Your goal is to provide the "Expert" solution:
1.  **Deep Reasoning:** Use your internal reasoning process to explore the problem depth.
2.  **Tool Usage:** If the problem requires calculation, simulation, or verification, you MUST write and execute Python code incrementally.
3.  **Iterative Execution:** 
    - NEVER generate all code upfront.
    - Write a small code block, execute it, see the result, then decide next.
4.  **Output Format:** 
    - You can freely mix natural language and code blocks.
    - If you need to run code, output a SINGLE JSON block for the tool at any point (usually at the start or after some reasoning).
    - If no tool is needed, provide a rigorous, proof-level derivation.

TOOL FORMAT:
```json
{
  "tool": "ipython",
  "code": "print('Hello World')",
  "mode": "exec",
  "continue": true
}
```

Formatting:
- Use LaTeX for math: \( x^2 \) or \[ \int f(x) dx \].
- Be concise but rigorous.
""".strip()

# -------------------------
# Utility helpers
# -------------------------
def safe_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)

def now_iso() -> str:
    return time.strftime("%Y%m%dT%H%M%S")

def save_session_history(session_id: str, data: dict):
    path = os.path.join(SESSION_DIR, f"{safe_filename(session_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_session_history(session_id: str) -> dict:
    path = os.path.join(SESSION_DIR, f"{safe_filename(session_id)}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    base = {
        "id": session_id,
        "created": now_iso(),
        "messages": [{"role":"system","content":SYSTEM_PROMPT}],
    }
    save_session_history(session_id, base)
    return base

def glm_proof_check(question: str, answer: str) -> dict:
    """
    Independent mathematical verification using GLM-4.7.
    Returns verdict, error location, and correction guidance.
    """

    prompt = f"""
You are an independent mathematical proof checker.

PROBLEM:
{question}

PROPOSED FINAL ANSWER:
{answer}

TASK:
1. Verify mathematical correctness.
2. If incorrect, identify:
   - The exact incorrect step
   - Why it is incorrect
   - What the correct approach should be
3. If correct, confirm rigor.

Respond STRICTLY in JSON:
{{
  "verdict": "correct" | "incorrect" | "uncertain",
  "error_step": "<exact faulty step or 'none'>",
  "reason": "<clear explanation>",
  "fix": "<how it should be done correctly>"
}}
"""

    resp = glm_client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700
    )

    try:
        content = resp.choices[0].message.content.strip()
        # Handle markdown JSON blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        return json.loads(content)
    except Exception as e:
        logger.error(f"Recheck Parse Error: {e} | Content: {resp.choices[0].message.content}")
        return {
            "verdict": "uncertain",
            "error_step": "unknown",
            "reason": f"GLM response not parseable: {str(e)}",
            "fix": "unknown"
        }

def fast_fallback(session, user_text):
    logger.warning("FALLBACK → FAST MODE")

    session.append_user(
        "Answer the question directly and clearly. "
        "Do NOT use any computational tools."
    )

    fast_answer = call_model_with_history(
        session.history,
        reasoning_level="low",   # FAST
        temperature=0.7
    )

    visible = fast_answer.strip()
    visible = wrap_pure_math(visible)

    return {
        "reply": visible,
        "fallback": "fast"
    }

def should_fallback_to_fast(text: str) -> bool:
    """
    Detect problems that mix calculation with explanation / reasoning /
    geometry / multi-part questions where forcing tools causes deadlock.
    """

    if not isinstance(text, str):
        return False

    t = text.lower()

    # -------------------------------------------------
    # 1. STRONG EXPLANATION / REASONING SIGNALS
    # -------------------------------------------------
    explanation_signals = [
        # explicit explanation
        "explain", "explain why", "explain how",
        "justify", "reason", "give reasons",
        "comment on", "discuss",

        # proof / derivation
        "prove", "hence prove", "show that",
        "derive", "deduce", "establish",
        "verify that",

        # locus / geometry wording
        "locus", "find the locus",
        "trace the path",

        # conceptual / theory
        "state and explain",
        "interpret", "explain the significance",
        "what do you observe",

        # MCQ reasoning
        "which of the following",
        "which statements are correct",
        "select the correct",
        "multiple correct",

        # step-based phrasing
        "step by step",
        "each step",
        "clearly explain"
    ]

    # -------------------------------------------------
    # 2. GEOMETRY / CONIC / DIAGRAM-HEAVY SIGNALS
    # -------------------------------------------------
    geometry_signals = [
        "parabola", "ellipse", "hyperbola", "circle",
        "focus", "directrix", "axis",
        "chord", "tangent", "normal",
        "midpoint", "vertex",
        "angle subtended", "right angle",
        "conic", "curve",
        "first quadrant", "region bounded",
        "area enclosed",
        "diagram", "figure"
    ]

    # -------------------------------------------------
    # 3. MULTI-PART QUESTION SIGNALS
    # -------------------------------------------------
    multipart_signals = [
        "(i)", "(ii)", "(iii)",
        "part (a)", "part (b)", "part (c)",
        "first find", "then find",
        "hence", "therefore",
        "and hence",
        "also find"
    ]

    # -------------------------------------------------
    # 4. SCHOOL / JEE / OLYMPIAD STYLE SIGNALS
    # -------------------------------------------------
    exam_style_signals = [
        "jee", "advanced", "main",
        "board exam", "olympiad",
        "assertion", "reason",
        "match the following",
        "column i", "column ii"
    ]

    # -------------------------------------------------
    # 5. CALCULATION HEURISTIC (already exists)
    # -------------------------------------------------
    needs_calc = requires_tool(text)

    # -------------------------------------------------
    # 6. FINAL DECISION LOGIC
    # -------------------------------------------------
    has_explanation = any(k in t for k in explanation_signals)
    has_geometry = any(k in t for k in geometry_signals)
    has_multipart = any(k in t for k in multipart_signals)
    has_exam_style = any(k in t for k in exam_style_signals)

    # 🔑 Core rule:
    # If it needs calculation AND any reasoning-heavy signal is present,
    # prefer FAST to avoid tool deadlock.
    return needs_calc and (
        has_explanation
        or has_geometry
        or has_multipart
        or has_exam_style
    )


# -------------------------
# Sandboxed Python executor
# -------------------------
def run_python_sandbox(code: str, timeout_s: int = 15) -> Tuple[str, str, int]:
    """
    Runs 'code' in a temporary python file with resource limits.
    """
    import resource
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        fname = f.name
        f.write("# -*- coding: utf-8 -*-\n")
        f.write("import json,sys,math\n")
        f.write(code)

    def preexec():
        try:
            # CPU Limit
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s + 1))
            # Memory Limit (1GB)
            mem = 1024 * 1024 * 1024 
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except: pass

    proc = subprocess.Popen(
        ["python3", fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec
    )
    try:
        out, err = proc.communicate(timeout=timeout_s + 2)
        return out, err, proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return "", "TimeoutExpired", -9
    except Exception as e:
        return "", str(e), -8

# -------------------------
# SymPy runner (from ooo.py)
# -------------------------
def run_sympy_check(expr_code: str, timeout_s: int = 8) -> Tuple[str, str, int]:
    """
    Wraps code in a SymPy safety harness to catch errors and format output as JSON.
    """
    wrapper = f"""
import json
import traceback
from fractions import Fraction
from sympy import symbols, simplify, Eq, solve, diff, integrate, limit, expand, factor
from sympy.abc import x, y, z, a, b, c, n

try:
    # User code goes here
{expr_code}
    # Implicitly assume user defined 'result'
    if 'result' not in locals():
        print(json.dumps({{"status": "error", "error": "No 'result' variable defined in scope"}}))
    else:
        out = {{}}
        out['status'] = 'ok'
        out['result'] = str(result) # Convert sympy objects to string
        print(json.dumps(out))
except Exception as e:
    print(json.dumps({{"status":"error", "error": str(e), "trace": traceback.format_exc()}}))
"""
    return run_python_sandbox(wrapper, timeout_s=timeout_s)

# -------------------------
# Model Call Wrappers
# -------------------------
def call_model_with_history(
    history: list, 
    reasoning_level: str = "high", 
    temperature: float = 0.5
) -> str:
    """
    Executes the API call with strict phase-aware temperature and reasoning control.
    Retries with backup keys on failure.
    """
    if len(history) > 80:
        history = [history[0]] + history[-60:]
    
    last_error = None
    for i, client in enumerate(cerebras_clients):
        try:
            logger.info(f"API Call (Key {i+1}) | Reasoning: {reasoning_level} | Temp: {temperature}")
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                max_tokens=4096,
                temperature=temperature,
                reasoning_effort=reasoning_level
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            logger.error(f"Cerebras API Error (Key {i+1}): {e}")
            if i < len(cerebras_clients) - 1:
                logger.info("Retrying with backup key...")
            continue
    
    raise last_error

def call_model_stream(
    history: list, 
    reasoning_level: str = "high", 
    temperature: float = 0.5
):
    """
    Executes the API call with streaming.
    Retries with backup keys on failure.
    """
    if len(history) > 80:
        history = [history[0]] + history[-60:]
    
    last_error = None
    for i, client in enumerate(cerebras_clients):
        try:
            logger.info(f"API Stream (Key {i+1}) | Reasoning: {reasoning_level} | Temp: {temperature}")
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                max_tokens=4096,
                temperature=temperature,
                reasoning_effort=reasoning_level,
                stream=True
            )
        except Exception as e:
            last_error = e
            logger.error(f"Cerebras API Stream Error (Key {i+1}): {e}")
            if i < len(cerebras_clients) - 1:
                logger.info("Retrying with backup key...")
            continue
            
    raise last_error

def call_expert_model(history: list, temperature: float = 1.0) -> str:
    """
    Calls the Expert Model (NVIDIA GLM5).
    """
    if len(history) > 80:
        history = [history[0]] + history[-60:]

    logger.info(f"EXPERT API Call | Model: {EXPERT_MODEL} | Temp: {temperature}")
    
    resp = expert_client.chat.completions.create(
        model=EXPERT_MODEL,
        messages=history,
        max_tokens=16384,
        temperature=temperature,
        top_p=1,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": True}
        },
        timeout=120
    )
    return resp.choices[0].message.content

def call_expert_model_stream(history: list, temp: float = 1.0):
    """
    Calls the Expert Model with streaming.
    """
    if len(history) > 80:
        history = [history[0]] + history[-60:]

    logger.info(f"EXPERT API Stream | Model: {EXPERT_MODEL} | Temp: {temp}")
    
    return expert_client.chat.completions.create(
        model=EXPERT_MODEL,
        messages=history,
        max_tokens=16384,
        temperature=temp,
        top_p=1,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": True}
        },
        stream=True,
        timeout=120
    )

# -------------------------
# Tool Parsing
# -------------------------
def parse_tool_instruction(text: str):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # Hard guard: must start with JSON
    if not text.startswith("{"):
        # EXPERT MODE: Sometimes wraps in ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            return None

    try:
        obj = json.loads(text)
        if obj.get("tool") == "python":
            return obj
    except Exception:
        return None

    return None


def extract_ipython_instruction(text: str):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # 1. Try to find Markdown JSON block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if obj.get("tool") == "ipython":
                return obj
        except:
            pass

    # 2. Try to find any JSON object that looks like a tool call
    start_indices = [m.start() for m in re.finditer(r'\{', text)]
    for start in start_indices:
        balance = 0
        for i in range(start, len(text)):
            char = text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
            if balance == 0:
                candidate = text[start:i+1]
                if '"tool"' in candidate and '"ipython"' in candidate: 
                    try:
                        obj = json.loads(candidate)
                        if obj.get("tool") == "ipython":
                            return obj
                    except:
                        pass
                break
                
    return None

# -------------------------
# Session Manager
# -------------------------
class ResearchSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self._data = load_session_history(self.id)
        self.history = self._data.get("messages", [])
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role":"system","content":SYSTEM_PROMPT})
        
        # Attach executor
        self.executor = get_executor(self.id)

    def append_user(self, text: str):
        self.history.append({"role":"user","content":text})
        self._save()

    def append_assistant(self, text: str):
        self.history.append({"role":"assistant","content":text})
        self._save()
    
    def set_system_prompt(self, prompt: str):
        """Overrides the system prompt for the current turn if needed."""
        if self.history and self.history[0].get("role") == "system":
            self.history[0]["content"] = prompt
        else:
            self.history.insert(0, {"role": "system", "content": prompt})

    def _save(self):
        self._data["messages"] = self.history
        save_session_history(self.id, self._data)

# -------------------------
# Core Logic: Phase-Aware Handling
# -------------------------
def wrap_pure_math(text: str) -> str:
    if not isinstance(text, str):
        return "_[Invalid response]_"

    stripped = text.strip()
    if not stripped:
        return "_[Empty response]_"

    # Detect pure LaTeX (no explanatory text)
    if stripped.startswith("\\[") or stripped.startswith("\\("):
        return "### Result\n\n" + text

    return text

def strip_tool_json(text: str) -> str:
    if not isinstance(text, str):
        return ""

    stripped = text.strip()

    # If model leaked tool JSON, suppress it entirely
    # Check for ```json ... ``` wrapper too
    stripped = re.sub(r"```json\s*\{.*?\}\s*```", "", stripped, flags=re.DOTALL)
    if stripped.startswith("{") and '"tool"' in stripped:
        return ""

    return text

def requires_tool(text: str) -> bool:
    """
    Refined heuristic detector for problems that REQUIRE
    verified computation or numerical confirmation.
    """

    if not isinstance(text, str):
        return False

    t = text.lower()

    # -------------------------------------------------
    # 1. STRONG NUMERIC / COMPUTATIONAL TRIGGERS
    # -------------------------------------------------
    numeric_keywords = [
        "compute", "calculate", "evaluate", "find value",
        "numerically", "approximate",
        "verify", "verification", "check numerically",
        "solve", "solution", "roots", "zeroes",
        "determinant", "det", "eigenvalue", "eigenvalues",
        "matrix", "polynomial", "characteristic",
        "integral", "differentiate", "derivative",
        "summation", "probability", "expectation", "variance",
        "trajectory", "dynamics",
        "wavefunction", "schrodinger",
        "numerical method", "newton method",
        "iteration", "convergence",
        "nullspace", "least squares",
        "extrema", "maxima", "minima", "optimization",
        "jacobian", "hessian", "gradient", "divergence", "curl",
        "laplacian", "taylor", "maclaurin", "fourier series",
        "hyperbola", "ellipse", "parabola", "tangent", "foci", "directrix"
    ]

    if any(k in t for k in numeric_keywords):
        return True

    # -------------------------------------------------
    # 2. SYMBOLIC / STRUCTURAL SIGNALS (More specific)
    # -------------------------------------------------
    symbolic_signals = [
        "==", "**", "[[", "]]", "dx", "dt", "∫", "∑", "∏", 
        "\\lim", "\\sqrt", "\\int", "\\sum"
    ]

    if any(s in t for s in symbolic_signals):
        return True

    # -------------------------------------------------
    # 3. DEGREE / SCALE HEURISTICS
    # -------------------------------------------------
    if re.search(r"\bdegree\s*[4-9]\b", t):
        return True

    if re.search(r"x\^\d{2,}", t):  # high-degree polynomial
        return True

    # -------------------------------------------------
    # 4. LARGE NUMERIC DATA / MATH EXPRESSIONS
    # -------------------------------------------------
    if re.search(r"\d+\s*[+\-*/=]\s*\d+", t): # basic arithmetic 1+1
        return True

    if re.search(r"\b\d+\.\d+\b", t):  # decimals
        return True

    return False


def requires_numeric_tool(text: str) -> bool:
    keywords = [
        "calculate", "compute", "numerical", "value of",
        "evaluate", "data", "probability", "statistics",
        "mean", "variance", "time taken", "integrate numerically"
    ]
    return any(k in text.lower() for k in keywords)


def allows_analytic_solution(text: str) -> bool:
    keywords = [
        "prove", "show that", "derive", "find the coefficient",
        "rolling", "without slipping", "using conservation",
        "by symmetry", "angular momentum"
    ]
    return any(k in text.lower() for k in keywords)


def has_visible_content(text: str) -> bool:
    if not isinstance(text, str):
        return False

    # Remove whitespace and math blocks
    stripped = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    stripped = re.sub(r"\\\(.*?\\\)", "", stripped, flags=re.DOTALL)

    return bool(stripped.strip())

def force_tool_result(tool_output: dict) -> str:
    """
    Guarantees visible output from tool execution.
    Used ONLY if model exposition is empty or generic.
    """
    if not tool_output:
        return ""

    stdout = (tool_output.get("stdout") or "").strip()
    stderr = (tool_output.get("stderr") or "").strip()

    if stdout:
        return (
            "### Result\n\n"
            "**Computed values:**\n\n"
            f"```\n{stdout}\n```"
        )

    if stderr:
        return (
            "### Result\n\n"
            "**Tool reported:**\n\n"
            f"```\n{stderr}\n```"
        )

    return ""

def handle_message(
    session: ResearchSession,
    user_text: str,
    reasoning_level: str,
    recheck: bool = False
):
    """
    Advanced iterative message handler (Unified Loop Version):
    Yields JSON events for the UI.
    """
    session.append_user(user_text)

    # Reasoning Config
    is_expert = (reasoning_level == "expert")
    if is_expert:
        session.set_system_prompt(SYSTEM_PROMPT_EXPERT)
        call_fn = call_expert_model_stream
    else:
        session.set_system_prompt(SYSTEM_PROMPT)
        call_fn = lambda hist, temp=0.5: call_model_stream(hist, reasoning_level=reasoning_level, temperature=temp)

    def process_stream(stream_obj, step_label=None):
        full_text = ""
        for chunk in stream_obj:
            if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            
            # NVIDIA GLM5 Reasoning - Yield as dedicated 'thinking' type
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "thinking", "content": reasoning}
            
            if delta.content:
                content = delta.content
                full_text += content
                yield {"type": "delta", "content": content, "step": step_label}
        return full_text

    # --- Unified Iterative Loop ---
    max_iterations = 10
    tool_outputs = []
    cont = True # Track if model wants to continue tool usage
    synthesis_started = False
    
    # Pre-loop logic: if it needs a tool, encourage it.
    if requires_tool(user_text):
        session.history.append({
            "role": "system",
            "content": "This problem likely requires computation or derivation. You can use the 'ipython' tool for verification, or provide a reasoned explanation if information is missing."
        })

    for i in range(max_iterations):
        if i == 0:
            if requires_tool(user_text):
                step_label = "Reasoning"
                yield {"type": "status", "content": "Analyzing request..."}
            else:
                step_label = "Final Answer"
        else:
            # If the previous tool call said 'continue: false', this is the final synthesis.
            if not cont:
                if synthesis_started:
                    break
                step_label = "Final Answer"
                yield {"type": "status", "content": "Synthesizing final answer..."}
                synthesis_started = True
            else:
                step_label = f"Step {i} Reasoning"
                yield {"type": "status", "content": f"Analyzing results (Step {i})..."}

        stream = call_fn(session.history, temp=TEMP_PLANNING if i == 0 else TEMP_EXECUTION)
        assistant_raw = ""
        
        # Stream the thought/plan
        for event in process_stream(stream, step_label):
            if isinstance(event, dict) and event["type"] == "delta":
                assistant_raw += event["content"]
            yield event
        
        # Look for tool instruction in the response
        tool_inst = extract_ipython_instruction(assistant_raw)
        
        if tool_inst:
            code = tool_inst.get("code", "")
            mode = tool_inst.get("mode", "exec")
            
            # Update continuation flag
            cont = tool_inst.get("continue", True) 
            
            logger.info(f"ITERATION {i+1} | Mode: {mode}")
            yield {"type": "tool_call", "code": code, "mode": mode, "step": i+1}
            
            result = session.executor.execute(code, mode=mode)
            tool_outputs.append(result)
            yield {"type": "tool_output", "output": result, "step": i+1}
            
            # Prepare feedback for next turn
            session.append_assistant(assistant_raw)
            stdout_vis = result["stdout"][:8000]
            stderr_vis = result["stderr"][:2000]
            feedback = f"EXECUTION OUTPUT (Step {i+1}):\nSTDOUT:\n{stdout_vis}\nSTDERR:\n{stderr_vis}"
            if result.get("figures"):
                feedback += f"\n[Captured {len(result['figures'])} matplotlib figures]"
            
            snapshot = session.executor.get_namespace_snapshot()
            if snapshot:
                feedback += f"\nCURRENT VARIABLES: {json.dumps(snapshot)}"

            session.history.append({"role": "system", "content": feedback})
            
            # If model explicitly said continue: false, we've fulfilled its tool needs.
            # We'll continue the loop one more time to let it see the results and provide the Final Answer.
            if not cont:
                logger.info("Model signaled end of tool chain. Moving to synthesis.")
        else:
            # No tool call found -> This was the final answer or a direct response
            # Append to history so recheck and future turns can see it.
            session.append_assistant(assistant_raw)
            break

    # Optional Recheck
    if recheck:
        yield {"type": "status", "content": "Performing independent recheck..."}
        # Get final text from history
        final_text = ""
        for m in reversed(session.history):
            if m["role"] == "assistant":
                final_text = m["content"]
                break
        
        if final_text:
            proof_check = glm_proof_check(user_text, final_text)
            verdict = proof_check.get("verdict", "uncertain")
            confidence = CONFIDENCE_BY_VERDICT.get(verdict, 0.6)
            reply_recheck = f"\n\n---\n**Proof Recheck:** `{verdict.upper()}`\n**Confidence:** {confidence}"
            yield {"type": "delta", "content": reply_recheck, "step": "Final Answer"}

    yield {"type": "done"}




# -------------------------------------------------------
# Canonical Math Normalizer (MathJax-safe, Regex-limited)
# -------------------------------------------------------

INLINE = r"\\\((.*?)\\\)"
BLOCK  = r"\\\[(.*?)\\\]"

RAW_MATH_HINT = re.compile(
    r"(?<!\\)"
    r"("
    r"[a-zA-Z]\s*\^\s*\{?\d+\}?"          # x^2, y^{2}
    r"|\\frac\s*\{.*?\}\s*\{.*?\}"        # \frac{a}{b}
    r"|\\sqrt\s*\{.*?\}"                  # \sqrt{3}
    r"|[a-zA-Z0-9]+\s*=\s*[a-zA-Z0-9]+"   # y^2 = x
    r"|[a-zA-Z]+_[a-zA-Z0-9]+"            # F_AB
    r")"
)

def tool_exposition_invalid(answer: str, tool_stdout: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return True

    low_info_phrases = [
        "computed successfully",
        "computation completed",
        "result computed",
        "calculation complete",
        "done",
        "finished"
    ]

    ans = answer.lower()

    # 1️⃣ Generic acknowledgments
    if any(p in ans for p in low_info_phrases):
        return True

    # 2️⃣ Tool produced numbers but answer has none
    has_numbers_in_tool = any(ch.isdigit() for ch in tool_stdout)
    has_numbers_in_answer = any(ch.isdigit() for ch in answer)

    if has_numbers_in_tool and not has_numbers_in_answer:
        return True

    # 3️⃣ Roots / eigenvalues / determinant mentioned but not stated
    critical_words = ["root", "roots", "eigen", "determinant", "value"]
    if any(w in ans for w in critical_words) and not has_numbers_in_answer:
        return True

    return False


def strip_text_from_math_blocks(text: str) -> str:
    if not isinstance(text, str):
        return ""

    extracted_text = []

    def clean_math_block(match):
        block = match.group(1)

        # If this is prose disguised as math → extract text
        if re.search(r"\\text", block):
            # Remove \text{} wrappers
            plain = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", block)
            plain = re.sub(r"\\text\{([^}]*)\}", r"\1", plain)
            extracted_text.append(plain)
            return ""  # remove math wrapper

        # Otherwise keep real math
        return f"\\[{block}\\]"

    text = re.sub(
        r"\\\[(.*?)\\\]",
        clean_math_block,
        text,
        flags=re.DOTALL
    )

    # Append extracted prose back
    if extracted_text:
        text = "\n\n".join(extracted_text) + "\n\n" + text

    return text

def is_proof_only(text: str) -> bool:
    if not isinstance(text, str):
        return False

    # Indicators of proof-style reasoning
    keywords = [
        "there exists", "suppose", "assume", "by contradiction",
        "by mean value theorem", "hence", "therefore", "let"
    ]

    return any(k in text.lower() for k in keywords)


def strip_latex_lists(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove enumerate/itemize blocks entirely
    text = re.sub(
        r"\\begin\{(enumerate|itemize)\}[\s\S]*?\\end\{\1\}",
        "",
        text
    )

    # Remove stray \item
    text = re.sub(r"\\item\s*", "- ", text)

    # Remove alignment markers that break MathJax
    text = re.sub(r"(?<!\\)&", "", text)

    return text

def strip_latex_math_layout(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove aligned / array environments (used incorrectly by LLMs)
    text = re.sub(
        r"\\begin\{(aligned|array|cases)\}[\s\S]*?\\end\{\1\}",
        "",
        text
    )

    # Remove standalone \text{} used outside inline math
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)

    # Remove spacing commands that pollute output
    text = re.sub(r"\\(quad|qquad|hfill|vspace\{.*?\})", "", text)

    return text

def normalize_math(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # -----------------------------
    # 1. Protect valid LaTeX blocks
    # -----------------------------
    protected = []

    def _protect(m):
        protected.append(m.group(0))
        return f"@@MATH{len(protected)-1}@@"

    # Protect MathJax-style blocks first
    text = re.sub(BLOCK, _protect, text, flags=re.S)
    text = re.sub(INLINE, _protect, text, flags=re.S)

    # -----------------------------
    # 1.5 EXTRA PROTECTION
    # Protect common environments if they appear
    # -----------------------------
    text = re.sub(
        r"\\begin\{(aligned|array|cases|matrix|pmatrix|bmatrix)\}.*?\\end\{\1\}",
        _protect,
        text,
        flags=re.S
    )

    # -----------------------------
    # 2. Fix common model mistakes
    # -----------------------------
    # \mathbf{F}{AB} → \mathbf{F}_{AB}
    text = re.sub(
        r"\\mathbf\{([A-Za-z])\}\s*\{([A-Za-z]+)\}",
        r"\\mathbf{\1}_{\2}",
        text
    )

    # \vec{v}{t} → \vec{v}(t)
    text = re.sub(
        r"\\vec\{([A-Za-z])\}\s*\{([A-Za-z]+)\}",
        r"\\vec{\1}(\2)",
        text
    )

    # -----------------------------
    # 3. Wrap raw math in \( ... \)
    # (ONLY when math symbols are detected)
    # -----------------------------
    def wrap_raw_math(m):
        expr = m.group(1).strip()
        return rf"\( {expr} \)"

    text = RAW_MATH_HINT.sub(wrap_raw_math, text)

    # -----------------------------
    # 3.5 EXTRA RAW-MATH CATCH
    # Detect orphaned equations like: a = b + c
    # -----------------------------
    text = re.sub(
        r"(?<!\\)\b([A-Za-z]\w*\s*=\s*[^.\n]+)",
        lambda m: rf"\( {m.group(1).strip()} \)",
        text
    )

    # -----------------------------
    # 4. Restore protected math
    # -----------------------------
    for i, m in enumerate(protected):
        text = text.replace(f"@@MATH{i}@@", m)

    # -----------------------------
    # 5. De-duplicate wrappers
    # -----------------------------
    text = re.sub(r"\\\(\s*\\\(", r"\\(", text)
    text = re.sub(r"\\\)\s*\\\)", r"\\)", text)
    text = re.sub(r"\\\[\s*\\\[", r"\\[", text)
    text = re.sub(r"\\\]\s*\\\]", r"\\]", text)

    # -----------------------------
    # 6. Final safety cleanup
    # -----------------------------
    # Remove empty math blocks
    text = re.sub(r"\\\(\s*\\\)", "", text)
    text = re.sub(r"\\\[\s*\\\]", "", text)

    return text.strip()


# -------------------------
# Routes
# -------------------------
@rigor_bp.route("/")
def home():
    return Response(HTML_PAGE, mimetype="text/html")

@rigor_bp.route("/chat", methods=["POST"])
def chat_route():
    try:
        data = request.json or {}

        sess_id = (
            data.get("session_id")
            or request.cookies.get("session_id")
            or str(uuid.uuid4())
        )

        msg = data.get("message", "").strip()

        # UI controls
        r_level = data.get("reasoning_level", "high")
        recheck = bool(data.get("recheck", False))

        if not msg:
            return jsonify({"error": "empty"}), 400

        sess = ResearchSession(sess_id)

        def stream_response():
            # Initial cookie or ID sync? We'll just yield sess_id if needed, but SSE usually uses response headers.
            # But we can't set cookies in a stream easily without a wrapper.
            # Flask's Response handles headers.
            
            generator = handle_message(
                sess,
                msg,
                r_level,
                recheck=recheck
            )
            
            for event in generator:
                yield f"data: {json.dumps(event)}\n\n"

        return Response(
            stream_with_context(stream_response()),
            mimetype="text/event-stream",
            headers={"X-Session-ID": sess_id}
        )

    except Exception as e:
        logger.exception("Chat Error")
        return jsonify({"error": str(e)}), 500


# -------------------------
# UI
# -------------------------

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ZYLO-RIGOR | Research Engine</title>

    <!-- Responsive -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <!-- Markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

    <!-- MathJax -->
    <script>
        window.MathJax = {
            tex: { 
                inlineMath: [['\\(', '\\)'], ['$', '$']], 
                displayMath: [['\\[', '\\]'], ['$$', '$$']] 
            },
            svg: { fontCache: 'global' }
        };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

    <style>
        :root {
            --bg-dark: #050509;
            --bg-deep: #020205;
            --panel: rgba(15, 23, 42, 0.6);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.15);
            --font-main: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

        body {
            margin: 0;
            background: radial-gradient(circle at top center, #111425 0%, var(--bg-deep) 80%);
            color: var(--text-main);
            font-family: var(--font-main);
            height: 100dvh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* ---------- SPLASH SCREEN ---------- */
        #splash {
            position: fixed;
            inset: 0;
            background: #020205;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: opacity 0.8s ease, visibility 0.8s;
        }

        .splash-content {
            text-align: center;
            animation: slideUpFade 1.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .splash-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }

        .splash-sub {
            color: var(--text-muted);
            font-size: 1rem;
            opacity: 0.8;
        }

        @keyframes slideUpFade {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* ---------- HEADER ---------- */
        header {
            padding: 1rem 1.5rem;
            background: rgba(5, 5, 9, 0.8);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            z-index: 10;
        }

        .brand {
            font-weight: 700;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.5px;
        }

        .brand span { color: var(--accent); }

        .controls {
            display: flex;
            gap: 0.8rem;
            align-items: center;
        }

        /* --- Styled pill appearance kept, native select replaced by custom component --- */
        .pill-select, .toggle {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-family: var(--font-main);
            cursor: pointer;
            transition: all 0.2s ease;
            -webkit-appearance: none;
            appearance: none;
        }

        .pill-select:hover, .toggle:hover {
            border-color: var(--border-highlight);
            background: rgba(255,255,255,0.06);
        }

        .toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            user-select: none;
        }

        .toggle input {
            accent-color: var(--accent);
            cursor: pointer;
            width: 1rem;
            height: 1rem;
        }

        /* ---------- CUSTOM SELECT (replaces native <select>) ---------- */
        .custom-select {
            position: relative;
            display: inline-block;
            min-width: 120px;
            font-family: var(--font-main);
        }

        .select-btn {
            display: inline-flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            width: 140px;
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.03);
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.85rem;
            text-align: left;
        }

        .select-btn:focus {
            outline: none;
            box-shadow: 0 6px 16px rgba(0,0,0,0.35);
            color: var(--text-main);
            border-color: var(--border-highlight);
        }

        .select-btn .caret {
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 7px solid var(--text-muted);
            opacity: 0.9;
        }

        .select-btn.open .caret {
            transform: rotate(180deg);
            border-top-color: var(--text-main);
        }

        .select-options {
            position: absolute;
            right: 0;
            left: 0;
            margin-top: 8px;
            background: linear-gradient(180deg, rgba(10,10,12,0.98), rgba(8,8,10,0.98));
            border: 1px solid var(--border);
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            list-style: none;
            padding: 6px;
            display: none;
            z-index: 50;
            max-height: 240px;
            overflow-y: auto;
        }

        .select-options.open { display: block; }

        .select-options li {
            padding: 10px 12px;
            margin: 4px 0;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-main);
            font-weight: 600;
            font-size: 0.95rem;
        }

        .select-options li:hover,
        .select-options li:focus {
            background: rgba(99,102,241,0.08);
            color: var(--text-main);
            outline: none;
        }

        .select-options li.selected {
            background: rgba(99,102,241,0.12);
            color: var(--accent);
            box-shadow: inset 0 0 0 1px rgba(99,102,241,0.06);
        }

        /* ---------- CHAT AREA ---------- */
        #chat {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            scroll-behavior: smooth;
        }

        .msg {
            max-width: 800px;
            margin: 0 auto 1.5rem auto;
            display: flex;
            flex-direction: column;
            animation: msgEnter 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            opacity: 0;
            transform: translateY(10px);
        }

        @keyframes msgEnter {
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 6px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* AI label — prominent */
        .assistant .msg-label { 
            color: var(--accent);
            font-size: 1.35rem;
            font-weight: 900;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        /* hide the user label (no "You") */
        .user .msg-label {
            display: none;
        }

        .msg-content {
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            line-height: 1.7;
            font-size: 0.95rem;
            position: relative;
            word-wrap: break-word;
        }

        .user { align-items: flex-end; }
        .user .msg-content {
            background: linear-gradient(135deg, #2e3558 0%, #1e293b 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .assistant { align-items: flex-start; }
        .assistant .msg-content {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-bottom-left-radius: 4px;
        }

        /* Tool Output Box */
        .tool-output {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            background: #09090b;
            color: #4ade80;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 0.8rem;
            border: 1px solid #222;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* Thinking Section */
        .thinking-container {
            margin-bottom: 1rem;
            border-left: 2px solid var(--accent);
            padding-left: 1rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 0 8px 8px 0;
            padding-top: 8px;
            padding-bottom: 8px;
        }

        .thinking-header {
            font-size: 0.8rem;
            color: var(--accent);
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .thinking-content {
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-style: italic;
            line-height: 1.5;
        }

        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid rgba(99, 102, 241, 0.2);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .step-label {
            display: inline-block;
            background: var(--accent);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 800;
            margin-top: 12px;
            margin-bottom: 4px;
        }

        /* ---------- TYPING INDICATOR ---------- */
        #typing-indicator {
            display: none;
            max-width: 800px;
            margin: 0 auto 1rem auto;
            padding-left: 0.5rem;
        }
        
        .typing-bubble {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            padding: 12px 20px;
            border-radius: 16px;
            border-bottom-left-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .typing-text {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-right: 8px;
            font-weight: 500;
        }

        .dot {
            width: 6px;
            height: 6px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .dot:nth-child(2) { animation-delay: -0.32s; }
        .dot:nth-child(3) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        /* ---------- INPUT AREA ---------- */
        #input-area {
            padding: 1.2rem;
            background: rgba(5,5,9,0.9);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            position: relative;
            z-index: 20;
        }

        #prompt {
            flex: 1;
            padding: 14px 18px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(20, 20, 25, 0.6);
            color: var(--text-main);
            font-size: 1rem;
            font-family: var(--font-main);
            outline: none;
            transition: all 0.2s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }

        #prompt:focus {
            border-color: var(--accent);
            background: rgba(20, 20, 25, 0.9);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        button {
            padding: 0 24px;
            border-radius: 12px;
            border: none;
            background: var(--accent);
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.1s ease, filter 0.2s;
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        button:active { transform: scale(0.96); }
        button:hover { filter: brightness(1.1); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }

        /* ---------- MOBILE ---------- */
        @media (max-width: 600px) {
            header { padding: 0.8rem 1rem; }
            .brand { font-size: 1rem; }
            /* show our custom select's button but reduce size */
            .select-btn { width: 120px; font-size: 0.82rem; padding: 8px 10px; }
            .msg-content { padding: 1rem 1.2rem; font-size: 0.9rem; }
            #input-area { padding: 1rem; }
            #prompt { font-size: 16px; } /* Prevent iOS zoom */
            .assistant .msg-label { font-size: 1.1rem; } /* scale AI label on small screens */
        }
    </style>
</head>

<body>

<div id="splash">
    <div class="splash-content">
        <div class="splash-title">Welcome!</div>
        <div class="splash-sub">Initializing Research Environment...</div>
    </div>
</div>

<header>
    <div class="brand">⚡ ZYLO <span>RIGOR</span></div>

    <div class="controls">
        <!-- Hidden input preserves existing JS expectations (reasoningSelect.value) -->
        <input type="hidden" id="reasoning" value="high">

        <!-- Custom visually styled select -->
        <div class="custom-select" id="customReasoning" tabindex="0" aria-haspopup="listbox" aria-expanded="false">
            <button type="button" class="select-btn" id="reasoningBtn" aria-haspopup="listbox" aria-expanded="false">
                Research
                <span class="caret"></span>
            </button>
            <ul class="select-options" id="reasoningOptions" role="listbox" aria-labelledby="reasoningBtn">
                <li role="option" data-value="low">Fast</li>
                <li role="option" data-value="medium">Balanced</li>
                <li role="option" data-value="high" class="selected">Research</li>
                <li role="option" data-value="expert">Expert</li>
            </ul>
        </div>

        <label class="toggle">
            <input type="checkbox" id="recheckToggle">
            <span>Recheck</span>
        </label>
    </div>
</header>

<div id="chat"></div>

<!-- Typing Indicator -->
<div id="typing-indicator">
    <div class="typing-bubble">
        <span class="typing-text">Typing</span>
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
    </div>
</div>

<div id="input-area">
    <input id="prompt" placeholder="Ask a complex question..." autocomplete="off">
    <button onclick="ask()" id="sendBtn">Send</button>
</div>

<script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('prompt');
    const sendBtn = document.getElementById('sendBtn');

    // reasoningSelect now a hidden input (keeps existing code)
    const reasoningSelect = document.getElementById('reasoning');
    const recheckToggle = document.getElementById('recheckToggle');
    const typingIndicator = document.getElementById('typing-indicator');

    // Custom select elements
    const reasoningBtn = document.getElementById('reasoningBtn');
    const reasoningOptions = document.getElementById('reasoningOptions');
    const customReasoning = document.getElementById('customReasoning');

    // open/close helper
    function closeReasoning() {
        reasoningOptions.classList.remove('open');
        reasoningBtn.classList.remove('open');
        customReasoning.setAttribute('aria-expanded', 'false');
        reasoningBtn.setAttribute('aria-expanded', 'false');
    }
    function openReasoning() {
        reasoningOptions.classList.add('open');
        reasoningBtn.classList.add('open');
        customReasoning.setAttribute('aria-expanded', 'true');
        reasoningBtn.setAttribute('aria-expanded', 'true');
    }

    // Toggle on button click
    reasoningBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (reasoningOptions.classList.contains('open')) closeReasoning();
        else openReasoning();
    });

    // Click outside closes
    document.addEventListener('click', (e) => {
        if (!customReasoning.contains(e.target)) closeReasoning();
    });

    // Option click handler
    reasoningOptions.querySelectorAll('li').forEach(li => {
        li.addEventListener('click', (ev) => {
            const v = li.getAttribute('data-value');
            const text = li.textContent.trim();
            // update hidden input
            reasoningSelect.value = v;
            // update visual selected
            reasoningOptions.querySelectorAll('li').forEach(n => n.classList.remove('selected'));
            li.classList.add('selected');
            reasoningBtn.firstChild.nodeValue = text; // set visible text (firstChild is text node)
            // ensure caret remains (rebuild)
            const caret = document.createElement('span');
            caret.className = 'caret';
            // fix button content to include caret after text
            reasoningBtn.innerHTML = text;
            reasoningBtn.appendChild(caret);
            closeReasoning();
            reasoningBtn.focus();
        });
    });

    // keyboard accessibility for custom select (Enter/Space to toggle, arrows to navigate)
    customReasoning.addEventListener('keydown', (e) => {
        const open = reasoningOptions.classList.contains('open');
        const items = Array.from(reasoningOptions.querySelectorAll('li'));
        const currentIndex = items.findIndex(i => i.classList.contains('selected'));
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (!open) openReasoning(); else {
                // if open and an item is focused (use document.activeElement), click it
                const focused = document.activeElement;
                if (reasoningOptions.contains(focused) && focused.tagName === 'LI') focused.click();
                else closeReasoning();
            }
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (!open) openReasoning();
            const nextIndex = (currentIndex + 1) % items.length;
            items[nextIndex].focus();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (!open) openReasoning();
            const prevIndex = (currentIndex - 1 + items.length) % items.length;
            items[prevIndex].focus();
        } else if (e.key === 'Escape') {
            closeReasoning();
            reasoningBtn.focus();
        }
    });

    // ensure list items are focusable for keyboard
    reasoningOptions.querySelectorAll('li').forEach(li => {
        li.tabIndex = 0;
        li.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                li.click();
            } else if (e.key === 'Escape') {
                closeReasoning();
                reasoningBtn.focus();
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const next = li.nextElementSibling || reasoningOptions.querySelector('li');
                next.focus();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const prev = li.previousElementSibling || reasoningOptions.querySelector('li:last-child');
                prev.focus();
            }
        });
    });

    // Splash Screen Logic
    window.addEventListener('load', () => {
        setTimeout(() => {
            const splash = document.getElementById('splash');
            splash.style.opacity = '0';
            setTimeout(() => {
                splash.style.visibility = 'hidden';
            }, 800);
        }, 1500); // Show for 1.5s
    });

    function renderContent(text) {
        // Hide JSON tool blocks from delta stream rendering to avoid redundancy
        text = text.replace(/```json\s*\{\s*"tool":\s*"ipython"[\s\S]*?\}\s*```/g, '');
        text = text.replace(/\{\s*"tool":\s*"ipython"[\s\S]*?\}/g, '');
        // Also catch cases where tool name might be slightly different or without quotes in some models
        text = text.replace(/```json\s*\{[\s\S]*?"tool":\s*"ipython"[\s\S]*?\}\s*```/g, '');

        const mathBlocks = [];

        // 1️⃣ Extract block math ($$ ... $$ and \[ ... \])
        text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => {
            mathBlocks.push({ type: "block", content: m });
            return `@@BLOCK_${mathBlocks.length - 1}@@`;
        });
        text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => {
            mathBlocks.push({ type: "block", content: m });
            return `@@BLOCK_${mathBlocks.length - 1}@@`;
        });

        // 2️⃣ Extract inline math ($ ... $ and \( ... \))
        text = text.replace(/\$([\s\S]*?)\$/g, (_, m) => {
            mathBlocks.push({ type: "inline", content: m });
            return `@@INLINE_${mathBlocks.length - 1}@@`;
        });
        text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => {
            mathBlocks.push({ type: "inline", content: m });
            return `@@INLINE_${mathBlocks.length - 1}@@`;
        });

        // 3️⃣ Markdown parse ONLY plain text
        let html = marked.parse(text, {
            mangle: false,
            headerIds: false
        });

        // 4️⃣ Restore math verbatim (NO Markdown)
        mathBlocks.forEach((m, i) => {
            const latex =
                m.type === "block"
                    ? (m.content.includes('$$') ? m.content : `$$${m.content}$$`)
                    : (m.content.includes('$') ? m.content : `$${m.content}$`);

            html = html
                .replace(`@@BLOCK_${i}@@`, latex)
                .replace(`@@INLINE_${i}@@`, latex);
        });

        return html;
    }

    async function ask() {
        const val = input.value.trim();
        if (!val) return;

        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;

        appendMsg('user', val);

        // Show Typing Indicator (initial)
        typingIndicator.style.display = 'block';
        chat.scrollTop = chat.scrollHeight;

        try {
            const response = await fetch('chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: val,
                    reasoning_level: reasoningSelect.value,
                    recheck: recheckToggle.checked
                })
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let lastType = null;
            let lastStep = null;
            let currentContentEl = null;
            let currentThinkingContainer = null;
            let responseHasLabel = false;
            let buffer = '';

            const stopSpinner = () => {
                const spinners = chat.querySelectorAll('.spinner');
                spinners.forEach(spinner => {
                    spinner.className = 'dot';
                    spinner.style.width = '8px';
                    spinner.style.height = '8px';
                    spinner.style.background = 'var(--success)';
                    spinner.style.borderRadius = '50%';
                    spinner.style.animation = 'none';
                    spinner.innerHTML = '<span style="color:var(--success); font-size:10px; position:relative; top:-2px;">✓</span>';
                    spinner.style.display = 'flex';
                    spinner.style.alignItems = 'center';
                    spinner.style.justifyContent = 'center';
                });
            };

            const ensureAssistantLabel = (msgDiv) => {
                if (!responseHasLabel) {
                    const label = document.createElement('div');
                    label.className = 'msg-label';
                    label.textContent = 'AI:';
                    msgDiv.prepend(label);
                    responseHasLabel = true;
                }
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep potential incomplete line in buffer

                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;
                    
                    let data;
                    try {
                        data = JSON.parse(trimmedLine.substring(6));
                    } catch(e) { 
                        console.error("JSON Parse Error:", e, trimmedLine);
                        continue; 
                    }

                    typingIndicator.style.display = 'none';

                    if (data.type === 'status') {
                        stopSpinner();
                        currentThinkingContainer = document.createElement('div');
                        currentThinkingContainer.className = 'msg assistant';
                        currentThinkingContainer.innerHTML = `
                            <div class="thinking-container">
                                <div class="thinking-header">
                                    <div class="spinner"></div>
                                    ${data.content}
                                </div>
                            </div>`;
                        ensureAssistantLabel(currentThinkingContainer);
                        chat.appendChild(currentThinkingContainer);
                        lastType = 'status';
                        lastStep = data.content;
                        currentContentEl = null;
                    } 
                    else if (data.type === 'thinking') {
                        if (lastType !== 'thinking' || !currentContentEl) {
                            stopSpinner();
                            const msgDiv = document.createElement('div');
                            msgDiv.className = 'msg assistant';
                            msgDiv.innerHTML = `
                                <div class="thinking-container">
                                    <div class="thinking-header">Thought</div>
                                    <div class="thinking-content msg-content"></div>
                                </div>`;
                            ensureAssistantLabel(msgDiv);
                            currentContentEl = msgDiv.querySelector('.msg-content');
                            chat.appendChild(msgDiv);
                            lastType = 'thinking';
                            lastStep = 'Thought';
                        }
                        currentContentEl.dataset.raw = (currentContentEl.dataset.raw || '') + data.content;
                        currentContentEl.innerHTML = renderContent(currentContentEl.dataset.raw);
                    }
                    else if (data.type === 'delta') {
                        const stepKey = data.step || 'Reasoning';
                        const isFinal = (stepKey === 'Final Answer' || stepKey === 'Recheck');
                        const isContinuationOfFinal = isFinal && (lastStep === 'Final Answer' || lastStep === 'Recheck');

                        if (!isContinuationOfFinal && (lastType !== 'delta' || lastStep !== stepKey || !currentContentEl)) {
                            stopSpinner();
                            const msgDiv = document.createElement('div');
                            msgDiv.className = 'msg assistant';
                            
                            if (isFinal) {
                                msgDiv.innerHTML = `<div class="msg-content"></div>`;
                                currentContentEl = msgDiv.querySelector('.msg-content');
                            } else {
                                msgDiv.innerHTML = `
                                    <div class="thinking-container">
                                        <div class="thinking-header">${stepKey}</div>
                                        <div class="thinking-content msg-content"></div>
                                    </div>`;
                                currentContentEl = msgDiv.querySelector('.msg-content');
                            }
                            ensureAssistantLabel(msgDiv);
                            chat.appendChild(msgDiv);
                        }
                        
                        lastType = 'delta';
                        lastStep = stepKey;
                        currentContentEl.dataset.raw = (currentContentEl.dataset.raw || '') + data.content;
                        currentContentEl.innerHTML = renderContent(currentContentEl.dataset.raw);
                    }
                    else if (data.type === 'tool_call') {
                        stopSpinner();
                        const toolDiv = document.createElement('div');
                        toolDiv.className = 'msg assistant';
                        toolDiv.innerHTML = `
                            <div class="step-label">Step ${data.step}</div>
                            <div class="tool-output" style="color:var(--text-muted)">$ ipython [Mode: ${data.mode}]\n${data.code}</div>`;
                        ensureAssistantLabel(toolDiv);
                        chat.appendChild(toolDiv);
                        lastType = 'tool';
                        lastStep = `Step ${data.step}`;
                        currentContentEl = null;
                    }
                    else if (data.type === 'tool_output') {
                        const tout = data.output;
                        const toolDiv = document.createElement('div');
                        toolDiv.className = 'msg assistant';
                        let figHTML = '';
                        if (tout.figures && tout.figures.length > 0) {
                            tout.figures.forEach(fig => {
                                figHTML += `<img src="data:image/png;base64,${fig}" style="max-width:100%; border-radius:8px; margin-top:10px; border:1px solid var(--border);">`;
                            });
                        }
                        toolDiv.innerHTML = `
                            <div class="tool-output">$ ipython [Result: ${tout.success ? 'OK' : 'FAIL'}]\n${tout.stdout}${tout.stderr ? '\nERR: ' + tout.stderr : ''}</div>
                            ${figHTML}`;
                        ensureAssistantLabel(toolDiv);
                        chat.appendChild(toolDiv);
                        lastType = 'output';
                        lastStep = `Step ${data.step}`;
                        currentContentEl = null;
                    }
                    else if (data.type === 'done') {
                        stopSpinner();
                        if (currentContentEl) {
                            MathJax.typesetPromise([currentContentEl.closest('.msg')]);
                        }
                    }
                    chat.scrollTop = chat.scrollHeight;
                }
            }

        } catch (err) {
            console.error(err);
            typingIndicator.style.display = 'none';
            appendMsg('assistant', 'Error: ' + err.message);
        }

        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }

    function appendMsg(role, text) {
        if (!text || !text.trim()) {
            text = "_[No textual explanation returned]_";
        }

        const div = document.createElement('div');
        div.className = `msg ${role}`;
        
        // Only render the label for assistant messages. user label removed as requested.
        const labelHTML = role === 'assistant' ? `<div class="msg-label">AI:</div>` : ``;

        div.innerHTML = `
            ${labelHTML}
            <div class="msg-content">${renderContent(text)}</div>
        `;
        
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
        MathJax.typesetPromise([div]);
    }

    // Auto-focus input when typing
    document.addEventListener('keydown', (e) => {
        if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            if (e.key.length === 1 && !e.ctrlKey && !e.altKey && !e.metaKey) {
                input.focus();
            }
        }
    });

    input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""

def _cleanup():
    logger.info("Shutdown.")

atexit.register(_cleanup)

if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(rigor_bp, url_prefix='/rigor')
    @app.route('/')
    def root(): return Response('<a href="/rigor">Go to ZYLO RIGOR</a>')
    app.run(host="0.0.0.0", port=PORT, debug=False)
            
