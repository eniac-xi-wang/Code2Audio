"""Execute a restricted waveform contract in a fresh, source-free subprocess."""

import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .io import write_audio

ALLOWED = {
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.fft",
    "scipy.ndimage",
    "scipy.special",
    "scipy.interpolate",
    "scipy.integrate",
    "scipy.linalg",
    "scipy.optimize",
    "scipy.stats",
    "math",
}
SCIPY_PARTS = {
    "signal",
    "fft",
    "ndimage",
    "special",
    "interpolate",
    "integrate",
    "linalg",
    "optimize",
    "stats",
}
BLOCKED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "input",
    "help",
    "breakpoint",
}
BLOCKED_ATTRS = {
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "savetxt",
    "fromfile",
    "tofile",
    "memmap",
    "ctypes",
    "f2py",
    "testing",
    "distutils",
    "DataSource",
    "get_include",
    "show_config",
    "lookfor",
    "info",
    "dump",
    "dumps",
    "genfromtxt",
    "fromregex",
    "recfromcsv",
    "recfromtxt",
    "ctypeslib",
    "lib",
}


def check_program(program, parameters):
    program, parameters = Path(program), Path(parameters)
    if program.is_symlink() or parameters.is_symlink():
        raise ValueError("Generated assets must not be symlinks")
    if program.stat().st_size > 100_000 or parameters.stat().st_size > 100_000:
        raise ValueError("Program and parameters must each be at most 100 KB")
    value = json.loads(parameters.read_text())
    if not isinstance(value, dict):
        raise ValueError("Parameters must be a JSON object")
    text = program.read_text()
    tree = ast.parse(text)
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "render" for n in tree.body
    ):
        raise ValueError("Missing render(duration, sample_rate, seed, parameters)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name not in ALLOWED for n in node.names):
                raise ValueError("Only numeric imports are allowed")
        if isinstance(node, ast.ImportFrom):
            if (
                node.level
                or node.module not in ALLOWED
                or any(
                    n.name.startswith("_")
                    or n.name == "*"
                    or n.name in BLOCKED_ATTRS
                    or n.name in BLOCKED_NAMES
                    for n in node.names
                )
            ):
                raise ValueError("Unsupported import")
            if node.module == "scipy" and any(
                n.name not in SCIPY_PARTS for n in node.names
            ):
                raise ValueError("Use only the supported SciPy numeric modules")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError("Unsupported builtin: " + node.id)
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in BLOCKED_ATTRS
        ):
            raise ValueError("Unsupported attribute: " + node.attr)
    return text, value


def render(
    program, parameters, output, duration=8.0, sample_rate=44100, seed=17, timeout=30
):
    if not 1 <= duration <= 60 or not 8000 <= sample_rate <= 96000:
        raise ValueError("Render supports 1–60 seconds at 8–96 kHz")
    check_program(program, parameters)
    with tempfile.TemporaryDirectory(prefix="c2a-render-") as tmp:
        stage = Path(tmp)
        shutil.copyfile(program, stage / "program.py")
        shutil.copyfile(parameters, stage / "parameters.json")
        env = {
            k: os.environ[k]
            for k in ["PATH", "SYSTEMROOT", "TMPDIR"]
            if k in os.environ
        }
        env.update(
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"
        )
        cmd = [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("worker.py")),
            str(stage),
            str(duration),
            str(sample_rate),
            str(seed),
            str(timeout),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=stage,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, error = proc.communicate(timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise RuntimeError("Render timed out") from None
        if proc.returncode:
            raise RuntimeError(
                "Render failed: " + error.decode(errors="replace")[-1500:]
            )
        x = np.load(stage / "output.npy", allow_pickle=False)
    if x.ndim == 1:
        x = x[:, None]
    if (
        x.shape[0] != round(duration * sample_rate)
        or x.ndim != 2
        or x.shape[1] not in (1, 2)
    ):
        raise ValueError("Renderer returned the wrong duration or channel layout")
    if (
        not np.isfinite(x).all()
        or np.max(abs(x)) >= 0.999
        or np.sqrt(np.mean(x * x)) < 1e-7
    ):
        raise ValueError("Renderer returned invalid, clipping or silent audio")
    write_audio(output, sample_rate, x)
    return {
        "frames": len(x),
        "channels": x.shape[1],
        "sample_rate": sample_rate,
        "duration": duration,
        "seed": seed,
        "peak": float(abs(x).max()),
    }
