"""Private subprocess entry point; used by render.py, not by generated programs."""

import builtins
import json
import sys
from pathlib import Path

import numpy as np


def main():
    stage = Path(sys.argv[1])
    duration = float(sys.argv[2])
    sr = int(sys.argv[3])
    seed = int(sys.argv[4])
    try:
        import resource

        seconds = max(1, int(float(sys.argv[5])))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    except ImportError:
        pass
    allowed = {
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

    def numeric_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in allowed:
            raise ImportError("Numeric imports only")
        return builtins.__import__(name, globals, locals, fromlist, level)

    names = [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "ValueError",
        "RuntimeError",
        "TypeError",
        "Exception",
    ]
    safe = {name: getattr(builtins, name) for name in names}
    safe["__import__"] = numeric_import
    namespace = {"__builtins__": safe, "__name__": "waveform_program"}
    exec(compile((stage / "program.py").read_text(), "program.py", "exec"), namespace)
    result = namespace["render"](
        duration, sr, seed, json.loads((stage / "parameters.json").read_text())
    )
    x = np.asarray(result)
    if x.dtype.kind not in "fiu" or x.size > round(duration * sr) * 2:
        raise ValueError("Expected a numeric mono/stereo waveform")
    np.save(stage / "output.npy", x.astype(np.float64), allow_pickle=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        raise SystemExit(1)
