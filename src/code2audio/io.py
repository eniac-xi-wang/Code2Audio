"""Small serialization and PCM helpers shared by the pipeline."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_audio(path):
    sr, raw = wavfile.read(path)
    if raw.dtype == np.uint8:
        x = (raw.astype(float) - 128) / 128
    elif np.issubdtype(raw.dtype, np.integer):
        x = raw.astype(float) / (np.iinfo(raw.dtype).max + 1)
    elif np.issubdtype(raw.dtype, np.floating):
        x = raw.astype(float)
    else:
        raise ValueError("Unsupported PCM type")
    if x.ndim == 1:
        x = x[:, None]
    if (
        x.ndim != 2
        or x.shape[1] not in (1, 2)
        or not len(x)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite nonempty mono/stereo audio")
    return sr, x


def write_audio(path, sr, x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or not x.size or np.max(abs(x)) >= 1:
        raise ValueError("Invalid or clipping waveform; fix the generator")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, np.rint(x * 32767).astype(np.int16))


def excerpt(source, destination, start, duration, sample_rate):
    sr, x = read_audio(source)
    if start < 0 or duration <= 0:
        raise ValueError("Invalid excerpt interval")
    x = x[round(start * sr) : round((start + duration) * sr)]
    if len(x) < round(duration * sr):
        raise ValueError("Reference is shorter than the requested excerpt")
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        x = signal.resample_poly(x, sample_rate // divisor, sr // divisor, axis=0)
    write_audio(destination, sample_rate, x[: round(duration * sample_rate)])
