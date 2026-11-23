"""Source-neutral measurements for an agent; no sound-class presets."""

import numpy as np
from scipy import signal, stats

from .io import read_audio


def analyze(path):
    sr, x = read_audio(path)
    mono = x.mean(1)
    hop = max(1, round(sr * 0.01))
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop)
    envelope = np.sqrt(np.mean(frames**2, axis=1))
    frequencies, power = signal.welch(mono, sr, nperseg=min(8192, len(mono)))
    edges = np.geomspace(40, sr / 2, 33)
    bands = [
        float(power[(frequencies >= a) & (frequencies < b)].sum())
        for a, b in zip(edges[:-1], edges[1:])
    ]
    total = max(float(power.sum()), 1e-20)
    rms = float(np.sqrt(np.mean(x * x)))
    peaks, _ = signal.find_peaks(power)
    peaks = peaks[np.argsort(power[peaks])[-8:]][::-1]
    return {
        "sample_rate": sr,
        "duration": len(x) / sr,
        "channels": x.shape[1],
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
        "peak": float(np.max(abs(x))),
        "clipped_samples": int(np.sum(abs(x) >= 1)),
        "envelope_cv": float(envelope.std() / max(envelope.mean(), 1e-12)),
        "mono_kurtosis": float(stats.kurtosis(mono, fisher=False))
        if mono.std() > 1e-8
        else 0.0,
        "stereo_correlation": float(np.corrcoef(x.T)[0, 1])
        if x.shape[1] == 2 and np.all(x.std(0) > 1e-12)
        else None,
        "band_edges_hz": edges.tolist(),
        "band_energy_fraction": [v / total for v in bands],
        "spectral_peak_hz": frequencies[peaks].tolist(),
        "interpretation": "Descriptive measurements, not source labels or a perceptual quality score.",
    }
