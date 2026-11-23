"""Estimate aggregate rain statistics for the example waveform program."""

import argparse

import numpy as np
from program import BANDS, EPS, NFFT, band_weights, mid_side, rms
from scipy import fft, ndimage, signal, stats

from code2audio.io import dump, read_audio


def fit(path):
    sr, x = read_audio(path)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    ms = mid_side(x)
    frequencies, power = signal.welch(
        ms, fs=sr, nperseg=NFFT, noverlap=NFFT // 2, axis=0
    )
    knots = np.unique(np.r_[0, np.geomspace(frequencies[1], sr / 2, 600)])
    curves = [
        np.exp(
            ndimage.gaussian_filter1d(
                np.interp(knots, frequencies, np.log(np.maximum(power[:, i], EPS))), 0.8
            )
        ).tolist()
        for i in range(2)
    ]
    transformed = fft.rfft(ms, axis=0)
    f = fft.rfftfreq(len(ms), 1 / sr)
    bands = []
    for weight in band_weights(f):
        band = fft.irfft(transformed * np.sqrt(weight[:, None]), n=len(ms), axis=0)
        hop = round(sr * 0.1)
        frames = band[: len(band) // hop * hop].reshape(-1, hop, 2)
        envelope = np.sqrt(np.mean(frames**2, axis=1))
        bands.append(
            {
                "kurtosis_ms": stats.kurtosis(band, axis=0, fisher=False).tolist(),
                "slow_envelope_cv_ms": (envelope.std(0) / envelope.mean(0)).tolist(),
            }
        )
    return {
        "sample_rate": sr,
        "frequency_hz": knots.tolist(),
        "psd_mid_side": curves,
        "rms_lr": rms(x, axis=0).tolist(),
        "bands_hz": BANDS,
        "band_statistics": bands,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args()
    dump(args.output, fit(args.reference))
