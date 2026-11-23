"""Render the README's paired rain spectrograms from the two example WAVs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy import signal

from code2audio.io import read_audio

PAPER = "#f8fafc"
INK = "#15171a"
MUTED = "#515861"
PALETTE = matplotlib.colormaps["magma"]


def plot(example, output):
    """Use one time range, frequency range and absolute power scale for both clips."""
    fig = plt.figure(figsize=(10, 4.25), dpi=180, facecolor=PAPER)
    fig.text(
        0.082, 0.825, "Steady rain", fontsize=31, fontfamily="DejaVu Serif", color=INK
    )
    fig.text(0.572, 0.918, "Spectral power density  (dB/Hz)", fontsize=10, color=MUTED)
    limits = Normalize(-100, -40)
    legend = fig.add_axes([0.572, 0.869, 0.344, 0.013])
    legend.imshow(
        np.linspace(limits.vmin, limits.vmax, 256)[None, :],
        extent=(limits.vmin, limits.vmax, 0, 1),
        aspect="auto",
        cmap=PALETTE,
        norm=limits,
    )
    legend.set_yticks([])
    legend.set_xticks([-100, -80, -60, -40])
    legend.tick_params(length=0, pad=4, labelsize=9, colors=MUTED)
    legend.spines[:].set_visible(False)

    panels = [
        ("reference.wav", "Field recording", 0.082),
        ("generated.wav", "Python synthesis", 0.555),
    ]
    duration = None
    for index, (filename, title, left) in enumerate(panels):
        sr, audio = read_audio(example / filename)
        seconds = len(audio) / sr
        if duration is None:
            duration = seconds
        if seconds != duration:
            raise ValueError("The comparison figure expects excerpts of equal length")
        freq, time, density = signal.spectrogram(
            audio.mean(axis=1),
            sr,
            window="hann",
            nperseg=2048,
            noverlap=1536,
            scaling="density",
            detrend=False,
        )
        ax = fig.add_axes([left, 0.16, 0.407, 0.49], facecolor=PALETTE(0.0))
        ax.pcolormesh(
            time,
            freq,
            10 * np.log10(np.maximum(density, 1e-16)),
            shading="nearest",
            cmap=PALETTE,
            norm=limits,
            rasterized=True,
        )
        ax.set(yscale="log", ylim=(100, 16000), xlim=(0, duration))
        ax.set_title(title, loc="left", pad=9, fontsize=14, color=INK)
        # The same logarithmic frequency axis applies to both panels.
        ax.set_yticks([125, 500, 2000, 8000], ["125 Hz", "500 Hz", "2 kHz", "8 kHz"])
        ax.tick_params(axis="y", labelleft=(index == 0), length=0, pad=7)
        ticks = np.linspace(0, duration, 5)
        ax.set_xticks(
            ticks, [f"{value:g}" for value in ticks[:-1]] + [f"{duration:g} s"]
        )
        ax.tick_params(axis="x", length=3, pad=5)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.minorticks_off()
        ax.spines[:].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=Path("assets/rain-analysis.png"))
    args = parser.parse_args()
    plot(args.example, args.output)
