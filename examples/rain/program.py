"""Rain from aggregate spectrum and event statistics; no recorded samples."""

import numpy as np
from scipy import fft, ndimage, signal

NFFT = 8192
EPS = 1e-20
BANDS = [(20, 300), (300, 1000), (1000, 3000), (3000, 8000), (8000, 22050)]


def rms(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def mid_side(x):
    return np.column_stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1])) / np.sqrt(2)


def band_weights(f):
    """Smooth overlapping frequency partitions, whose power weights sum to 1."""
    transitions = []
    for boundary in (300, 1000, 3000, 8000):
        position = np.clip(np.log2(np.maximum(f, 1) / boundary) / 0.5 + 0.5, 0, 1)
        transitions.append(0.5 - 0.5 * np.cos(np.pi * position))
    return np.array(
        [
            1 - transitions[0],
            transitions[0] - transitions[1],
            transitions[1] - transitions[2],
            transitions[2] - transitions[3],
            transitions[3],
        ]
    )


def minimum_phase_kernel(amplitude):
    """Real-cepstrum construction: colored impulses decay after each impact."""
    log_amplitude = np.log(np.maximum(amplitude, 1e-10))
    cepstrum = fft.irfft(log_amplitude, n=NFFT)
    causal = np.zeros(NFFT)
    causal[0], causal[NFFT // 2] = cepstrum[0], cepstrum[NFFT // 2]
    causal[1 : NFFT // 2] = 2 * cepstrum[1 : NFFT // 2]
    return fft.irfft(np.exp(fft.rfft(causal)), n=NFFT)


def synthesize(profile, duration, seed, intensity=1.0):
    """Colored diffuse noise + random minimum-phase drop impulses in five bands.

    The Gaussian bed has kurtosis 3. A Bernoulli/Gaussian impact train has
    excess kurtosis 3/p - 3. Filtering multiplies that by sum(h^4)/sum(h^2)^2;
    mixing at impact power fraction a multiplies it by a^2. This gives an
    analytic event probability from measured excess kurtosis, without copying
    the recording. Shared event candidates correlate adjacent frequency bands.
    """
    sr = profile["sample_rate"]
    n = round(duration * sr)
    warmup = sr
    length = n + 2 * warmup
    rng = np.random.default_rng(seed)
    frequencies = fft.rfftfreq(NFFT, 1 / sr)
    weights = band_weights(frequencies)
    ms = np.zeros((length, 2))
    model_details = []
    for channel in range(2):
        spectrum = np.exp(
            np.interp(
                frequencies,
                profile["frequency_hz"],
                np.log(np.maximum(profile["psd_mid_side"][channel], EPS)),
            )
        )
        # A tiny DC component is suppressed; the measured high-frequency rolloff is retained.
        spectrum[0] = 0
        candidates = rng.random(length)
        impact_amplitude = rng.standard_normal(length)
        for k, weight in enumerate(weights):
            kernel = minimum_phase_kernel(np.sqrt(spectrum * sr / 2 * weight))
            kernel_energy = np.sum(kernel**2)
            filter_kurtosis_ratio = np.sum(kernel**4) / max(kernel_energy**2, EPS)
            target_excess = max(
                0, profile["band_statistics"][k]["kurtosis_ms"][channel] - 3
            )
            impact_fraction = 0.60 if target_excess > 0.1 else 0.0
            probability = 3 / (
                target_excess / max(impact_fraction**2 * filter_kurtosis_ratio, EPS) + 3
            )
            probability = float(
                np.clip(probability * intensity, 1 / (duration * sr), 1)
            )
            impacts = (
                (candidates < probability) * impact_amplitude / np.sqrt(probability)
            )
            innovation = np.sqrt(1 - impact_fraction) * rng.standard_normal(length)
            innovation += np.sqrt(impact_fraction) * impacts
            band = signal.fftconvolve(innovation, kernel, mode="full")[:length]
            # New slow fluctuations; the original amplitude envelope is not reused.
            slow_cv = profile["band_statistics"][k]["slow_envelope_cv_ms"][channel]
            slow_cv = min(0.06, slow_cv * 0.2)
            control_times = np.arange(0, length + sr, sr // 4)
            control = ndimage.gaussian_filter1d(
                rng.standard_normal(len(control_times)), 2
            )
            control /= max(np.std(control), 1e-12)
            slow_gain = np.exp(
                slow_cv * np.interp(np.arange(length), control_times, control)
            )
            band *= slow_gain
            measured_rms = rms(band[warmup : warmup + n])
            band *= np.sqrt(kernel_energy) / max(measured_rms, 1e-12)
            ms[:, channel] += band
            model_details.append(
                {
                    "component": "mid" if channel == 0 else "side",
                    "band_hz": BANDS[k],
                    "impact_candidates_per_second": probability * sr,
                    "impact_power_fraction": impact_fraction,
                    "target_excess_kurtosis": target_excess,
                }
            )
    out = mid_side(
        ms[warmup : warmup + n]
    )  # The orthonormal transform is its own inverse.
    out *= np.asarray(profile["rms_lr"]) / rms(out, axis=0)
    fade = min(round(0.06 * sr), n // 2)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    out[:fade] *= ramp[:, None]
    out[-fade:] *= ramp[::-1, None]
    peak = np.max(np.abs(out))
    safety_gain = min(1.0, 0.95 / max(peak, 1e-12))
    out *= safety_gain
    return out, {
        "seed": seed,
        "intensity": intensity,
        "peak_safety_gain": safety_gain,
        "components": model_details,
    }


def render(duration, sample_rate, seed, parameters):
    """Return floating-point stereo samples, using only parameters and seed."""
    audio, _ = synthesize(
        parameters, duration, seed, intensity=parameters.get("event_density", 1.0)
    )
    source_rate = parameters["sample_rate"]
    if source_rate != sample_rate:
        from math import gcd

        divisor = gcd(source_rate, sample_rate)
        audio = signal.resample_poly(
            audio, sample_rate // divisor, source_rate // divisor, axis=0
        )
    return audio[: round(duration * sample_rate)] * parameters.get("level", 1.0)
