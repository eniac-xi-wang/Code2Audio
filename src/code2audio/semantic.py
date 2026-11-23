"""Independent-caption CLAP verification; no audio preference model."""

import re
from pathlib import Path

import numpy as np
from scipy import signal

from .io import digest, read_audio


class Clap:
    def __init__(self, directory):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from transformers.audio_utils import (
            mel_filter_bank,
            spectrogram,
            window_function,
        )

        directory = Path(directory)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        ort.disable_telemetry_events()
        self.audio = ort.InferenceSession(
            str(directory / "onnx/audio_model.onnx"), sess_options=options
        )
        self.text = ort.InferenceSession(
            str(directory / "onnx/text_model_quantized.onnx"), sess_options=options
        )
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.filters = mel_filter_bank(
            513, 64, 50, 14000, 48000, norm="slaney", mel_scale="slaney"
        )
        self.window = window_function(1024, "hann")
        self.spectrogram = spectrogram
        self.identity = {
            "name": "Xenova/clap-htsat-unfused",
            "audio_weights_sha256": digest(directory / "onnx/audio_model.onnx"),
            "text_weights_sha256": digest(directory / "onnx/text_model_quantized.onnx"),
            "tokenizer_sha256": digest(directory / "tokenizer.json"),
        }

    def texts(self, captions):
        outputs = []
        for caption in captions:
            ids = self.tokenizer(
                [caption],
                padding=False,
                truncation=True,
                max_length=77,
                return_tensors="np",
            )["input_ids"].astype(np.int64)
            y = self.text.run(["text_embeds"], {"input_ids": ids})[0][0]
            outputs.append(y / np.linalg.norm(y))
        return np.stack(outputs)

    def waveform(self, x, sr):
        if x.ndim == 2:
            x = x.mean(1)
        x = signal.resample_poly(x, 48000, sr).astype(np.float32)
        if len(x) < 480000:
            repeats = max(1, 480000 // len(x))
            x = np.pad(np.tile(x, repeats), (0, 480000 - len(x) * repeats))
        x = x[:480000]
        x *= min(
            0.95 / max(float(abs(x).max()), 1e-12),
            0.08 / max(float(np.sqrt(np.mean(x * x))), 1e-12),
        )
        mel = self.spectrogram(
            x,
            self.window,
            frame_length=1024,
            hop_length=480,
            power=2.0,
            mel_filters=self.filters,
            log_mel="dB",
        ).T
        y = self.audio.run(
            ["audio_embeds"], {"input_features": mel[None, None].astype(np.float32)}
        )[0][0]
        return y / np.linalg.norm(y)

    def score_audio(self, path, texts):
        sr, x = read_audio(path)
        scores = []
        weights = []
        windows = []
        for start in range(0, len(x), sr * 10):
            clip = x[start : start + sr * 10]
            scores.append(self.waveform(clip, sr) @ texts.T)
            weights.append(len(clip))
            windows.append([start / sr, (start + len(clip)) / sr])
        return np.average(scores, axis=0, weights=weights), np.asarray(scores), windows


UNSUITABLE = re.compile(
    r"\b(no|not|without|none|silent|silence|absent|lack|lacks|lacking|may|might|possibly|perhaps|suggesting)\b",
    re.I,
)


def caption_pool(documents, target_caption=None):
    pool = []
    index = {}
    for name, record in documents.items():
        description = record["description"]
        entries = [description["caption"]] + description["attributes"]
        if name == "reference" and target_caption:
            entries = [target_caption] + entries
        for value in entries:
            text = value.strip()
            if not text:
                continue
            key = " ".join(text.lower().split()).strip(" .")
            if key in index:
                row = pool[index[key]]
                if name not in row["origins"]:
                    row["origins"].append(name)
                continue
            index[key] = len(pool)
            pool.append(
                {
                    "text": text,
                    "origins": [name],
                    "eligible": not bool(UNSUITABLE.search(text)),
                }
            )
    return pool


def verify(model, reference, candidates, documents, config, target_caption=None):
    paths = {
        "reference": Path(reference),
        **{k: Path(v) for k, v in candidates.items()},
    }
    for name, path in paths.items():
        if documents[name]["audio_sha256"] != digest(path):
            raise ValueError("Caption does not belong to " + name)
        if documents[name].get("request_type") != "independent_audio_caption":
            raise ValueError("Only independent captions are valid")
    pool = caption_pool(documents, target_caption)
    target = np.array([p["eligible"] and "reference" in p["origins"] for p in pool])
    if not target.any():
        raise ValueError("No affirmative reference captions; sample is unresolved")
    vectors = model.texts([p["text"] for p in pool])
    scores = {}
    windows = {}
    for name, path in paths.items():
        mean, per_window, times = model.score_audio(path, vectors)
        scores[name] = mean
        windows[name] = {"times": times, "scores": per_window.tolist()}
    reference_score = float(scores["reference"][target].mean())
    threshold = max(
        config["minimum_target_similarity"],
        reference_score - config["maximum_reference_drop"],
    )
    target_scores = {
        name: float(value[target].mean())
        for name, value in scores.items()
        if name != "reference"
    }
    reference_supported = reference_score >= config["minimum_target_similarity"]
    passed = reference_supported and all(
        value >= threshold for value in target_scores.values()
    )
    return {
        "schema_version": 1,
        "model": model.identity,
        "caption_pool": pool,
        "audio_sha256": {n: digest(p) for n, p in paths.items()},
        "scores": {n: s.astype(float).tolist() for n, s in scores.items()},
        "windows": windows,
        "reference_target_match": reference_score,
        "candidate_target_match": target_scores,
        "mean_target_match": float(np.mean(list(target_scores.values()))),
        "minimum_candidate_match": min(target_scores.values()),
        "threshold": threshold,
        "reference_supported": reference_supported,
        "semantic_pass": bool(passed),
        "policy": {
            "minimum_target_similarity": config["minimum_target_similarity"],
            "maximum_reference_drop": config["maximum_reference_drop"],
        },
        "caption_score_gaps": {
            n: (s - scores["reference"]).astype(float).tolist()
            for n, s in scores.items()
            if n != "reference"
        },
        "interpretation": "Semantic screening under configured thresholds, not a probability, naturalness score or proof of exact reconstruction.",
    }
