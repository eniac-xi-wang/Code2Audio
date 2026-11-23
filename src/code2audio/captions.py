"""One audio per Qwen request. Descriptions only; never comparative judgments."""

import base64
import hashlib
import io
import json
import os
import re
import ssl
from pathlib import Path
from urllib import error, request

import certifi
import numpy as np
from scipy import signal
from scipy.io import wavfile

from .io import digest, dump, load, read_audio


def env_file(path):
    if path is None:
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {
            "QWEN_API_KEY",
            "QWEN_BASE_URL",
            "QWEN_MODEL",
            "CLAP_MODEL_DIR",
        }:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def clean_text(text, key):
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text.replace(key, "[REDACTED]"))


def validate(doc):
    if (
        not isinstance(doc, dict)
        or not isinstance(doc.get("caption"), str)
        or not doc["caption"].strip()
    ):
        raise ValueError("Missing independent caption")
    if not isinstance(doc.get("attributes"), list) or not all(
        isinstance(v, str) for v in doc["attributes"]
    ):
        raise ValueError("Expected a list of attribute sentences")
    if not isinstance(doc.get("uncertainties"), list):
        raise ValueError("Missing uncertainty list")
    return doc


def describe(audio, cache=None):
    prompt = (Path(__file__).parent / "prompts/caption.txt").read_text()
    model = os.environ.get("QWEN_MODEL", "qwen3-omni-flash")
    base = os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    key = os.environ.get("QWEN_API_KEY", "").strip()
    fingerprint = hashlib.sha256(
        (digest(audio) + prompt + model + base).encode()
    ).hexdigest()
    cached = Path(cache) / f"{fingerprint}.json" if cache else None
    if cached and cached.exists():
        result = load(cached)
        validate(result["description"])
        return result
    if not key:
        raise ValueError("Set QWEN_API_KEY or provide a validated cached caption")
    if not base.startswith("https://"):
        raise ValueError("The caption endpoint must use HTTPS")
    sr, x = read_audio(audio)
    # All available sound is captioned, with a bounded request payload.
    if len(x) / sr > 20:
        raise ValueError("Captioning supports excerpts up to 20 seconds")
    mono = signal.resample_poly(x.mean(1), 24000, sr)
    stream = io.BytesIO()
    wavfile.write(
        stream, 24000, np.rint(np.clip(mono, -0.999, 0.999) * 32767).astype(np.int16)
    )
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64," + encoded,
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "modalities": ["text"],
        "stream": True,
        "enable_thinking": False,
        "max_tokens": 1600,
    }
    req = request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    parts = []
    finish = None
    try:
        with request.urlopen(
            req, timeout=120, context=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                message = json.loads(data)
                if "error" in message:
                    raise ValueError("Caption API returned an error")
                for choice in message.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if isinstance(text, str):
                        parts.append(text)
                    finish = choice.get("finish_reason") or finish
    except error.HTTPError as exc:
        raise RuntimeError("Caption API HTTP " + str(exc.code)) from None
    text = clean_text("".join(parts), key)
    if finish != "stop":
        raise ValueError("Incomplete caption response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    doc = validate(json.loads(text))
    result = {
        "description": doc,
        "model": model,
        "audio_sha256": digest(audio),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_processing": "full excerpt, mono PCM16 at 24 kHz",
        "request_type": "independent_audio_caption",
    }
    if cached:
        dump(cached, result)
    return result
