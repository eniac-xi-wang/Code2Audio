"""Bounded agent -> render -> describe -> verify -> select loop."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from .analysis import analyze
from .captions import describe
from .io import digest, dump, excerpt, load
from .render import render
from .semantic import Clap, verify

DEFAULTS = {
    "duration": 8.0,
    "sample_rate": 44100,
    "seeds": [17, 31],
    "max_rounds": 3,
    "patience": 2,
    "minimum_improvement": 0.005,
    "minimum_target_similarity": 0.35,
    "maximum_reference_drop": 0.08,
    "render_timeout_seconds": 30,
    "agent_timeout_seconds": 600,
    "test_fraction": 0.01,
}


def configuration(path=None):
    config = {**DEFAULTS, **(load(path) if path else {})}
    if not 1 <= config["duration"] <= 20 or not 8000 <= config["sample_rate"] <= 96000:
        raise ValueError("Use excerpts of 1–20 seconds at 8–96 kHz")
    if not config["seeds"] or any(
        not isinstance(s, int) or s < 0 for s in config["seeds"]
    ):
        raise ValueError("Provide nonnegative integer seeds")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be distinct")
    if config["max_rounds"] < 1 or config["patience"] < 1:
        raise ValueError("Invalid iteration budget")
    if not 0 <= config["test_fraction"] < 1:
        raise ValueError("Invalid test split")
    return config


class Services:
    def __init__(self, model_dir, cache):
        self.model = Clap(model_dir)
        self.cache = cache

    def caption(self, path):
        return describe(path, self.cache)

    def evaluate(self, reference, candidates, captions, config, target):
        return verify(self.model, reference, candidates, captions, config, target)


def command_agent(command, task, directory, timeout):
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(v, str) for v in command)
    ):
        raise ValueError("Agent command must be a JSON array of arguments")
    env = dict(os.environ)
    env.pop("QWEN_API_KEY", None)
    with (directory / "agent.log").open("wb") as log:
        proc = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            proc.communicate(
                json.dumps(task, ensure_ascii=False).encode(), timeout=timeout
            )
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise RuntimeError("Agent timed out") from None
    if proc.returncode:
        raise RuntimeError("Agent command failed; inspect the local attempt log")


def load_manifest(path):
    path = Path(path).resolve()
    rows = []
    ids = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", name) or name in ids:
            raise ValueError("Item IDs must be unique safe names")
        ids.add(name)
        audio = (path.parent / row["audio"]).resolve()
        if not audio.is_file():
            raise ValueError("Missing reference for " + name)
        row["audio"] = str(audio)
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def build(
    manifest, output, config, command=None, services=None, agent_fn=None, resume=False
):
    output = Path(output).resolve()
    rows = load_manifest(manifest)
    signature = hashlib.sha256(
        json.dumps(
            {
                "rows": rows,
                "config": config,
                "source_hashes": [digest(r["audio"]) for r in rows],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if output.exists() and any(output.iterdir()):
        if (
            not resume
            or not (output / "run.json").exists()
            or load(output / "run.json")["signature"] != signature
        ):
            raise ValueError("Existing run differs or --resume was not supplied")
    output.mkdir(parents=True, exist_ok=True)
    if services is None:
        directory = os.environ.get("CLAP_MODEL_DIR")
        if not directory:
            raise ValueError("Set CLAP_MODEL_DIR or pass --model-dir")
        services = Services(directory, output / "caption_cache")
    model_identity = getattr(getattr(services, "model", None), "identity", None)
    if (
        resume
        and (output / "run.json").exists()
        and load(output / "run.json").get("verification_model") != model_identity
    ):
        raise ValueError("Verification model changed; use a new run")
    dump(
        output / "run.json",
        {
            "schema_version": 1,
            "signature": signature,
            "config": config,
            "verification_model": model_identity,
            "item_ids": [r["id"] for r in rows],
        },
    )
    if agent_fn is None:
        command = command or [sys.executable, "-m", "code2audio.codex_agent"]

        def agent_fn(task, directory):
            return command_agent(
                command, task, directory, config["agent_timeout_seconds"]
            )

    summaries = []
    for row in rows:
        item = output / row["id"]
        item.mkdir(exist_ok=True)
        if resume and (item / "status.json").exists():
            status = load(item / "status.json")
            if status.get("complete") and status.get("status") != "unresolved":
                summaries.append(status)
                continue
        # Each unfinished item can resume at the next numbered attempt.
        status = {
            "id": row["id"],
            "status": "unresolved",
            "complete": False,
            "attempts": [],
            "selected": None,
        }
        if (item / "status.json").exists():
            status = load(item / "status.json")
        try:
            reference = item / "reference.wav"
            if not reference.exists():
                excerpt(
                    row["audio"],
                    reference,
                    row.get("start", 0),
                    config["duration"],
                    config["sample_rate"],
                )
            caption_path = item / "reference_caption.json"
            reference_caption = (
                load(caption_path)
                if caption_path.exists()
                else services.caption(reference)
            )
            if reference_caption["audio_sha256"] != digest(reference):
                raise ValueError("Stored reference caption does not match the excerpt")
            dump(caption_path, reference_caption)
            analysis = analyze(reference)
            dump(item / "analysis.json", analysis)
            dump(
                item / "source.json",
                {
                    "source_id": row.get("source_id") or digest(row["audio"]),
                    "source_sha256": digest(row["audio"]),
                    "reference_sha256": digest(reference),
                    "source_uri": row.get("source_uri"),
                    "target_caption": row.get("caption"),
                    "start": row.get("start", 0),
                    "duration": config["duration"],
                },
            )
            previous = status.get("feedback")
            stale = status.get("stale", 0)
            for number in range(len(status["attempts"]) + 1, config["max_rounds"] + 1):
                attempt = item / f"attempt_{number:03d}"
                if attempt.exists():
                    # Preserve interrupted partial work; never reuse its outputs.
                    archived = item / f"interrupted_{number:03d}"
                    if archived.exists():
                        raise ValueError(
                            "Repeated interrupted attempt; inspect the run"
                        )
                    attempt.rename(archived)
                attempt.mkdir()
                task = {
                    "id": row["id"],
                    "reference_audio": str(reference),
                    "reference_description": reference_caption["description"],
                    "target_caption": row.get("caption"),
                    "analysis": analysis,
                    "duration": config["duration"],
                    "sample_rate": config["sample_rate"],
                    "seeds": config["seeds"],
                    "previous_feedback": previous,
                    "previous_program": None,
                    "output_contract": {
                        "program": "program.py",
                        "parameters": "parameters.json",
                        "model": "model.json",
                    },
                }
                if status["selected"]:
                    task["previous_program"] = str(
                        item / status["selected"]["attempt"] / "program.py"
                    )
                elif (
                    number > 1
                    and (item / f"attempt_{number - 1:03d}" / "program.py").exists()
                ):
                    task["previous_program"] = str(
                        item / f"attempt_{number - 1:03d}" / "program.py"
                    )
                dump(attempt / "request.json", task)
                result = {"attempt": attempt.name, "semantic_pass": False}
                try:
                    agent_fn(task, attempt)
                    model_note = load(attempt / "model.json")
                    if not isinstance(model_note, dict) or not {
                        "components",
                        "controls",
                        "assumptions",
                    } <= set(model_note):
                        raise ValueError(
                            "model.json must document components, controls and assumptions"
                        )
                    candidates = {}
                    captions = {"reference": reference_caption}
                    checks = {}
                    for seed in config["seeds"]:
                        name = f"seed_{seed}"
                        path = attempt / (name + ".wav")
                        checks[name] = render(
                            attempt / "program.py",
                            attempt / "parameters.json",
                            path,
                            config["duration"],
                            config["sample_rate"],
                            seed,
                            config["render_timeout_seconds"],
                        )
                        candidates[name] = path
                        captions[name] = services.caption(path)
                        dump(attempt / (name + "_caption.json"), captions[name])
                    first = config["seeds"][0]
                    rerender = attempt / "reproducibility.wav"
                    render(
                        attempt / "program.py",
                        attempt / "parameters.json",
                        rerender,
                        config["duration"],
                        config["sample_rate"],
                        first,
                        config["render_timeout_seconds"],
                    )
                    exact = digest(rerender) == digest(candidates[f"seed_{first}"])
                    rerender.unlink()
                    if not exact:
                        raise ValueError("Same seed did not reproduce identical PCM")
                    report = services.evaluate(
                        reference, candidates, captions, config, row.get("caption")
                    )
                    report["render_checks"] = checks
                    report["deterministic"] = True
                    report["asset_sha256"] = {
                        name: digest(attempt / name)
                        for name in ["program.py", "parameters.json", "model.json"]
                    }
                    report["audio_sha256"] = {
                        "reference": digest(reference),
                        **{name: digest(path) for name, path in candidates.items()},
                    }
                    report["caption_sha256"] = {
                        "reference": digest(caption_path),
                        **{
                            name: digest(attempt / (name + "_caption.json"))
                            for name in candidates
                        },
                    }
                    dump(attempt / "verification.json", report)
                    result.update(
                        semantic_pass=report["semantic_pass"],
                        score=report["mean_target_match"],
                    )
                    previous = {
                        "semantic": report,
                        "candidate_measurements": {
                            k: analyze(v) for k, v in candidates.items()
                        },
                    }
                    old_score = (
                        status["selected"]["score"]
                        if status["selected"]
                        else -float("inf")
                    )
                    if (
                        report["semantic_pass"]
                        and report["mean_target_match"]
                        > old_score + config["minimum_improvement"]
                    ):
                        status["selected"] = {
                            "attempt": attempt.name,
                            "score": report["mean_target_match"],
                        }
                        stale = 0
                    else:
                        stale += 1
                except Exception as exc:
                    result["error"] = str(exc)[:1500]
                    previous = {"execution_error": result["error"]}
                    stale += 1
                status["attempts"].append(result)
                status["feedback"] = previous
                status["stale"] = stale
                dump(item / "status.json", status)
                if stale >= config["patience"]:
                    break
            status["status"] = (
                "accepted_semantic"
                if status["selected"]
                else (
                    "rejected"
                    if any("score" in a for a in status["attempts"])
                    else "unresolved"
                )
            )
        except Exception as exc:
            status["error"] = str(exc)[:1500]
            status["status"] = "unresolved"
        status["complete"] = True
        dump(item / "status.json", status)
        summaries.append(status)
    dump(
        output / "summary.json",
        [{k: v for k, v in s.items() if k not in ["feedback"]} for s in summaries],
    )
    return summaries
