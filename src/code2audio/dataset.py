"""Export accepted code/audio pairs, grouped by source for the train/test split."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from .io import digest, dump, load


def split_for(source_id, fraction):
    number = int(hashlib.sha256(source_id.encode()).hexdigest()[:16], 16) / 2**64
    return "test" if number < fraction else "train"


def export(run, destination):
    run = Path(run)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Export destination must be new")
    config = load(run / "run.json")["config"]
    accepted = []
    seen_audio = set()
    source_groups = {}
    for name in load(run / "run.json")["item_ids"]:
        item = run / name
        status = load(item / "status.json")
        if status["status"] != "accepted_semantic" or not status.get("selected"):
            continue
        source = load(item / "source.json")
        attempt = item / status["selected"]["attempt"]
        report = load(attempt / "verification.json")
        if not report["semantic_pass"] or not report.get("deterministic"):
            continue
        for filename in ["program.py", "parameters.json", "model.json"]:
            if report.get("asset_sha256", {}).get(filename) != digest(
                attempt / filename
            ):
                raise ValueError(
                    "Selected asset changed after verification: " + filename
                )
        if (
            report.get("audio_sha256", {}).get("reference")
            != source["reference_sha256"]
        ):
            raise ValueError("Verification reference does not match source provenance")
        if report.get("caption_sha256", {}).get("reference") != digest(
            item / "reference_caption.json"
        ):
            raise ValueError("Reference description changed after verification")
        group = source["source_id"]
        source_hash = source["source_sha256"]
        if source_hash in source_groups and source_groups[source_hash] != group:
            raise ValueError(
                "The same source has conflicting group IDs; fix the manifest"
            )
        source_groups[source_hash] = group
        for seed in config["seeds"]:
            audio = attempt / f"seed_{seed}.wav"
            audio_hash = digest(audio)
            caption = load(attempt / f"seed_{seed}_caption.json")
            if (
                report["audio_sha256"].get(f"seed_{seed}") != audio_hash
                or caption["audio_sha256"] != audio_hash
            ):
                raise ValueError("Selected audio changed after verification")
            if report.get("caption_sha256", {}).get(f"seed_{seed}") != digest(
                attempt / f"seed_{seed}_caption.json"
            ):
                raise ValueError("Selected description changed after verification")
            if audio_hash in seen_audio:
                continue
            seen_audio.add(audio_hash)
            accepted.append((name, seed, source, attempt, report, audio_hash))
    # Validate every source before making a partial export.
    for _, _, _, attempt, _, _ in accepted:
        for filename in ["program.py", "parameters.json", "model.json"]:
            path = attempt / filename
            if path.is_symlink():
                raise ValueError("Generated assets must not be symlinks")
            if re.search(r"(?:/Users/|/home/|sk-[A-Za-z0-9_-]{16,})", path.read_text()):
                raise ValueError(
                    "Generated assets contain a local home path or credential-like value"
                )
    destination.mkdir(parents=True)
    records = []
    for name, seed, source, attempt, report, audio_hash in accepted:
        sample = destination / "samples" / name
        sample.mkdir(parents=True, exist_ok=True)
        for filename in ["program.py", "parameters.json", "model.json"]:
            shutil.copyfile(attempt / filename, sample / filename)
        shutil.copyfile(attempt / f"seed_{seed}.wav", sample / f"{seed}.wav")
        dump(sample / "verification.json", report)
        caption_doc = load(attempt / f"seed_{seed}_caption.json")
        reference_doc = load(run / name / "reference_caption.json")
        dump(sample / f"{seed}.caption.json", caption_doc)
        dump(sample / "reference.caption.json", reference_doc)
        prefix = f"samples/{name}/"
        records.append(
            {
                "schema_version": 1,
                "id": f"{name}-{seed}",
                "caption": source.get("target_caption")
                or reference_doc["description"]["caption"],
                "observed_caption": caption_doc["description"]["caption"],
                "code": prefix + "program.py",
                "parameters": prefix + "parameters.json",
                "audio": prefix + f"{seed}.wav",
                "model": prefix + "model.json",
                "verification": prefix + "verification.json",
                "source_id": source["source_id"],
                "source_sha256": source["source_sha256"],
                "reference_sha256": source["reference_sha256"],
                "split": split_for(source["source_id"], config["test_fraction"]),
                "render": {
                    "duration": config["duration"],
                    "sample_rate": config["sample_rate"],
                    "seed": seed,
                },
                "hashes": {
                    "audio": audio_hash,
                    "code": digest(sample / "program.py"),
                    "parameters": digest(sample / "parameters.json"),
                    "model": digest(sample / "model.json"),
                },
                "label_status": "automatic_semantic_screen",
            }
        )
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    dump(
        destination / "dataset.json",
        {
            "schema_version": 1,
            "records": len(records),
            "split_group": "source_id",
            "test_fraction": config["test_fraction"],
            "reference_audio_included": False,
            "label_status": "automatic_semantic_screen",
            "counts": {
                s: sum(r["split"] == s for r in records) for s in ["train", "test"]
            },
        },
    )
    return records
