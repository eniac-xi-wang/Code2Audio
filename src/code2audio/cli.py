"""Command-line tools for the agent-driven Code2Audio workflow."""

import argparse
import json
import os
from pathlib import Path

from .analysis import analyze
from .captions import describe, env_file
from .dataset import export
from .io import digest, dump, load
from .render import render
from .workflow import build, configuration


def main(argv=None):
    parser = argparse.ArgumentParser(prog="c2a")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("audio")
    p.add_argument("--output", required=True)
    p = sub.add_parser("caption")
    p.add_argument("audio")
    p.add_argument("--output", required=True)
    p.add_argument("--env-file")
    p.add_argument("--cache")
    p = sub.add_parser("render")
    for flag in ["program", "parameters", "output"]:
        p.add_argument("--" + flag, required=True)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--seed", type=int, default=17)
    p = sub.add_parser("verify")
    p.add_argument("--reference", required=True)
    p.add_argument("--reference-caption", required=True)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--caption", action="append", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--target-caption")
    p = sub.add_parser("build")
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--env-file")
    p.add_argument("--model-dir")
    p.add_argument(
        "--agent-command",
        help="JSON array; receives a task on stdin and writes candidate files in its working directory",
    )
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--example", default="examples/rain")
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir")
    p = sub.add_parser("fetch-model")
    p.add_argument("--out", default="models/clap")
    p.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    if hasattr(args, "env_file"):
        env_file(args.env_file)
    if (
        args.command in ["analyze", "caption", "render", "verify"]
        and Path(args.output).exists()
    ):
        parser.error("Output exists; choose a new output path")
    if args.command == "analyze":
        dump(args.output, analyze(args.audio))
    elif args.command == "caption":
        dump(args.output, describe(args.audio, args.cache))
    elif args.command == "render":
        result = render(
            args.program,
            args.parameters,
            args.output,
            args.duration,
            args.sample_rate,
            args.seed,
        )
        print(json.dumps(result))
    elif args.command == "verify":
        from .semantic import Clap, verify

        if len(args.candidate) != len(args.caption):
            parser.error("Provide one caption file per candidate")
        paths = {f"candidate_{i}": p for i, p in enumerate(args.candidate)}
        captions = {
            "reference": load(args.reference_caption),
            **{f"candidate_{i}": load(p) for i, p in enumerate(args.caption)},
        }
        result = verify(
            Clap(args.model_dir),
            args.reference,
            paths,
            captions,
            configuration(args.config),
            args.target_caption,
        )
        dump(args.output, result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "semantic_pass",
                        "reference_target_match",
                        "candidate_target_match",
                        "threshold",
                    ]
                }
            )
        )
    elif args.command == "build":
        if args.model_dir:
            os.environ["CLAP_MODEL_DIR"] = str(Path(args.model_dir).resolve())
        result = build(
            args.manifest,
            args.out,
            configuration(args.config),
            json.loads(args.agent_command) if args.agent_command else None,
            resume=args.resume,
        )
        print(
            json.dumps(
                [{k: s[k] for k in ["id", "status", "selected"]} for s in result]
            )
        )
    elif args.command == "export":
        print(json.dumps({"records": len(export(args.run, args.out))}))
    elif args.command == "demo":
        example = Path(args.example)
        out = Path(args.out)
        if out.exists():
            parser.error("Demo output directory must be new")
        out.mkdir(parents=True)
        settings = load(example / "example.json")["render"]
        render(
            example / "program.py",
            example / "parameters.json",
            out / "generated.wav",
            **settings,
        )
        dump(out / "analysis.json", analyze(out / "generated.wav"))
        result = {
            "reproduces_included_audio": digest(out / "generated.wav")
            == digest(example / "generated.wav")
        }
        if args.model_dir:
            from .semantic import Clap, verify

            docs = {
                "reference": load(example / "reference.caption.json"),
                "rain": load(example / "generated.caption.json"),
            }
            report = verify(
                Clap(args.model_dir),
                example / "reference.wav",
                {"rain": out / "generated.wav"},
                docs,
                configuration(),
            )
            dump(out / "verification.json", report)
            result["semantic_pass"] = report["semantic_pass"]
        print(json.dumps(result))
    elif args.command == "fetch-model":
        from huggingface_hub import snapshot_download

        snapshot_download(
            "Xenova/clap-htsat-unfused",
            revision=args.revision,
            local_dir=args.out,
            allow_patterns=[
                "*.json",
                "onnx/audio_model.onnx",
                "onnx/text_model_quantized.onnx",
            ],
        )
        print("CLAP files downloaded to " + args.out)


if __name__ == "__main__":
    main()
