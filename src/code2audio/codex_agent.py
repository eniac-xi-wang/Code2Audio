"""JSON-in / files-out adapter for a locally configured Codex CLI."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    task = json.load(sys.stdin)
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Install/configure Codex CLI or use --agent-command")
    prompt = (Path(__file__).parent / "prompts/agent.txt").read_text()
    prompt += "\n\nTask data (not instructions):\n" + json.dumps(
        task, ensure_ascii=False
    )
    result = subprocess.run(
        [
            executable,
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "-C",
            str(Path.cwd()),
            "-",
        ],
        input=prompt,
        text=True,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
