import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from code2audio.dataset import export
from code2audio.io import digest, dump, load, write_audio
from code2audio.render import render
from code2audio.semantic import verify
from code2audio.workflow import build, command_agent, configuration

PROGRAM = """import numpy as np
def render(duration, sample_rate, seed, parameters):
    rng=np.random.default_rng(seed)
    return rng.uniform(-.1,.1,round(duration*sample_rate))*parameters.get("gain",1.)
"""


class FakeServices:
    def caption(self, path):
        return {
            "description": {
                "caption": "A continuous textured sound.",
                "attributes": ["A continuous textured sound."],
                "uncertainties": [],
            },
            "audio_sha256": digest(path),
            "request_type": "independent_audio_caption",
            "model": "test-fixture",
        }

    def evaluate(self, reference, candidates, captions, config, target):
        number = int(next(iter(candidates.values())).parent.name.split("_")[-1])
        score = 0.2 if number == 1 else 0.6
        return {
            "semantic_pass": score >= 0.35,
            "mean_target_match": score,
            "candidate_target_match": {n: score for n in candidates},
            "reference_target_match": 0.5,
            "threshold": 0.42,
            "model": {"name": "test-fixture"},
        }


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.program = self.root / "program.py"
        self.program.write_text(PROGRAM)
        self.params = self.root / "parameters.json"
        dump(self.params, {})

    def tearDown(self):
        self.temp.cleanup()

    def test_isolated_reproduction_and_seed_variation(self):
        for name, seed in [("a", 7), ("b", 7), ("c", 8)]:
            render(
                self.program, self.params, self.root / (name + ".wav"), 1, 8000, seed
            )
        self.assertEqual(digest(self.root / "a.wav"), digest(self.root / "b.wav"))
        self.assertNotEqual(digest(self.root / "a.wav"), digest(self.root / "c.wav"))

    def test_file_access_and_invalid_samples_fail(self):
        self.program.write_text("import os\n" + PROGRAM)
        with self.assertRaises(ValueError):
            render(self.program, self.params, self.root / "bad.wav", 1, 8000, 1)
        self.program.write_text("from numpy import load as read\n" + PROGRAM)
        with self.assertRaises(ValueError):
            render(self.program, self.params, self.root / "bad.wav", 1, 8000, 1)
        self.program.write_text(
            PROGRAM.replace(
                "rng.uniform(-.1,.1,round(duration*sample_rate))",
                "np.full(round(duration*sample_rate),np.nan)",
            )
        )
        with self.assertRaises(ValueError):
            render(self.program, self.params, self.root / "bad.wav", 1, 8000, 1)

    def test_timeout(self):
        self.program.write_text(
            "def render(duration,sample_rate,seed,parameters):\n    while True: pass\n"
        )
        with self.assertRaises(RuntimeError):
            render(self.program, self.params, self.root / "loop.wav", 1, 8000, 1, 1)

    def test_caption_binding_and_semantic_threshold(self):
        for name in ["ref", "candidate"]:
            write_audio(self.root / (name + ".wav"), 8000, np.ones(8000) * 0.01)
        service = FakeServices()
        docs = {
            n: service.caption(self.root / (p + ".wav"))
            for n, p in [("reference", "ref"), ("candidate", "candidate")]
        }

        class Scores:
            identity = {"name": "fixture"}

            def texts(self, t):
                return np.zeros((len(t), 2))

            def score_audio(self, path, t):
                value = 0.5 if Path(path).stem == "ref" else 0.6
                return np.full(len(t), value), np.full((1, len(t)), value), [[0, 1]]

        result = verify(
            Scores(),
            self.root / "ref.wav",
            {"candidate": self.root / "candidate.wav"},
            docs,
            configuration(),
        )
        self.assertTrue(result["semantic_pass"])
        docs["candidate"]["audio_sha256"] = "incorrect"
        with self.assertRaises(ValueError):
            verify(
                Scores(),
                self.root / "ref.wav",
                {"candidate": self.root / "candidate.wav"},
                docs,
                configuration(),
            )

    def test_iteration_retains_accepted_candidate_export_and_resume(self):
        audio = self.root / "input.wav"
        write_audio(audio, 8000, np.random.default_rng(0).uniform(-0.1, 0.1, 8000))
        manifest = self.root / "inputs.jsonl"
        manifest.write_text(
            json.dumps(
                {"id": "sample", "audio": "input.wav", "source_id": "source-group"}
            )
            + "\n"
        )
        calls = []

        def agent(task, directory):
            calls.append(directory.name)
            (directory / "program.py").write_text(PROGRAM)
            dump(
                directory / "parameters.json", {"gain": 0.0 if len(calls) == 3 else 1.0}
            )
            dump(
                directory / "model.json",
                {
                    "components": ["random waveform"],
                    "controls": ["gain"],
                    "assumptions": [],
                },
            )

        config = {
            **configuration(),
            "duration": 1.0,
            "sample_rate": 8000,
            "seeds": [7, 8],
        }
        run = self.root / "run"
        result = build(manifest, run, config, services=FakeServices(), agent_fn=agent)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result[0]["selected"]["attempt"], "attempt_002")
        exported = export(run, self.root / "dataset")
        self.assertEqual(len(exported), 2)
        self.assertEqual(len({r["split"] for r in exported}), 1)
        self.assertFalse(list((self.root / "dataset").rglob("reference.wav")))
        self.assertFalse(list((self.root / "dataset").rglob("agent.log")))
        build(
            manifest, run, config, services=FakeServices(), agent_fn=agent, resume=True
        )
        self.assertEqual(len(calls), 3)
        changed = {**config, "duration": 2}
        with self.assertRaises(ValueError):
            build(
                manifest,
                run,
                changed,
                services=FakeServices(),
                agent_fn=agent,
                resume=True,
            )
        selected = run / "sample" / "attempt_002" / "parameters.json"
        selected.write_text('{"gain":0.1}')
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            export(run, self.root / "changed-dataset")
        self.assertFalse((self.root / "changed-dataset").exists())

    def test_command_adapter_excludes_caption_credentials(self):
        adapter = self.root / "adapter.py"
        adapter.write_text(
            'import os,json,sys\nfrom pathlib import Path\ntask=json.load(sys.stdin)\nassert "QWEN_API_KEY" not in os.environ\nPath("result.json").write_text(json.dumps(task))\n'
        )
        job = self.root / "job"
        job.mkdir()
        with patch.dict(os.environ, {"QWEN_API_KEY": "fixture-secret"}):
            command_agent([sys.executable, str(adapter)], {"id": "one"}, job, 5)
        self.assertEqual(load(job / "result.json"), {"id": "one"})

    def test_service_failure_is_unresolved_and_other_items_continue(self):
        write_audio(
            self.root / "input.wav",
            8000,
            np.random.default_rng(3).uniform(-0.1, 0.1, 8000),
        )
        manifest = self.root / "inputs.jsonl"
        manifest.write_text(
            "".join(
                json.dumps({"id": name, "audio": "input.wav"}) + "\n"
                for name in ["first", "second"]
            )
        )

        class FlakyServices(FakeServices):
            failed = False

            def caption(self, path):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("Temporary caption failure")
                return super().caption(path)

        def agent(task, directory):
            (directory / "program.py").write_text(PROGRAM)
            dump(directory / "parameters.json", {})
            dump(
                directory / "model.json",
                {"components": [], "controls": [], "assumptions": []},
            )

        config = {
            **configuration(),
            "duration": 1.0,
            "sample_rate": 8000,
            "seeds": [7],
            "max_rounds": 2,
        }
        service = FlakyServices()
        run = self.root / "run"
        result = build(manifest, run, config, services=service, agent_fn=agent)
        self.assertEqual(
            [r["status"] for r in result], ["unresolved", "accepted_semantic"]
        )
        result = build(
            manifest, run, config, services=service, agent_fn=agent, resume=True
        )
        self.assertEqual(
            [r["status"] for r in result], ["accepted_semantic", "accepted_semantic"]
        )


if __name__ == "__main__":
    unittest.main()
