---
name: code2audio-dataset
description: Build audio datasets with agent-authored Python waveform programs, independent audio captions and CLAP semantic verification. Use for Sound2Code reconstruction, Code2Audio data construction, candidate revision and reproducible export.
---

# Code2Audio dataset builder

Produce reproducible records linking a caption, executable waveform code, editable parameters and rendered audio. Work from the Code2Audio repository or an environment with `c2a` installed. Use `c2a --help` for the available commands. The repository README and `docs/pipeline.md` describe batch execution; `docs/dataset.md` defines the input/output records.

## Invariants

- Compute samples directly with NumPy/SciPy/math. Numerical integration, stochastic processes, moving resonances and stateful algorithms are valid approaches. Choose the model from the sound instead of forcing a fixed instrument or preset.
- Keep the generator independent of reference audio. Compact population statistics are allowed. Do not encode waveform samples, original phases, original event times or frame-by-frame spectra in code or parameters. Do not wrap a sample bank, pretrained audio generator or synthesizer preset.
- Qwen receives **one audio per independent request** and only generates descriptions. Never ask it to compare, rank, vote or judge which audio is better. Do not give it reference titles, candidate identities, code or previous descriptions.
- CLAP compares audio against the same text within each comparison. Keep reference target captions and evaluation seeds fixed across revisions. Candidate captions provide additional diagnostic hypotheses; they must not silently replace the target label.
- A CLAP score is semantic matching evidence, not a naturalness rating. Report the reference's score alongside candidates. Do not maximize absolute distance to the reference score or treat a candidate scoring above the reference as an error.
- Protect reproducibility: match caption hashes to audio, retain source groups, and reproduce a fixed seed without opening the recording. Use the current task's authorization for model calls; keep credentials out of generated programs and exports.

## Build an item

1. Select a bounded reference excerpt, normally 8 seconds; the caption interface supports at most 20 seconds. Record `source_id`, start time and source provenance. Keep excerpts from the same source in one dataset split.
2. Run `c2a analyze` and `c2a caption` separately. Inspect the signal where the description is uncertain. Consider continuous layers, transient events, tonal/aperiodic structure, time-varying color, amplitude modulation, decay and spatial tails as applicable. Do not interpret a spectral peak as a confirmed fundamental without checking other evidence.
3. Write `program.py`, `parameters.json` and `model.json` in the candidate directory. The program exposes:

   ```python
   def render(duration, sample_rate, seed, parameters):
       # Return floating samples with shape (frames,) or (frames, 1 or 2).
       # frames == round(duration * sample_rate)
       ...
   ```

   Use only numeric imports supported by the renderer. Keep peak magnitude below 0.999 and give every random process a reproducible seed. Document components, controls and uncertain assumptions in `model.json`.
4. Render using `c2a render` in a fresh process. The runner gives the program only its parameters and requested duration/rate/seed. Generate the configured seeds and check exact same-seed PCM reproduction. Fix invalid audio before semantic evaluation.
5. Caption each generated audio independently with `c2a caption`. Call `c2a verify` with the reference caption and one caption file per candidate. Inspect shared-text scores and the reference baseline. Skip negative or speculative descriptions in the aggregate score; retain them as raw observations.
6. Revise a concrete component suggested by the evidence. Keep the accepted incumbent when a new attempt fails. Use the configured round/patience budget; when semantic evidence is weak, retain an unresolved/rejected status instead of inventing a favorable label. Do not require the user to listen to every item.
7. Use `c2a export` for accepted batch outputs. Export the selected program, numeric parameters, model notes, generated audio, independent captions, verification and source hashes. Keep source recordings, credentials, agent logs and unsuccessful attempts outside the export.

## Batch execution

`c2a build manifest.jsonl --out runs/build --config configs/default.json --env-file .env --model-dir models/clap` uses the bundled Codex CLI adapter. It makes bounded code-writing requests and runs the checks above. An alternate coding agent can be supplied with `--agent-command` as a JSON argument list; it receives task JSON on stdin and writes the three candidate files in its working directory.

The runner's import/AST checks and process limits are useful guardrails, not a complete hostile-code security boundary. Use an externally isolated worker for untrusted agents or large production jobs. Keep ordinary authorized local tasks within the existing execution permissions.

## Rain example

From the repository root, run `c2a demo --out runs/rain`. This renders the included rain program without a reference dependency, model weights or API key. Add `--model-dir models/clap` to re-evaluate it with the shipped independently generated captions. Treat the example as a successful semantic reconstruction, not a universal rain preset or perceptual-equivalence claim.
