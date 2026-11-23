# Pipeline and agent interface

`c2a build` handles each manifest item independently. It copies the requested excerpt, measures it, obtains one independent reference caption, and gives the coding agent the description, measurements and reference path. The agent writes a waveform program and editable parameters. The pipeline then renders, checks reproducibility, captions each result independently and evaluates it with CLAP.

## Agent contract

The default adapter runs a locally configured `codex exec` with its normal workspace-write execution policy. Model selection and agent authentication belong to the local Codex configuration. The caption API key is removed from the agent subprocess environment.

Another agent can be connected through an argument list:

```bash
c2a build inputs.jsonl --out runs/build --env-file .env \
  --model-dir models/clap \
  --agent-command '["python", "/absolute/path/to/my_agent.py"]'
```

The command receives JSON on stdin and runs with the candidate directory as its working directory. It must exit successfully and write:

| File | Contract |
| --- | --- |
| `program.py` | `render(duration, sample_rate, seed, parameters)` returns finite floating mono/stereo samples. |
| `parameters.json` | A JSON object with compact, editable parameters. |
| `model.json` | An object containing `components`, `controls`, and `assumptions`. |

Task JSON includes `reference_description`, `analysis`, `reference_audio`, an optional `target_caption`, the render configuration, `previous_feedback`, and a `previous_program` path when available. The bundled prompt lives in `src/code2audio/prompts/agent.txt`.

The renderer executes a copied program and parameters in a fresh temporary directory. It passes no caption credentials or reference file into that directory. Static import/access checks reject common file, network and introspection operations. A timeout and process resource limits bound execution. These checks are not a complete security boundary against hostile Python; production workers processing untrusted code should provide external container/OS isolation.

## Caption and verification protocol

Qwen sees exactly one audio per stateless request. The prompt requests a caption, affirmative audible attributes and uncertainties. It never requests preferences, comparisons or quality scores. Audio is converted to mono 24 kHz PCM16 for captioning; the full excerpt is sent. Cached captions are keyed by the audio hash, prompt, model and endpoint. Exports retain the audio/prompt hashes and model name, not credentials or raw requests.

CLAP evaluates every current audio against the same pool of independent descriptions. Reference captions and an optional supplied target caption define the fixed target rows. Candidate captions add diagnostic rows, not new target labels. Negative and speculative text is retained in the report but omitted from the target aggregate.

For reference target texts \(T\), the target score is the mean audio–text cosine:

\[
s(x)=\frac{1}{|T|}\sum_{t\in T}\cos\bigl(e_a(x), e_t(t)\bigr).
\]

Longer excerpts are evaluated in consecutive windows of at most 10 seconds, weighted by their actual duration. Short windows use the model's repeat/pad preprocessing. Texts are encoded independently without cross-caption padding. The model weight/tokenizer hashes are saved with each report.

The default screening rule requires a supported reference score and **every configured seed** to satisfy:

\[
s(x)\geq\max(0.35,\ s(x_{ref})-0.08).
\]

These are configurable operating thresholds, not calibrated probabilities or a perceptual-equivalence test. A generated audio can score higher than the reference against a caption. Per-caption differences help localize a mismatch; their absolute average is not used as a quality score.

## Iteration and selection

Defaults are three attempts, two render seeds, and patience of two attempts without an accepted improvement. An accepted candidate replaces the incumbent only when its mean target match improves by more than `minimum_improvement`. A later invalid or weaker attempt leaves the incumbent intact. Validation includes finite/non-silent audio, correct shape/duration, no clipping, and exact same-seed PCM reproduction.

`accepted_semantic` means the configured checks passed. `rejected` means at least one candidate was evaluated and the attempt budget ended without a passing candidate. `unresolved` means preparation failed or no attempt reached a semantic evaluation. `--resume` checks the manifest, source hashes, configuration and CLAP model identity; it reuses the stored reference caption. Completed accepted/rejected items are preserved. Unresolved preparation failures can be retried; attempted failures still consume the round budget. Use a new run for a changed manifest, verification model or evaluation policy.

## Local work and dataset output

The run directory holds reference excerpts, attempts, captions and feedback for debugging/resume. It is ignored by Git. `c2a export` writes only selected records and their required assets to a new directory. Source recordings and unsuccessful attempts are not copied into the dataset. Nothing in the pipeline publishes, pushes or uploads a dataset.
