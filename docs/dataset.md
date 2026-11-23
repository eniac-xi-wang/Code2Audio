# Data format

## Input JSONL

Each line names a local WAV reference. Relative paths resolve against the manifest location.

```json
{"id":"rain-001","audio":"audio/rain.wav","start":60,"source_id":"recording-001","source_uri":"https://example.org/recording-001","caption":"Steady rain with dense individual impacts."}
```

`id` and `audio` are required. `start` defaults to zero. The configured duration is applied to every item. `source_id` groups related excerpts for splitting; when omitted, the full source-file hash is used. Supply one shared ID for multiple files from the same original recording. `caption` is an optional target/intent supplied by the source dataset; otherwise the reference's independent caption becomes the label. The original target is never replaced just because a candidate failed to match it.

## Exported JSONL

`records.jsonl` has one record per selected program and render seed:

```json
{
  "schema_version": 1,
  "id": "rain-001-17",
  "caption": "Steady rain with dense individual impacts.",
  "observed_caption": "Continuous rain falling with a granular patter.",
  "code": "samples/rain-001/program.py",
  "parameters": "samples/rain-001/parameters.json",
  "model": "samples/rain-001/model.json",
  "audio": "samples/rain-001/17.wav",
  "verification": "samples/rain-001/verification.json",
  "source_id": "recording-001",
  "split": "train",
  "render": {"duration": 8.0, "sample_rate": 44100, "seed": 17},
  "label_status": "automatic_semantic_screen"
}
```

Actual records also contain the original source/reference hashes and hashes of the code, parameters, component notes and generated audio. Export verifies that these assets still match the files that passed evaluation. The independent reference/generated caption records preserve model and prompt provenance. All asset paths are relative to the dataset root.

Splits are deterministic by `source_id`, with a configurable test fraction (default 1%). Multiple excerpts, revisions and seeds from a source stay in the same split. Duplicate generated audio hashes are exported once. Conflicting source-group IDs for the same original file are rejected. Very small datasets can legitimately have no test records; the pipeline does not move related samples across splits to fill a quota.

Load the code and parameters through the record paths, then call `c2a render` using the stored render settings to reproduce the WAV. The reference recording is not needed for rendering and is not included in exported datasets.

`automatic_semantic_screen` documents how the label was assigned. It does not assert human review, precise reconstruction or calibrated perceptual quality. `dataset.json` records counts and the split policy. Keep original-source usage terms and provenance with the input collection when preparing a public dataset.
