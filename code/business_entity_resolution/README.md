# Business entity resolution

Offline candidate retrieval and supervised LightGBM classification for the Amazon
ML Challenge 2026. Predicts zero, one, or many Source 2 / Source 3 records per
Source 1 entity. All countries in the input are supported.

## Environment

Tested with Python 3.12 on Windows, CPU only. Use a 64-bit Python and 16 GB RAM.
Allow substantial disk space for the approximately 2.5 GB source dataset, index,
candidate database, outputs, and package. Runtime depends on CPU and candidate
volume; this is a multi-million-record batch job, not an instant demo.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The supplied data directory must contain `train/train_source1.tsv`,
`train/train_source2.tsv`, `train/train_source3.tsv`,
`train/train_ground_truth.tsv`, and the three corresponding test source files.
It is not included in the submission archive.

## Reproduce

Run from this code folder, replacing the example data and output paths:

```powershell
.venv/Scripts/python.exe -u src/pipeline.py prepare --data ../../student_resource/dataset --artifacts artifacts --sample 40000
.venv/Scripts/python.exe -u src/pipeline.py train --artifacts artifacts
.venv/Scripts/python.exe -u src/pipeline.py predict --data ../../student_resource/dataset --artifacts artifacts --output ../../output
.venv/Scripts/python.exe src/validate_outputs.py --data ../../student_resource/dataset/test --output ../../output
.venv/Scripts/python.exe -m unittest discover -s src -p test_pipeline.py -v
```

Paths are relative to the working directory. Use absolute paths to avoid ambiguity.
Alternatively `all` runs preparation, training, and prediction in sequence.
The saved `artifacts/model.txt` and `config.json` allow prediction without retraining.
When extracted from the final package, the trained artifacts are already in this
folder. The raw source data must be supplied separately.

The workspace root also contains `run.ps1` for the existing installed environment:
`./run.ps1 test`, `./run.ps1 all`, or `./run.ps1 validate`.

## Method

1. Transliterate Unicode using AnyAscii, normalize punctuation/case, remove common
   legal suffix tokens, and expand a small fixed address abbreviation dictionary.
2. Build country-scoped keys for normalized names, reordered names, pairs of name
   token prefixes, normalized addresses, address numbers plus words, cross-field
   name/address tokens, and postal-code-like numbers plus name prefixes.
3. Store hashed keys and reference row indexes in sorted NumPy arrays. Ignore a
   key when it retrieves more than 80 reference records. Union the remaining
   retrieval passes and remove duplicate pairs. A final cheap filter requires
   name similarity >= 85, address similarity >= 85, or name >= 50 and address
   >= 45 (maximum of ratio and token-sort ratio, scale 0-100). No ground truth
   enters retrieval. The output candidates reflect this final filter.
4. Score every resulting candidate with a 350-tree LightGBM classifier using 29
   name/address similarity, missingness, number agreement, and conflict features.
   Country is only a normalized equality feature, not a fixed country vocabulary.
   Record IDs never become predictive features.
5. Write the exact scored candidates and matches passing the tuned threshold.
   No transitive or one-to-one constraints are imposed.

## Validation design

A fixed-seed random sample of 40,000 Source 1 entities is split 60/20/20 into
training, threshold tuning, and untouched holdout entities. Candidate retrieval
searches both full training target sources and uses the full reference index, so
block-size suppression is evaluated at the real dataset scale. All naturally
retrieved negatives are retained. Missed true matches remain in the metric's
denominator. Entities with zero candidates remain in the macro average.

Only the tuning subset selects the threshold. The model is intentionally not
refitted on holdout data, so the reported holdout corresponds to the delivered
model. `artifacts/validation.json` contains the metric, candidate recall,
singleton accuracy, country breakdown, threshold sweep, and error examples.

Local validation is not a leaderboard score. France has no supplied training
labels, so French test accuracy cannot be measured locally. Abbreviations,
transliterations, country label errors, very common names, and heavily corrupted
addresses can cause retrieval misses or false matches. This is an implemented
baseline, not a claim of research novelty or competition rank.

## Files and recovery

- `src/pipeline.py`: deterministic preparation, training, and full prediction.
- `src/validate_outputs.py`: streaming validation including target existence.
- `src/test_pipeline.py`: unit and output-contract tests.
- `src/package_submission.py`: validated package creation with SHA-256 manifest.
- `artifacts/model.txt`: LightGBM text model; no untrusted pickle loading.
- `artifacts/config.json`: chosen threshold and retrieval settings.
- `artifacts/pairs.npz`, `sample.json`: intermediate training sample.
- `artifacts/predictions.sqlite`: disk-backed inference audit table.
- `artifacts/prediction_part_*.sqlite`: country-specific inference audit tables.
- `output/matching_results.tsv`: leaderboard file.
- `output/candidate_pairs.tsv`: exact model-scored pairs.

Prediction defaults to three independent country partitions. Each worker holds
only its country's normalized references/index, avoiding duplicated full indexes.
Use `--workers 1` to process the partitions sequentially on machines with less RAM.

Rerunning `prepare` or `predict` rebuilds that stage; resumption mid-stage is not
implemented. TSVs are replaced only after complete generation. Inspect process
exit status and rerun validation after every prediction run; old complete TSVs
may remain if a newer run fails.

## Data and licensing

Only the provided dataset is used for entity matching. No web lookup, geocoding,
business registry, external labeled data, embeddings, or pretrained entity model
is used. AnyAscii supplies general character transliteration, not business facts.
The video transcription used for understanding instructions is separate from
the model and is not a prediction dependency.

The solution code and trained model are released under MIT. LightGBM is MIT
licensed: https://github.com/lightgbm-org/LightGBM . A 350-tree model is far below
the challenge's 8-billion-parameter ceiling. Dependencies retain their respective
licenses. Supplied challenge data/materials retain their original ownership.
