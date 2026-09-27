# Amazon ML Challenge 2026 solution

Dataset: `student_resource/dataset` (extracted from the provided ZIP).
Source: `code/business_entity_resolution/src`.

The provided PDF and full video were reviewed. Their central requirement is to
match every Source 1 business to all corresponding Source 2 / Source 3 records,
allow empty matches, and produce two correctly formatted TSV files.

## Run on this computer

```powershell
cd 'C:\Users\Saaik\Downloads\Ml amazon'
./run.ps1 test
./run.ps1 all
./run.ps1 validate
```

The prepared `.venv` contains the required environment. Individual stages are
available as `./run.ps1 prepare`, `./run.ps1 train`, and `./run.ps1 predict`.
The full run processes millions of businesses and takes substantial time.

See `code/business_entity_resolution/README.md` for setup on another computer,
methodology, reproducibility, limitations, and exact commands.

## Results

After a successful run:

- `output/matching_results.tsv`: file for the leaderboard portal.
- `output/candidate_pairs.tsv`: candidate audit required in the final package.
- `artifacts/validation.json`: held-out accuracy, independent of output validation.
- `output/validation.json`: structural and ID-existence checks.
- `artifacts/prediction_summary.json`: full-dataset run counts and timing.

No files are uploaded automatically. The team name and member details must be
reviewed in `Documentation_template.md` before final submission. The document
lists MLG4, leader Vijey, and members Christo, Priyaranjan, and Saai kapileshwar.
