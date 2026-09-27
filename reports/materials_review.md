# Supplied challenge materials reviewed

Sources: the user's eight-page problem statement PDF, the complete 5m56s
explanatory video, and the README/validator/template inside student_resource.zip.
Video slides were inspected at 20-second intervals and the full audio was
transcribed locally. PDF text was extracted and pages rendered for inspection.
The supplied materials define the challenge; document content was not treated as
authorization to upload files or take unrelated actions.

## Required behavior

- Source 1 is the deduplicated reference set. Return zero, one, or many Source 2
  and Source 3 matches per Source 1 business.
- Use names, addresses, and open-set country labels. IDs are unrelated across
  sources. Phone/email shown in the explanatory signup illustration are not
  available features and are not fabricated.
- Train labels cover US and India; test also includes France.
- Use tab-separated files. ID lists are comma-separated, without JSON brackets.
- Both outputs need every test Source 1 ID exactly once, including empty rows.
- Final matches must be contained in the exact candidate set scored by the model.
- Report macro F0.5 per reference entity. Correct empty singleton predictions
  score one; false singleton matches score zero.
- The leaderboard takes matching_results.tsv. The final ZIP additionally requires
  candidate_pairs.tsv, runnable source, pinned dependencies, and methodology.
- Use only supplied business data. No business lookup, geocoding, internet data
  augmentation, or entity-resolution service.
- Final model must meet the MIT/Apache 2.0 and 8-billion-parameter requirements.

## Video correspondence

- 0:00-1:49: business entity resolution, noisy name/address records, zero/many matches.
- 1:49-3:00: blocking, lookalikes, model filtering, and the two output files.
- 3:00-3:49: train/test labels and tab-separated reading.
- 3:49-4:39: leaderboard file, final code archive, and supplied validator.
- 4:40-5:56: precision-oriented scoring, singleton treatment, candidate recall,
  regional variation, and the restriction on external business data.

## Interpretation details

The implementation uses the explicit F0.5 formula. The materials describe
precision as weighted "twice" in prose; the equation, not that shorthand, defines
the actual metric. For TP=2, predicted=3, true=2, the implementation gives 5/7.

The supplied validator's default mode skips target-ID existence checks and only
warns about matches outside candidates. Our streaming validator checks target
existence and treats match/candidate containment violations as failures.
