# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** MLG4  
**Team Leader:** Vijey  
**Team Members:** Vijey, Christo, Priyaranjan, Saai kapileshwar  
**Preparation Date:** 27 September 2026

## 1. Executive Summary

We use country-scoped, multi-pass blocking followed by a supervised LightGBM
classifier on name and address similarity features. The decision threshold is
chosen using entity-level macro F0.5 on a separate tuning subset. Every test
Source 1 entity receives a row, including entities for which no match is predicted.

This is an engineering solution using established methods, not a claim of a novel
algorithm. All business information comes from the supplied challenge dataset.

## 2. Methodology

### 2.1 Problem Analysis

The training data has 2,206,821 reference businesses, 5,034,616 Source 2 records,
and 5,285,603 Source 3 records. The test data has 1,732,544 reference businesses,
4,887,273 Source 2 records, and 5,082,316 Source 3 records. Test references include
663,106 US, 809,986 India, and 259,452 France businesses. France is absent from
the labeled training set.

There are 123,247 singleton training references (5.5848%). Predicting an empty
list for every training reference would achieve macro F0.5 of 0.0558482. The
training labels contain 7,638,365 true links in total.

Records have inconsistent legal suffixes, punctuation, address abbreviations,
ordering, and scripts. Names and addresses can each be misleading in isolation:
similar names can belong to different businesses, and multiple businesses can
share an address. Entity IDs are opaque identifiers, not modeling features.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted classifier.

Normalize case and punctuation, transliterate Unicode using AnyAscii, remove a
fixed set of common company suffixes from names, and expand common address
abbreviations. Retain a normalized name before suffix removal as an additional
feature. Country labels are normalized strings with no closed country vocabulary.

Build compact sorted indexes from Source 1, then stream the other sources. This
reverses the lookup direction for memory efficiency but still produces all
retrieved Source 1-to-target pairs. Store scored pairs on disk and group them by
Source 1 when writing the outputs. Prediction partitions the work by normalized
country label and processes up to three partitions concurrently, with no shared
mutable worker state. The final merge restores the original Source 1 row order.

## 3. Candidate Generation (Blocking)

The union of these country-scoped passes supplies coarse candidates:

- Exact compact normalized name and token-reordered normalized name.
- Pairs of four-character prefixes from up to four longer name tokens.
- Exact token-reordered normalized address.
- Address number plus prefix of a longer address word.
- Name-word prefix plus address-word prefix.
- Five- or six-digit address number plus name-token prefix.

Keys are represented by 64-bit BLAKE2b hashes. A key retrieving more than 80
reference records is skipped. Hash collisions can only introduce coarse
candidates; full string features are still evaluated.

A final inexpensive blocking filter retains pairs with normalized name ratio
or token-sort ratio at least 85/100, address ratio or token-sort ratio at least
85/100, or both name similarity at least 50/100 and address similarity at least
45/100. Empty strings do not count as agreement. This filter is identical in
training and prediction. `candidate_pairs.tsv` contains the exact unique pairs
remaining after this filter, which are all scored by the classifier.

Training preparation first computes block frequencies over the full reference
set, then retains the sampled reference rows for efficient lookup. This preserves
the full-data block-size rule in local evaluation. Missed true matches are not
injected into training or validation candidates; they lower measured recall.

**Candidate pairs generated:** Recorded in `artifacts/prediction_summary.json`
after the completed test run.

**Recall measurement:** Candidate recall is measured against all labeled true
matches for held-out reference entities. Multi-pass retrieval mitigates misses,
but there is no guarantee of complete recall.

## 4. Matching Model

**Model type:** LightGBM binary classifier, 350 trees, 31 leaves per tree,
learning rate 0.05, minimum 40 samples per leaf, L2 regularization 2.0, feature
fraction 0.9, deterministic seed 2026.

**Features:** 29 numeric features covering name and address edit similarity,
token-sort similarity, token-set similarity, partial similarity, token Jaccard
and overlap, exact agreement, length ratios, acronym agreement, address-number
overlap and conflicts, postal-code-like number agreement and conflicts,
missingness, country equality, normalized original-name similarity, first-number
agreement, and name token counts. IDs and fixed country encodings are excluded.

**Training:** A seeded random sample of 40,000 reference entities. Split 60% for
model fitting, 20% for threshold selection, and 20% for final held-out evaluation.
All retrieved negative pairs for these entities are retained. Both target sources
are searched in full.

**Threshold selection:** Sweep 0.05 through 0.99 in increments of 0.01 on the tuning
subset, maximizing entity-level macro F0.5. Ties favor the higher threshold.
Correctly empty singleton predictions receive 1.0. Non-singleton score is
`1.25 * true_positives / (predicted_count + 0.25 * true_count)`.

The saved model is the same model measured on the untouched holdout. It is not
refit using holdout labels. No global one-to-one constraint or graph propagation
is imposed; multiple target matches are allowed.

**Licensing:** LightGBM and this solution's source code/model use MIT licensing.
The tree model has far fewer than 8 billion learned values. No pretrained
business-identity model is used.

## 5. Results & Error Analysis

Measured results are saved in `code/business_entity_resolution/artifacts/validation.json`
in the submission package. The local workspace uses `artifacts/validation.json`.
The results include macro F0.5, pair precision/recall, candidate recall, singleton
accuracy, country breakdowns, an empty-output baseline, and actual error examples.
Final measured figures will be inserted here when the run completes.

Potential false merges include businesses with similar names sharing a location
and short generic names. Potential missed matches include heavy corruption of
both name and address, transliteration differences, blocks suppressed for excessive
size, and differing country labels. Error-example records distinguish observed
mistakes from these anticipated failure modes.

There are no French training labels, so French test accuracy cannot be reported
locally. No public/private leaderboard result is claimed. Passing an output
validator establishes formatting and ID correctness, not predictive accuracy.

## 6. Conclusion

The pipeline makes full-scale entity matching feasible using indexed candidate
retrieval, a supervised precision-oriented decision rule, and a complete candidate
audit. Independent holdout evaluation and strict output validation address
different requirements and are reported separately.

## Appendix

### A. Code Artefacts

The complete pipeline is in `code/business_entity_resolution/src/pipeline.py`.
Use `prepare`, `train`, and `predict` subcommands, followed by
`src/validate_outputs.py`. The README contains exact setup and reproduction
commands, pinned dependencies, hardware guidance, and recovery behavior. Unit
tests live in `src/test_pipeline.py`. Saved model/config files permit inference
without repeating training preparation.

### B. Additional Results

Machine-readable validation and prediction summaries are included with the model.
`output/matching_results.tsv` is the leaderboard upload, and
`output/candidate_pairs.tsv` is the exact classifier input candidate audit.

### C. Data Sources and Tools

Entity training and prediction use only the supplied TSV data. There are no
business lookups, geocoding calls, external entity databases, or external labeled
business data. General-purpose software libraries handle string normalization,
numeric computation, and classification. Local speech transcription was used only
to review the supplied explanatory video; it is not part of the matching model.
