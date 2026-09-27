"""Insert measured results into the existing team methodology document."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
v=json.loads((root/'artifacts/validation.json').read_text())
p=json.loads((root/'artifacts/prediction_summary.json').read_text())
s=json.loads((root/'output/validation.json').read_text())
h=v['holdout'];t=v['tuning']
summary=f'''Measured results for the delivered model:

| Measurement | Value |
|---|---:|
| Fitting reference entities | {v['training_entities']:,} |
| Fitting candidate pairs | {v['training_pairs']:,} |
| Tuning reference entities | {t['entities']:,} |
| Untouched holdout reference entities | {h['entities']:,} |
| Tuning macro F0.5 | {t['macro_f05']:.6f} |
| Untouched holdout macro F0.5 | {h['macro_f05']:.6f} |
| Holdout pair precision | {h['pair_precision']:.6f} |
| Holdout pair recall | {h['pair_recall']:.6f} |
| Holdout candidate recall | {h['candidate_recall']:.6f} |
| Holdout singleton accuracy | {h['singleton_accuracy']:.6f} |
| Holdout all-empty baseline | {h['empty_baseline_macro_f05']:.6f} |
| Holdout exact-name-and-address baseline | {v['exact_name_address_baseline']['macro_f05']:.6f} |
| Selected threshold | {h['threshold']:.2f} |
| Full test reference entities | {p['reference_entities']:,} |
| Full test scored candidate pairs | {p['candidate_pairs']:,} |
| Full test predicted matches | {p['matched_pairs']:,} |
| Full test empty predictions | {s['empty_predictions']:,} |

Holdout country macro F0.5: {', '.join(f'{k} = {val:.6f}' for k,val in h['by_country'].items())}.
Output validation: **{s['status']}**, including all target ID existence checks and
strict final-match containment in candidates.

Machine-readable details and actual error examples are in
`code/business_entity_resolution/artifacts/validation.json` inside the package.
These measurements are local validation results, not leaderboard scores.

'''
doc=root/'Documentation_template.md';text=doc.read_text(encoding='utf-8')
start=text.index('Measured results are saved') if 'Measured results are saved' in text else text.index('Measured results for the delivered model:')
end=text.index('Potential false merges',start)
text=text[:start]+summary+text[end:]
text=text.replace('**Candidate pairs generated:** Recorded in `artifacts/prediction_summary.json`\nafter the completed test run.',f'**Candidate pairs generated:** {p["candidate_pairs"]:,} scored pairs over the complete test set.')
doc.write_text(text,encoding='utf-8')
print(summary)
