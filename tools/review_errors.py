"""Join held-out error examples back to supplied records for manual inspection."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
report=json.loads((root/'artifacts/validation.json').read_text())
examples=report['holdout']['error_examples']
needed={x['source1_entity_id'] for x in examples}
for x in examples:needed.update(x['truth']);needed.update(x['predicted'])
records={}
for source in (1,2,3):
    with (root/f'student_resource/dataset/train/train_source{source}.tsv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['entity_id'] in needed:records[row['entity_id']]=row
joined=[]
for x in examples:
    item=dict(x);item['reference_record']=records.get(x['source1_entity_id'])
    item['target_records']={t:records.get(t) for t in sorted(set(x['truth'])|set(x['predicted']))}
    joined.append(item)
(root/'reports/error_review.json').write_text(json.dumps(joined,indent=2,ensure_ascii=False),encoding='utf-8')
print(f'Wrote {len(joined)} held-out error examples with original records.')
