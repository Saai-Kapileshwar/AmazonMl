"""Apply the supplied validator's unmodified checks in bounded-memory batches.

The separate strict validator checks global uniqueness, ordering, and all target
IDs. This wrapper avoids retaining hundreds of millions of candidate ID strings
at once in the supplied validator's whole-file dictionaries.
"""
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_validator',root/'student_resource/utils/validate_submission.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
strict=json.loads((root/'output/validation.json').read_text())
if strict.get('status')!='PASS' or not strict.get('all_target_ids_exist'):raise ValueError('Run strict validator first')

with tempfile.TemporaryDirectory(prefix='mlg4_validate_') as tmp:
    tmp=Path(tmp);m_path=tmp/'matching_results.tsv';c_path=tmp/'candidate_pairs.tsv'
    with (root/'student_resource/dataset/test/test_source1.tsv').open(encoding='utf-8') as sf,(root/'output/matching_results.tsv').open(encoding='utf-8') as mf,(root/'output/candidate_pairs.tsv').open(encoding='utf-8') as cf:
        next(sf);mh=next(mf);ch=next(cf);rows=chunks=0
        while True:
            references=list(itertools.islice(sf,10000))
            if not references:break
            required={line.split('\t',1)[0] for line in references}
            m_lines=list(itertools.islice(mf,len(references)));c_lines=list(itertools.islice(cf,len(references)))
            if len(m_lines)!=len(references) or len(c_lines)!=len(references):raise ValueError('Missing rows')
            m_path.write_text(mh+''.join(m_lines),encoding='utf-8');c_path.write_text(ch+''.join(c_lines),encoding='utf-8')
            errors=[]
            m=official.validate_id_list_file(str(m_path),official.MATCHING_HEADER,'matched_entity_ids',required,None,errors)
            c=official.validate_id_list_file(str(c_path),official.CANDIDATE_HEADER,'candidate_entity_ids',required,None,errors)
            if errors:raise ValueError('\n'.join(errors))
            if any(ids-c.get(eid,set()) for eid,ids in m.items()):raise ValueError('Match outside candidate set')
            rows+=len(references);chunks+=1
        if next(mf,None) is not None or next(cf,None) is not None:raise ValueError('Extra output rows')
result={'status':'PASS','rows':rows,'chunks':chunks,'mode':'Unmodified supplied validator functions in 10000-reference batches','target_existence':'Verified separately by strict full-data validator','global_uniqueness':'Verified separately by strict full-data validator'}
(root/'reports/official_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
