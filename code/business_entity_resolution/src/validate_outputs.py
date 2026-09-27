"""Strict streaming output validator, including target existence and containment."""
import argparse
import csv
import itertools
import json
import hashlib
from pathlib import Path
from pipeline import read_tsv

def validate(data,output):
    data=Path(data);output=Path(output)
    targets=set()
    for name,prefix in [('test_source2.tsv','S2-'),('test_source3.tsv','S3-')]:
        for row in read_tsv(data/name):
            eid=row['entity_id']
            if not eid.startswith(prefix):raise ValueError(f'Invalid source ID: {eid}')
            if eid in targets:raise ValueError(f'Duplicate source ID: {eid}')
            targets.add(eid)
    required=read_tsv(data/'test_source1.tsv');seen=set();rows=matches=candidates=empty=0
    with (output/'matching_results.tsv').open(encoding='utf-8',newline='') as fm,(output/'candidate_pairs.tsv').open(encoding='utf-8',newline='') as fc:
        mr=csv.reader(fm,delimiter='\t');cr=csv.reader(fc,delimiter='\t')
        if next(mr,None)!=['source1_entity_id','matched_entity_ids']:raise ValueError('Invalid matching header')
        if next(cr,None)!=['source1_entity_id','candidate_entity_ids']:raise ValueError('Invalid candidate header')
        for ref,m,c in itertools.zip_longest(required,mr,cr):
            if ref is None or m is None or c is None:raise ValueError('Output row counts do not match test source1')
            eid=ref['entity_id']
            if len(m)!=2 or len(c)!=2:raise ValueError(f'Wrong number of columns at {eid}')
            if m[0]!=eid or c[0]!=eid:raise ValueError(f'Output must follow source1 order: {eid}')
            if eid in seen:raise ValueError(f'Duplicate Source 1 ID: {eid}')
            seen.add(eid)
            ml=m[1].split(',') if m[1] else [];cl=c[1].split(',') if c[1] else []
            ms=set(ml);cs=set(cl)
            if len(ms)!=len(ml) or len(cs)!=len(cl):raise ValueError(f'Duplicate list IDs: {eid}')
            if not ms<=cs:raise ValueError(f'Matches outside candidates: {eid}')
            if not cs<=targets:raise ValueError(f'Unknown candidate IDs: {eid}: {cs-targets}')
            rows+=1;matches+=len(ml);candidates+=len(cl);empty+=not ml
    report={'status':'PASS','source1_rows':rows,'candidate_pairs':candidates,'matched_pairs':matches,'empty_predictions':empty,'all_target_ids_exist':True,'matches_subset_candidates':True}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=validate(a.data,a.output)
    result['sha256']={}
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        h=hashlib.sha256()
        with (Path(a.output)/name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        result['sha256'][name]=h.hexdigest()
    print(json.dumps(result,indent=2))
    (Path(a.output)/'validation.json').write_text(json.dumps(result,indent=2))
