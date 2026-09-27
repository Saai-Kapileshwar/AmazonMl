"""Build an auditable submission ZIP only after successful strict validation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--team',default='MLG4');a=p.parse_args()
    if not a.team or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.team):raise ValueError('Use letters, digits, underscore or hyphen in team filename')
    root=Path(a.root).resolve();code=root/'code/business_entity_resolution';art=root/'artifacts';out=root/'output'
    validation=json.loads((out/'validation.json').read_text())
    if validation.get('status')!='PASS':raise ValueError('Run strict output validation first')
    if not validation.get('sha256'):raise ValueError('Validation must include output hashes')
    for name,expected in validation['sha256'].items():
        if digest(out/name)!=expected:raise ValueError(f'Output changed after validation: {name}')
    files=[]
    for path in sorted((code/'src').glob('*.py')):files.append((path,'code/business_entity_resolution/src/'+path.name))
    for name in ('README.md','requirements.txt','LICENSE'):files.append((code/name,'code/business_entity_resolution/'+name))
    for name in ('model.txt','config.json','validation.json','prediction_summary.json'):
        files.append((art/name,'code/business_entity_resolution/artifacts/'+name))
    for name in ('matching_results.tsv','candidate_pairs.tsv'):files.append((out/name,'output/'+name))
    files.append((root/'Documentation_template.md','Documentation_template.md'))
    files.append((out/'validation.json','code/business_entity_resolution/artifacts/output_validation.json'))
    manifest={dest:{'bytes':path.stat().st_size,'sha256':digest(path)} for path,dest in files}
    target=root/f'{a.team}_submission.zip';temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for path,dest in files:z.write(path,dest)
        z.writestr('code/business_entity_resolution/artifacts/package_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temporary) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'Archive integrity failure: {bad}')
    temporary.replace(target)
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'files':len(files)+1,'status':'PASS'},indent=2))

if __name__=='__main__':main()
