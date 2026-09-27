"""Exercise the entire CLI and compare serial/parallel inference on synthetic data.

These fixtures test software behavior only, never competition accuracy.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PIPELINE=ROOT/'code/business_entity_resolution/src/pipeline.py'
VALIDATOR=ROOT/'code/business_entity_resolution/src/validate_outputs.py'

def write(path,header,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(header);w.writerows(rows)

def run(*args):
    result=subprocess.run([sys.executable,'-X','utf8',*map(str,args)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout

def main():
    with tempfile.TemporaryDirectory(prefix='mlg4_smoke_') as tmp:
        root=Path(tmp);data=root/'dataset';art=root/'artifacts';out=root/'output';serial=root/'serial'
        for split in ('train','test'):
            folder=data/split;folder.mkdir(parents=True)
            rows={1:[],2:[],3:[]};truth=[]
            for i in range(300 if split=='train' else 120):
                country=('US','India')[i%2] if split=='train' else ('US','India','France')[i%3]
                name=f'Zenith {i} Engineering';address=f'{100+i} Main Road, Town {i%9}'
                rows[1].append([f'S1-{i}',name+' Corporation',address,country]);matches=[]
                for source in (2,3):
                    if i%5:
                        eid=f'S{source}-{i}';rows[source].append([eid,name+' Corp.',address.replace('Road','Rd'),country]);matches.append(eid)
                    rows[source].append([f'S{source}-neg{i}',f'Zenith {i} Bakery',f'{500+i} Other Avenue, Town {i%9}',country])
                truth.append([f'S1-{i}',','.join(matches)])
            for source in (1,2,3):write(folder/f'{split}_source{source}.tsv',['entity_id','business_name','business_address','country'],rows[source])
            if split=='train':write(folder/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],truth)
        common=['--data',data,'--artifacts',art,'--output',out,'--sample','300','--threads','2']
        run(PIPELINE,'prepare',*common)
        with np.load(art/'pairs.npz') as generated:
            parallel_features={name:generated[name].copy() for name in generated.files}
        run(PIPELINE,'prepare',*common,'--workers','1')
        with np.load(art/'pairs.npz') as generated:
            for name,values in parallel_features.items():assert np.array_equal(generated[name],values),name
        run(PIPELINE,'train',*common)
        run(PIPELINE,'predict',*common,'--workers','3')
        checked=json.loads(run(VALIDATOR,'--data',data/'test','--output',out))
        assert checked['source1_rows']==120 and checked['status']=='PASS'
        parallel={name:(out/name).read_bytes() for name in ('matching_results.tsv','candidate_pairs.tsv')}
        run(PIPELINE,'predict','--data',data,'--artifacts',art,'--output',serial,'--workers','1','--threads','2')
        for name,content in parallel.items():assert (serial/name).read_bytes()==content,name
        official=run(ROOT/'student_resource/utils/validate_submission.py','--matching',out/'matching_results.tsv','--candidate',out/'candidate_pairs.tsv','--test-dir',data/'test','--check-ids')
        print(official)
        shutil.copytree(ROOT/'code/business_entity_resolution',root/'code/business_entity_resolution',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT/'Documentation_template.md',root/'Documentation_template.md')
        run(ROOT/'code/business_entity_resolution/src/package_submission.py','--root',root,'--team','MLG4')
        with zipfile.ZipFile(root/'MLG4_submission.zip') as archive:
            assert archive.read('output/matching_results.tsv')==parallel['matching_results.tsv']
            assert archive.read('output/candidate_pairs.tsv')==parallel['candidate_pairs.tsv']
            assert 'code/business_entity_resolution/artifacts/model.txt' in archive.namelist()
        with (out/'matching_results.tsv').open('a') as f:f.write('\n')
        changed=subprocess.run([sys.executable,str(ROOT/'code/business_entity_resolution/src/package_submission.py'),'--root',str(root)],capture_output=True,text=True)
        assert changed.returncode!=0 and 'changed after validation' in changed.stderr
        print('PASS: complete CLI, serial/parallel preparation and prediction equivalence, both validators, archive integrity, and stale-output rejection.')

if __name__=='__main__':main()
