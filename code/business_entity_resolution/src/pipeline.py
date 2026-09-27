"""Offline entity resolution. MIT licensed. No external business data or ID features."""
from __future__ import annotations
import argparse
from array import array
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from functools import lru_cache

import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz

csv.field_size_limit(10_000_000)
LEGAL = set('inc incorporated llc llp ltd limited pvt private corp corporation co company plc gmbh sarl sas sa eurl the and'.split())
ABBREV = {'rd':'road','st':'street','ave':'avenue','blvd':'boulevard','dr':'drive','ln':'lane','hwy':'highway','ste':'suite','apt':'apartment','nr':'near','opp':'opposite','n':'north','s':'south','e':'east','w':'west'}
WORDS = re.compile(r'[a-z0-9]+')
NUMBERS = re.compile(r'\b\d+\b')
FEATURE_NAMES = ['name_ratio','name_sort','name_set','name_partial','name_jaccard','name_overlap','name_exact','name_length_ratio','name_acronym','address_ratio','address_sort','address_set','address_partial','address_jaccard','address_overlap','address_exact','address_length_ratio','number_jaccard','number_overlap','numbers_conflict','postcode_equal','postcode_conflict','name_missing','address_missing','country_equal','raw_name_ratio','first_number_equal','name_tokens_min','name_tokens_max']

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')

def normalize(text):
    return ' '.join(WORDS.findall(anyascii(text).lower().replace('&', ' and ')))

@dataclass(slots=True)
class Record:
    id: str
    name: str
    address: str
    country: str
    rawname: str

def record(row):
    raw=normalize(row['business_name'])
    name=' '.join(t for t in raw.split() if t not in LEGAL)
    # Preserve the original normalized name if stripping removes everything.
    name=name or raw
    address=' '.join(ABBREV.get(t,t) for t in normalize(row['business_address']).split())
    return Record(row['entity_id'],name,address,normalize(row['country']),raw)

def block_keys(r):
    """Country-scoped multi-pass blocking; generic to previously unseen countries."""
    keys=set()
    nt=sorted(set(t for t in r.name.split() if len(t)>2), key=lambda t:(-len(t),t))[:4]
    at=sorted(set(t for t in r.address.split() if len(t)>3 and not t.isdigit()), key=lambda t:(-len(t),t))[:2]
    nums=NUMBERS.findall(r.address)
    if r.name:
        keys.add('n:'+r.name.replace(' ',''))
        keys.add('s:'+' '.join(sorted(r.name.split())))
    for a,b in itertools.combinations(sorted(set(t[:4] for t in nt)),2):
        keys.add('p:'+a+':'+b)
    if r.address:
        keys.add('a:'+' '.join(sorted(r.address.split())))
    for t in at:
        for n in nums[:2]: keys.add('d:'+n+':'+t[:6])
        for n in nt[:2]: keys.add('x:'+n[:5]+':'+t[:6])
    for p in (n for n in nums if len(n) in (5,6)):
        for n in nt[:2]: keys.add('z:'+p+':'+n[:3])
    prefix=r.country+'|'
    return [int.from_bytes(hashlib.blake2b((prefix+k).encode(),digest_size=8).digest(),'little') for k in sorted(keys)]

def overlap(a,b):
    if not a or not b: return (0.,0.)
    common=len(a & b)
    return common/len(a|b), common/min(len(a),len(b))

@lru_cache(maxsize=20000)
def word_parts(text):
    words=tuple(text.split())
    return words,frozenset(words),''.join(t[0] for t in words)

@lru_cache(maxsize=20000)
def number_parts(text):
    numbers=tuple(NUMBERS.findall(text));unique=frozenset(numbers)
    return numbers,unique,frozenset(n for n in unique if len(n) in (5,6))

def text_features(a,b):
    if not a or not b:return [0.]*8
    j,o=overlap(word_parts(a)[1],word_parts(b)[1])
    return [fuzz.ratio(a,b)/100,fuzz.token_sort_ratio(a,b)/100,fuzz.token_set_ratio(a,b)/100,fuzz.partial_ratio(a,b)/100,j,o,float(a==b),min(len(a),len(b))/max(len(a),len(b))]

def features(a,b):
    nt1,_,ac1=word_parts(a.name);nt2,_,ac2=word_parts(b.name)
    acronym=float(bool(nt1 and nt2) and (ac1==b.name.replace(' ','') or ac2==a.name.replace(' ','')))
    num1,s1,p1=number_parts(a.address);num2,s2,p2=number_parts(b.address)
    return text_features(a.name,b.name)+[acronym]+text_features(a.address,b.address)+list(overlap(s1,s2))+[float(bool(s1 and s2) and not s1&s2),float(bool(p1&p2)),float(bool(p1 and p2) and not p1&p2),float(not a.name or not b.name),float(not a.address or not b.address),float(a.country==b.country),fuzz.ratio(a.rawname,b.rawname)/100,float(bool(num1 and num2) and num1[0]==num2[0]),min(len(nt1),len(nt2)),max(len(nt1),len(nt2))]

class Index:
    def __init__(self, refs, max_block=80):
        keys=array('Q'); rows=array('I')
        for i,r in enumerate(refs):
            k=block_keys(r);keys.extend(k);rows.extend([i]*len(k))
            if (i+1)%250000==0:log(f'Index: {i+1:,} reference records')
        k=np.frombuffer(keys,dtype=np.uint64);v=np.frombuffer(rows,dtype=np.uint32)
        order=np.argsort(k,kind='stable')
        self.keys=k[order];self.rows=v[order];self.max_block=max_block
        log(f'Index ready: {len(refs):,} records / {len(k):,} keys')

    def select_references(self,selected):
        """Retain sampled references only AFTER applying full-index block counts."""
        if not len(self.keys):return
        starts=np.r_[0,np.flatnonzero(self.keys[1:]!=self.keys[:-1])+1]
        counts=np.diff(np.r_[starts,len(self.keys)])
        allowed=np.repeat(counts<=self.max_block,counts)
        keep=allowed & selected[self.rows]
        self.keys=self.keys[keep];self.rows=self.rows[keep]
        log(f'Sample index ready: {len(self.keys):,} keys; full-data block limits preserved')

    def query_batch(self, targets, selected=None):
        flat=[];offsets=[0]
        for r in targets:
            flat.extend(block_keys(r));offsets.append(len(flat))
        query=np.asarray(flat,dtype=np.uint64)
        lo=np.searchsorted(self.keys,query,'left');hi=np.searchsorted(self.keys,query,'right')
        active=np.flatnonzero((hi>lo)&(hi-lo<=self.max_block))
        owners=np.searchsorted(offsets,active,side='right')-1
        found=[set() for _ in targets]
        for k,j in zip(active,owners):
            matches=self.rows[lo[k]:hi[k]]
            if selected is not None:matches=matches[selected[matches]]
            found[j].update(map(int,matches))
        for matches in found:yield sorted(matches)

def plausible(a,b):
    """Cheap final blocking filter, applied identically before training/inference."""
    ns=max(fuzz.ratio(a.name,b.name),fuzz.token_sort_ratio(a.name,b.name)) if a.name and b.name else 0
    ads=max(fuzz.ratio(a.address,b.address),fuzz.token_sort_ratio(a.address,b.address)) if a.address and b.address else 0
    return ns>=85 or ads>=85 or (ns>=50 and ads>=45)

def load_refs(path):
    refs=[]
    for row in read_tsv(path):
        refs.append(record(row))
        if len(refs)%500000==0:log(f'Loaded {len(refs):,} references')
    return refs

def batches(paths,size=4000,country=None):
    batch=[]
    for path in paths:
        for row in read_tsv(path):
            if country is not None and normalize(row['country'])!=country:continue
            batch.append(record(row))
            if len(batch)==size:yield batch;batch=[]
    if batch:yield batch

def entity_f05(truth, predicted):
    truth=set(truth);predicted=set(predicted)
    if not truth:return float(not predicted)
    tp=len(truth&predicted)
    return 1.25*tp/(len(predicted)+.25*len(truth))

def report_scores(ids,truth,groups,target_ids,probs,threshold,refs):
    predictions={int(i):set() for i in ids}
    candidates={int(i):set() for i in ids}
    for i,t,p in zip(groups,target_ids,probs):
        i=int(i)
        if i in predictions:
            candidates[i].add(t)
            if p>=threshold:predictions[i].add(t)
    values=[];country={};tp=fp=fn=singletons=singletons_ok=retrieved=positives=0
    errors=[]
    for i in ids:
        i=int(i);actual=truth.get(refs[i].id,set());pred=predictions[i]
        score=entity_f05(actual,pred);values.append(score)
        country.setdefault(refs[i].country,[]).append(score)
        tp+=len(actual&pred);fp+=len(pred-actual);fn+=len(actual-pred)
        positives+=len(actual);retrieved+=len(actual&candidates[i])
        singletons+=not actual;singletons_ok+=not actual and not pred
        if score<1 and len(errors)<30:errors.append({'source1_entity_id':refs[i].id,'name':refs[i].name,'truth':sorted(actual),'predicted':sorted(pred),'score':score,'missed_by_blocking':sorted(actual-candidates[i]),'rejected_by_classifier':sorted((actual&candidates[i])-pred),'false_merges':sorted(pred-actual)})
    return {'entities':len(ids),'macro_f05':float(np.mean(values)),'pair_precision':tp/max(1,tp+fp),'pair_recall':tp/max(1,tp+fn),'candidate_recall':retrieved/max(1,positives),'singleton_accuracy':singletons_ok/max(1,singletons),'singleton_count':int(singletons),'empty_baseline_macro_f05':singletons/len(ids),'by_country':{c:float(np.mean(v)) for c,v in country.items()},'threshold':float(threshold),'error_examples':errors}

_PREP=None

def init_prepare_worker(refs,keys,rows,max_block,truth):
    global _PREP
    index=Index.__new__(Index);index.keys=keys;index.rows=rows;index.max_block=max_block
    _PREP=(refs,index,truth)

def prepare_batch(batch,refs,index,truth):
    x=array('f');groups=array('I');labels=array('B');tids=[]
    for target,found in zip(batch,index.query_batch(batch)):
        for i in found:
            if not plausible(refs[i],target):continue
            x.extend(features(refs[i],target));groups.append(i);tids.append(target.id)
            labels.append(target.id in truth[refs[i].id])
    return len(batch),x,groups,labels,tids

def prepare_worker(batch):
    return prepare_batch(batch,*_PREP)

def prepared_batches(args,root,refs,index,truth):
    stream=batches([root/'train_source2.tsv',root/'train_source3.tsv'])
    if args.workers==1:
        for batch in stream:yield prepare_batch(batch,refs,index,truth)
        return
    compact={i:r for i,r in enumerate(refs) if r is not None}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_prepare_worker,initargs=(compact,index.keys,index.rows,index.max_block,truth)) as executor:
        pending=deque()
        for batch in stream:
            pending.append(executor.submit(prepare_worker,batch))
            if len(pending)>=args.workers*2:yield pending.popleft().result()
        while pending:yield pending.popleft().result()

def prepare(args):
    """Build features for a reproducible random S1 sample, searching all S2/S3."""
    out=Path(args.artifacts);out.mkdir(parents=True,exist_ok=True)
    root=Path(args.data)/'train'
    refs=load_refs(root/'train_source1.tsv')
    rng=np.random.default_rng(args.seed)
    sample=np.sort(rng.choice(len(refs),size=min(args.sample,len(refs)),replace=False))
    selected=np.zeros(len(refs),dtype=bool);selected[sample]=True
    sample_ids={refs[i].id for i in sample}
    truth={r['source1_entity_id']:set(filter(None,r['matched_entity_ids'].split(','))) for r in read_tsv(root/'train_ground_truth.tsv') if r['source1_entity_id'] in sample_ids}
    if len(truth)!=len(sample_ids):raise ValueError('Missing ground truth')
    index=Index(refs,args.max_block)
    index.select_references(selected)
    # Only sampled rows are consulted after index reduction; release other strings.
    refs=[r if selected[i] else None for i,r in enumerate(refs)]
    # Full reference index keeps block-frequency suppression identical to inference.
    groups=array('I');labels=array('B');X=array('f');tids=[]
    scanned=0;started=time.monotonic()
    for count,x,g,y,t in prepared_batches(args,root,refs,index,truth):
        X.extend(x);groups.extend(g);labels.extend(y);tids.extend(t)
        scanned+=count
        if scanned%200000==0:log(f'Train retrieval: {scanned:,} targets; {len(groups):,} pairs; {time.monotonic()-started:.0f}s')
    np.savez_compressed(out/'pairs.npz',X=np.frombuffer(X,dtype=np.float32).reshape(-1,len(FEATURE_NAMES)),y=np.asarray(labels),groups=np.asarray(groups),target_ids=np.asarray(tids),sample=sample)
    (out/'sample.json').write_text(json.dumps({'refs':{str(i):{'id':refs[i].id,'name':refs[i].name,'country':refs[i].country} for i in sample},'truth':{k:sorted(v) for k,v in truth.items()},'seed':args.seed,'max_block':args.max_block,'sample':len(sample),'scanned_targets':scanned}),encoding='utf-8')
    log('Feature preparation complete')

def train(args):
    from lightgbm import LGBMClassifier
    from types import SimpleNamespace
    art=Path(args.artifacts);data=np.load(art/'pairs.npz')
    meta=json.loads((art/'sample.json').read_text(encoding='utf-8'))
    refs={int(k):SimpleNamespace(**v) for k,v in meta['refs'].items()}
    truth={k:set(v) for k,v in meta['truth'].items()}
    sample=data['sample'].copy();np.random.default_rng(meta['seed']).shuffle(sample)
    n=len(sample);fit=sample[:int(n*.6)];tune=sample[int(n*.6):int(n*.8)];hold=sample[int(n*.8):]
    # IDs group entities only. They never enter the feature matrix.
    masks=[np.isin(data['groups'],s) for s in (fit,tune,hold)]
    clf=LGBMClassifier(n_estimators=350,learning_rate=.05,num_leaves=31,max_depth=-1,min_child_samples=40,colsample_bytree=.9,reg_lambda=2.,random_state=meta['seed'],n_jobs=args.threads,verbosity=-1,deterministic=True,force_col_wise=True)
    clf.fit(data['X'][masks[0]],data['y'][masks[0]],feature_name=FEATURE_NAMES)
    probs=clf.predict_proba(data['X'])[:,1]
    # Efficient macro threshold sweep, including references with no retrieved pairs.
    tune_g=data['groups'][masks[1]];tune_y=data['y'][masks[1]];tune_p=probs[masks[1]]
    positions={int(i):j for j,i in enumerate(tune)}
    mapped=np.array([positions[int(i)] for i in tune_g])
    truth_counts=np.array([len(truth[refs[int(i)].id]) for i in tune])
    grid=[]
    for threshold in np.linspace(.05,.99,95):
        keep=tune_p>=threshold
        pred=np.bincount(mapped[keep],minlength=len(tune));tp=np.bincount(mapped[keep],weights=tune_y[keep],minlength=len(tune))
        den=pred+.25*truth_counts
        scores=np.divide(1.25*tp,den,out=np.zeros(len(tune)),where=den!=0)
        scores[(truth_counts==0)&(pred==0)]=1
        grid.append((float(np.mean(scores)),float(threshold)))
    _,threshold=max(grid)
    reports={}
    for name,ids,mask in [('tuning',tune,masks[1]),('holdout',hold,masks[2])]:
        reports[name]=report_scores(ids,truth,data['groups'][mask],data['target_ids'][mask],probs[mask],threshold,refs)
    exact=((data['X'][masks[2],FEATURE_NAMES.index('name_exact')]==1)&(data['X'][masks[2],FEATURE_NAMES.index('address_exact')]==1)).astype(float)
    reports['exact_name_address_baseline']=report_scores(hold,truth,data['groups'][masks[2]],data['target_ids'][masks[2]],exact,.5,refs)
    reports.update({'training_entities':len(fit),'training_pairs':int(masks[0].sum()),'training_positive_pairs':int(data['y'][masks[0]].sum()),'sample_entities':n,'features':FEATURE_NAMES,'feature_importance':dict(zip(FEATURE_NAMES,map(int,clf.feature_importances_))),'threshold_grid':[{'macro_f05':s,'threshold':t} for s,t in grid],'limitations':['France is absent from labeled training data; its accuracy is unmeasured.','Metrics are a random S1 holdout, not the hidden leaderboard.','Blocks over max_block references are skipped; missed candidates cannot be recovered by the classifier.']})
    clf.booster_.save_model(str(art/'model.txt'))
    (art/'config.json').write_text(json.dumps({'threshold':threshold,'max_block':meta['max_block'],'seed':meta['seed'],'features':FEATURE_NAMES},indent=2))
    (art/'validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    log(f'Untouched holdout macro F0.5: {reports["holdout"]["macro_f05"]:.6f}; threshold {threshold:.2f}')

def score_batch(batch,refs,index,model,threads):
    feat=[];links=[]
    for target,found in zip(batch,index.query_batch(batch)):
        for i in found:
            if plausible(refs[i],target):feat.append(features(refs[i],target));links.append((i,target.id))
    if not links:return len(batch),[]
    probabilities=model.predict(np.asarray(feat,dtype=np.float32),num_threads=threads)
    return len(batch),[(i,t,float(p)) for (i,t),p in zip(links,probabilities)]

def predict_country(job):
    """Independent country partition: keeps total worker memory bounded."""
    import lightgbm as lgb
    import sqlite3
    root,art,country,part,config,threads=job
    root=Path(root);art=Path(art)
    refs=[];global_rows=[]
    for i,row in enumerate(read_tsv(root/'test_source1.tsv')):
        if normalize(row['country'])==country:refs.append(record(row));global_rows.append(i)
    log(f'{country}: {len(refs):,} reference records')
    index=Index(refs,config['max_block']);model=lgb.Booster(model_file=str(art/'model.txt'))
    if config['features']!=FEATURE_NAMES or model.num_feature()!=len(FEATURE_NAMES):raise ValueError('Model feature schema does not match pipeline')
    path=art/f'prediction_part_{part}.sqlite'
    db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS pairs (ref INTEGER, target TEXT, probability REAL)');db.execute('DELETE FROM pairs');db.commit()
    scanned=pairs=matches=0;started=time.monotonic()
    for batch in batches([root/'test_source2.tsv',root/'test_source3.tsv'],2000,country):
        count,scored=score_batch(batch,refs,index,model,threads)
        if scored:
            db.executemany('INSERT INTO pairs VALUES (?,?,?)',((global_rows[i],t,p) for i,t,p in scored))
            pairs+=len(scored);matches+=sum(p>=config['threshold'] for _,_,p in scored)
        scanned+=count;db.commit()
        if scanned%100000==0:log(f'{country}: {scanned:,} targets; {pairs:,} candidates; {matches:,} matches; {time.monotonic()-started:.0f}s')
    db.close()
    result={'country':country,'part':part,'reference_entities':len(refs),'target_entities':scanned,'candidate_pairs':pairs,'matched_pairs':matches,'seconds':time.monotonic()-started}
    (art/f'prediction_part_{part}.json').write_text(json.dumps(result,indent=2));log(f'{country}: complete')
    return result

def predict(args):
    import sqlite3
    art=Path(args.artifacts);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    root=Path(args.data)/'test';config=json.loads((art/'config.json').read_text())
    countries={normalize(r['country']) for r in read_tsv(root/'test_source1.tsv')}
    started=time.monotonic()
    jobs=[(str(root),str(art),country,i,config,max(1,args.threads//args.workers)) for i,country in enumerate(sorted(countries))]
    if args.workers==1:parts=[predict_country(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:parts=list(executor.map(predict_country,jobs))
    db=sqlite3.connect(art/'predictions.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS pairs (ref INTEGER, target TEXT, probability REAL)');db.execute('DELETE FROM pairs');db.commit()
    for part in parts:
        log(f'Merging {part["country"]} candidate audit')
        db.execute('ATTACH DATABASE ? AS shard',(str(art/f'prediction_part_{part["part"]}.sqlite'),))
        db.execute('INSERT INTO pairs SELECT * FROM shard.pairs');db.commit();db.execute('DETACH DATABASE shard')
    log('Sorting candidates and writing complete outputs')
    db.execute('CREATE INDEX IF NOT EXISTS pair_ref ON pairs(ref,target)');db.commit()
    cursor=iter(db.execute('SELECT ref,target,probability FROM pairs ORDER BY ref,target'));current=next(cursor,None)
    paths=[out/'matching_results.tsv',out/'candidate_pairs.tsv']
    with paths[0].with_suffix('.tsv.tmp').open('w',encoding='utf-8',newline='') as fm,paths[1].with_suffix('.tsv.tmp').open('w',encoding='utf-8',newline='') as fc:
        mw=csv.writer(fm,delimiter='\t',lineterminator='\n');cw=csv.writer(fc,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids']);cw.writerow(['source1_entity_id','candidate_entity_ids'])
        rows=0
        for i,r in enumerate(read_tsv(root/'test_source1.tsv')):
            cand=[];matched=[]
            while current is not None and current[0]==i:
                _,t,p=current;cand.append(t)
                if p>=config['threshold']:matched.append(t)
                current=next(cursor,None)
            mw.writerow([r['entity_id'],','.join(sorted(set(matched)))]);cw.writerow([r['entity_id'],','.join(sorted(set(cand)))]);rows+=1
    for path in paths:os.replace(path.with_suffix('.tsv.tmp'),path)
    summary={'reference_entities':rows,'target_entities':sum(p['target_entities'] for p in parts),'candidate_pairs':sum(p['candidate_pairs'] for p in parts),'matched_pairs':sum(p['matched_pairs'] for p in parts),'threshold':config['threshold'],'seconds':time.monotonic()-started,'countries':{p['country']:p['reference_entities'] for p in parts},'partitions':parts}
    (art/'prediction_summary.json').write_text(json.dumps(summary,indent=2));db.close();log('Prediction complete')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','predict','all'])
    p.add_argument('--data',default='student_resource/dataset');p.add_argument('--artifacts',default='artifacts');p.add_argument('--output',default='output')
    p.add_argument('--sample',type=int,default=40000);p.add_argument('--seed',type=int,default=2026);p.add_argument('--max-block',type=int,default=80);p.add_argument('--threads',type=int,default=6);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.sample<100:raise ValueError('Use at least 100 sample entities')
    if args.workers<1 or args.threads<1:raise ValueError('workers and threads must be positive')
    for name,fn in [('prepare',prepare),('train',train),('predict',predict)]:
        if args.command in (name,'all'):fn(args)

if __name__=='__main__':main()
