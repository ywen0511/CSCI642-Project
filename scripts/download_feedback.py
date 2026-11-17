#!/usr/bin/env python
"""Download and validate the pinned feedback CSVs (standard library only; no GPU)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen

COLUMNS=['subject_id','trial_id','label','split','model','decoder_seed','p_error_raw',
         'p_error','checkpoint_id','preprocessing_id','calibration_id']


def verify_table(path, split, profile):
    by_model={m:{} for m in profile['models']}
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=COLUMNS:raise ValueError('Unexpected CSV schema')
        for r in reader:
            m=r['model']
            if m not in by_model:raise ValueError('Unexpected model')
            spec=profile['models'][m]
            if r['split']!=split or r['subject_id'] not in profile['splits'][split]:
                raise ValueError('Unexpected split/participant')
            if int(r['decoder_seed'])!=profile['decoder_seed'] or r['label'] not in {'0','1'}:
                raise ValueError('Unexpected seed/label')
            for key in ['checkpoint_id','calibration_id','preprocessing_id']:
                if r[key]!=spec[key]:raise ValueError(f'Unexpected {key}')
            for col in ['p_error','p_error_raw']:
                p=float(r[col])
                if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
            if r['trial_id'] in by_model[m]:raise ValueError('Duplicate trial ID')
            by_model[m][r['trial_id']]=(r['subject_id'],r['label'])
    values=list(by_model.values())
    if any(len(v)!=profile['tables'][split]['trials_per_model'] or v!=values[0] for v in values):
        raise ValueError('Incomplete or non-aligned model tables')
    for v in values:
        if {s for s,label in v.values()}!=set(profile['splits'][split]):
            raise ValueError('Incomplete participant coverage')
    return {'rows':sum(map(len,values)),'trials_per_model':len(values[0]),'models':list(by_model),'ok':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',default='configs/rl_feedback.json')
    ap.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    profile=json.loads(Path(a.profile).read_text())
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    base=profile['release_url'].replace('/tag/','/download/')+'/'
    assets={v['asset']:v for v in profile['tables'].values()}
    # the manifest asset name defaults to the historical 'manifest.json'; a profile may name another asset
    assets[profile['manifest'].get('asset','manifest.json')]=profile['manifest']
    report={'release_url':profile['release_url'],'scope':'feedback CSV and manifest checks; no checkpoint inference',
            'assets':{},'tables':{},'ok':False}
    try:
        for name,expected in assets.items():
            p=out/name
            # Existing files are checked, never silently replaced.
            data=p.read_bytes() if p.exists() else urlopen(base+name,timeout=120).read()
            digest=hashlib.sha256(data).hexdigest()
            if digest!=expected['sha256'] or len(data)!=expected['bytes']:
                raise ValueError(f'Checksum or size mismatch: {name}')
            if not p.exists():p.write_bytes(data)
            report['assets'][name]={'bytes':len(data),'sha256':digest,'ok':True}
        for split,spec in profile['tables'].items():
            report['tables'][split]=verify_table(out/spec['asset'],split,profile)
        report['ok']=True
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    (out/'feedback_download_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['ok'] else 1

if __name__=='__main__':raise SystemExit(main())
