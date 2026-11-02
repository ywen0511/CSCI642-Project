#!/usr/bin/env python
"""Verify Release checksums, then reproduce both main models' predictions.
Writes a JSON report and exits nonzero on any failed check. Asset failure stops
before extraction or checkpoint loading. --assets-only explicitly skips inference.
Resumption is a separate check using scripts/check_resume.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
TOL = 1e-6
REQUIRED = {'manifest.json', 'validation_predictions.csv', 'test_predictions.csv'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def check_assets(dl, sizes):
    sums = {}
    for line in (dl/'SHA256SUMS').read_text().splitlines():
        if not line.strip(): continue
        h, name = line.split(maxsplit=1)
        name = name.strip().removeprefix('*')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', h) or Path(name).name != name or name in {'.','..'} or name in sums:
            raise ValueError('Invalid or duplicate checksum entry')
        sums[name] = h.lower()
    if not REQUIRED <= sums.keys() or not any(n.endswith('.tar.gz') for n in sums):
        raise ValueError('Checksum inventory is empty or lacks required Release files')
    if sizes and set(sizes)-{'SHA256SUMS'} != set(sums):
        raise ValueError('Checksum inventory differs from the GitHub API asset inventory')
    unlisted = {p.name for p in dl.glob('*.tar.gz')}-set(sums)
    if unlisted: raise ValueError(f'Unlisted archives in download directory: {sorted(unlisted)}')
    rows = {}
    for n,h in sums.items():
        p=dl/n
        r={'present':p.is_file()}
        if p.is_file():
            r.update(bytes=p.stat().st_size,sha256=sha256(p))
            r['sha256_ok']=r['sha256']==h
            if n in sizes:r['size_matches_api']=sizes[n]==r['bytes']
        rows[n]=r
    ok=all(r.get('sha256_ok',False) and r.get('size_matches_api',True) for r in rows.values())
    return rows, ok


def reproduce(dl, ex, prep_dir, split_path, device):
    import numpy as np
    import pandas as pd
    import torch
    from errp.data import load_splits
    from errp.evaluate import margins_from_ckpt
    from errp.calibration import apply_calibration
    splits=load_splits(split_path)
    manifest=json.loads((dl/'manifest.json').read_text())
    entries={k:e for k,e in manifest['checkpoints'].items() if e['group']=='main' and e['best_ckpt']}
    if len(entries)!=2 or {e['model'] for e in entries.values()} != {'eegnet','cbramod'}:
        raise ValueError('Both main EEGNet and CBraMod checkpoints are required')
    results={}
    for ckpt_id,e in entries.items():
        ck=ex/e['best_ckpt']
        if not ck.resolve().is_relative_to(ex.resolve()): raise ValueError('Checkpoint path outside extraction directory')
        digest=sha256(ck)
        rec={'best_ckpt':str(ck),'best_ckpt_sha256':digest,
             'sha256_matches_manifest':digest==e['files'][e['best_ckpt']]['sha256']}
        if not rec['sha256_matches_manifest']:
            rec['ok']=False;results[ckpt_id]=rec;continue
        out,st=margins_from_ckpt(ck,prep_dir,splits,device)
        cal=manifest['calibrations'][f"{e['run_id']}:temperature"]
        for split in ['validation','test']:
            d,z=out[split]
            tab=pd.read_csv(dl/f'{split}_predictions.csv',dtype={'trial_id':str,'subject_id':str})
            tab=tab[tab.checkpoint_id==ckpt_id].set_index('trial_id')
            if not tab.index.is_unique or set(tab.index)!=set(d.trial_ids):
                raise ValueError(f'Duplicate, missing, or extra trials: {ckpt_id}/{split}')
            tab=tab.loc[d.trial_ids]
            raw,calibrated=1/(1+np.exp(-z)),apply_calibration(z,cal)
            dr,dc=float(np.abs(raw-tab.p_error_raw.values).max()),float(np.abs(calibrated-tab.p_error.values).max())
            rec[split]={'n':len(z),'max_abs_diff_p_error_raw':dr,'max_abs_diff_p_error':dc,
                'labels_match':bool((d.y==tab.label.values).all()),
                'subjects_match':bool((d.subject_ids==tab.subject_id.values).all()),
                'split_matches':bool((tab.split==split).all()),
                'within_tolerance':bool(np.isfinite(dr) and np.isfinite(dc) and dr<=TOL and dc<=TOL)}
        rec['ok']=all(rec[s][k] for s in ['validation','test']
                       for k in ['within_tolerance','labels_match','subjects_match','split_matches'])
        results[ckpt_id]=rec
    return results,torch.__version__


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download-dir',required=True)
    ap.add_argument('--extract-dir')
    ap.add_argument('--prep-dir')
    ap.add_argument('--splits',default='configs/splits.json')
    ap.add_argument('--expected-sizes',help='JSON {asset: bytes} from the GitHub API')
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--assets-only',action='store_true',help='Check full asset inventory only; no inference claim')
    a=ap.parse_args(argv)
    if not a.assets_only and (not a.extract_dir or not a.prep_dir):
        ap.error('--extract-dir and --prep-dir are required unless --assets-only is set')
    res={'tolerance_abs_p':TOL,'mode':'assets_only' if a.assets_only else 'assets_and_inference',
         'assets':{},'reproduction':{},'all_assets_ok':False,'all_reproduction_ok':None,'ok':False}
    try:
        dl=Path(a.download_dir)
        sizes=json.loads(Path(a.expected_sizes).read_text()) if a.expected_sizes else {}
        res['assets'],res['all_assets_ok']=check_assets(dl,sizes)
        if not res['all_assets_ok']:
            res['error']='Asset verification failed; extraction and inference skipped'
        elif a.assets_only:
            res['ok']=True
        else:
            ex=Path(a.extract_dir)
            ex.mkdir(parents=True,exist_ok=False)
            for name in sorted(res['assets']):
                if name.endswith('.tar.gz'):
                    with tarfile.open(dl/name) as tf:tf.extractall(ex,filter='data')
            res['reproduction'],res['torch']=reproduce(dl,ex,a.prep_dir,a.splits,a.device)
            res['all_reproduction_ok']=bool(res['reproduction']) and all(r['ok'] for r in res['reproduction'].values())
            res['ok']=bool(res['all_assets_ok'] and res['all_reproduction_ok'])
    except Exception as exc:
        res['error']=f'{type(exc).__name__}: {exc}'
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps({k:res[k] for k in ['mode','all_assets_ok','all_reproduction_ok','ok']}))
    return 0 if res['ok'] else 1

if __name__=='__main__':
    raise SystemExit(main())
