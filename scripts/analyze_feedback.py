#!/usr/bin/env python
"""Audit frozen seed-42 feedback tables and generate CPU-only closeout reports.
No fitting, recalibration, checkpoint selection, or EEG/GPU access is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAG = 'eeg-stage1-20261101-bccbb8a'
RELEASE = f'https://github.com/ywen0511/CSCI642-Project/releases/tag/{TAG}'
MODELS = ['eegnet', 'cbramod']
COLS = ['subject_id','trial_id','label','split','model','decoder_seed','p_error_raw',
        'p_error','checkpoint_id','preprocessing_id','calibration_id']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid labels or probabilities')
    pos, neg = int(y.sum()), int((1-y).sum())
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = (ranks[y == 1].sum()-pos*(pos+1)/2)/(pos*neg) if pos and neg else None
    q = np.clip(p, 1e-12, 1-1e-12)
    return dict(n=len(y), n_error=pos, n_nonerror=neg, auroc=auc,
                brier=float(np.mean((p-y)**2)),
                nll=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))))


def validate_table(df, split, trials, manifest):
    if list(df.columns) != COLS or df.empty:
        raise ValueError('Unexpected prediction schema or empty table')
    if set(df.model) != set(MODELS) or set(df.decoder_seed) != {42} or set(df.split) != {split}:
        raise ValueError('Unexpected model, decoder seed, or split')
    expected = trials[(trials.retained == 1) & (trials.split == split)].set_index('trial_id')
    for model, g in df.groupby('model'):
        if not g.trial_id.is_unique or set(g.trial_id) != set(expected.index):
            raise ValueError('Missing, duplicate, or extra trials')
        e = expected.loc[g.trial_id]
        if not np.array_equal(g.label, e.label) or not np.array_equal(g.subject_id, e.subject_id):
            raise ValueError('Prediction labels or participants disagree with the trial manifest')
        run = manifest['selection'][model]['selected_run_id']
        if set(g.checkpoint_id) != {run+':best'} or set(g.calibration_id) != {run+':temperature'}:
            raise ValueError('Unexpected checkpoint or calibration')
        if not set(g.preprocessing_id) <= set(manifest['preprocessing']):
            raise ValueError('Unknown preprocessing identifier')
        for col in ['p_error', 'p_error_raw']:
            scores(g.label, g[col])


def md_table(df, cols):
    out = ['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |']
    for row in df[cols].itertuples(index=False, name=None):
        out.append('| '+' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row)+' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-dir', required=True)
    ap.add_argument('--trial-manifest', default='reports/data/trial_manifest.csv')
    ap.add_argument('--out-dir', default='reports/feedback_closeout')
    ap.add_argument('--profile', default='configs/rl_feedback.json')
    a = ap.parse_args()
    src, out = Path(a.release_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src/'manifest.json').read_text())
    sums = {line.split()[1]: line.split()[0] for line in (src/'SHA256SUMS').read_text().splitlines()}
    files = ['manifest.json', 'metrics_summary.json', 'validation_predictions.csv', 'test_predictions.csv']
    provenance = {}
    for name in files:
        digest = sha(src/name)
        if digest != sums[name]: raise ValueError(f'Checksum mismatch: {name}')
        provenance[name] = {'sha256':digest, 'bytes':(src/name).stat().st_size}
    trials = pd.read_csv(a.trial_manifest, dtype={'subject_id':str, 'trial_id':str})
    prep = next(iter(manifest['preprocessing'].values()))
    if sha(a.trial_manifest) != prep['files']['trial_manifest.csv']['sha256']:
        raise ValueError('Trial manifest differs from the frozen release')
    sets = {s:set(manifest['splits'][s]) for s in ['train','validation','test']}
    if any(sets[x] & sets[y] for x,y in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Participant overlap')
    for s, ids in sets.items():
        if set(trials.loc[(trials.retained == 1) & (trials.split == s), 'subject_id']) != ids:
            raise ValueError('Participant split disagrees with the frozen manifest')
    train = trials[(trials.retained == 1) & (trials.split == 'train')]
    prior = float(train.label.mean())
    rows, stats, gaps, tables = [], [], [], {}
    metric_differences = []
    released_metrics = json.loads((src/'metrics_summary.json').read_text())
    for split in ['validation','test']:
        df = pd.read_csv(src/f'{split}_predictions.csv', dtype={'subject_id':str,'trial_id':str})
        validate_table(df, split, trials, manifest)
        tables[split] = df
        for model in MODELS:
            g = df[df.model == model]
            for subject in ['pooled']+sorted(g.subject_id.unique()):
                sub = g if subject == 'pooled' else g[g.subject_id == subject]
                baseline = scores(sub.label, np.full(len(sub), prior))
                for version, col in [('raw','p_error_raw'), ('calibrated','p_error')]:
                    met = scores(sub.label, sub[col])
                    rows.append(dict(split=split, subject_id=subject, model=model, probability=version, **met,
                                     delta_brier_vs_constant=met['brier']-baseline['brier'],
                                     delta_nll_vs_constant=met['nll']-baseline['nll']))
                    if split == 'test' and subject == 'pooled':
                        run = manifest['selection'][model]['selected_run_id']
                        recorded = released_metrics[run]['test_pooled'][version+'@0.5']
                        for key in ['auroc','brier','nll']:
                            delta = float(abs(met[key]-recorded[key]))
                            tolerance = 1e-5 if key == 'nll' else 1e-7
                            metric_differences.append(dict(model=model,probability=version,metric=key,abs_difference=delta,tolerance=tolerance))
                            if delta > tolerance:
                                raise ValueError(f'Released metrics disagree: {model}/{version}/{key}')
                for label in [0,1]:
                    p = sub.loc[sub.label == label, 'p_error']
                    q = p.quantile([.05,.25,.5,.75,.95]).tolist()
                    stats.append(dict(split=split,subject_id=subject,model=model,label=label,n=len(p),
                        mean=float(p.mean()),sd=float(p.std(ddof=1)),
                        **dict(zip(['q05','q25','median','q75','q95'],q)),
                        fraction_045_to_055=float(p.between(.45,.55).mean())))
                gaps.append(dict(split=split,subject_id=subject,model=model,
                    mean_error=float(sub.loc[sub.label==1,'p_error'].mean()),
                    mean_nonerror=float(sub.loc[sub.label==0,'p_error'].mean()),
                    mean_gap=float(sub.loc[sub.label==1,'p_error'].mean()-sub.loc[sub.label==0,'p_error'].mean())))
        g = df[df.model == MODELS[0]]
        for subject in ['pooled']+sorted(g.subject_id.unique()):
            sub = g if subject == 'pooled' else g[g.subject_id == subject]
            rows.append(dict(split=split,subject_id=subject,model='constant_train_prevalence',probability='constant',
                             **scores(sub.label,np.full(len(sub),prior)),delta_brier_vs_constant=0.,delta_nll_vs_constant=0.))
        subjects = ['pooled']+sorted(df.subject_id.unique())
        fig, axes = plt.subplots(len(subjects),2,figsize=(11,2.6*len(subjects)),sharex=True,constrained_layout=True)
        for i,subject in enumerate(subjects):
            for j,model in enumerate(MODELS):
                ax = axes[i,j]
                sub = df[df.model == model]
                if subject != 'pooled':sub=sub[sub.subject_id == subject]
                for label,color,name in [(0,'#2467a8','Non-error'),(1,'#c85424','Error')]:
                    p = sub.loc[sub.label == label,'p_error']
                    ax.hist(p,bins=np.linspace(0,1,26),density=True,histtype='step',linewidth=1.8,color=color,
                            label=f'{name} (n={len(p)})')
                    ax.axvline(p.mean(),color=color,linestyle=':',linewidth=1)
                ax.axvline(.5,color='#888888',linestyle='--',linewidth=.8)
                ax.set(title=f'{model.upper()} | {subject}',ylabel='Conditional density',xlim=(0,1))
                ax.legend(fontsize=8,loc='upper right')
                if i==len(subjects)-1:ax.set_xlabel('Frozen calibrated P(error)')
        fig.suptitle(f'{split.capitalize()} feedback distributions | decoder seed 42\nEach class normalized separately; dotted lines show class means',fontsize=12)
        fig.savefig(out/f'{split}_distributions.png',dpi=160)
        fig.savefig(out/f'{split}_distributions.pdf',metadata={'CreationDate':None,'ModDate':None})
        plt.close(fig)
    metrics, summary, gap = pd.DataFrame(rows), pd.DataFrame(stats), pd.DataFrame(gaps)
    for df,name in [(metrics,'probability_metrics'),(summary,'conditional_probability_stats'),(gap,'class_mean_gaps')]:
        df.to_csv(out/f'{name}.csv',index=False,float_format='%.10g')
    check = {'scope':'CPU analysis of downloaded seed-42 CSVs; no checkpoint inference or training rerun',
        'release_url':RELEASE,'release_tag':TAG,'downloaded_assets':provenance,
        'trial_manifest_sha256':sha(a.trial_manifest),'train_n':len(train),'train_n_error':int(train.label.sum()),
        'constant_probability':prior,'checks':{'download_checksums':True,'disjoint_participants':True,
        'trial_label_alignment':True,'probability_ranges':True,'checkpoint_and_calibration_ids':True,
        'test_metrics_match_release':True},'metric_reproduction':metric_differences,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                       'matplotlib':matplotlib.__version__}}
    (out/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    profile={'schema':'csci642-rl-feedback/v1','release_url':RELEASE,'release_tag':TAG,
        'release_source_commit':'bccbb8a2918c0a08381852feb1c6b1eb81e19548',
        'checkpoint_training_commit':manifest['source_sha'],'decoder_seed':42,
        'label_mapping':{'0':'non-error','1':'machine error'},'probability_column':'p_error',
        'splits':{s:manifest['splits'][s] for s in sets},
        'tables':{s:{'asset':f'{s}_predictions.csv',**provenance[f'{s}_predictions.csv'],
                     'trials_per_model':int(len(tables[s])//2)} for s in tables},
        'manifest':provenance['manifest.json'],'models':{},'constant_train_prevalence':prior}
    for model in MODELS:
        run=manifest['selection'][model]['selected_run_id'];e=manifest['checkpoints'][run+':best']
        profile['models'][model]={'checkpoint_id':run+':best','archive':f'run_{run}.tar.gz',
            'archive_sha256':sums[f'run_{run}.tar.gz'],'best_ckpt':e['best_ckpt'],
            'best_ckpt_sha256':e['files'][e['best_ckpt']]['sha256'],
            'preprocessing_id':tables['test'].query('model == @model').preprocessing_id.iloc[0],
            'calibration_id':run+':temperature','calibration':manifest['calibrations'][run+':temperature']}
    dest=Path(a.profile);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(profile,indent=2)+'\n')
    pooled=metrics[metrics.subject_id=='pooled']
    report=['# Frozen EEG feedback quality','',f'Source: [{TAG}]({RELEASE}). Decoder seed 42 only.',
        '',f'The constant baseline is fitted only from the training labels: {int(train.label.sum())}/{len(train)} = **{prior:.9f}**. '
        'It is applied unchanged to validation and test, including every participant. It does not use evaluation prevalence.',
        '', '## Probability scores', '', 'Lower Brier and NLL are better. Negative deltas mean improvement over the constant baseline. '
        'NLL uses natural logarithms and probabilities clipped to [1e-12, 1-1e-12]. AUROC uses average ranks for ties. Metrics are recomputed from the released 10-significant-digit CSVs: AUROC/Brier tolerance is 1e-7 and NLL tolerance is 1e-5 because rounding extreme probabilities amplifies log-loss differences. Exact differences are recorded in verification.json.', '',
        md_table(pooled,['split','model','probability','n','auroc','brier','nll','delta_brier_vs_constant','delta_nll_vs_constant']),
        '', '## Label-conditional feedback', '',
        'Mean gap = mean P(error) on error trials minus mean P(error) on non-error trials. Positive gaps describe useful direction, '
        'not causal effects or evidence that RL will improve. Both classes overlap substantially.', '',
        md_table(gap,['split','subject_id','model','mean_nonerror','mean_error','mean_gap']), '',
        'Full per-participant metrics: [probability_metrics.csv](probability_metrics.csv). '
        'Counts, sample SD (ddof=1), means, 5/25/50/75/95 percentiles and fraction in [0.45, 0.55]: '
        '[conditional_probability_stats.csv](conditional_probability_stats.csv). Quantiles use linear interpolation.', '',
        '![Validation distributions](validation_distributions.png)', '', '![Test distributions](test_distributions.png)', '',
        '## Interpretation and scope', '',
        '- EEGNet seed 42 is only slightly better than the constant baseline in test Brier; its test NLL is worse. '
        'CBraMod is worse than the baseline on both probability scores. AUROC measures ranking and does not establish calibrated probabilities.',
        '- The baseline is a diagnostic comparator, not a replacement selected using the test set. No model, seed, temperature, threshold or feedback table was changed.',
        '- Validation participants were already used for decoder selection and calibration. Their scores are not unbiased final estimates; subsequent RL tuning may use them under the agreed protocol.',
        '- These are descriptive pooled and participant results on one frozen split. Five training seeds are not five independent participant samples. No significance claim is made.',
        '- The plots describe the stored error/non-error trial library. RL action-to-label mapping, reward transformation, feedback frequency, sampling policy and RL random seeds belong to the later stage.',
        '- [verification.json](verification.json) records the downloaded-file checks and reproduction environment for this analysis. '
        'This CPU check does not independently verify checkpoint loading or continuation; original GPU verification is reported separately in [artifact_index.md](../artifact_index.md).', '']
    (out/'README.md').write_text('\n'.join(report))
    print(pooled[['split','model','probability','brier','nll']].to_string(index=False))

if __name__ == '__main__':
    main()
