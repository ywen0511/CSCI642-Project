"""CPU regression checks for verification gates and frozen-feedback analysis."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import verify_release as verify
import analyze_feedback as feedback
import download_feedback as download
import csv


class VerifyReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.dl=self.root/'dl';self.dl.mkdir()
        for name in verify.REQUIRED:(self.dl/name).write_text('{}')
        with tarfile.open(self.dl/'run.tar.gz','w:gz'):pass
        self.refresh()
        self.out=self.root/'report.json'
        self.ex=self.root/'extracted'

    def refresh(self):
        self.sizes={p.name:p.stat().st_size for p in self.dl.iterdir() if p.name!='SHA256SUMS'}
        (self.dl/'SHA256SUMS').write_text(''.join(f'{verify.sha256(self.dl/n)}  {n}\n' for n in sorted(self.sizes)))

    def args(self):
        return ['--download-dir',str(self.dl),'--extract-dir',str(self.ex),'--prep-dir','unused','--out',str(self.out)]

    def result(self):return json.loads(self.out.read_text())

    def test_corrupt_asset_blocks_extraction_and_inference(self):
        (self.dl/'manifest.json').write_text('corrupted')
        with patch.object(verify,'reproduce') as run:
            self.assertEqual(verify.main(self.args()),1);run.assert_not_called()
        self.assertFalse(self.ex.exists());self.assertFalse(self.result()['ok'])

    def test_missing_asset_fails(self):
        (self.dl/'test_predictions.csv').unlink()
        self.assertEqual(verify.main(self.args()),1)

    def test_empty_checksum_inventory_fails(self):
        (self.dl/'SHA256SUMS').write_text('')
        self.assertEqual(verify.main(self.args()),1)

    def test_size_mismatch_fails(self):
        self.sizes['manifest.json']+=1
        p=self.root/'sizes.json';p.write_text(json.dumps(self.sizes))
        self.assertEqual(verify.main(self.args()+['--expected-sizes',str(p)]),1)

    def test_missing_api_inventory_entry_fails(self):
        self.sizes['missing.tar.gz']=10
        p=self.root/'sizes.json';p.write_text(json.dumps(self.sizes))
        self.assertEqual(verify.main(self.args()+['--expected-sizes',str(p)]),1)

    def test_unlisted_archive_fails(self):
        (self.dl/'extra.tar.gz').write_bytes(b'extra')
        self.assertEqual(verify.main(self.args()),1)

    def test_reproduction_failure_returns_nonzero(self):
        with patch.object(verify,'reproduce',return_value=({'eegnet':{'ok':True},'cbramod':{'ok':False}},'test')):
            self.assertEqual(verify.main(self.args()),1)
        self.assertFalse(self.result()['all_reproduction_ok'])

    def test_empty_reproduction_fails(self):
        with patch.object(verify,'reproduce',return_value=({},'test')):
            self.assertEqual(verify.main(self.args()),1)

    def test_inference_exception_saved_and_fails(self):
        with patch.object(verify,'reproduce',side_effect=RuntimeError('prediction mismatch')):
            self.assertEqual(verify.main(self.args()),1)
        self.assertIn('prediction mismatch',self.result()['error'])

    def test_success_returns_zero(self):
        with patch.object(verify,'reproduce',return_value=({'eegnet':{'ok':True},'cbramod':{'ok':True}},'test')):
            self.assertEqual(verify.main(self.args()),0)
        self.assertTrue(self.result()['ok'])

    def test_assets_only_does_not_claim_inference(self):
        with patch.object(verify,'reproduce') as run:
            self.assertEqual(verify.main(self.args()+['--assets-only']),0);run.assert_not_called()
        self.assertIsNone(self.result()['all_reproduction_ok']);self.assertFalse(self.ex.exists())

    def test_actual_cli_failure_exit_code(self):
        (self.dl/'manifest.json').write_text('bad')
        p=subprocess.run([sys.executable,str(ROOT/'scripts/verify_release.py'),*self.args()],capture_output=True)
        self.assertEqual(p.returncode,1)


class ProbabilityTests(unittest.TestCase):
    def test_constant_known_scores_and_auc_ties(self):
        s=feedback.scores([0,0,1,1],[.5]*4)
        self.assertAlmostEqual(s['brier'],.25);self.assertAlmostEqual(s['nll'],np.log(2))
        self.assertEqual(s['auroc'],.5)

    def test_perfect_ranking(self):
        self.assertEqual(feedback.scores([0,1,0,1],[.1,.9,.2,.8])['auroc'],1)

    def test_invalid_probabilities_rejected(self):
        for p in [[.1,np.nan],[-.1,.7],[.1,1.1]]:
            with self.assertRaises(ValueError):feedback.scores([0,1],p)

    def test_label_corruption_rejected(self):
        rows=[]
        for model in feedback.MODELS:
            rows.append(['s01','t1',1,'test',model,42,.3,.4,model+':best','p',model+':temperature'])
        df=pd.DataFrame(rows,columns=feedback.COLS)
        trials=pd.DataFrame({'subject_id':['s01'],'trial_id':['t1'],'label':[0],'retained':[1],'split':['test']})
        manifest={'selection':{m:{'selected_run_id':m} for m in feedback.MODELS},'preprocessing':{'p':{}}}
        with self.assertRaisesRegex(ValueError,'labels'):feedback.validate_table(df,'test',trials,manifest)


class DownloadTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'test.csv'
        self.profile={'models':{m:{'checkpoint_id':m+':best','calibration_id':m+':temperature','preprocessing_id':'p'} for m in feedback.MODELS},
                      'decoder_seed':42,'splits':{'test':['s01']},'tables':{'test':{'trials_per_model':1}}}
        self.rows=[['s01','t1','0','test',m,'42','.3','.4',m+':best','p',m+':temperature'] for m in feedback.MODELS]
    def save(self):
        with self.path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(download.COLUMNS);w.writerows(self.rows)
    def test_valid_aligned_tables(self):
        self.save();self.assertTrue(download.verify_table(self.path,'test',self.profile)['ok'])
    def test_duplicate_trial_rejected(self):
        self.rows.append(self.rows[0]);self.save()
        with self.assertRaisesRegex(ValueError,'Duplicate'):download.verify_table(self.path,'test',self.profile)
    def test_cross_model_label_mismatch_rejected(self):
        self.rows[1][2]='1';self.save()
        with self.assertRaisesRegex(ValueError,'aligned'):download.verify_table(self.path,'test',self.profile)
    def test_wrong_checkpoint_rejected(self):
        self.rows[0][8]='wrong';self.save()
        with self.assertRaisesRegex(ValueError,'checkpoint'):download.verify_table(self.path,'test',self.profile)


if __name__=='__main__':unittest.main()
