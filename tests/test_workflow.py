"""Synthetic checks for the published three-task workflow; no database downloads."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
from typing import Optional
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import evaluate_downstream_strategies as base
import evaluate_true_truncation_results as training
import evaluate_signal_fragment_results as signal_evaluation
import evaluate_window_distillation_results as window_evaluation
from encode_positional_inputs import select_segments
from pool_domain_random import interval_mask, contiguous_runs, random_matched_mask
from build_crossfit_window_teacher import window_distribution
from build_signal_fragment_coordinates import rolling_mass
from train_window_student import WindowStudentCNN, predict_distribution


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(17)
        cls.x = rng.normal(size=(60, 8)).astype(np.float32)
        cls.y = (cls.x[:, :3] > 0).astype(int)

    def test_all_python_files_compile(self):
        for path in (REPO / 'scripts').glob('*.py'):
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_threshold_tie_chooses_lowest(self):
        # All grid values yield the same binary prediction, hence the same F1.
        thresholds = base.select_multilabel_thresholds(np.array([[1], [0]]), np.array([[1.], [0.]]))
        np.testing.assert_allclose(thresholds, [.1])

    def test_logistic_thresholds_precede_combined_refit(self):
        fit_sizes = []
        original = base.fit_lasso
        validation_probs = []
        original_thresholds = base.select_multilabel_thresholds

        def fit(*args):
            fit_sizes.append(len(args[1]) + len(args[3]))
            return original(*args)

        def choose(y, probability):
            validation_probs.append(probability.copy())
            return original_thresholds(y, probability)

        with patch.object(base, 'fit_lasso', side_effect=fit), patch.object(base, 'select_multilabel_thresholds', side_effect=choose):
            model, best, thresholds = training.choose_lasso('multilabel', self.x[:40], self.y[:40], self.x[40:50], self.y[40:50], 42)
        self.assertEqual(fit_sizes, [40, 50])
        self.assertEqual(len(validation_probs), 1)
        np.testing.assert_allclose(thresholds, original_thresholds(self.y[40:50], validation_probs[0]))
        # The final logistic model's scaler must include the validation inputs.
        np.testing.assert_allclose(model.estimators_[0].named_steps['standardscaler'].mean_, self.x[:50].mean(0), atol=1e-7)

    def test_xgboost_and_dnn_fit_and_predict(self):
        xgb = dict(n_estimators=8, learning_rate=.1, max_depth=2, subsample=1., colsample_bytree=1.)
        with patch.object(base, 'xgb_param_candidates', return_value=[xgb]):
            model, best, threshold = training.choose_xgboost('multilabel', self.x[:40], self.y[:40], self.x[40:50], self.y[40:50], 42, 1, 1)
        pred, prob = training.predict_model('multilabel', 'xgboost', model, self.x[50:], threshold)
        self.assertEqual(pred.shape, (10, 3))
        self.assertTrue(np.isfinite(prob).all())
        args = types.SimpleNamespace(dnn_trials=1, dnn_max_epochs=2, dnn_patience=1, batch_size=16)
        with patch.object(base, 'mlp_candidates', return_value=[{'hidden':[8], 'dropout':0., 'lr':.01}]):
            model, scaler, best, threshold = training.choose_dnn('multilabel', self.x[:40], self.y[:40], self.x[40:50], self.y[40:50], 42, args)
        self.assertEqual(best['final_epochs'], best['best_epoch'])
        np.testing.assert_allclose(scaler.mean_, self.x[:50].mean(0), atol=1e-7)
        pred, prob = training.predict_model('multilabel', 'dnn', model, self.x[50:], threshold, scaler)
        self.assertEqual(pred.shape, (10, 3))

    def test_domain_random_residue_counts(self):
        mask = interval_mask(51, [[2,10], [8,16], [31,37]])
        for seed in [101,102,103,104,105]:
            control = random_matched_mask(51, contiguous_runs(mask), np.random.default_rng(seed))
            self.assertEqual(mask.sum(), control.sum())

    def test_three_evaluation_entry_points(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'splits').mkdir()
            entries = np.asarray([f'P{i:03}' for i in range(60)])
            np.savez(root/'splits/EC_level2_split_entries.npz', train_entries=entries[:40],
                     val_entries=entries[40:50], test_entries=entries[50:],
                     y_train=self.y[:40], y_val=self.y[40:50], y_test=self.y[50:], label_cols=np.asarray(['a','b','c']))
            files = [root/'embeddings_full_length/esm2/EC_level2/full_length_mean.npz',
                     root/'embeddings_signal_fragments/esm2/EC_level2/task_global/300/high_signal/embeddings.npz',
                     root/'embeddings_window_distillation/esm2/EC_level2/student_window/embeddings.npz']
            for path in files:
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez(path, **dict(zip(entries,self.x)))
            args = types.SimpleNamespace(outdir=root/'outputs', overwrite=False, random_state=42,
                xgb_trials=1,num_workers=1,dnn_trials=1,dnn_max_epochs=2,dnn_patience=1,batch_size=16,
                max_samples_per_split=None,smoke_test=True,selection_mode='task_global',fragment_length=300)
            with patch.object(base,'ROOT',root), patch.object(training,'ROOT',root), \
                 patch.object(base,'xgb_param_candidates',return_value=[dict(n_estimators=8,learning_rate=.1,max_depth=2,subsample=1.,colsample_bytree=1.)]), \
                 patch.object(base,'mlp_candidates',return_value=[dict(hidden=[8],dropout=0.,lr=.01)]):
                for model in ['lasso','xgboost','dnn']:
                    training.run_one(args,'EC_level2','esm2','full_length_mean',model)
                    args.embedding_root=root/'embeddings_signal_fragments'
                    signal_evaluation.run_one(args,'EC_level2','esm2','high_signal',model)
                    args.embedding_root=root/'embeddings_window_distillation'
                    window_evaluation.run_one(args,'EC_level2','student_window',model)
            results = list((root/'outputs').rglob('metrics.json'))
            self.assertEqual(len(results),9)
            for path in results:
                metrics=json.loads(path.read_text())
                self.assertEqual(metrics['validation_protocol'],'train_only_tuning_frozen_val_thresholds_train_plus_val_refit')
                with np.load(path.with_name('predictions.npz')) as saved:
                    np.testing.assert_array_equal(saved['y_true'],self.y[50:])

    def test_positional_boundary_rules(self):
        # Load the actual coordinate helpers without loading a PLM dependency.
        tree = ast.parse((REPO/'scripts/prott5_input_utils.py').read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['clean_seq','fixed_slice','select_domain_region']]
        ns = {'re':re, 'WINDOW_SIZE':1000, 'Optional':Optional}
        exec(compile(ast.Module(body=funcs,type_ignores=[]), '<coordinate helpers>', 'exec'), ns)
        utils = types.SimpleNamespace(**ns)
        sequence = 'A'*1500
        self.assertEqual(len(select_segments(sequence, [[1300,1500]], 'domain_center_longest', 'prott5', 'Subcellular', utils)[0]),600)
        self.assertEqual(len(select_segments(sequence, [[1300,1500]], 'domain_center_longest', 'prott5', 'EC_level2', utils)[0]),1000)
        for strategy in ['head1000','mid1000','tail1000','splice300_400_300']:
            self.assertEqual(len(select_segments(sequence, [], strategy, 'prott5', 'EC_level2', utils)[0]),1000)

    def test_teacher_and_student_distributions(self):
        signal = np.array([0., 2., 5., 1., 0.])
        np.testing.assert_allclose(rolling_mass(signal, 2), [2.,7.,6.,1.])
        teacher = window_distribution(signal, 2, 1.)
        self.assertEqual(int(np.argmax(teacher)), 1)
        self.assertAlmostEqual(float(teacher.sum()), 1., places=6)
        model = WindowStudentCNN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 25601)
        probability = predict_distribution(model, torch.device('cpu'), 'ACDEFGHIKL'*60, 500)
        self.assertEqual(len(probability), 101)
        self.assertAlmostEqual(float(probability.sum()), 1., places=5)

    def test_preparation_and_saved_metric_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = pd.DataFrame({'Entry':[f'P{i:03}' for i in range(60)], 'Length':[1100]*60,
                                  'Sequence':['A'*1100]*60, 'final_ec2':['2.1' if i%2 else '3.1' for i in range(60)]})
            frame.to_csv(tmp/'candidate.csv',index=False)
            frame[['Entry']].to_csv(tmp/'entries.csv',index=False)
            (tmp/'domains.json').write_text(json.dumps({e:[[10,40]] for e in frame.Entry}))
            env = dict(os.environ, PROTEIN_INPUT_WORKDIR=str(tmp/'work'))
            cmd = [sys.executable,str(REPO/'scripts/prepare_datasets.py'),'--task','EC_level2','--sequences',str(tmp/'candidate.csv'),'--domains',str(tmp/'domains.json'),'--available-entries',str(tmp/'entries.csv')]
            result = subprocess.run(cmd,capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            with np.load(tmp/'work/splits/EC_level2_split_entries.npz',allow_pickle=True) as z:
                groups = [set(z[f'{s}_entries']) for s in ['train','val','test']]
                self.assertEqual([len(g) for g in groups],[48,6,6])
                self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
            truth=np.array([[1,0,0],[0,1,0],[1,0,0]])
            pred=np.array([[1,0,0],[1,1,0],[0,0,0]])
            np.savez(tmp/'predictions.npz',y_true=truth,y_pred=pred)
            result=subprocess.run([sys.executable,str(REPO/'scripts/evaluate_predictions.py'),'--predictions',str(tmp/'predictions.npz')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertAlmostEqual(json.loads(result.stdout)['macro_f1'],f1_score(truth,pred,average='macro',zero_division=0))


if __name__ == '__main__':
    unittest.main()
