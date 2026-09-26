"""Regression tests for Phase 2A validation-only safety and helpers."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.evaluation.phase2a import (load_validation_predictions, require_validation_predictions,
    select_sensitivity_constrained_threshold, validate_per_view_export)

class Phase2ATests(unittest.TestCase):
    def test_calibration_loader_refuses_test_predictions(self):
        with self.assertRaisesRegex(ValueError,'test predictions are forbidden'):
            require_validation_predictions('models/resnet18/test_predictions.csv')

    def test_sensitivity_constraint_selects_highest_feasible_threshold(self):
        # Positives at .90, .80, .20; target 2/3 means .80 is the highest feasible threshold.
        row=select_sensitivity_constrained_threshold([1,1,1,0,0],[.90,.80,.20,.70,.10],2/3)
        self.assertAlmostEqual(float(row['threshold']),.8)
        self.assertGreaterEqual(float(row['sensitivity']),2/3)

    def test_validation_loader_preserves_patient_study_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'validation_predictions.csv'
            pd.DataFrame({'patient_id':['p1','p2'],'study_id':['s1','s2'],'label':[0,1],'num_views':[1,2],'abnormal_probability':[.1,.9]}).to_csv(path,index=False)
            loaded=load_validation_predictions(path)
            self.assertEqual(loaded.patient_id.tolist(),['p1','p2'])
            self.assertEqual(loaded.study_id.tolist(),['s1','s2'])

    def test_validation_loader_rejects_duplicate_study_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'validation_predictions.csv'
            pd.DataFrame({'patient_id':['p1','p1'],'study_id':['s1','s1'],'label':[0,0],'num_views':[1,1],'abnormal_probability':[.1,.2]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'one row per study'):
                load_validation_predictions(path)

    def test_per_view_export_rejects_padded_or_missing_view(self):
        frame=pd.DataFrame({'patient_id':['p1'],'study_id':['s1'],'image_path':[''],'image_view_id':[2],
                            'study_label':[1],'view_probability':[.8],'num_views':[2]})
        with self.assertRaisesRegex(ValueError,'Padded/nonexistent'):
            validate_per_view_export(frame)

if __name__=='__main__': unittest.main()
