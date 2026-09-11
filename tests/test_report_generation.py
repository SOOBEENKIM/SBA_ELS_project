"""Regression checks using temporary copies of the recorded result files.

Run after scripts/restore_artifacts.py with:
python -m unittest discover -s tests -p 'test_report_generation.py' -v
No MC simulation, model inference or training is performed.
"""
from pathlib import Path
import importlib.util
import json
import shutil
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report', ROOT / 'scripts/summarize_results.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
A = report.V3 / 'results'
B = report.V4 / 'results'


class ReportGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='els-report-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        files = [report.V3 / 'protocol.json', report.V4 / 'protocol.json', A / 'model_selection.json',
                 A / 'mc_labels.csv', A / 'model_metrics.csv', A / 'predictions.csv',
                 B / 'mc_labels.csv', B / 'model_metrics.csv', B / 'mc_effect_summary.csv', B / 'predictions.csv',
                 Path('analysis/mc_v3_audit_20260910/results/independent_mc.csv')]
        for name in files:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)

    def test_recorded_prices_and_errors_are_preserved(self):
        text, plot = report.render(self.root)
        # Frozen result values, not expectations calculated through the renderer.
        self.assertIn('| 연 쿠폰 +1%p | +56.29원 (128개) | +100.50원 (128개) |', text)
        self.assertIn('| 일반형 | 가격 + 증분 손실 + 쿠폰 선형 구조 (선택) | 0.9703 | 64.84 | 7.43 | 98.37% |', text)
        self.assertIn('일반형 **14.92→7.43원**으로 감소했다.', text)
        self.assertIn('월지급형 **11.70→6.41원**으로 감소했다.', text)
        self.assertIn('MC **−9.55원**, 선택 DeepONet **+9.44원**으로 방향이 달랐다.', text)
        self.assertIn('모델 평균 **−375.97원**, 증분 MAE **384.78원**', text)
        self.assertEqual(set(plot.model), {'affine_coupon_delta'})

    def test_changed_metrics_update_prose_and_reversed_improvement(self):
        path = self.root / A / 'model_metrics.csv'
        frame = pd.read_csv(path)
        for cohort, mae, r2 in [('regular_test', 32.10, .4321), ('monthly_test', 21.98, .5432)]:
            mask = frame.cohort.eq(cohort) & frame.model.eq('affine_coupon_delta') & frame.seed.eq('ensemble') & frame.axis.eq('ALL')
            frame.loc[mask, ['delta_mae', 'price_r2']] = [mae, r2]
        frame.to_csv(path, index=False)
        text, _ = report.render(self.root)
        self.assertIn('일반형 **14.92→32.10원**으로 증가했다.', text)
        self.assertIn('월지급형 **11.70→21.98원**으로 증가했다.', text)
        self.assertIn('가격 R²는 일반형 0.4321, 월지급형 0.5432였다.', text)
        self.assertNotIn('14.92→7.43', text)
        self.assertNotIn('11.70→6.41', text)

    def test_selection_changes_all_selected_model_references(self):
        path = self.root / A / 'model_selection.json'
        decision = json.loads(path.read_text())
        decision['selected_arm'] = 'augmented_delta'
        path.write_text(json.dumps(decision))
        text, plot = report.render(self.root)
        selected_rows = [line for line in text.splitlines() if line.startswith('|') and '(선택)' in line]
        self.assertEqual(len(selected_rows), 2)
        self.assertTrue(all('| 가격 + 증분 손실 (선택) |' in row for row in selected_rows))
        self.assertIn('일반형 **14.92→8.92원**으로 감소했다.', text)
        self.assertIn('월지급형 **11.70→6.19원**으로 감소했다.', text)
        self.assertIn('선택 DeepONet **+9.46원**', text)
        self.assertEqual(set(plot.model), {'augmented_delta'})

    def test_changed_predictions_update_case_direction_and_schedule_means(self):
        path = self.root / A / 'predictions.csv'
        frame = pd.read_csv(path)
        mask = frame.family.eq('published_KR6MD0003TN4') & frame.model.eq('affine_coupon_delta') & frame.seed.eq('ensemble') & frame.axis.eq('first_strike') & frame.offset.eq(.05)
        frame.loc[mask, 'pred_delta_krw'] = -12.34
        frame.to_csv(path, index=False)
        path = self.root / B / 'predictions.csv'
        frame = pd.read_csv(path)
        mask = frame.cohort.eq('monthly_test_128') & frame.model.eq('affine_coupon_delta') & frame.training_seed.eq('ensemble') & frame.axis.eq('nobs_middle')
        frame.loc[mask, 'pred_delta_krw'] = 42.25
        frame.to_csv(path, index=False)
        text, _ = report.render(self.root)
        self.assertIn('선택 DeepONet **−12.34원**으로 방향이 같았다.', text)
        self.assertIn('모델 평균 **+42.25원**', text)
        self.assertNotIn('−375.97', text)

    def test_missing_or_duplicate_result_fails_before_overwriting_report(self):
        path = self.root / A / 'model_metrics.csv'
        original = pd.read_csv(path)
        mask = original.cohort.eq('regular_test') & original.model.eq('affine_coupon_delta') & original.seed.eq('ensemble') & original.axis.eq('ALL')
        readme = self.root / 'README.md'
        readme.write_text('Original report must survive invalid input.\n')
        for frame in [original.loc[~mask], pd.concat([original, original.loc[mask]])]:
            with self.subTest(rows=len(frame)):
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, 'Expected exactly one result'):
                    report.main(self.root)
                self.assertEqual(readme.read_text(), 'Original report must survive invalid input.\n')
                self.assertFalse((self.root / 'docs/RESULTS.md').exists())


if __name__ == '__main__':
    unittest.main()
