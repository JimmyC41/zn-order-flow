import csv
import datetime as dt
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from conditional_forecast import ConditionalHawkes, Segment
from run_validation_forecast import days_from_csv,evaluate_day
from summarize_results import validate_score_source

HERE=Path(__file__).resolve().parent


class RunnerTests(unittest.TestCase):
    def csv_file(self,folder,rows,fields=('day','t')):
        path=Path(folder)/'events.csv'
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(fields);w.writerows(rows)
        return path

    def test_rejects_dates_outside_validation(self):
        for day in ['2026-04-30','2026-09-01']:
            with tempfile.TemporaryDirectory() as d:
                p=self.csv_file(d,[(day,1)])
                with self.assertRaises(ValueError):list(days_from_csv(p,dt.date(2026,5,1),dt.date(2026,8,31),24000))

    def test_rejects_bad_format_and_time_order(self):
        cases=[([('2026-05-04',1)],('date','time')),
               ([('2026-05-04',2),('2026-05-04',1)],('day','t')),
               ([('2026-05-04','nan')],('day','t')),
               ([('20260508',1)],('day','t')),
               ([('2026-W19-5',1)],('day','t')),
               ([('2026-05-04',24000)],('day','t')),
               ([('2026-05-05',1),('2026-05-04',1)],('day','t'))]
        for rows,fields in cases:
            with tempfile.TemporaryDirectory() as d:
                p=self.csv_file(d,rows,fields)
                with self.assertRaises(ValueError):list(days_from_csv(p,dt.date(2026,5,1),dt.date(2026,8,31),24000))

    def command(self,orders,out,*extra):
        return [sys.executable,str(HERE/'run_validation_forecast.py'),'--orders',str(orders),
                '--output',str(out),'--horizons','600','--input-label','synthetic_smoke',*extra]

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.csv_file(d,[('2026-05-04',1)]);out=Path(d)/'existing';out.mkdir()
            sentinel=out/'summary.csv';sentinel.write_text('preserve this')
            run=subprocess.run(self.command(p,out),capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);self.assertEqual(sentinel.read_text(),'preserve this')

    def test_rejects_altered_split_and_zero_nan_delay(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.csv_file(d,[('2026-05-04',1)]);config=json.loads((HERE.parents[1]/'analysis/config.json').read_text())
            config['validate_end']='2026-10-31';cp=Path(d)/'changed.json';cp.write_text(json.dumps(config))
            cases=[('--config',str(cp)),('--delays-ms','0'),('--delays-ms','nan')]
            for i,args in enumerate(cases):
                run=subprocess.run(self.command(p,Path(d)/str(i),*args),capture_output=True,text=True)
                self.assertNotEqual(run.returncode,0)

    def test_fixed_grid_and_future_prefix_invariance(self):
        def run(events):
            m=ConditionalHawkes([.2,.3],[2,20]);saved=m.forecast;seen=[]
            def tracked(state,cut,start,end,schedule):
                result=saved(state,cut,start,end,schedule);seen.append((start,result['expected_events_in_interval']))
                return result
            m.forecast=tracked
            rows=evaluate_day('2026-05-04',events,m,[Segment(0,30,2)],[10],[.001,.1,1])
            self.assertTrue(all(r['n_windows']==2 for r in rows))
            return [x for x in seen if x[0]==10]
        self.assertEqual(run([1,9,12]),run([1,9,10,12,25,26,27]))

    def test_half_open_targets_and_exact_cutoff(self):
        model = ConditionalHawkes([0.25], [2])
        captured = []
        forecast = model.forecast

        def observe(state, cutoff, start, end, schedule):
            captured.append((cutoff, tuple(state)))
            return forecast(state, cutoff, start, end, schedule)

        model.forecast = observe
        events = [9, 9.5, 10, 10, 19, 20, 29]
        rows = evaluate_day("2026-05-04", events, model, [Segment(0, 30, 1)], [10], [1])
        # [10,20) contains three arrivals and [20,30) contains two.
        self.assertTrue(all(row["mean_observed_count"] == 2.5 for row in rows))
        self.assertEqual(captured[0], (9, (0.25,)))
        self.assertAlmostEqual(captured[1][1][0], model.state_from_history(events[:5], 19)[0])

    def test_rejects_delays_that_expose_the_target(self):
        model = ConditionalHawkes([0.25], [2])
        for delay in [0.0, 1e-20, -1.0]:
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                evaluate_day("2026-05-04", [10.], model, [Segment(0, 30, 1)], [10], [delay])

    def test_failed_run_can_retry_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "scores"
            broken = self.csv_file(d, [("2026-05-04", 24000)])
            rejected = subprocess.run(self.command(broken, out), capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(out.exists())
            repaired = self.csv_file(d, [("2026-05-04", 2)])
            retried = subprocess.run(self.command(repaired, out), capture_output=True, text=True)
            self.assertEqual(retried.returncode, 0, retried.stderr)

    def test_complete_table_checks_empirical_score_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            scores = path / "daily_scores.csv"
            scores.write_text("example score rows")
            score_hash = hashlib.sha256(scores.read_bytes()).hexdigest()
            manifest = {"n_orders": 123, "source_hashes": {"output/model.json": "a",
                        "analysis/config.json": "b", "data/news_releases.csv": "c"}}
            provenance = {"input_label": "validation_export", "n_orders": 123,
                          "daily_scores_sha256": score_hash, "validation_range": ["2026-05-01", "2026-08-31"],
                          "horizons_s": [1, 10, 600], "delays_s": [0.001, 0.1, 1],
                          "input_sha256": {"orders": "d", "model": "a", "config": "b", "news": "c"}}
            provenance_path = path / "provenance.json"
            # Manifest format and production input-hash validation have separate tests.
            with patch("summarize_results.validate_export_manifest"), patch("summarize_results.verify_export_input") as check_input:
                provenance_path.write_text(json.dumps(provenance))
                validate_score_source(path, manifest, scores)
                check_input.assert_called_once_with(manifest, "validation_orders.csv.gz", "d")
                for key, value in [("input_label", "synthetic_smoke"), ("n_orders", 10),
                                   ("daily_scores_sha256", "wrong"), ("horizons_s", [10])]:
                    with self.subTest(key=key):
                        provenance_path.write_text(json.dumps({**provenance, key: value}))
                        with self.assertRaises(ValueError):
                            validate_score_source(path, manifest, scores)

    def test_cli_valid_synthetic_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.csv_file(d,[('2026-05-04',1),('2026-05-04',3)]);out=Path(d)/'new'
            run=subprocess.run(self.command(p,out),capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            prov=json.loads((out/'provenance.json').read_text())
            self.assertEqual(prov['input_label'],'synthetic_smoke')
            self.assertTrue((out/'daily_scores.csv').is_file())

if __name__=='__main__':unittest.main(verbosity=2)
