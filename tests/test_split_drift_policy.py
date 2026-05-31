import unittest


class SplitDriftPolicyTests(unittest.TestCase):
    def test_evaluate_acceptance_warn_on_low_ks_pvalue(self) -> None:
        from harchoc.split_drift_policy import DriftAcceptanceConfig, evaluate_acceptance

        report = {
            "comparisons": {
                "train_vs_val": {
                    "labels": {"class_dist_l1": 0.01, "class_jsd_nats": 0.01},
                    "images": {
                        "width_ks": {"available": True, "pvalue": 0.001},
                    },
                }
            }
        }
        cfg = DriftAcceptanceConfig()
        acc = evaluate_acceptance(report, cfg=cfg)
        self.assertEqual(acc["status"], "fail")

    def test_emit_plots_skipped_without_matplotlib(self) -> None:
        from harchoc.split_drift_plots import emit_split_drift_plots
        from pathlib import Path
        import tempfile

        report = {
            "comparisons": {
                "train_vs_val": {
                    "labels": {"class_dist_l1": 0.2, "class_jsd_nats": 0.1},
                }
            }
        }
        with tempfile.TemporaryDirectory() as td:
            r = emit_split_drift_plots(report, out_dir=Path(td) / "plots")
            self.assertIn(r["status"], ("ok", "skipped"))


if __name__ == "__main__":
    unittest.main()
