import json
import unittest
from pathlib import Path


class NowTodosSmokeTests(unittest.TestCase):
    def test_cpu_smoke_bundle(self) -> None:
        from harchoc.now_todos_smoke import run_now_todos_smoke

        repo = Path(__file__).resolve().parents[1]
        payload, rc = run_now_todos_smoke(repo, stage_group="verify")
        self.assertEqual(rc, 0, payload)
        self.assertEqual(payload["n_fail"], 0)

        payload, rc = run_now_todos_smoke(repo, stage_group="cpu")
        self.assertEqual(rc, 0, payload)
        self.assertEqual(payload["n_fail"], 0)
        report = repo / "reports/manuscript/now_todos_smoke.json"
        self.assertTrue(report.is_file())
        doc = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], "now_todos_smoke_run.v1")
        ids = {s["id"] for s in doc["stages"]}
        self.assertIn("tables_repro", ids)
        self.assertIn("post_zoo_queue_dry", ids)

    def test_experiment_cli_cpu(self) -> None:
        from scripts.experiment import main

        repo = Path(__file__).resolve().parents[1]
        rc = main(["now-todos-smoke", "--stage", "verify"])
        self.assertEqual(rc, 0)
        rc = main(["now-todos-smoke", "--stage", "cpu"])
        self.assertEqual(rc, 0)

    def test_post_zoo_smoke_manifest_loads(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(
            repo / "configs/experiments/gpu_queue_post_zoo_smoke.json", repo_root=repo
        )
        self.assertNotIn("require_before", m)
        kinds = [j.get("kind") for j in m["jobs"]]
        self.assertIn("finetune_tray", kinds)


if __name__ == "__main__":
    unittest.main()
