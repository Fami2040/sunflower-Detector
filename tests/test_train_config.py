import json
import unittest
from pathlib import Path


class TrainConfigExtendsTests(unittest.TestCase):
    def test_train_bench_yolov8n_merges_base(self) -> None:
        from harchoc.bench_config import _bench_to_train_config, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs/bench/yolov8n_default.yaml")
        doc = _bench_to_train_config(cfg, weights_path="yolov8n.pt")
        merged = doc["train"]
        self.assertEqual(merged["model"], "yolov8n.pt")
        self.assertEqual(merged["batch"], 1)
        self.assertEqual(merged["epochs"], 100)
        self.assertEqual(merged["aug_config"], "configs/aug/robustness_minimal.yaml")
        self.assertEqual(doc["eval"]["max_det"], 3000)
        self.assertNotIn("extends", merged)

    def test_train_bench_yolov8m_merged_eval_cpu_max_det(self) -> None:
        from harchoc.bench_config import _bench_to_train_config, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs/bench/yolov8m_default.yaml")
        doc = _bench_to_train_config(cfg, weights_path="yolov8m.pt")
        merged = doc["train"]
        self.assertEqual(merged["model"], "yolov8m.pt")
        self.assertEqual(doc["eval"]["max_det"], 3000)
        self.assertEqual(doc["eval"]["device"], "cpu")

    def test_train_rtdetr_smoke_15ep_merged_eval_skip(self) -> None:
        from harchoc.train_config import load_train_config_json

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "configs" / "experiments" / "train_rtdetr_smoke_15ep.json"
        merged = load_train_config_json(path, repo_root=repo_root)
        self.assertTrue(merged["eval"]["skip"])

    def test_train_rtdetr_queries_smoke_15ep_merged(self) -> None:
        from harchoc.rtdetr_limits import validate_rtdetr_query_cap
        from harchoc.train_config import load_train_config_json

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "configs" / "experiments" / "train_rtdetr_queries_smoke_15ep.json"
        merged = load_train_config_json(path, repo_root=repo_root)
        self.assertEqual(merged["epochs"], 15)
        self.assertEqual(merged["num_queries"], 1024)
        self.assertEqual(merged["model"], "configs/models/rtdetr-l_nq1024.yaml")
        self.assertTrue(merged["eval"]["skip"])
        self.assertEqual(merged["eval"]["max_det"], 1024)
        self.assertFalse(merged["accept_rtdetr_query_truncation"])
        warnings = validate_rtdetr_query_cap(
            model=str(merged["model"]),
            train_json=merged,
            train_json_path=str(path),
            fail=True,
        )
        self.assertEqual(warnings, [])

    def test_bench_to_train_config_after_extends(self) -> None:
        from harchoc.bench_config import _bench_to_train_config, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8m_default.yaml")
        doc = _bench_to_train_config(cfg, weights_path="/tmp/yolov8m.pt")
        self.assertEqual(doc["train"].get("max_det"), 3000)
        self.assertEqual(doc.get("eval", {}).get("max_det"), 3000)
        self.assertEqual(doc["train"]["optimizer"], "AdamW")

    def test_train_bench_configs_only_differ_from_baseline_in_allowed_fields(self) -> None:
        from harchoc.train_config import (
            BENCH_AUG_CONFIG_RELPATH,
            BENCH_PARITY_ALLOWED_DIFF_KEYS,
            load_train_config_json,
            normalize_train_config_for_bench_parity,
        )

        repo_root = Path(__file__).resolve().parents[1]
        baseline_path = repo_root / "configs" / "experiments" / "train_yolov8m_baseline.json"
        baseline = load_train_config_json(baseline_path, repo_root=repo_root)
        expected = normalize_train_config_for_bench_parity(
            baseline, repo_root=repo_root, bench_config_name="train_yolov8m_baseline.json"
        )

        exp_dir = repo_root / "configs" / "experiments"
        for path in sorted(exp_dir.glob("train_bench_*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if "model_id" in raw:
                continue

            resolved = load_train_config_json(path, repo_root=repo_root)
            norm = normalize_train_config_for_bench_parity(
                resolved, repo_root=repo_root, bench_config_name=path.name
            )
            self.assertEqual(
                expected,
                norm,
                msg=f"{path.name} drifts from {baseline_path.name} outside "
                f"{sorted(BENCH_PARITY_ALLOWED_DIFF_KEYS)} / aug_config",
            )
            if "aug_config" in resolved:
                self.assertEqual(resolved["aug_config"], BENCH_AUG_CONFIG_RELPATH)

    def test_train_bench_json_files_have_explicit_cache_false(self) -> None:
        from harchoc.train_config import validate_train_bench_raw_cache

        repo_root = Path(__file__).resolve().parents[1]
        exp_dir = repo_root / "configs" / "experiments"
        for path in sorted(exp_dir.glob("train_bench_*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            validate_train_bench_raw_cache(raw, path_label=path.name)

    def test_s13_patience_smoke_extends_s1_only_patience(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import validate_epochs_patience_close_mosaic

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")
        s1 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S1"), repo_root=repo_root)
        s13 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S13"), repo_root=repo_root)
        self.assertEqual(s1["patience"], 12)
        self.assertEqual(s13["patience"], 5)
        diff = {
            k: (s1.get(k), s13.get(k))
            for k in sorted(set(s1) | set(s13))
            if s1.get(k) != s13.get(k)
        }
        self.assertEqual(set(diff) - {"notes"}, {"patience"})
        validate_epochs_patience_close_mosaic(s13, repo_root=repo_root, label="aug_smoke S13")

    def test_train_schedules_satisfy_close_mosaic_guard(self) -> None:
        from harchoc.train_config import (
            load_train_config_json,
            validate_epochs_patience_close_mosaic,
        )

        repo_root = Path(__file__).resolve().parents[1]
        exp_dir = repo_root / "configs" / "experiments"
        paths = sorted(exp_dir.glob("train_bench_*.json"))
        paths.append(exp_dir / "train_yolov8m_baseline.json")
        for path in paths:
            resolved = load_train_config_json(path, repo_root=repo_root)
            validate_epochs_patience_close_mosaic(
                resolved, repo_root=repo_root, label=path.name
            )

    def test_aug_smoke_configs_close_mosaic_guard(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import (
            SMOKE_EPOCHS_MICRO,
            effective_train_aug_merged,
            scale_close_mosaic_for_epochs,
            validate_epochs_patience_close_mosaic,
        )

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")
        micro_close = scale_close_mosaic_for_epochs(SMOKE_EPOCHS_MICRO)
        micro_aug = "tests/fixtures/aug/micro_close1.yaml"
        smoke_expectations: dict[str, dict[str, float | int | str]] = {
            "S0": {
                "close_mosaic": micro_close,
                "mosaic": 0.1,
            },
            "S1": {
                "close_mosaic": micro_close,
                "mosaic": 0.1,
                "aug_config": micro_aug,
            },
            "S2": {
                "close_mosaic": 0,
                "mosaic": 0.0,
                "translate": 0.05,
                "scale": 0.15,
                "fliplr": 0.5,
                "hsv_s": 0.35,
            },
            "S3": {
                "close_mosaic": 0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "fliplr": 0.0,
                "erasing": 0.2,
                "hsv_s": 0.45,
                "hsv_v": 0.40,
            },
            "S4": {
                "close_mosaic": micro_close,
                "mosaic": 0.1,
                "translate": 0.10,
                "aug_config": "tests/fixtures/aug/micro_mosaic01.yaml",
            },
            "S5": {
                "close_mosaic": micro_close,
                "mosaic": 0.3,
                "aug_config": "tests/fixtures/aug/micro_close1_mosaic03.yaml",
            },
            "S6": {
                "close_mosaic": 0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "fliplr": 0.0,
                "erasing": 0.0,
                "hsv_s": 0.45,
                "hsv_v": 0.40,
            },
            "S7": {
                "close_mosaic": 0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "fliplr": 0.0,
                "erasing": 0.3,
                "hsv_s": 0.45,
                "hsv_v": 0.40,
            },
        }
        for sid, expected_aug in smoke_expectations.items():
            entry = find_smoke_entry(index, sid)
            resolved = resolve_aug_smoke_train_raw(entry, repo_root=repo_root)
            resolved["epochs"] = SMOKE_EPOCHS_MICRO
            resolved["patience"] = 1
            aug_override = expected_aug.pop("aug_config", None)
            if aug_override is not None:
                resolved["aug_config"] = aug_override
            self.assertEqual(resolved["epochs"], SMOKE_EPOCHS_MICRO)
            validate_epochs_patience_close_mosaic(
                resolved, repo_root=repo_root, label=sid
            )
            merged = effective_train_aug_merged(resolved, repo_root=repo_root)
            for key, value in expected_aug.items():
                self.assertEqual(merged[key], value, msg=f"{sid} {key}")

    def test_aug_smoke_s9_no_aug_yaml_resolves_without_aug_config(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import effective_train_aug_merged

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")
        resolved = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S9"), repo_root=repo_root)
        self.assertIsNone(resolved.get("aug_config"))
        self.assertEqual(resolved["epochs"], 15)
        self.assertEqual(resolved["patience"], 12)

        s0 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S0"), repo_root=repo_root)
        self.assertIsNotNone(s0.get("aug_config"))
        self.assertEqual(s0["aug_config"], "configs/aug/robustness_minimal.yaml")
        merged_s9 = effective_train_aug_merged(resolved, repo_root=repo_root)
        merged_s0 = effective_train_aug_merged(s0, repo_root=repo_root)
        self.assertNotIn("close_mosaic", merged_s9)
        self.assertEqual(merged_s0["close_mosaic"], 3)

    def test_smoke_rank_15ep_fragment_merges_baseline_schedule(self) -> None:
        from harchoc.train_config import SMOKE_EPOCHS_RANK, load_train_config_json

        repo_root = Path(__file__).resolve().parents[1]
        merged = load_train_config_json(
            repo_root / "configs/experiments/train_smoke_rank_15ep.json",
            repo_root=repo_root,
        )
        self.assertEqual(merged["epochs"], SMOKE_EPOCHS_RANK)
        self.assertEqual(merged["patience"], 12)
        self.assertEqual(merged["model"], "yolov8m.pt")
        self.assertEqual(merged["aug_config"], "configs/aug/robustness_minimal.yaml")

    def test_aug_s_smoke_raw_configs_use_rank_schedule_fragment(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        exp_dir = repo_root / "configs" / "experiments"
        rank_fragments = {
            "configs/experiments/train_smoke_rank_15ep.json",
            "configs/experiments/train_smoke_rank_yolo11s_15ep.json",
        }
        chained_parent = {
            "train_aug_s11_musgd_smoke.json": "configs/experiments/train_aug_s10_yolo11s_smoke.json",
        }
        for path in sorted(exp_dir.glob("train_aug_s*_smoke.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("epochs", raw, msg=path.name)
            self.assertNotIn("patience", raw, msg=path.name)
            extends = raw.get("extends")
            if path.name in chained_parent:
                self.assertEqual(extends, chained_parent[path.name])
            else:
                self.assertIn(extends, rank_fragments, msg=f"{path.name} extends={extends!r}")

    def test_aug_smoke_index_train_overrides_s9_s12_s13(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")
        s9 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S9"), repo_root=repo_root)
        s12 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S12"), repo_root=repo_root)
        s13 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S13"), repo_root=repo_root)
        self.assertIsNone(s9.get("aug_config"))
        self.assertFalse(s12.get("amp"))
        self.assertEqual(s13.get("patience"), 5)
        self.assertEqual(s9["epochs"], 15)
        self.assertEqual(s12["epochs"], 15)

    def test_train_hyperparams_common_merges_into_bench_and_baseline(self) -> None:
        from harchoc.train_config import load_train_config_json

        repo_root = Path(__file__).resolve().parents[1]
        base = load_train_config_json(
            repo_root / "configs/experiments/train_bench_base.json",
            repo_root=repo_root,
        )
        baseline = load_train_config_json(
            repo_root / "configs/experiments/train_yolov8m_baseline.json",
            repo_root=repo_root,
        )
        for cfg in (base, baseline):
            self.assertEqual(cfg["epochs"], 100)
            self.assertEqual(cfg["imgsz"], 1280)
            self.assertEqual(cfg["patience"], 50)
            self.assertEqual(cfg["seed"], 0)
        self.assertEqual(baseline["model"], "yolov8m.pt")
        self.assertEqual(baseline["batch"], 1)

    def test_aug_smoke_s0_s4_distinctness(self) -> None:
        import hashlib

        import yaml

        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import SMOKE_EPOCHS_RANK, effective_train_aug_merged

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")

        def _ultralytics_hash(relpath: str) -> str:
            doc = yaml.safe_load((repo_root / relpath).read_text(encoding="utf-8"))
            payload = json.dumps(doc.get("ultralytics") or {}, sort_keys=True).encode()
            return hashlib.sha256(payload).hexdigest()

        def _merged(sid: str) -> tuple[dict[str, object], dict[str, object]]:
            cfg = resolve_aug_smoke_train_raw(find_smoke_entry(index, sid), repo_root=repo_root)
            return cfg, effective_train_aug_merged(cfg, repo_root=repo_root)

        cases: tuple[tuple[str, str, str], ...] = (
            ("S0", "S1", "s0_minimal_vs_s1_close3_yaml"),
            ("S1", "S4", "s1_vs_s4_translate"),
            ("S2", "S3", "s2_mosaic0_vs_s3_photometric"),
        )
        for left_sid, right_sid, case_id in cases:
            with self.subTest(case_id=case_id, left=left_sid, right=right_sid):
                left_cfg, left = _merged(left_sid)
                right_cfg, right = _merged(right_sid)
                self.assertNotEqual(left_cfg["aug_config"], right_cfg["aug_config"])
                self.assertNotEqual(
                    _ultralytics_hash(str(left_cfg["aug_config"])),
                    _ultralytics_hash(str(right_cfg["aug_config"])),
                )
                if case_id == "s0_minimal_vs_s1_close3_yaml":
                    self.assertEqual(left_cfg["aug_config"], "configs/aug/robustness_minimal.yaml")
                    self.assertEqual(
                        right_cfg["aug_config"], "configs/aug/robustness_smoke_close3.yaml"
                    )
                    self.assertEqual(left["close_mosaic"], 3)
                    self.assertEqual(right["close_mosaic"], 3)
                    self.assertEqual(left["mosaic"], right["mosaic"])
                elif case_id == "s1_vs_s4_translate":
                    self.assertEqual(left["close_mosaic"], 3)
                    self.assertEqual(right["close_mosaic"], 3)
                    self.assertEqual(left["mosaic"], right["mosaic"])
                    self.assertEqual(left["translate"], 0.05)
                    self.assertEqual(right["translate"], 0.10)
                elif case_id == "s2_mosaic0_vs_s3_photometric":
                    self.assertEqual(left["mosaic"], 0.0)
                    self.assertEqual(left["close_mosaic"], 0)
                    self.assertEqual(right["mosaic"], 0.0)
                    self.assertEqual(right["close_mosaic"], 0)
                    self.assertNotEqual(
                        (
                            left["translate"],
                            left["scale"],
                            left["fliplr"],
                            left["hsv_s"],
                            left["hsv_v"],
                        ),
                        (
                            right["translate"],
                            right["scale"],
                            right["fliplr"],
                            right["hsv_s"],
                            right["hsv_v"],
                        ),
                    )
                    self.assertEqual(left["translate"], 0.05)
                    self.assertEqual(right["translate"], 0.0)
                    self.assertEqual(right["fliplr"], 0.0)
                self.assertEqual(left["epochs"], SMOKE_EPOCHS_RANK)

    def test_apply_close_mosaic_epoch_scale(self) -> None:
        from harchoc.bench_config import _bench_to_train_config, load_bench_config
        from harchoc.train_config import (
            SMOKE_EPOCHS_RANK,
            apply_close_mosaic_epoch_scale,
            effective_train_aug_merged,
        )

        repo_root = Path(__file__).resolve().parents[1]
        scaled = apply_close_mosaic_epoch_scale(
            {"epochs": SMOKE_EPOCHS_RANK, "patience": 12, "close_mosaic": 15}
        )
        self.assertEqual(scaled["close_mosaic"], 3)
        self.assertEqual(
            apply_close_mosaic_epoch_scale(
                {"epochs": SMOKE_EPOCHS_RANK, "patience": 12, "close_mosaic": 10}
            )["close_mosaic"],
            2,
        )
        self.assertEqual(
            apply_close_mosaic_epoch_scale(
                {"epochs": SMOKE_EPOCHS_RANK, "patience": 12, "close_mosaic": 3}
            )["close_mosaic"],
            3,
        )
        bench = _bench_to_train_config(
            load_bench_config(repo_root / "configs/bench/yolov8m_default.yaml"),
            weights_path="yolov8m.pt",
        )["train"]
        self.assertEqual(
            effective_train_aug_merged(bench, repo_root=repo_root)["close_mosaic"],
            15,
        )

    def test_15ep_smoke_configs_satisfy_close_mosaic_guard(self) -> None:
        import io
        from contextlib import redirect_stderr

        from harchoc.train_config import (
            SMOKE_EPOCHS_RANK,
            load_train_config_json,
            validate_epochs_patience_close_mosaic,
            warn_train_schedule_close_mosaic,
        )

        repo_root = Path(__file__).resolve().parents[1]
        exp_dir = repo_root / "configs" / "experiments"
        smoke_paths = [
            "train_amp_on_15ep_smoke.json",
            "train_sg_yolo_nas_s_smoke_15ep.json",
            "train_aug_mosaic_sweep_smoke_15ep.json",
            "train_rtdetr_smoke_15ep.json",
            "train_rtdetr_queries_smoke_15ep.json",
        ]
        for name in smoke_paths:
            path = exp_dir / name
            cfg = load_train_config_json(path, repo_root=repo_root)
            validate_epochs_patience_close_mosaic(
                cfg, repo_root=repo_root, label=name
            )
        for path in sorted(exp_dir.glob("train_aug_s*_smoke.json")):
            cfg = load_train_config_json(path, repo_root=repo_root)
            self.assertEqual(cfg["epochs"], SMOKE_EPOCHS_RANK, msg=path.name)
            validate_epochs_patience_close_mosaic(
                cfg, repo_root=repo_root, label=path.name
            )
            buf = io.StringIO()
            with redirect_stderr(buf):
                warn_train_schedule_close_mosaic(
                    cfg, repo_root=repo_root, label=path.name
                )
            self.assertEqual(buf.getvalue(), "", msg=path.name)

    def test_warn_train_schedule_raises_on_long_misconfig(self) -> None:
        from harchoc.train_config import warn_train_schedule_close_mosaic

        repo_root = Path(__file__).resolve().parents[1]
        cfg = {
            "epochs": 50,
            "patience": 45,
            "close_mosaic": 15,
        }
        with self.assertRaises(SystemExit):
            warn_train_schedule_close_mosaic(cfg, repo_root=repo_root, label="unit")

    def test_scale_close_mosaic_for_epochs(self) -> None:
        from harchoc.train_config import (
            CLOSE_MOSAIC_PRODUCTION_DEFAULT,
            CLOSE_MOSAIC_PRODUCTION_EPOCHS,
            SMOKE_EPOCHS_MICRO,
            SMOKE_EPOCHS_RANK,
            scale_close_mosaic_for_epochs,
        )

        self.assertEqual(scale_close_mosaic_for_epochs(SMOKE_EPOCHS_RANK), 3)
        self.assertEqual(
            scale_close_mosaic_for_epochs(SMOKE_EPOCHS_MICRO),
            1,
        )
        self.assertEqual(
            scale_close_mosaic_for_epochs(CLOSE_MOSAIC_PRODUCTION_EPOCHS),
            CLOSE_MOSAIC_PRODUCTION_DEFAULT,
        )
        self.assertEqual(scale_close_mosaic_for_epochs(100, production_close_mosaic=10), 10)

    def test_robustness_smoke_close10_distinct_from_close3(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import (
            effective_train_aug_merged,
            load_train_config_json,
            validate_epochs_patience_close_mosaic,
        )

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")
        close3 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S1"), repo_root=repo_root)
        arm = next(
            a
            for a in (index.get("sweeps_15ep") or {}).get("arms") or []
            if str(a.get("id")) == "close10"
        )
        close10 = resolve_aug_smoke_train_raw(arm, repo_root=repo_root)
        validate_epochs_patience_close_mosaic(close10, repo_root=repo_root)
        m3 = effective_train_aug_merged(close3, repo_root=repo_root)
        m10 = effective_train_aug_merged(close10, repo_root=repo_root)
        self.assertEqual(m3["close_mosaic"], 3)
        self.assertEqual(m10["close_mosaic"], 2)

    def test_robustness_smoke_close25_distinct_from_close3(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import (
            effective_train_aug_merged,
            load_train_config_json,
            validate_epochs_patience_close_mosaic,
        )

        repo_root = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo_root / "configs/experiments/aug_smoke_index.json")
        close3 = resolve_aug_smoke_train_raw(find_smoke_entry(index, "S1"), repo_root=repo_root)
        arm = next(
            a
            for a in (index.get("sweeps_15ep") or {}).get("arms") or []
            if str(a.get("id")) == "close25"
        )
        close25 = resolve_aug_smoke_train_raw(arm, repo_root=repo_root)
        validate_epochs_patience_close_mosaic(close25, repo_root=repo_root)
        m3 = effective_train_aug_merged(close3, repo_root=repo_root)
        m25 = effective_train_aug_merged(close25, repo_root=repo_root)
        self.assertEqual(m3["close_mosaic"], 3)
        self.assertEqual(m25["close_mosaic"], 4)
        self.assertEqual(m25["translate"], m3["translate"])

    def test_robustness_smoke_extends_base_merge(self) -> None:
        from harchoc.aug_config import merge_aug_yaml, resolve_aug_yaml

        repo_root = Path(__file__).resolve().parents[1]
        base_path = repo_root / "configs/aug/robustness_smoke_base.yaml"
        base_ultra = resolve_aug_yaml(base_path, repo_root=repo_root).get("ultralytics") or {}
        self.assertEqual(base_ultra.get("fliplr"), 0.5)
        self.assertEqual(base_ultra.get("erasing"), 0.2)

        aug_dir = repo_root / "configs" / "aug"
        for path in sorted(aug_dir.glob("robustness_smoke_*.yaml")):
            if path.name == "robustness_smoke_base.yaml":
                continue
            raw = path.read_text(encoding="utf-8")
            self.assertIn("extends:", raw, msg=f"{path.name} must extend smoke base")
            with self.subTest(yaml=path.name):
                obj = resolve_aug_yaml(path, repo_root=repo_root)
                ultra = obj.get("ultralytics")
                self.assertIsInstance(ultra, dict)
                assert isinstance(ultra, dict)
                self.assertEqual(ultra.get("fliplr"), base_ultra.get("fliplr"))
                self.assertEqual(ultra.get("erasing"), base_ultra.get("erasing"))
                merged = merge_aug_yaml({}, path, repo_root=repo_root)
                self.assertEqual(merged["fliplr"], 0.5)
                if path.name == "robustness_smoke_close3.yaml":
                    self.assertEqual(merged["close_mosaic"], 3)
                if path.name == "robustness_smoke_mosaic01.yaml":
                    self.assertEqual(merged["translate"], 0.10)
                if path.name == "robustness_smoke_close10.yaml":
                    self.assertEqual(merged["close_mosaic"], 2)

    def test_mosaic_sweep_template_close_mosaic_guard(self) -> None:
        from harchoc.train_config import (
            CLOSE_MOSAIC_SWEEP_100EP,
            MOSAIC_SWEEP_VALUES,
            load_train_config_json,
            validate_epochs_patience_close_mosaic,
        )

        repo_root = Path(__file__).resolve().parents[1]
        from harchoc.aug_smoke_train import TRAIN_AUG_MOSAIC_SWEEP_TEMPLATE

        template = load_train_config_json(
            repo_root / TRAIN_AUG_MOSAIC_SWEEP_TEMPLATE,
            repo_root=repo_root,
        )
        self.assertEqual(template["epochs"], 100)
        self.assertEqual(template["patience"], 30)
        self.assertEqual(MOSAIC_SWEEP_VALUES, (0.0, 0.1, 0.3))
        self.assertEqual(CLOSE_MOSAIC_SWEEP_100EP, (10, 15, 25))
        for close_mosaic in CLOSE_MOSAIC_SWEEP_100EP:
            cfg = dict(template)
            cfg["close_mosaic"] = close_mosaic
            validate_epochs_patience_close_mosaic(
                cfg, repo_root=repo_root, label=f"template close_mosaic={close_mosaic}"
            )

    def test_aug_winner_100ep_config(self) -> None:
        from harchoc.train_config import load_train_config_json, validate_epochs_patience_close_mosaic

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_train_config_json(
            repo_root / "configs/experiments/train_aug_winner_100ep.json",
            repo_root=repo_root,
        )
        self.assertEqual(cfg["epochs"], 100)
        self.assertEqual(cfg["patience"], 30)
        self.assertEqual(cfg["max_det"], 3000)
        self.assertEqual(cfg["eval"]["max_det"], 3000)
        self.assertEqual(cfg["aug_config"], "configs/aug/robustness_minimal.yaml")
        validate_epochs_patience_close_mosaic(
            cfg, repo_root=repo_root, label="train_aug_winner_100ep"
        )


class AugSmokeConfigValidateTests(unittest.TestCase):
    def test_validate_aug_smoke_configs_clean_on_production_index(self) -> None:
        from harchoc.aug_smoke_train import validate_aug_smoke_configs

        repo_root = Path(__file__).resolve().parents[1]
        errors = validate_aug_smoke_configs(repo_root)
        self.assertEqual(errors, [], msg="\n".join(errors))

    def test_validate_aug_smoke_configs_catches_runtime_smoke_train_config(self) -> None:
        import json
        import tempfile

        from harchoc.aug_smoke_train import validate_aug_smoke_configs

        repo_root = Path(__file__).resolve().parents[1]
        prod = json.loads(
            (repo_root / "configs/experiments/aug_smoke_index.json").read_text(encoding="utf-8")
        )
        prod = dict(prod)
        prod["smokes"] = [
            dict(
                {**entry, "train_config": "configs/experiments/train_smoke_rank_15ep.json"}
                if str(entry.get("id")).upper() == "S0"
                else dict(entry)
            )
            for entry in prod.get("smokes") or []
        ]
        with tempfile.TemporaryDirectory(dir=repo_root / "tests") as td:
            bad = Path(td) / "bad_index.json"
            bad.write_text(json.dumps(prod), encoding="utf-8")
            rel = str(bad.relative_to(repo_root))
            errors = validate_aug_smoke_configs(repo_root, index_path=rel)
        self.assertTrue(any("S0" in e and "train_config" in e for e in errors))


class TrainConfigIsoConfigAuditTests(unittest.TestCase):
    """Iso-config audit: bench overlays may differ only on documented parity keys."""

    def test_bench_parity_allowed_keys_are_documented(self) -> None:
        from harchoc.train_config import (
            BENCH_EVAL_PARITY_ALLOWED_DIFF_KEYS,
            BENCH_PARITY_ALLOWED_DIFF_KEYS,
        )

        self.assertEqual(
            BENCH_PARITY_ALLOWED_DIFF_KEYS,
            frozenset({"model", "batch", "notes", "aug_config", "amp", "grad_clip"}),
        )
        self.assertEqual(
            BENCH_EVAL_PARITY_ALLOWED_DIFF_KEYS,
            frozenset({"notes", "device", "max_det"}),
        )


if __name__ == "__main__":
    unittest.main()
