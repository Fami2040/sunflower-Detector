from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harchoc.yaml_minimal import (
    parse_minimal_yaml,
    parse_minimal_yaml_flat,
    parse_names_and_nc,
    safe_load,
)


def _legacy_bench_parse_minimal_yaml(path: Path) -> dict[str, object]:
    def _parse_scalar(raw: str) -> str | int:
        s = raw.strip()
        if s and ((s[0] == s[-1]) and s[0] in ("'", '"')):
            s = s[1:-1]
        if s.isdigit():
            try:
                return int(s)
            except ValueError:
                return s
        return s

    def _is_mapping_value(v: object) -> bool:
        return isinstance(v, dict)

    lines = path.read_text("utf-8").splitlines()
    out: dict[str, object] = {}
    current_map_key: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith(("  ", "\t")):
            if current_map_key is None:
                i += 1
                continue
            inner = line.lstrip()
            if ":" not in inner:
                i += 1
                continue
            k, rest = inner.split(":", 1)
            k = k.strip()
            rest = rest.strip()
            m = out.get(current_map_key)
            if not _is_mapping_value(m):
                m = {}
                out[current_map_key] = m
            assert isinstance(m, dict)
            m[k] = _parse_scalar(rest) if rest else ""
            i += 1
            continue
        if line.startswith((" ", "\t")):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        current_map_key = None
        if rest in (">", "|"):
            i += 1
            block: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith(("  ", "\t")):
                    block.append(nxt.lstrip())
                    i += 1
                elif not nxt.strip():
                    block.append("")
                    i += 1
                else:
                    break
            out[key] = (
                " ".join([b for b in block if b != ""]).strip()
                if rest == ">"
                else "\n".join(block).rstrip()
            )
            continue
        if rest == "":
            out[key] = {}
            current_map_key = key
            i += 1
            continue
        out[key] = _parse_scalar(rest)
        i += 1
    return out


def _legacy_flat_parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("-", "[")):
            continue
        if ":" not in line:
            continue
        k, rest = line.split(":", 1)
        k = k.strip()
        v = rest.strip().strip("'\"")
        if v:
            out[k] = v
    return out


def _legacy_names_and_nc(path: Path) -> tuple[dict[int, str] | None, int | None]:
    nc: int | None = None
    names: dict[int, str] | None = None
    current_map_key: str | None = None
    for raw in path.read_text("utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(("  ", "\t")):
            if current_map_key != "names":
                continue
            inner = line.lstrip()
            if ":" not in inner:
                continue
            k, v = inner.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            try:
                ki = int(k)
            except ValueError:
                continue
            if names is None:
                names = {}
            names[ki] = v
            continue
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        current_map_key = None
        if key == "nc":
            try:
                nc = int(rest)
            except ValueError:
                nc = None
        elif key == "names" and rest == "":
            current_map_key = "names"
            if names is None:
                names = {}
    return names, nc


class YamlMinimalParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]

    def test_safe_load_alias(self) -> None:
        p = self.repo / "configs" / "bench" / "yolov8n_default.yaml"
        self.assertEqual(safe_load(p), parse_minimal_yaml(p))

    def test_bench_yaml_load_bench_config_unchanged(self) -> None:
        from harchoc.bench_config import _resolve_bench_includes, is_bench_row_config
        from harchoc.yaml_minimal import parse_minimal_yaml
        from scripts.benchmark_matrix import load_bench_config

        bench_dir = self.repo / "configs" / "bench"
        for p in sorted(bench_dir.glob("*.yaml")):
            with self.subTest(path=p.name):
                if not is_bench_row_config(p):
                    continue
                legacy_obj = parse_minimal_yaml(p)
                if legacy_obj.get("include"):
                    legacy_obj = _resolve_bench_includes(legacy_obj, p)
                cfg = load_bench_config(p)
                self.assertEqual(cfg.epochs, legacy_obj.get("epochs"))
                self.assertEqual(cfg.patience, legacy_obj.get("patience"))
                self.assertEqual(cfg.seed, legacy_obj.get("seed"))
                self.assertEqual(cfg.model, legacy_obj.get("model"))
                self.assertEqual(cfg.model_id, legacy_obj.get("model_id"))
                self.assertEqual(cfg.backend, legacy_obj.get("backend"))

    def test_aug_merge_matches_legacy_parser(self) -> None:
        from harchoc.aug_config import _coerce_aug_scalar, merge_aug_yaml

        p = self.repo / "configs" / "aug" / "robustness_minimal.yaml"
        legacy = _legacy_bench_parse_minimal_yaml(p)
        merged = merge_aug_yaml({}, p)
        ultra = legacy.get("ultralytics")
        self.assertIsInstance(ultra, dict)
        assert isinstance(ultra, dict)
        for k, v in ultra.items():
            if v is not None and v != "":
                self.assertEqual(merged[k], _coerce_aug_scalar(v), msg=k)

    def test_flat_data_yaml_matches_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.yaml"
            p.write_text(
                "\n".join(
                    [
                        "# comment",
                        "path: /data/root",
                        'train: "images/train"',
                        "val: images/val",
                        "nc: 2",
                        "names:",
                        "  0: a",
                        "- ignored",
                        "",
                    ]
                ),
                "utf-8",
            )
            self.assertEqual(parse_minimal_yaml_flat(p), _legacy_flat_parse(p))

    def test_names_and_nc_matches_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.yaml"
            p.write_text(
                "\n".join(
                    [
                        "nc: 2",
                        "names:",
                        "  0: developed",
                        "  1: aborted",
                        "",
                    ]
                ),
                "utf-8",
            )
            self.assertEqual(parse_names_and_nc(p), _legacy_names_and_nc(p))

    def test_block_scalar_folded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.yaml"
            p.write_text("notes: >\n  line one\n  line two\n", "utf-8")
            obj = parse_minimal_yaml(p)
            self.assertEqual(obj["notes"], "line one line two")
