#!/usr/bin/env python3
"""Copy thesis-ready figures from a saved report package manifest."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy thesis-ready uncertainty figures from a report package")
    p.add_argument("--report-manifest", required=True, help="Path to evaluation/report_packages/.../manifest.json")
    p.add_argument("--thesis-dir", default="thesis", help="Thesis directory root")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _resolve_artifact(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    report_manifest = Path(args.report_manifest).expanduser().resolve()
    thesis_dir = Path(args.thesis_dir).expanduser().resolve()
    fig_dir = thesis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    package = _load_json(report_manifest)
    copied: dict[str, str] = {}

    for key, rel_path in sorted((package.get("summary_figures") or {}).items()):
        src = _resolve_artifact(rel_path)
        copied[key] = _copy_if_exists(src, fig_dir / src.name) or ""

    for method, payload in sorted((package.get("method_reports") or {}).items()):
        for fig_key, rel_path in sorted((payload.get("figures") or {}).items()):
            src = _resolve_artifact(rel_path)
            copied[f"{method}_{fig_key}"] = _copy_if_exists(src, fig_dir / src.name) or ""

    manifest = {
        "report_manifest": str(report_manifest),
        "copied_figures": copied,
    }
    with open(fig_dir / "thesis_figures_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
