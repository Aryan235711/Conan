"""Zip everything a Kaggle session needs: code, the consolidated training set, held-out splits.

    python3 training/make_consolidated.py        # data/kaggle/sft_all.jsonl
    python3 training/make_kaggle_bundle.py       # data/kaggle/conan_kaggle.zip (+ conan_private_eval.zip)

The main zip holds no private cases. The hand-written reliability suite and
spot-check cases go into a second, optional zip: upload it only as a private
Kaggle dataset, or leave it out and run those two checks on the Mac.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "kaggle"
EVAL = ["mixed_test", "combo_test", "test_ood", "liar_test", "mliar_test", "prose_test2", "prose_mixed_test"]
PRIVATE = ["benchmarks/reliability/cases.jsonl", "data/generated/spotcheck.jsonl"]


def main() -> None:
    files: list[Path] = []
    for folder in ("detective_engine", "training"):
        files += [f for f in (ROOT / folder).rglob("*") if f.is_file() and "__pycache__" not in f.parts
                  and f.suffix in (".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml")]
    files += [ROOT / "benchmarks" / "lookup_probe.py", ROOT / "benchmarks" / "interval_probe.py",
              OUT / "sft_all.jsonl", ROOT / "docs" / "KAGGLE.md"]
    files += [ROOT / "data" / "generated" / f"{s}.jsonl" for s in EVAL]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing: {[str(m.relative_to(ROOT)) for m in missing]}")
    main_zip = OUT / "conan_kaggle.zip"
    with zipfile.ZipFile(main_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("Conan") / f.relative_to(ROOT))
    print(f"{main_zip.relative_to(ROOT)}: {len(files)} files, {main_zip.stat().st_size / 1e6:.1f} MB")
    priv = [ROOT / p for p in PRIVATE if (ROOT / p).exists()]
    if priv:
        pz = OUT / "conan_private_eval.zip"
        with zipfile.ZipFile(pz, "w", zipfile.ZIP_DEFLATED) as z:
            for f in priv:
                z.write(f, Path("Conan") / f.relative_to(ROOT))
        print(f"{pz.relative_to(ROOT)}: {len(priv)} private files (upload only to a PRIVATE dataset, or skip)")


if __name__ == "__main__":
    main()
