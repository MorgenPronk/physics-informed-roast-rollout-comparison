# -*- coding: utf-8 -*-
"""Build the sanitized public release tree for the SciRep resubmission.

Allowlist snapshot of the working repo with:
  * no cohort CSV, no raw/interim data
  * roast IDs neutralized everywhere (deterministic map, P-label preserved)
  * measured 'actual' traces stripped from all JSON artifacts except the
    single representative roast shown in the manuscript figure
  * old manufacturer-adapted schematic excluded
Private ID map written back to the working repo (data/metadata/, gitignored dir).
"""
import json
import re
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\pronk\codex_projects\thesis_to_paper")
REL = SRC / "artifacts" / "release_staging"

ID_RE = re.compile(r"ROAST_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}_(P\d+)")

ROOT_FILES = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "requirements.txt", ".gitignore"]
TREES = ["src", "scripts"]
MANUSCRIPT_DIR = "manuscript/scientific_reports/submission_latex"
MANUSCRIPT_KEEP_SUFFIX = {".tex", ".bib", ".cls", ".sty", ".ldf", ".bst", ".ps1", ".pdf"}
FIGURE_EXCLUDE = {"roaster_schematic_2.png"}
REPORT_DIRS = ["reports/manuscript_hpo", "reports/r1_seed_campaign", "reports/synthetic_study"]
REPORT_EXCLUDE_DIRS = {"checkpoints"}
# Old assets with no builder script and embedded roast IDs: exclude rather than regenerate.
REPORT_EXCLUDE_FILES = {
    "main_cohort_per_roast_rollout_r2_hpo.png", "main_cohort_per_roast_rollout_r2_hpo.pdf",
    "main_cohort_representative_rollouts_hpo.png", "main_cohort_representative_rollouts_hpo.pdf",
}
DOCS = ["docs/training_stability_notes.md"]
TEXT_SUFFIX = {".json", ".jsonl", ".md", ".txt", ".csv", ".cff", ".py", ".tex"}


def copy_tree(src: Path, dst: Path, exclude_dirs=(), exclude_files=()):
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        relpath = p.relative_to(src)
        if any(part in exclude_dirs for part in relpath.parts):
            continue
        if relpath.name in exclude_files:
            continue
        if any(part == "__pycache__" for part in relpath.parts):
            continue
        target = dst / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def main():
    if REL.exists():
        shutil.rmtree(REL)
    REL.mkdir(parents=True)

    for f in ROOT_FILES:
        if (SRC / f).exists():
            shutil.copy2(SRC / f, REL / f)
    for t in TREES:
        copy_tree(SRC / t, REL / t)
    for d in DOCS:
        (REL / d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / d, REL / d)

    msrc = SRC / MANUSCRIPT_DIR
    for p in msrc.rglob("*"):
        if p.is_dir():
            continue
        relpath = p.relative_to(msrc)
        if p.name in FIGURE_EXCLUDE:
            continue
        if relpath.parts[0] == "figures":
            pass  # keep all figures except excluded
        elif p.suffix.lower() not in MANUSCRIPT_KEEP_SUFFIX:
            continue  # drop aux/log/fls/out build junk
        target = REL / MANUSCRIPT_DIR / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    for d in REPORT_DIRS:
        copy_tree(SRC / d, REL / d, exclude_dirs=REPORT_EXCLUDE_DIRS,
                  exclude_files=REPORT_EXCLUDE_FILES)

    # Data placeholders + the synthetic benchmark cohort.
    for sub in ["raw", "interim", "processed", "metadata"]:
        (REL / "data" / sub).mkdir(parents=True, exist_ok=True)
        (REL / "data" / sub / ".gitkeep").write_text("")
    if (SRC / "data/raw/README.md").exists():
        shutil.copy2(SRC / "data/raw/README.md", REL / "data/raw/README.md")
    copy_tree(SRC / "data/synthetic", REL / "data/synthetic")

    # ---- Build the neutral ID map over the whole release tree ----
    all_ids = set()
    text_files = [p for p in REL.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIX]
    for p in text_files:
        try:
            all_ids.update(m.group(0) for m in ID_RE.finditer(p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass

    # The canonical map is the band-preserving 221-roast map written by
    # scripts/build_deidentified_cohort.py; run that first. Using one shared
    # map keeps released artifact keys consistent with the de-identified
    # cohort CSV, should it be published.
    map_path = SRC / "data" / "metadata" / "roast_id_neutralization_map_PRIVATE.json"
    if not map_path.exists():
        raise SystemExit("run scripts/build_deidentified_cohort.py first (map missing)")
    id_map = json.loads(map_path.read_text(encoding="utf-8"))
    unmapped = sorted(all_ids - set(id_map))
    if unmapped:
        raise SystemExit(f"IDs in artifacts missing from map: {unmapped[:5]}")
    print(f"neutralizing {len(all_ids)} roast IDs via shared band-preserving map")

    def neutralize(text: str) -> str:
        return ID_RE.sub(lambda m: id_map[m.group(0)], text)

    for p in text_files:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        t2 = neutralize(t)
        if t2 != t:
            p.write_text(t2, encoding="utf-8")

    # ---- Strip measured traces ----
    def strip_actuals(obj, keep_key=None):
        removed = 0
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k == "actual" and isinstance(obj[k], list) and keep_key is not True:
                    del obj[k]
                    removed += 1
                else:
                    removed += strip_actuals(obj[k])
        elif isinstance(obj, list):
            for v in obj:
                removed += strip_actuals(v)
        return removed

    # seed11_rollouts.json: keep 'actual' only for the representative roast
    # (deterministic-longest, same rule as the figure builder).
    sr = REL / "reports/manuscript_hpo/seed11_rollouts.json"
    d = json.loads(sr.read_text(encoding="utf-8"))
    mech = d["models"]["mechanistic"]
    rep = max(mech, key=lambda r: len(mech[r]["actual"]))
    n_kept = 0
    for model, roasts in d["models"].items():
        for rid in list(roasts):
            if rid == rep:
                n_kept += 1
                continue
            roasts[rid].pop("actual", None)
    d["representative_roast_with_measured_trace"] = rep
    d["note"] = ("Measured 'actual' traces are retained only for the representative "
                 "roast shown in the manuscript figure; the full telemetry cohort is "
                 "available upon reasonable request (see Data Availability).")
    sr.write_text(json.dumps(d), encoding="utf-8")
    print(f"seed11_rollouts: representative roast = {rep}, kept {n_kept} traces")

    # All other JSONs: remove every 'actual' array. Synthetic artifacts are
    # exempt: their traces are simulated, not partner telemetry.
    for p in REL.rglob("*.json"):
        if p == sr:
            continue
        rel = p.relative_to(REL).as_posix()
        if rel.startswith(("reports/synthetic_study", "data/synthetic")):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n = strip_actuals(obj)
        if n:
            p.write_text(json.dumps(obj), encoding="utf-8")
            print(f"stripped {n} 'actual' arrays from {p.relative_to(REL)}")

    # Verify: no raw IDs, no stray 'actual' outside the representative roast file.
    leftovers = []
    for p in REL.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIX:
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        if ID_RE.search(t):
            leftovers.append(str(p.relative_to(REL)))
    print("files with raw IDs remaining:", leftovers or "NONE")


if __name__ == "__main__":
    main()
