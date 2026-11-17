"""Loading the large dataset's layout: subfolders, samples, several folders.

Run with:  python -m pytest tests -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_data import load_corpus, load_patents  # noqa: E402


def record(number):
    return {"doc_number": str(number), "title": f"WHEEL {number}", "abstract": "A wheel.",
            "classification": "A61B5/097", "filename": f"US{number}A1.XML",
            "claims": ["1 . A wheel comprising a rim."],
            "detailed_description": ["The wheel has a rim and a hub joined by spokes."]}


def write_dataset(root, files, per_file, start=1):
    number = start
    for f in range(files):
        folder = root / f"job_{f // 2:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        records = [record(number + i) for i in range(per_file)]
        number += per_file
        (folder / f"patents_batch_{f:04d}.json").write_text(json.dumps(records))
    return number - start


def test_files_in_subfolders_are_found(tmp_path):
    total = write_dataset(tmp_path, files=4, per_file=5)
    patents, report = load_patents(str(tmp_path))
    assert len(patents) == total == 20 and report["files"] == 4


def test_a_limit_samples_across_the_whole_dataset(tmp_path):
    write_dataset(tmp_path, files=10, per_file=100)
    patents, report = load_patents(str(tmp_path), limit=120)
    assert len(patents) == 120
    assert report["files"] == 3                          # about 50 per file, spread out
    numbers = sorted(int(p.patent_id) for p in patents)
    assert numbers[0] <= 100 and numbers[-1] > 900       # first and last file both used
    again, _ = load_patents(str(tmp_path), limit=120)
    assert [p.patent_id for p in again] == [p.patent_id for p in patents]   # repeatable


def test_several_folders_with_a_sample_size_and_duplicates(tmp_path):
    small, large = tmp_path / "small", tmp_path / "large"
    write_dataset(small, files=1, per_file=3)                 # patents 1-3
    write_dataset(large, files=4, per_file=50, start=2)       # patents 2-201, overlaps 2-3
    patents, _, report = load_corpus([str(small), f"{large}:60"])
    ids = [p.patent_id for p in patents]
    assert len(ids) == len(set(ids))                          # each patent once
    assert {"1", "2", "3"} <= set(ids) and len(ids) <= 63
    assert patents[0].url == "https://patents.google.com/patent/US1A1"


def test_a_classification_written_as_na_counts_as_missing(tmp_path):
    from src.load_data import parse_record
    patent = parse_record(dict(record(7), classification="N/A"), "")
    assert patent.classification == ""


def test_other_json_files_are_ignored_and_bad_batches_reported(tmp_path, capsys):
    write_dataset(tmp_path, files=2, per_file=5)
    (tmp_path / "sample.json").write_text(json.dumps({"title": "an example, not a batch"}))
    (tmp_path / "job_000" / "patents_batch_0099.json").write_text(json.dumps({"oops": 1}))
    patents, report = load_patents(str(tmp_path))
    assert len(patents) == 10                       # sample.json is not read at all
    assert report["files"] == 3 and report["files_not_a_list"] == 1
    assert "patents_batch_0099.json is not a list" in capsys.readouterr().out
