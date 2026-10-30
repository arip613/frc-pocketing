import json
import sys
import zipfile
from pathlib import Path

import cadquery as cq

from frcpocket import extract, ingestManual
from tests.test_extract import knownPlate


def runIngest(monkeypatch, inbox, outDir, logPath):
    monkeypatch.setattr(sys, "argv", ["ingestManual", "--inbox", str(inbox), "--out", str(outDir), "--log", str(logPath)])
    ingestManual.main()


def test_ingest_step_and_zip(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    stepPath = inbox / "1690 gearbox plate.step"
    cq.exporters.export(knownPlate().val().scale(25.4), str(stepPath))
    with zipfile.ZipFile(inbox / "254 plates.zip", "w") as archive:
        archive.write(stepPath, "nested/folder/side plate.step")
        archive.writestr("../../escape.step", "not cad")  # must not land outside the target folder
        archive.writestr("readme.txt", "ignored")
    outDir = tmp_path / "manual"
    logPath = tmp_path / "sources.jsonl"

    runIngest(monkeypatch, inbox, outDir, logPath)
    entries = [json.loads(line) for line in logPath.read_text().splitlines()]
    assert len(entries) == 3  # one STEP + two CAD members of the zip
    assert {entry["team"] for entry in entries} == {1690, 254}
    assert all(outDir in Path(entry["localPath"]).parents for entry in entries)
    assert not (tmp_path / "escape.step").exists()

    runIngest(monkeypatch, inbox, outDir, logPath)  # second run is a no-op
    assert len(logPath.read_text().splitlines()) == 3

    plateRecords = extract.platesFromStep(Path(entries[0]["localPath"]))
    assert plateRecords[0]["stats"]["pocketCount"] == 2
