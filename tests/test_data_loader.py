from pathlib import Path

import pytest

from data_loader import resolve_dataset_path


def test_resolves_workbook_from_data_directory(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    workbook = data / "hand_hygiene.xlsx"
    workbook.write_bytes(b"test")
    assert resolve_dataset_path(tmp_path, "Hand hygiene") == workbook


def test_resolves_workbook_from_nested_project_directory(tmp_path: Path):
    project = tmp_path / "repo"
    nested_data = project / "idea_discovery_lab" / "data"
    nested_data.mkdir(parents=True)
    workbook = nested_data / "antimicrobial_stewardship.xlsx"
    workbook.write_bytes(b"test")
    assert resolve_dataset_path(project, "Antimicrobial stewardship") == workbook


def test_missing_workbook_has_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Commit the data/ folder"):
        resolve_dataset_path(tmp_path, "Hand hygiene")
