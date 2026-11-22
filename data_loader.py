from __future__ import annotations

from pathlib import Path

DATASET_FILES = {
    "Hand hygiene": "hand_hygiene.xlsx",
    "Antimicrobial stewardship": "antimicrobial_stewardship.xlsx",
}


def resolve_dataset_path(app_dir: Path, dataset_name: str) -> Path:
    """Find a bundled workbook across common Streamlit/GitHub layouts.

    The repository root should contain ``data/`` next to ``app.py``. The
    additional locations support deployments where the project folder was
    uploaded one level deeper or where workbooks were placed beside app.py.
    Matching is case-insensitive.
    """
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown built-in dataset: {dataset_name!r}")

    filename = DATASET_FILES[dataset_name]
    root = Path(app_dir).resolve()
    candidate_dirs = (
        root / "data",
        root,
        root.parent / "data",
        root.parent / "idea_discovery_lab" / "data",
    )

    for directory in candidate_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate

    # Fall back to a bounded project-tree search for case differences or
    # an accidentally nested data directory.
    target = filename.casefold()
    for directory in (root,):
        if directory.exists():
            for candidate in directory.glob(f"**/{filename}"):
                if candidate.is_file():
                    return candidate
            for candidate in directory.glob("**/*.xlsx"):
                if candidate.is_file() and candidate.name.casefold() == target:
                    return candidate

    visible = []
    for directory in candidate_dirs:
        if directory.is_dir():
            visible.extend(str(item.relative_to(root)) if item.is_relative_to(root) else str(item)
                           for item in directory.iterdir() if item.is_file())
    listing = ", ".join(sorted(set(visible))) or "no Excel files found in expected locations"
    raise FileNotFoundError(
        f"Missing bundled dataset '{filename}'. Expected it at '{root / 'data' / filename}'. "
        f"Files found in checked locations: {listing}. "
        "Commit the data/ folder and both .xlsx workbooks to the GitHub repository "
        "at the same level as app.py."
    )
