import pathlib
import re

import homicsx


ROOT = pathlib.Path(__file__).resolve().parents[1]


def project_version():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_runtime_version_matches_project_metadata():
    assert homicsx.__version__ == project_version()


def test_citation_version_matches_project_metadata():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'version: "{project_version()}"' in citation


def test_source_manifest_includes_reproducibility_material():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include examples" in manifest
    assert "recursive-include tests" in manifest
    assert "recursive-include validation" in manifest
    assert "recursive-include docs" in manifest
