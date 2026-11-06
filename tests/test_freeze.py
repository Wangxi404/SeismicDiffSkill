import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("freeze",Path(__file__).resolve().parents[1]/"tools/freeze.py")
freeze=importlib.util.module_from_spec(spec);spec.loader.exec_module(freeze)


def packet(tmp_path):
    p=tmp_path/"packet";p.mkdir();(p/"SKILL.md").write_text("name: example\n")
    (p/"scripts").mkdir();(p/"scripts/check.py").write_text("print('real helper')\n")
    return p


def test_freeze_verifies_and_refuses_overwrite(tmp_path):
    p=packet(tmp_path);r=tmp_path/"release"
    manifest=freeze.create(p,r,"0.1.0")
    assert freeze.verify(r,manifest["packet_sha256"])["files"]==manifest["files"]
    with pytest.raises(FileExistsError):freeze.create(p,r,"0.1.0")
    with pytest.raises(ValueError):freeze.verify(r,"0"*64)


def test_helper_tampering_is_detected(tmp_path):
    r=tmp_path/"release";freeze.create(packet(tmp_path),r,"0.1.0")
    helper=r/"skill/scripts/check.py";helper.chmod(0o644);helper.write_text("print('tampered')\n")
    with pytest.raises(ValueError):freeze.verify(r)


def test_external_symlink_is_not_a_frozen_dependency(tmp_path):
    p=packet(tmp_path);external=tmp_path/"external";external.write_text("secret")
    (p/"link").symlink_to(external)
    with pytest.raises(ValueError):freeze.inventory(p)
