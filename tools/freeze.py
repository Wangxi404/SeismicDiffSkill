#!/usr/bin/env python3
"""Create or verify a content-addressed frozen skill packet."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

IGNORED = {"__pycache__", ".pytest_cache", ".DS_Store"}


def inventory(root):
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p in IGNORED for p in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"A frozen packet must contain real files: {relative}")
        if path.is_file():
            entries.append({"path":relative.as_posix(), "bytes":path.stat().st_size,
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    if not entries or not (root/"SKILL.md").is_file():
        raise ValueError("Not a skill packet with SKILL.md")
    return entries


def digest(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create(packet, output, version):
    packet, output = Path(packet).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Freeze already exists: {output}")
    if output == packet or packet in output.parents:
        raise ValueError("Put release output outside the operative packet")
    original = inventory(packet)
    output.mkdir(parents=True)
    shutil.copytree(packet, output/"skill", ignore=shutil.ignore_patterns(*IGNORED))
    copied = inventory(output/"skill")
    if original != copied:
        raise ValueError("Packet changed while being frozen")
    manifest = {"schema_version":1, "skill_name":packet.name, "version":version,
                "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "packet_sha256":digest(copied), "files":copied,
                "generated_paths_excluded":sorted(IGNORED)}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with tarfile.open(output/"skill.tar", "w", format=tarfile.PAX_FORMAT) as tar:
        for entry in copied:
            path=output/"skill"/entry["path"]
            info=tar.gettarinfo(str(path), arcname=packet.name+"/"+entry["path"])
            info.uid=info.gid=0; info.uname=info.gname=""; info.mtime=0; info.mode=0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    for path in sorted((output/"skill").rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    (output/"skill").chmod(0o555)
    (output/"manifest.json").chmod(0o444)
    (output/"skill.tar").chmod(0o444)
    return manifest


def verify(release, expected_sha256=None):
    release = Path(release)
    manifest=json.loads((release/"manifest.json").read_text())
    actual=inventory(release/"skill")
    if actual != manifest["files"] or digest(actual) != manifest["packet_sha256"]:
        raise ValueError("Frozen skill contents differ from the manifest")
    if expected_sha256 and manifest["packet_sha256"] != expected_sha256:
        raise ValueError("Manifest differs from the independently recorded packet hash")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command", required=True)
    c=sub.add_parser("create");c.add_argument("--packet",type=Path,required=True);c.add_argument("--output",type=Path,required=True);c.add_argument("--version",required=True)
    v=sub.add_parser("verify");v.add_argument("release",type=Path);v.add_argument("--expected-sha256")
    args=p.parse_args()
    result=create(args.packet,args.output,args.version) if args.command=="create" else verify(args.release,args.expected_sha256)
    print(json.dumps({"version":result["version"],"packet_sha256":result["packet_sha256"],"files":len(result["files"])}))


if __name__=="__main__":
    main()
