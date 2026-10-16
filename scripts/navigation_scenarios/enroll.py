#!/usr/bin/env python3
"""Enroll a user-approved capture and publish its visualization soft link."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import re

import yaml


def enroll(scene_id, manifest):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", scene_id):
        raise ValueError("Invalid scene ID")
    manifest = Path(manifest)
    config = yaml.safe_load(manifest.read_text())
    root = Path(config["data_root"])
    pending = root / "pending" / scene_id
    target = root / "scenes" / scene_id
    if pending.exists() and target.exists():
        raise ValueError("Both pending and scenes contain this ID")
    source = pending if pending.exists() else target
    scenario = json.loads((source / "scenario.json").read_text())
    verification = json.loads((source / "verification.json").read_text())
    for folder in ("bag", "visualization_bag"):
        metadata = yaml.safe_load((source / folder / "metadata.yaml").read_text())[
            "rosbag2_bagfile_information"
        ]
        files = metadata["relative_file_paths"]
        if metadata["storage_identifier"] != "mcap" or not files:
            raise ValueError(f"Missing MCAP data: {folder}")
        for filename in files:
            path = source / folder / filename
            if not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Invalid bag file: {path}")
    if len(files) != 1:
        raise ValueError("Visualization bag must contain one MCAP file")
    relative_file = Path("visualization_bag") / files[0]
    link = root / "remote_mcap" / f"{scene_id}_visualization.mcap"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() not in (
            (source / relative_file).resolve(), (target / relative_file).resolve()
        ):
            raise ValueError(f"Refusing to replace unrelated file: {link}")
    entries = config.setdefault("scenes", [])
    matches = [entry for entry in entries if entry["id"] == scene_id]
    if len(matches) > 1:
        raise ValueError("Duplicate scene IDs in manifest")
    entry = matches[0] if matches else {
        "id": scene_id,
        "name": scenario.get("task_description") or scene_id,
        "task_description": scenario.get("task_description", ""),
        "data_quality": {
            "verification_status": verification["status"],
            "scan_tf_coverage": verification.get("scan_tf_coverage"),
            "known_issues": verification.get("errors", []) + verification.get("warnings", []),
        },
    }
    if not matches:
        entries.append(entry)
    entry.update(enabled=True, inclusion_decision="accepted",
                 scene_dir=f"scenes/{scene_id}")
    entry.setdefault("included_on", date.today().isoformat())
    # Validate everything before moving data; repeated calls repair partial enrollment.
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        source.rename(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(target / relative_file, link.parent))
    temp = manifest.with_suffix(".yaml.pending")
    temp.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    temp.replace(manifest)
    return target, link


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_id", help="Scene explicitly approved by the user")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("test_set.yaml"))
    args = parser.parse_args()
    for path in enroll(args.scene_id, args.manifest):
        print(path)
