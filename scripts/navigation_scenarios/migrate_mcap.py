#!/usr/bin/env python3
"""Convert a finalized bag in place, verify bytes/QoS, retain its SQLite backup.
Stop readers/writers before running. No ROS topics are published.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from bag_io import reader, messages, writer


def fingerprint(bag):
    result = {}
    counts = Counter()
    for topic, raw, stamp in messages(bag):
        digest = result.setdefault(topic, hashlib.sha256())
        digest.update(stamp.to_bytes(8, "little"))
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        counts[topic] += 1
    return {
        topic: (counts[topic], digest.hexdigest()) for topic, digest in result.items()
    }


def migrate(bag):
    import yaml

    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())[
        "rosbag2_bagfile_information"
    ]
    if metadata["storage_identifier"] == "mcap":
        return {"bag": str(bag), "status": "already_mcap"}
    backup = bag.with_name(bag.name + "_sqlite3_backup")
    target = bag.with_name(bag.name + "_mcap_pending")
    if backup.exists() or target.exists():
        raise RuntimeError("Migration target/backup already exists: " + str(bag))
    source_topics = reader(bag).get_all_topics_and_types()
    output = writer(target)
    for topic in source_topics:
        output.create_topic(topic)
    for topic, raw, stamp in messages(bag):
        output.write(topic, raw, stamp)
    output = None
    expected = fingerprint(bag)
    assert fingerprint(target) == expected, "Message bytes/timestamps/count mismatch"

    def topics(path):
        return sorted(
            (t.name, t.type, t.serialization_format, t.offered_qos_profiles)
            for t in reader(path).get_all_topics_and_types()
        )

    assert topics(bag) == topics(target), "Topic/QoS mismatch"
    counts = {
        t["topic_metadata"]["name"]: t["message_count"]
        for t in metadata["topics_with_message_count"]
    }
    assert all(
        expected.get(t, (0,))[0] == n for t, n in counts.items()
    ), "Source metadata count mismatch"
    # Do not replace active bags. Original data is retained for recovery.
    bag.rename(backup)
    try:
        target.rename(bag)
    except Exception:
        backup.rename(bag)
        raise
    return {
        "bag": str(bag),
        "backup": str(backup),
        "status": "migrated",
        "messages": sum(v[0] for v in expected.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", type=Path, nargs="+")
    args = parser.parse_args()
    for bag in args.bags:
        print(json.dumps(migrate(bag)), flush=True)


if __name__ == "__main__":
    main()
