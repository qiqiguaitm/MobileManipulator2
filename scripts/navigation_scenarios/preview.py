#!/usr/bin/env python3
"""Create a visualization-only MCAP, retaining three camera samples per stream."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from bag_io import messages, reader, writer

IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}


def camera_group(topic):
    for part in (
        "/aligned_depth_to_color/",
        "/color/",
        "/rgb/",
        "/depth/",
        "/infra1/",
        "/infra2/",
    ):
        if part in topic:
            return topic.split(part, 1)[0]
    return topic.rsplit("/", 1)[0]


def select_frames(streams, samples=3, tolerance_ns=50_000_000):
    """Select start/middle/end camera stamps, matching sibling streams by stamp."""
    selected = {topic: set() for topic in streams}
    groups = defaultdict(list)
    for topic, stamps in streams.items():
        if stamps:
            groups[camera_group(topic)].append(topic)
    for topics in groups.values():
        anchor = min(topics, key=lambda t: (not ("/color/" in t or "/rgb/" in t), t))
        stamps = streams[anchor]
        count = min(samples, len(stamps))
        targets = [
            stamps[0] + (stamps[-1] - stamps[0]) * i // max(1, count - 1)
            for i in range(count)
        ]
        indices = {
            min(range(len(stamps)), key=lambda i: abs(stamps[i] - t)) for t in targets
        }
        selected[anchor] = indices
        for topic in topics:
            if topic == anchor:
                continue
            for index in indices:
                candidates = streams[topic]
                match = min(
                    range(len(candidates)),
                    key=lambda i: abs(candidates[i] - stamps[index]),
                )
                if abs(candidates[match] - stamps[index]) <= tolerance_ns:
                    selected[topic].add(match)
    return selected


def fingerprint_update(digest, topic, raw, stamp):
    name = topic.encode()
    digest.update(len(name).to_bytes(8, "little") + name)
    digest.update(stamp.to_bytes(8, "little") + len(raw).to_bytes(8, "little") + raw)


def generate_preview(scene, samples=3):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    if samples < 1:
        raise ValueError("samples must be positive")
    scene = Path(scene)
    source = scene / "bag"
    output = scene / "visualization_bag"
    pending = scene / "visualization_bag_pending"
    if output.exists() or pending.exists():
        raise FileExistsError("Preview or pending output exists; refusing to overwrite")
    metadata = reader(source).get_all_topics_and_types()
    image_types = {
        m.name: get_message(m.type) for m in metadata if m.type in IMAGE_TYPES
    }
    stamps = {topic: [] for topic in image_types}
    if image_types:
        print("Preview: selecting camera frames...", flush=True)
        for topic, raw, _ in messages(source, image_types):
            msg = deserialize_message(raw, image_types[topic])
            stamps[topic].append(
                msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            )
    selected = select_frames(stamps, samples)
    sink = writer(pending)
    for meta in metadata:
        sink.create_topic(meta)
    indices, counts, original_counts = Counter(), Counter(), Counter()
    expected = hashlib.sha256()
    for topic, raw, stamp in messages(source):
        original_counts[topic] += 1
        if topic in selected:
            index = indices[topic]
            indices[topic] += 1
            if index not in selected[topic]:
                continue
        sink.write(topic, raw, stamp)
        counts[topic] += 1
        fingerprint_update(expected, topic, raw, stamp)
    del sink
    actual = hashlib.sha256()
    for topic, raw, stamp in messages(pending):
        fingerprint_update(actual, topic, raw, stamp)
    if actual.digest() != expected.digest():
        raise RuntimeError(
            "Preview message verification failed; source bag is unchanged"
        )
    pending.rename(output)
    report = {
        "purpose": "visualization_only; use bag/ for simulation",
        "source": "bag",
        "output": "visualization_bag",
        "samples_per_camera_stream": samples,
        "pairing_tolerance_seconds": 0.05,
        "source_counts": dict(original_counts),
        "preview_counts": dict(counts),
        "camera_frames": {t: len(indices) for t, indices in selected.items()},
        "selected_header_stamps_ns": {
            t: [stamps[t][i] for i in sorted(indices)]
            for t, indices in selected.items()
        },
        "preserved_message_bytes_timestamps_verified": True,
        "source_bytes": sum(p.stat().st_size for p in source.glob("*.mcap")),
        "preview_bytes": sum(p.stat().st_size for p in output.glob("*.mcap")),
    }
    (scene / "visualization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def generate_preview_safe(scene):
    try:
        generate_preview(scene)
    except Exception as exc:
        print(
            f"Preview generation failed: {exc}. Full recording remains at {scene}/bag.",
            flush=True,
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    generate_preview(args.scene, args.samples)


if __name__ == "__main__":
    main()
