"""Storage-neutral ROS bag reads; all new bags use indexed MCAP."""

import os
from pathlib import Path


def configure_storage():
    # Optional user-owned plugin overlay, for robots without sudo access.
    prefix = Path(
        os.environ.get(
            "NAVIGATION_MCAP_PREFIX", str(Path.home() / ".local/opt/rosbag2_mcap")
        )
    )
    if prefix.is_dir():
        for key, value in [
            ("AMENT_PREFIX_PATH", str(prefix)),
            ("LD_LIBRARY_PATH", str(prefix / "lib")),
        ]:
            parts = os.environ.get(key, "").split(":")
            if value not in parts:
                os.environ[key] = ":".join([value, *filter(None, parts)])
        # dlopen reads LD_LIBRARY_PATH at process startup; preload the overlay
        # dependency when a Python entry point configures it after startup.
        vendor = prefix / "lib/libmcap.so"
        if vendor.exists():
            import ctypes

            ctypes.CDLL(str(vendor), mode=ctypes.RTLD_GLOBAL)


def reader(bag, topics=None):
    configure_storage()
    import rosbag2_py
    import yaml

    bag = Path(bag)
    if bag.is_dir():
        storage = yaml.safe_load((bag / "metadata.yaml").read_text())[
            "rosbag2_bagfile_information"
        ]["storage_identifier"]
    else:
        storage = "mcap" if bag.suffix == ".mcap" else "sqlite3"
    result = rosbag2_py.SequentialReader()
    result.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    if topics is not None:
        result.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return result


def messages(bag, topics=None, start=None, end=None):
    source = reader(bag, topics)
    while source.has_next():
        topic, raw, stamp = source.read_next()
        if end is not None and stamp > end:
            break
        if start is None or stamp >= start:
            yield topic, raw, stamp


def require_mcap():
    configure_storage()
    import rosbag2_py

    if "mcap" not in rosbag2_py.get_registered_writers():
        raise RuntimeError(
            "MCAP storage plugin missing: install ros-humble-rosbag2-storage-mcap or set NAVIGATION_MCAP_PREFIX"
        )


def writer(bag):
    require_mcap()
    import rosbag2_py

    result = rosbag2_py.SequentialWriter()
    result.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return result
