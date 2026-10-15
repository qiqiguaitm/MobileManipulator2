"""Display messages written at acquisition time, without publishing to hardware."""

import copy
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker


class RecordedDisplay:
    def __init__(self, base_frame="base_link", path_topic="/path"):
        self.path_topic = path_topic
        self.base_frame = base_frame
        self.path = Path()

    def sample(self, pose):
        self.path.header = copy.deepcopy(pose.header)
        self.path.poses.append(copy.deepcopy(pose))
        box = Marker()
        box.header = copy.deepcopy(pose.header)
        box.header.frame_id = self.base_frame
        box.ns = "navigation_footprint_box"
        box.id = 0
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.frame_locked = True
        box.pose.orientation.w = 1.0
        box.pose.position.x = (0.434 - 0.375) / 2
        box.pose.position.z = 0.125
        # Navigation footprint; height is a display extrusion, not arm geometry.
        box.scale.x, box.scale.y, box.scale.z = 0.809, 0.62, 0.25
        box.color.r, box.color.g, box.color.b, box.color.a = 1.0, 0.55, 0.12, 0.4
        return [(self.path_topic, self.path), ("/robot_box", box)]


def detected_target_messages(xyz_map, stamp):
    """Detected object position, without inventing an object orientation."""
    import math

    if len(xyz_map) != 3 or not all(math.isfinite(v) for v in xyz_map):
        raise ValueError("Detected target must have three finite map coordinates")
    point = PointStamped()
    point.header.frame_id = "map"
    point.header.stamp = copy.deepcopy(stamp)
    point.point.x, point.point.y, point.point.z = map(float, xyz_map)
    marker = Marker()
    marker.header = copy.deepcopy(point.header)
    marker.ns = "detected_target"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = copy.deepcopy(point.point)
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 0.18
    marker.color.r, marker.color.b, marker.color.a = 1.0, 1.0, 1.0
    return [("/detected_target/position", point), ("/detected_target", marker)]
