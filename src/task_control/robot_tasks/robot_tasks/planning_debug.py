"""Planning evidence and RViz markers. Diagnostics never command the robot."""

import json
import math
from datetime import datetime
from pathlib import Path

from geometry_msgs.msg import Point
from nav2_msgs.srv import GetCostmap
from rcl_interfaces.srv import GetParameters
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def grid_cell(grid, x, y):
    ox, oy, yaw = grid["origin"]
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - ox, y - oy
    ix = math.floor((c * dx + s * dy) / grid["resolution"])
    iy = math.floor((-s * dx + c * dy) / grid["resolution"])
    if 0 <= ix < grid["width"] and 0 <= iy < grid["height"]:
        return ix, iy
    return None


def cell_cost(grid, x, y):
    cell = grid_cell(grid, x, y) if grid else None
    return grid["data"][cell[1] * grid["width"] + cell[0]] if cell else None


def cell_world(grid, ix, iy):
    ox, oy, yaw = grid["origin"]
    r = grid["resolution"]
    x, y = (ix + 0.5) * r, (iy + 0.5) * r
    return (
        ox + math.cos(yaw) * x - math.sin(yaw) * y,
        oy + math.sin(yaw) * x + math.cos(yaw) * y,
    )


def cost_name(value):
    if value is None:
        return "OUTSIDE/UNAVAILABLE"
    if value == 255:
        return "UNKNOWN"
    if value == 254:
        return "OBSTACLE"
    if value == 253:
        return "INSCRIBED"
    return "FREE" if value == 0 else "INFLATED"


class PlanningDebug:
    def __init__(self, node):
        self.node = node
        self.record = None
        self.markers = None
        self.pub = node.create_publisher(
            MarkerArray,
            "/nc/planning_debug",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.costmap = node.create_client(
            GetCostmap, "/global_costmap/get_costmap", callback_group=node.group
        )
        self.params = node.create_client(
            GetParameters,
            "/global_costmap/global_costmap/get_parameters",
            callback_group=node.group,
        )
        self.timer = node.create_timer(1.0, self.republish, callback_group=node.group)
        self.folder = Path.home() / ".ros/robot_tasks/planning"

    def begin(self, robot, target, candidates, name, parent):
        self.path = self.folder / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
        )
        self.record = dict(
            frame=self.node.map_frame,
            name=name,
            robot=list(robot),
            target=list(target),
            status="PLANNING",
            candidates=[dict(pose=list(p), result="PENDING") for p in candidates],
            grid=None,
            footprint=[],
            diagnostic_errors=[],
        )
        try:
            if self.costmap.service_is_ready():
                msg = self.node.call(
                    self.costmap, GetCostmap.Request(), parent, 2.0
                ).map
                meta = msg.metadata
                q = meta.origin.orientation
                if msg.header.frame_id != self.node.map_frame:
                    raise ValueError("Costmap frame differs from NC map frame")
                yaw = math.atan2(
                    2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
                )
                self.record["grid"] = dict(
                    width=meta.size_x,
                    height=meta.size_y,
                    resolution=meta.resolution,
                    origin=[meta.origin.position.x, meta.origin.position.y, yaw],
                    data=list(msg.data),
                )
            else:
                self.record["diagnostic_errors"].append("Costmap service unavailable")
            if self.params.service_is_ready():
                vals = self.node.call(
                    self.params,
                    GetParameters.Request(names=["footprint", "footprint_padding"]),
                    parent,
                    2.0,
                ).values
                footprint = json.loads(vals[0].string_value)
                padding = vals[1].double_value
                self.record["footprint"] = [
                    [
                        x + math.copysign(padding, x) if x else x,
                        y + math.copysign(padding, y) if y else y,
                    ]
                    for x, y in footprint
                ]
        except Exception as error:
            self.record["diagnostic_errors"].append(str(error))
        # Diagnostics must never turn an unavailable map service into permission to move.
        self.node.check(parent)
        grid = self.record["grid"]
        self.record["start_cost"] = cell_cost(grid, *robot[:2])
        for item in self.record["candidates"]:
            item["cost"] = cell_cost(grid, *item["pose"][:2])
        self.render()

    def candidate(self, index, result, path=None):
        item = self.record["candidates"][index]
        item["result"] = result
        if path is not None:
            item["path"] = [[p.pose.position.x, p.pose.position.y] for p in path]
        self.render()

    def finish(self, status, selected=None):
        if self.record is None:
            return
        self.record["status"] = status
        if selected is not None:
            self.record["selected"] = selected
        self.render()

    def republish(self):
        if self.markers is not None:
            self.pub.publish(self.markers)

    def render(self):
        record = self.record
        output = MarkerArray()

        def marker(ns, kind, color, ident=0):
            m = Marker()
            m.header.frame_id = record["frame"]
            m.ns = ns
            m.id = ident
            m.type = kind
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = color
            output.markers.append(m)
            return m

        def point(x, y, z=0.035):
            return Point(x=float(x), y=float(y), z=float(z))

        def label(ns, xy, text, color=(1.0, 1.0, 1.0, 1.0)):
            m = marker(ns, Marker.TEXT_VIEW_FACING, color, 1)
            m.pose.position = point(*xy, 0.24)
            m.scale.z = 0.11
            m.text = text

        def pose(ns, p, color, text):
            x, y, yaw = p
            m = marker(ns, Marker.ARROW, color)
            m.points = [
                point(x, y, 0.12),
                point(x + 0.33 * math.cos(yaw), y + 0.33 * math.sin(yaw), 0.12),
            ]
            m.scale.x = 0.04
            m.scale.y = 0.09
            m.scale.z = 0.09
            if record["footprint"]:
                m = marker(ns, Marker.LINE_STRIP, color, 2)
                m.scale.x = 0.015
                pts = record["footprint"] + [record["footprint"][0]]
                m.points = [
                    point(
                        x + math.cos(yaw) * a - math.sin(yaw) * b,
                        y + math.sin(yaw) * a + math.cos(yaw) * b,
                        0.06,
                    )
                    for a, b in pts
                ]
            label(ns, (x - 0.42 * math.cos(yaw), y - 0.42 * math.sin(yaw)), text, color)

        robot = record["robot"]
        target = record["target"]
        grid = record["grid"]
        pose(
            "start",
            robot,
            (0.0, 1.0, 1.0, 1.0),
            "START cost=" + str(record["start_cost"]),
        )
        m = marker("object", Marker.SPHERE, (1.0, 0.0, 1.0, 1.0))
        m.pose.position = point(*target[:2], 0.15)
        m.scale.x = m.scale.y = m.scale.z = 0.13
        label("object", (target[0], target[1] + 0.3), "OBJECT")
        xmin = min(robot[0], target[0]) - 1.2
        xmax = max(robot[0], target[0]) + 1.2
        ymin = min(robot[1], target[1]) - 1.2
        ymax = max(robot[1], target[1]) + 1.2
        if grid:
            colors = {
                253: ("inscribed_253", (1.0, 0.55, 0.0, 0.65)),
                254: ("obstacle_254", (1.0, 0.05, 0.05, 0.75)),
                255: ("unknown_255", (0.3, 0.35, 0.6, 0.3)),
            }
            groups = {}
            for value, (ns, color) in colors.items():
                m = marker("grid/" + ns, Marker.CUBE_LIST, color)
                m.scale.x = m.scale.y = grid["resolution"]
                m.scale.z = 0.018
                groups[value] = m
            for iy in range(grid["height"]):
                for ix in range(grid["width"]):
                    value = grid["data"][iy * grid["width"] + ix]
                    if value not in groups:
                        continue
                    x, y = cell_world(grid, ix, iy)
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        groups[value].points.append(point(x, y, 0.02))
        for i, item in enumerate(record["candidates"]):
            status = item["result"]
            ns = "candidate_" + str(i + 1)
            color = (
                (0.0, 1.0, 0.2, 1.0)
                if status == "REACHABLE"
                else (
                    (1.0, 0.15, 0.15, 1.0)
                    if status != "PENDING"
                    else (1.0, 1.0, 0.0, 1.0)
                )
            )
            if record.get("selected") == i:
                color = (0.0, 1.0, 1.0, 1.0)
            pose(ns, item["pose"], color, f'C{i+1} cost={item["cost"]}\n{status}')
            if item.get("path"):
                m = marker(ns, Marker.LINE_STRIP, color, 3)
                m.scale.x = 0.025
                m.points = [point(x, y, 0.08) for x, y in item["path"]]
        label(
            "legend",
            (robot[0], ymin - 0.25),
            record["status"]
            + " | RED grid: obstacle 254 | ORANGE: inscribed 253\nCyan: start/selected | Magenta: object | Arrows: candidate heading",
        )
        # Replace the previous attempt once; cached markers then republish without deleting namespaces.
        clear = Marker()
        clear.action = Marker.DELETEALL
        self.pub.publish(MarkerArray(markers=[clear] + output.markers))
        self.markers = output
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(record, ensure_ascii=False))
