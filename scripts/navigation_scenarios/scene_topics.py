"""Resolve recorded inputs; route only explicit reference topics into simulation."""

from bag_io import reader, messages


def resolve_topics(names):
    names = set(names)
    mapping = {}
    for name in names:
        canonical = (
            name.removeprefix("/reference") if name.startswith("/reference/") else name
        )
        if canonical in ("/trajectory", "/current_pose"):
            canonical = "/current_pose"
        candidates = [canonical, "/reference" + canonical]
        if canonical == "/current_pose":
            candidates += ["/trajectory", "/reference/trajectory"]
        mapping[canonical] = next((n for n in candidates if n in names), name)
    return mapping


class SceneInput:
    def __init__(self, bag):
        self.bag = bag
        self.metadata = {t.name: t for t in reader(bag).get_all_topics_and_types()}
        self.topics = resolve_topics(self.metadata)

    def messages(self, topics, start=None, end=None):
        selected = {self.topics[t]: t for t in topics if t in self.topics}
        if not selected:
            return
        for topic, raw, stamp in messages(self.bag, selected, start=start, end=end):
            yield selected[topic], raw, stamp


# Never publish original commands/goals to active simulation topics.
REFERENCE_TOPICS = {
    "/current_pose": "geometry_msgs/msg/PoseStamped",
    "/plan": "nav_msgs/msg/Path",
    "/goal_pose": "geometry_msgs/msg/PoseStamped",
    "/cmd_vel": "geometry_msgs/msg/Twist",
    "/detected_target": "visualization_msgs/msg/Marker",
    "/detected_target/position": "geometry_msgs/msg/PointStamped",
    "/nc/navigation_goal": "geometry_msgs/msg/PoseStamped",
}
