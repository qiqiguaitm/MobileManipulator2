#!/usr/bin/env python3
"""Trim a finalized scene at operator confirmation, retaining TF/map context."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene', type=Path)
    a = p.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from geometry_msgs.msg import PoseStamped
    from tf2_msgs.msg import TFMessage
    src = a.scene
    dst = src.with_name(src.name + '_trimmed')
    if dst.exists():
        raise SystemExit('Output already exists: ' + str(dst))
    control = json.loads((src/'operator_confirmation.json').read_text())
    start = control['confirmation_received_ros_ns']
    end = control['stop_confirmation_received_ros_ns']
    dbs = list((src/'bag').glob('*.db3'))
    if len(dbs) != 1:
        raise SystemExit('Expected one finalized SQLite file')
    db = sqlite3.connect('file:'+str(dbs[0])+'?mode=ro', uri=True)
    topics = {row[0]: row for row in db.execute('select id,name,type,serialization_format,offered_qos_profiles from topics')}
    ids = {row[1]: key for key,row in topics.items()}
    seeds = []
    for topic in ['/reference/map', '/scenario/task', '/reference/camera/top/color/camera_info', '/reference/camera/top/aligned_depth_to_color/camera_info', '/reference/camera/top/extrinsics/depth_to_color']:
        row = db.execute('select data from messages where topic_id=? and timestamp<? order by timestamp desc,id desc limit 1', (ids.get(topic,-1),start)).fetchone()
        if row: seeds.append((topic,row[0]))
    static = {}
    for (raw,) in db.execute('select data from messages where topic_id=? and timestamp<? order by timestamp,id', (ids['/reference/tf_static'],start)):
        for tf in deserialize_message(raw,TFMessage).transforms:
            static[tf.child_frame_id] = tf
    seeds.append(('/reference/tf_static',serialize_message(TFMessage(transforms=list(static.values())))))
    for (raw,) in db.execute('select data from messages where topic_id=? and timestamp>=? and timestamp<? order by timestamp,id',(ids['/reference/tf'],start-2_000_000_000,start)):
        seeds.append(('/reference/tf',raw))
    dst.mkdir()
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(dst/'bag'),storage_id='sqlite3'),rosbag2_py.ConverterOptions('cdr','cdr'))
    for _,name,kind,fmt,qos in topics.values():
        writer.create_topic(rosbag2_py.TopicMetadata(name=name,type=kind,serialization_format=fmt,offered_qos_profiles=qos))
    counts = Counter()
    def write(topic,raw,stamp):
        writer.write(topic,raw,stamp)
        counts[topic] += 1
    for topic,raw in seeds: write(topic,raw,start)
    poses=[]
    for tid,raw,stamp in db.execute('select topic_id,data,timestamp from messages where timestamp>=? and timestamp<=? order by timestamp,id',(start,end)):
        topic=topics[tid][1]
        if topic=='/scenario/start': continue
        write(topic,raw,stamp)
        if topic=='/reference/trajectory':
            msg=deserialize_message(raw,PoseStamped)
            if not poses:write('/scenario/start',raw,stamp)
            poses.append(msg)
    writer=None
    db.close()
    if not poses:raise RuntimeError('No trajectory in selected interval')
    def pose(msg):
        q=msg.pose.orientation
        return dict(stamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,frame_id=msg.header.frame_id,x=msg.pose.position.x,y=msg.pose.position.y,yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
    manifest=json.loads((src/'scenario.json').read_text())
    manifest.update(start=pose(poses[0]),recorded_endpoint=pose(poses[-1]),started_ros_ns=start,started_utc=datetime.fromtimestamp(start/1e9,timezone.utc).isoformat(),elapsed_wall_seconds=(end-start)/1e9,message_counts=dict(counts),estimated_travel_m=sum(math.hypot(b.pose.position.x-a.pose.position.x,b.pose.position.y-a.pose.position.y) for a,b in zip(poses,poses[1:])))
    manifest['trim'] = dict(source=str(src),start_ns=start,end_ns=end,note='Boundary is assistant receipt of operator confirmation; exact UI click time unavailable. Original header stamps preserved. Map/static TF/calibration and 2-second TF pre-roll seeded at bag start; no pre-start images or trajectory retained.')
    (dst/'scenario.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (dst/'operator_confirmation.json').write_text(json.dumps(control,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(dst),duration=(end-start)/1e9,counts=dict(counts)),ensure_ascii=False))

if __name__=='__main__':main()
