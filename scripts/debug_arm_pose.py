#!/usr/bin/env python3
"""诊断: Piper 末端位姿读取"""
import sys
import time

sys.path.insert(0, '/data/workspace/MobileManipulator2/scripts/arm_robot/src')
from piper_sdk import C_PiperInterface_V2

CAN = 'can0'

print(f"=== Step 1: ConnectPort on {CAN} ===")
piper = C_PiperInterface_V2(CAN)
piper.ConnectPort()
time.sleep(0.5)

print(f"\n=== Step 2: Read pose BEFORE enable (expect zeros) ===")
for i in range(3):
    msg = piper.GetArmEndPoseMsgs()
    p = msg.end_pose
    print(f"  [{i}] X={p.X_axis} Y={p.Y_axis} Z={p.Z_axis} "
          f"RX={p.RX_axis} RY={p.RY_axis} RZ={p.RZ_axis} "
          f"Hz={msg.Hz:.1f} ts={msg.time_stamp:.3f}")
    time.sleep(0.2)

print(f"\n=== Step 3: Check motor enable status ===")
enable_status = piper.GetArmEnableStatus()
print(f"  Motor enable: {enable_status}")

print(f"\n=== Step 4: EnablePiper (loop up to 5s) ===")
t0 = time.time()
enabled = False
for attempt in range(500):
    if piper.EnablePiper():
        enabled = True
        break
    time.sleep(0.01)
elapsed = time.time() - t0
print(f"  EnablePiper returned: {enabled} (took {elapsed:.2f}s)")

print(f"\n=== Step 5: Check motor enable status AFTER ===")
enable_status = piper.GetArmEnableStatus()
print(f"  Motor enable: {enable_status}")

print(f"\n=== Step 6: Read pose AFTER enable (wait 2s for feedback) ===")
for wait in [0.2, 0.5, 1.0, 2.0]:
    time.sleep(wait)
    msg = piper.GetArmEndPoseMsgs()
    p = msg.end_pose
    print(f"  [+{wait}s] X={p.X_axis} Y={p.Y_axis} Z={p.Z_axis} "
          f"RX={p.RX_axis} RY={p.RY_axis} RZ={p.RZ_axis} "
          f"Hz={msg.Hz:.1f} ts={msg.time_stamp:.3f}")

print(f"\n=== Step 7: Continuous read (2s @ 10Hz) ===")
for i in range(20):
    msg = piper.GetArmEndPoseMsgs()
    p = msg.end_pose
    vals = [p.X_axis, p.Y_axis, p.Z_axis, p.RX_axis, p.RY_axis, p.RZ_axis]
    nonzero = sum(1 for v in vals if v != 0)
    print(f"  [{i:02d}] nonzero={nonzero}/6 "
          f"X={p.X_axis/1000:.1f}mm Z={p.Z_axis/1000:.1f}mm "
          f"Hz={msg.Hz:.1f}")
    time.sleep(0.1)

print(f"\n=== Step 8: Raw CAN sniff (check if 0x152-0x154 arrive) ===")
try:
    import can
    bus = can.interface.Bus(channel=CAN, bustype='socketcan')
    found = {0x152: False, 0x153: False, 0x154: False}
    t0 = time.time()
    while time.time() - t0 < 3.0:
        frame = bus.recv(timeout=0.5)
        if frame and frame.arbitration_id in found:
            found[frame.arbitration_id] = True
            print(f"  CAN 0x{frame.arbitration_id:03X}: {frame.data.hex()}")
        if all(found.values()):
            break
    for cid, ok in found.items():
        status = "OK" if ok else "MISSING"
        print(f"  0x{cid:03X}: {status}")
    bus.shutdown()
except Exception as e:
    print(f"  Raw CAN check skipped: {e}")

print("\n=== Done ===")
