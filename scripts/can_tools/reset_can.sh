#!/bin/bash
# Reset CAN interfaces with optimized settings

# CAN0: Piper arm (1Mbps)
sudo ip link set can0 down
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up type can bitrate 1000000 restart-ms 100

# CAN1: Chassis (500kbps)
sudo ip link set can1 down
sudo ip link set can1 txqueuelen 1000
sudo ip link set can1 up type can bitrate 500000 restart-ms 100

echo "CAN interfaces reset:"
ip -brief link show can0 can1
