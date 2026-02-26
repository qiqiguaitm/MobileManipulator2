#export DISPLAY=:99  # 设置环境变量指向虚拟显示

demo:
	PYTHONPATH=src python demo.py

slam:
	PYTHONPATH=src python slam.py

screen:
	-pkill -f Xvfb
	-rm -f /tmp/.X99-lock
	Xvfb :99 -screen 0 1024x768x24 &  # 创建虚拟显示器（分辨率 1024x768，色深 24）

set-route:
	ip route show
	-sudo ip rule add from 192.168.1.100 lookup eth0-route
	-sudo ip route add default via 192.168.1.1 dev enx00e04c68004c table eth0-route
	-sudo ip route add 192.168.1.0/24 dev enx00e04c68004c table eth0-route
	-sudo ip route add default via 192.168.215.1 dev wlp0s20f3 metric 100
	ip route show

save-log:
	scp grasp.* jiangxiaoke@dgx:/comp_robot/dino-3D/grasp/Ours/xarm/