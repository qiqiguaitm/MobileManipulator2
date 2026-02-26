from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# 不声明 packages，避免与 ROS 生成的 perception.msg 包冲突
# 我们的 Python 脚本通过 catkin_install_python 安装
d = generate_distutils_setup()

setup(**d)
