from setuptools import setup
from glob import glob

setup(
    name="robot_tasks",
    version="0.1.0",
    packages=["robot_tasks"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/robot_tasks"]),
        ("share/robot_tasks", ["package.xml", "README.md"]),
        ("share/robot_tasks/launch", glob("launch/*.py")),
        ("share/robot_tasks/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "gc=robot_tasks.gc:main",
            "nc=robot_tasks.nc:main",
            "scene_perception=robot_tasks.scene:main",
            "task=robot_tasks.cli:main",
            "approach=robot_tasks.approach_cli:main",
            "session=robot_tasks.session:main",
            "mock_backends=robot_tasks.mock_backends:main",
        ]
    },
)
