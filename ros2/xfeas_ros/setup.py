from glob import glob

from setuptools import setup

setup(
    name="xfeas_ros",
    version="0.1.0",
    packages=["xfeas_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/xfeas_ros"]),
        ("share/xfeas_ros", ["package.xml"]),
        ("share/xfeas_ros/launch", glob("launch/*.launch.py")),
        ("share/xfeas_ros/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": ["xfeas_node = xfeas_ros.node:main"]},
)
