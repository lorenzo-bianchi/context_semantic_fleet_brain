from glob import glob
from setuptools import setup

package_name = 'redis_ros_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'redis'],
    zip_safe=True,
    maintainer='Lorenzo Bianchi',
    maintainer_email='lnz.bnc@gmail.com',
    description='Bridge node between Redis task queue and ROS 2 network for Semantic Fleet Brain',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge = redis_ros_bridge.bridge:main',
        ],
    },
)