from setuptools import setup, find_packages

setup(
    name='semantic_visualizer',
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/semantic_visualizer']),
        ('share/semantic_visualizer', ['package.xml']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'visualizer = semantic_visualizer.visualizer:main',
        ],
    },
)