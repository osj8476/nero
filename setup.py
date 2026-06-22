from setuptools import setup

package_name = 'nero_ai'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'requests',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    zip_safe=True,
    maintainer='nero',
    maintainer_email='nero@example.com',
    description='NERO 박스 인식 perception (박스 전용 단순화 버전)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = nero_ai.perception_node:main',
        ],
    },
)
