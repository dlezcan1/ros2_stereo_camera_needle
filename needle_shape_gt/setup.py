from setuptools import setup
import glob

package_name = 'needle_shape_gt'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/launch", glob.glob('launch/*.launch.*')),
        ('share/' + package_name + "/config", glob.glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yera',
    maintainer_email='yernar.zhetpissov@gmail.com',
    description='Needle Stereo Shape Reconstruction Python ROS 2 package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_needle_shape_node = needle_shape_gt.shape_reconstruction_node:main',
            'transformed_shape_node = needle_shape_gt.transformed_shape_node:main',
        ],
    },
)
