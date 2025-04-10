#!/usr/bin/env python

from setuptools import setup, find_packages
import os
import subprocess
import sys

# Check if we're running the 'install' command
INSTALL = len(sys.argv) > 1 and sys.argv[1] == 'install'

# Function to run CMake build
def build_cmake():
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    # Run CMake
    subprocess.check_call(['cmake', '..'], cwd=build_dir)
    
    # Build with make
    cores = 1
    subprocess.check_call(['make', f'-j{cores}'], cwd=build_dir)
    
    # Check for orientation file and download if needed
    if not os.path.exists(os.path.join(build_dir, '100MilOrients.bin')):
        subprocess.check_call(['make', 'download_orientation_file'], cwd=build_dir)

# If installing, build the C/CUDA components first
if INSTALL:
    build_cmake()

# Get the requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Setup
setup(
    name="LaueMatching",
    version="0.1.0",
    author="Hemant Sharma",
    author_email="hsharma@anl.gov",
    description="Find orientations in Laue Diffraction Images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AdvancedPhotonSource/LaueMatching",
    py_modules=["RunImage"],
    package_data={
        '': ['*.py', '*.bin'],
    },
    include_package_data=True,
    data_files=[
        ('bin', [
            os.path.join('build', 'bin', 'LaueMatchingCPU'),
            os.path.join('build', 'bin', 'LaueMatchingGPU'),
        ] if os.path.exists(os.path.join('build', 'bin')) else []),
        ('', ['requirements.txt']),
    ],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.6',
)