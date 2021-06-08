import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='Autometacal',
    description='Metacal implementationmentation in TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='CosmoStat',
    url='https://github.com/CosmoStat/autometacal',
    license='MIT',
    packages=find_packages(),
    install_requires=['tfa-nightly', 'tfg-nightly','tensorflow-datasets'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    keywords='astronomy',
    use_scm_version=True,
    setup_requires=['setuptools_scm']
)
