#!/usr/bin/env python
from setuptools import setup, find_packages

from mochi import __author__, __version__, __license__, IS_PYTHON_34


install_requires = ['rply>=0.7.2',
                    'pyrsistent>=0.6.2',
                    'eventlet>=0.15.2']

if not IS_PYTHON_34:
    install_requires.append('pathlib>=1.0.1')

setup(
    name='mochi',
    version=__version__,
    description='Dynamically typed functional programming language',
    license=__license__,
    author=__author__,
    url='https://github.com/i2y/mochi',
    platforms=['any'],
    entry_points={
        'console_scripts': [
            'mochi = mochi.mochi:main'
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Compilers",
    ]
)