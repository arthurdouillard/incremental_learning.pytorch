#!/usr/bin/env python3

from setuptools import find_packages, setup

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

lines = list(parse_requirements("requirements.txt", session=PipSession()))
install_requires = [str(l.req) for l in lines if l.original_link is None]

setup(
    name='inclearn',
    version='0.1',
    description='An Incremental Learning library.',
    author='Arthur Douillard',
    url='https://github.com/arthurdouillard/incremental_learning.pytorch',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={'console_scripts': ['inclearn = inclearn.__main__:main']},
)
