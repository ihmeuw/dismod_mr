#!/usr/bin/env python
import os
import shutil
import subprocess

from setuptools import setup, find_packages


def check_environment():
    conda_path = shutil.which('conda')
    if not conda_path:
        raise EnvironmentError('You must have conda installed on your system to use this library.'
                               'See installation instructions at https://docs.conda.io/en/latest/miniconda.html')
    cp = subprocess.run([conda_path, 'list'], stdout=subprocess.PIPE, check=True)

    versions = {}
    for line in cp.stdout.decode().split('\n')[3:-1]:
        lib, version, *_ = line.split()
        versions[lib] = version

    if 'pymc' not in versions:
        raise EnvironmentError(f'PyMC is not installed in your conda environment. Run "conda install pymc" and '
                               f'try installation again.')
    # TODO: Add version check?


if __name__ == "__main__":
    check_environment()

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "dismod_mr", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'numpy>=1.7.1',
        'scipy>=0.12.0',
        'pymc>=2.3.6',
        'networkx>=1.8',
        'pandas>=0.23.4',
        'numba>=0.44.0',
        'matplotlib',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Natural Language :: English",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
            "Operating System :: POSIX :: BSD",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
            "Topic :: Software Development :: Libraries",
        ],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,

        zip_safe=False,
    )
