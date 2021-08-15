"""ml module setup."""
import pathlib
from typing import Dict

from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

version: Dict[str, float] = {}
with open('src/ml/version.py') as fp:
    exec(fp.read(), version)

setup(
    name='ml',
    version=version['__version__'],
    long_description=long_description,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
