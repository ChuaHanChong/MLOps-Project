from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

version = {}
with open("src/ml/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='ml',
    version=version['__version__'],
    long_description=long_description,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
