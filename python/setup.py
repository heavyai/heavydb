import os
from codecs import open

from setuptools import setup, find_packages

CONDA_BUILD = int(os.environ.get('CONDA_BUILD', '0'))

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

import importlib.util
spec = importlib.util.spec_from_file_location("omnisci_version", os.path.join(here, 'omnisci', 'version.py'))
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)
VERSION = version_module.get_source_version()  # gets version from ../CMakeLists.txt
assert VERSION is not None

install_requires = [
    "thrift == 0.13.0",
    "numpy",
    "sqlalchemy >= 1.3",
    "packaging >= 20.0",
    "requests >= 2.23.0",
    "rbc-project >= 0.2.2",
]

# Optional Requirements
doc_requires = ['sphinx', 'numpydoc', 'sphinx-rtd-theme']
test_requires = ['coverage', 'pytest', 'pytest-mock', 'pandas']
dev_requires = doc_requires + test_requires + ['pre-commit']
complete_requires = dev_requires

extra_requires = {
    "docs": doc_requires,
    "test": test_requires,
    "dev": dev_requires,
    "complete": complete_requires,
} if not CONDA_BUILD else {}  # CONDA deps are specified in meta.yaml

setup(
    name="pyomniscidb",
    description="A DB API 2 compatible client for OmniSci (formerly MapD).",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url="https://github.com/omnisci/omniscidb",
    author="OmniSci",
    author_email="community@omnisci.com",
    license="Apache Software License",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(exclude=('tests',)),
    version=VERSION,
    install_requires=install_requires,
    extras_require=extra_requires,
)
