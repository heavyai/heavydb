# HeavyDB Developer Documentation

User-facing product documentation is published at:

https://docs.nvidia.com/heavyai

When built via ``dev-tools``, developer documentation HTML is written to
``build/<distro>/docs/html/`` (for example ``build/ubuntu22.04/docs/html/``).
The CMake and manual venv paths below still write to ``docs/build/html/``.

## Sphinx docs

[Sphinx](https://www.sphinx-doc.org) is a Python-based tool for generating
documentation. Here it generates HTML pages for the HeavyDB developer guide
using the [NVIDIA Sphinx theme](https://pypi.org/project/nvidia-sphinx-theme/)
(``nvidia_sphinx_theme``).

## Building docs

Documentation can be built locally using ``dev-tools``, a CMake target, a
manual Docker invocation, or a Python virtual environment.

* In the below steps, replace `make html` with `make livehtml` to have the build watch for changes and provide a live-preview.

* If you would like to add a version number to the docs, run the following from the root of this repo to export version number into the $VER variable:
```
export VER=$(scripts/parse-version.sh)
```
and then append it to the `make html` command like so:
```
make html SPHINXOPTS="-D version=$VER"
```
These steps are not required when building via ``dev-tools`` or the CMake
target, as those gather the version themselves.

### Building with dev-tools (recommended)

From the repository root:

```
dev-tools/dev.sh build docs
```

HTML output: ``build/<distro>/docs/html/`` (default distro: ``ubuntu22.04``).

The default ``dev-tools/dev.sh build`` / ``build all`` targets also build docs
after heavydb. Docs run outside the heavydb deps container (Sphinx uses the
docs Docker image). Doxygen runs only when a configured heavydb build
(``Doxyfile``) exists and ``doxygen`` is on the host PATH; otherwise Sphinx
continues with placeholder C++ API pages.

Set ``HEAVYDB_SPHINX_IMAGE`` to override the default Sphinx image name
(``heavydb-sphinx-doc``). The image is built from ``docs/Dockerfile`` only when
missing; after editing ``docs/Dockerfile`` or ``docs/requirements.txt``, rebuild
explicitly:

```
docker build -t heavydb-sphinx-doc docs/
```

#### Building with make target

From the repository root, run the `sphinx` target after configuring the
top-level `build` directory with CMake:

```
cmake --build build --target sphinx
```

### Building with Docker (manual)

Docker can be used to build the documentation locally without installing any
dependencies to the host system. Prefer ``dev-tools/dev.sh build docs`` when
possible; the steps below are the underlying image and ``docker run`` flow.

Build a local image from the ``Dockerfile`` in this directory (Python 3.11,
``nvidia-sphinx-theme``, graphviz, and plantuml):

```
docker build -t heavydb-sphinx-doc .
```

Then, from inside this ``docs`` directory, run (example writing into a heavydb
build tree):

```
docker run --rm \
  -v "$PWD:/doc" -v "$PWD/../build/ubuntu22.04:/build" \
  -w /doc heavydb-sphinx-doc make html BUILDDIR=/build/docs
```

If ``/build/doxygen/xml`` is present (relative to the mounted heavydb build
dir), ``make html`` will also generate breathe C++ API pages under
``source/api/``. Otherwise it writes a placeholder API page and continues.

If there are any changes to dependencies, rebuild the local image with the ``Dockerfile`` in this directory.


### Building manually

Sphinx requires Python 3.10 or newer (``nvidia-sphinx-theme``). Install the
packages listed in `requirements.txt` with pip.

From this ``docs`` directory, create a virtual environment and build HTML:

```
python3 -m venv sphinx-env
. sphinx-env/bin/activate
pip install -r requirements.txt

make html SPHINXOPTS="-D version=$(../scripts/parse-version.sh)"
deactivate
```

This writes to ``docs/build/html/``.

### Previewing Locally

Once the docs are built via ``dev-tools``, running `python -m http.server` from
``build/<distro>/docs/html`` will allow for viewing the docs at `localhost:8000`.
For the CMake/venv paths, use ``docs/build/html`` instead.

#### VSCode Live Preview

Install the [reStructuredText extension](https://github.com/vscode-restructuredtext/vscode-restructuredtext)

Point ``settings.json`` to the correct Python path. Assuming the manual
``sphinx-env`` virtualenv from above:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/docs/sphinx-env/bin/python"
}
```

`reStructuredText: Open Locked Preview to the Side` will give a live preview window with the generated sphinx docs.
