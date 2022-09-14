# OmniSciDB Developer Documentation

Documentation is available at:

https://www.mapd.com/docs/

## Sphinx docs

[Sphinx](http://www.sphinx-doc.org) is a python-based tool to generate documentation. Here it will be used to generate HTML pages with the OmniSciDB Developer Documentation.

## Building docs

Documentation can be build locally using a make target on the host machine, using a docker container, or manually.

* In the below steps, replace `make html` with `make livehtml` to have the build watch for changes and provide a live-preview.

* If you would like to add a version number to the docs, run the following from the root of this repo to export version number into the $VER variable:
```
export VER=$(scripts/parse-version.sh)
```
and then append it to the `make html` command like so:
```
make html SPHINXOPTS="-D version=$VER"
```
these steps are not required when building using the make target, as it will gather the version itself.

#### Building with make target

There is a make target, `make sphinx` that can be ran from the top-level `../build/` directory after initialized with `cmake`.


### Building Manually

#### Requirements

Sphinx requires Python 3 (tested on Python 3.7) and the required python packages are installed with pip. See requirements.txt for list of required packages.

#### Building Sphinx docs

This will take the source docs files from the `./source/` directory and output HTML site files into the `./build/` directory.

#### Manually

The following steps use a python virtual environment, from inside this `docs` directory run:

```
python3 -m venv sphinx-env
. sphinx-env/bin/activate
pip install -r requirements.txt

make html SPHINXOPTS="-D version=$(../scripts/parse-version.sh)"
deactivate
```

### Previewing Locally

Once the docs are built, running `python -m http.server` from the `docs/build/html` directory will
allow for viewing the docs at `localhost:8000`.

#### VSCode Live Preview

Install the [reStructuredText extension](https://github.com/vscode-restructuredtext/vscode-restructuredtext)

Point `settings.json` to the the correct Python path. 
Assuming the manual using `sphinx-env` virtualenv from above:

```json
{
    "python.pythonPath": "${workspaceFolder}/docs/sphinx-env/bin/python"
}
```

`reStructuredText: Open Locked Preview to the Side` will give a live preview window with the generated sphinx docs.
