# OmniSciDB Developer Documentation

Documentation is available at:

https://www.mapd.com/docs/

## Sphinx docs

[Sphinx](http://www.sphinx-doc.org) is a python-based tool to generate documentation. Here it will be used to generate HTML pages with the OmniSciDB Developer Documentation.

### Requirements

Sphinx requires python3 (tested on python3.7) and the required python packages are installed with pip. See requirements.txt for list of required packages.

### Building Sphinx docs

This will take the source docs files from the `./source/` directory and output HTML site files into the `./build/` directory.

#### Manually

The following steps use a python virtual environment:

```
python3 -m venv sphinx-env
. sphinx-env/bin/activate
pip install -r requirements.txt
make html
deactivate
```

#### With top-level make target

Alternatively, there is a make target to do the above steps, use `make sphinx` from the top-level `../build/` directory.

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
