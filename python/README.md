![PyPI](https://img.shields.io/pypi/v/pyheavydb?style=for-the-badge)

## pyheavydb

A python [DB API](https://www.python.org/dev/peps/pep-0249/) compliant
interface for [HeavyDB](https://www.heavy.ai/) (formerly OmniSci and MapD).

### Regenerate thrift files
You need to install [thrift](https://thrift.apache.org) compiler. Then
you can do:

```
make thrift
```

### Release
Update first the version numbers and make sure thrift files are up-to-date.
Releasing on PyPi assume you have a PyPi token in your environment.

```
# this will generate thrift files, make a distribution and upload on PyPi
make publish
```
