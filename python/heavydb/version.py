import os
import importlib_metadata


def get_source_version():
    d = dict(MAJOR='6', MINOR='0', MICRO='0', EXTRA='none')
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        f = open(os.path.join(here, '..', '..', 'CMakeLists.txt'))
    except FileNotFoundError:
        return None
    for line in f.readlines():
        if line.lstrip().startswith('set(MAPD_VERSION_'):
            k = line.split()[0].rsplit('_', 1)[-1]
            n = line.split('"')[1]
            d[k] = n
    return '{MAJOR}.{MINOR}.{MICRO}{EXTRA}'.format(**d)


def get_package_version():
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:
        # package is not installed
        return get_source_version()
