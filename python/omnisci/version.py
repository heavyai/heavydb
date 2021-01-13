def get_source_version():
    import os
    d = dict(MAJOR='5', MINOR='6', MICRO='0', EXTRA='none')
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
    from pkg_resources import get_distribution, DistributionNotFound
    try:
        return get_distribution(__name__).version
    except DistributionNotFound:
        # package is not installed
        return get_source_version()
