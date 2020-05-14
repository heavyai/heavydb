from distutils.core import setup, Extension
from Cython.Build import cythonize

import os
import numpy as np
import pyarrow as pa


ext_modules = [Extension("dbe",
                     ["dbe.pyx"],
                     language='c++',
                     )]


for ext in ext_modules:
    # The Numpy C headers are currently required
    ext.include_dirs.append(np.get_include())
    ext.include_dirs.append(pa.get_include())
    ext.libraries.extend(pa.get_libraries())
    ext.library_dirs.extend(pa.get_library_dirs())

    ext.include_dirs.append('../')
    ext.include_dirs.append('/usr/local/mapd-deps/include')
    ext.libraries.append('DBEngine')
    ext.library_dirs.append('./')
    ext.library_dirs.append('/usr/local/mapd-deps/lib')
    ext.library_dirs.append('/lib/x86_64-linux-gnu')
    ext.library_dirs.append('/usr/lib/x86_64-linux-gnu')

    ext.extra_compile_args.append('-std=c++17')

    # Try uncommenting the following line on Linux
    # if you get weird linker errors or runtime crashes
    #    ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

setup(
  name = 'dbe',
  ext_modules = cythonize(ext_modules, compiler_directives={'c_string_type': "str", 'c_string_encoding': "utf8", 'language_level': "3"})
)