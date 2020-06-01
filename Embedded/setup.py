from distutils.core import setup, Extension
from Cython.Build import cythonize

import os
import numpy as np
import pyarrow as pa


ext_modules = [Extension("dbe",
                     ["${CMAKE_CURRENT_SOURCE_DIR}/dbe.pyx"],
                     language='c++',
                     )]


for ext in ext_modules:
    # The Numpy C headers are currently required
    ext.include_dirs.append(np.get_include())
    ext.include_dirs.append(pa.get_include())
    ext.libraries.extend(pa.get_libraries())
    ext.library_dirs.extend(pa.get_library_dirs())

    ext.include_dirs.append("${CMAKE_SOURCE_DIR}")
    ext.libraries.append('DBEngine')
    ext.library_dirs.append("${CMAKE_CURRENT_BINARY_DIR}")

    ext.extra_compile_args.append('-std=c++17')

setup(
  name = 'dbe',
  ext_modules = cythonize(ext_modules, compiler_directives={'c_string_type': "str", 'c_string_encoding': "utf8", 'language_level': "3"})
)