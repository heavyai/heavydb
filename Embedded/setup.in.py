from Cython.Build import cythonize
from distutils.core import setup, Extension

import os
import numpy as np
import pyarrow as pa

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbe = Extension("dbe",
                ["dbe.pyx"],
                language='c++17',
                include_dirs=[
                  np.get_include(),
                  pa.get_include(),
                  root,
                  "@CMAKE_SOURCE_DIR@",
                  "@CMAKE_CURRENT_SOURCE_DIR@"
                ],
                library_dirs=pa.get_library_dirs() + ['.'],
                runtime_library_dirs=pa.get_library_dirs() + ['$ORIGIN/../../'], # lib/python3.*/site-packages/../../
                libraries=pa.get_libraries() + ['DBEngine', 'boost_system'],
                extra_compile_args=['-std=c++17'],
              )
# Try uncommenting the following line on Linux
# if you get weird linker errors or runtime crashes
#    dbe.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

setup(
  name = 'dbe',
  version='0.1',
  ext_modules = cythonize(dbe,
    compiler_directives={'c_string_type': "str", 'c_string_encoding': "utf8", 'language_level': "3"}
  ),
  data_files=[
    ("lib", ["$<TARGET_FILE:DBEngine>"]),
    ('include', [
      "DBEngine.h",
      "DBEngine.pxd",
    ])
  ],
)
