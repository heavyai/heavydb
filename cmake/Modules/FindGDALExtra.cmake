#.rst:
# FindGDALExtra.cmake
# -------------
#
# Find additional libraries required by GDAL.
# This is only relevant when static linking GDAL.
#
# find_package(GDALExtra ...)
#
# The following are set after the configuration is done:
#
# ::
#
#   GDALExtra_FOUND            - Set to TRUE if the libraries were found.
#   GDALExtra_INCLUDE_DIRS     - Include directories
#   GDALExtra_LIBRARIES        - Path to the libraries.
#   GDALExtra_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(GDALExtra)
#    if(GDALExtra_FOUND)
#      target_link_libraries(<YourTarget> ${GDALExtra_LIBRARIES})
#    endif()

if(PREFER_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

set(GDALExtra_LIBRARIES "")
function(find_static_lib name)
  find_library(${name}_LIBRARY
    NAMES ${ARGN}
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)
endfunction()

if(PREFER_STATIC_LIBS)
  find_static_lib(EXPAT expat)
  find_static_lib(KMLDOM kmldom)
  find_static_lib(MINIZIP minizip)
  find_static_lib(KMLENGINE kmlengine)
  find_static_lib(KMLBASE kmlbase)
  find_static_lib(URIPARSER uriparser)
  find_static_lib(PROJ proj)
  set(GDALExtra_LIBRARIES ${KMLDOM_LIBRARY} ${EXPAT_LIBRARY} ${KMLENGINE_LIBRARY} ${KMLBASE_LIBRARY} ${MINIZIP_LIBRARY} ${URIPARSER_LIBRARY} ${PROJ_LIBRARY})
endif()

if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
if(PREFER_STATIC_LIBS)
  find_package_handle_standard_args(GDALExtra REQUIRED_VARS EXPAT_LIBRARY KMLDOM_LIBRARY MINIZIP_LIBRARY KMLENGINE_LIBRARY KMLBASE_LIBRARY URIPARSER_LIBRARY PROJ_LIBRARY)
endif()
