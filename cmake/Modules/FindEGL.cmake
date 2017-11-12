# Find EGL
#
# EGL_INCLUDE_DIR
# EGL_LIBRARY
# EGL_FOUND

find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h PATHS ${CMAKE_SOURCE_DIR}/ThirdParty/egl /usr/include PATH_SUFFIXES nvidia nvidia-375 nvidia-367 nvidia-365 nvidia-361 nvidia-381 nvidia-384 aarch64-linux-gnu)
find_library(EGL_LIBRARY NAMES egl EGL libEGL PATHS /usr/lib64 /usr/lib PATH_SUFFIXES nvidia nvidia-375 nvidia-367 nvidia-365 nvidia-361 nvidia-381 nvidia-384 aarch64-linux-gnu NO_DEFAULT_PATH)

set(EGL_INCLUDE_DIRS ${EGL_INCLUDE_DIR})
set(EGL_LIBRARIES ${EGL_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EGL REQUIRED_VARS EGL_INCLUDE_DIR EGL_LIBRARY)
#find_package_handle_standard_args(EGL DEFAULT_MSG EGL_INCLUDE_DIR EGL_LIBRARY)

mark_as_advanced(EGL_INCLUDE_DIR EGL_LIBRARY)
