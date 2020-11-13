# Found at https://gitlab.kitware.com/vtk/vtk-m/blob/783867eeb05e0a6538f9c520af02c3615651b4ed/CMake/FindThrust.cmake
# should look at extending this to list available backends, tho some backends would be forced by other enable flags, like ENABLE_CUDA

##============================================================================
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2014 Sandia Corporation.
##  Copyright 2014 UT-Battelle, LLC.
##  Copyright 2014 Los Alamos National Security.
##
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
##  Laboratory (LANL), the U.S. Government retains certain rights in
##  this software.
##============================================================================

#
# FindThrust
#
# This module finds the Thrust header files and extrats their version.  It
# sets the following variables.
#
# THRUST_INCLUDE_DIR -  Include directory for thrust header files.  (All header
#                       files will actually be in the thrust subdirectory.)
# THRUST_VERSION -      Version of thrust in the form "major.minor.patch".
#

find_path( THRUST_INCLUDE_DIR
  HINTS
    /usr/local/cuda/include
    /usr/include/cuda
    /usr/local/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
  NAMES thrust/version.h
  DOC "Thrust headers"
  )
if( THRUST_INCLUDE_DIR )
  list( REMOVE_DUPLICATES THRUST_INCLUDE_DIR )
endif( THRUST_INCLUDE_DIR )

# Find thrust version
if (THRUST_INCLUDE_DIR)
  file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
    version
    REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define THRUST_VERSION[ \t]+"
    ""
    version
    "${version}"
    )

  string( REGEX MATCH "^[0-9]" major ${version} )
  string( REGEX REPLACE "^${major}00" "" version "${version}" )
  string( REGEX MATCH "^[0-9]" minor ${version} )
  string( REGEX REPLACE "^${minor}0" "" version "${version}" )
  set( THRUST_VERSION "${major}.${minor}.${version}")
  set( THRUST_MAJOR_VERSION "${major}")
  set( THRUST_MINOR_VERSION "${minor}")
endif()

# Check for required components
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Thrust
  REQUIRED_VARS THRUST_INCLUDE_DIR
  VERSION_VAR THRUST_VERSION
  )

set(THRUST_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
mark_as_advanced(THRUST_INCLUDE_DIR)
