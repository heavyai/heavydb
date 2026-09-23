# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

#.rst:
# FindOpenSaml.cmake
# -------------
#
# Find an opensaml installation.
#
# This module finds if opensaml is installed and selects a default
# configuration to use.
#
# find_package(OpenSaml ...)
#
#
# The following variables control which libraries are found::
#
#   LibOpenSaml_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   OpenSaml_FOUND            - Set to TRUE if opensaml libraries were found.
#   OpenSaml_INCLUDE_DIRS     - Include directories.
#   OpenSaml_LIBRARIES        - Path to the opensaml libraries.
#
#
# Sample usage:
#
# ::
#
#    find_package(OpenSaml)
#    if(OpenSaml_FOUND)
#      target_link_libraries(<YourTarget> ${LibOpenSaml_LIBRARIES})
#    endif()

if(LibOpenSaml_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

macro(find_lib libname libprefix)
  find_library(${libprefix}_LIBRARY
    NAMES ${libname}
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_${libprefix}}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)
endmacro()

find_lib(saml SAML)
find_lib(xmltooling XMLTOOLING)
find_lib(xerces-c XERCES_C)
find_lib(xml-security-c XML_SECURITY_C)

find_package(OpenSSL REQUIRED)

set(OpenSaml_LIBRARIES ${SAML_LIBRARY} ${XMLTOOLING_LIBRARY} ${XERCES_C_LIBRARY} ${XML_SECURITY_C_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})

macro(find_include filename libprefix)
  get_filename_component(LIBRARY_DIR ${${libprefix}_LIBRARY} DIRECTORY)
  find_path(${libprefix}_INCLUDE_DIR
    NAMES ${filename}
    HINTS ${LIBRARY_DIR}/../include
    HINTS ${PREFIX_${libprefix}}/include
    PATHS
    /include
    /usr/include
    /usr/local/include
    /usr/local/homebrew/include
    /opt/local/include
    )
endmacro()

find_include(saml/SAMLConfig.h SAML)
find_include(xmltooling/XMLObject.h XMLTOOLING)
find_include(xercesc/util/Xerces_autoconf_config.hpp XERCES_C)

set(OpenSaml_INCLUDE_DIRS ${SAML_INCLUDE_DIR} ${XMLTOOLING_INCLUDE_DIR} ${XERCES_C_INCLUDE_DIR})

#message("Saml includes = ${OpenSaml_INCLUDE_DIRS}")
#message("Saml libs = ${OpenSaml_LIBRARIES}")

if(LibOpenSaml_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenSaml REQUIRED_VARS OpenSaml_LIBRARIES OpenSaml_INCLUDE_DIRS SAML_LIBRARY XMLTOOLING_LIBRARY XERCES_C_LIBRARY XML_SECURITY_C_LIBRARY SAML_INCLUDE_DIR XMLTOOLING_INCLUDE_DIR XERCES_C_INCLUDE_DIR)

