########################################################################
# Extract version
########################################################################
set(MUPARSERX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/muparserx)
file(READ "${MUPARSERX_SOURCE_DIR}/mpDefines.h" mpDefines_h)
string(REGEX MATCH "\\#define MUP_PARSER_VERSION _T\\(\"([0-9]+\\.[0-9]+\\.[0-9]+) \\(" MUPARSERX_VERSION_MATCHES "${mpDefines_h}")
if(NOT MUPARSERX_VERSION_MATCHES)
    message(FATAL_ERROR "Failed to extract version number from mpDefines.h")
endif(NOT MUPARSERX_VERSION_MATCHES)
set(MUPARSERX_VERSION ${CMAKE_MATCH_1})

########################################################################
# Compiler specific flags
########################################################################
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

    #C++11 is a required language feature for this project
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_STD_CXX11)
    if(HAS_STD_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else(HAS_STD_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()

endif(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))

#enable c++11 extensions for OSX
if (APPLE)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wc++11-extensions")
endif(APPLE)

if(MSVC)
    add_compile_options(/MP) #multi-core build
endif(MSVC)

########################################################################
# Build library
# Defaults to static, set BUILD_SHARED_LIBS=ON for shared
########################################################################
file(GLOB MUPARSERX_SOURCES "${MUPARSERX_SOURCE_DIR}/*.cpp")
include_directories(${MUPARSERX_SOURCE_DIR})
add_library(muparserx ${MUPARSERX_SOURCES})
set_target_properties(muparserx PROPERTIES VERSION ${MUPARSERX_VERSION})
set_property(TARGET muparserx PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(muparserx PROPERTIES SOVERSION ${MUPARSERX_VERSION})
set_target_properties(muparserx PROPERTIES VERSION ${MUPARSERX_VERSION})

#link with lib math when found
find_library(
    M_LIBRARY NAMES m
    PATHS /usr/lib /usr/lib64
)
if(M_LIBRARY)
    target_link_libraries(muparserx ${M_LIBRARY})
endif(M_LIBRARY)

########################################################################
# Print summary
########################################################################
message(STATUS "Building muparserx version: ${MUPARSERX_VERSION}")
