# Copyright (c) 2002-2006 Marcus Geelnard
# Copyright (c) 2006-2019 Camilla Löwy
#
# This software is provided 'as-is', without any express or implied warranty.
# In no event will the authors be held liable for any damages arising from the
# use of this software. Permission is granted to anyone to use this software for
# any purpose, including commercial applications, and to alter it and
# redistribute it freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not claim
#    that you wrote the original software. If you use this software in a product,
#    an acknowledgment in the product documentation would be appreciated but is
#    not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#
# Derived from GLFW: https://github.com/glfw/glfw
#
# Find Vulkan
#
# VULKAN_INCLUDE_DIR
# VULKAN_LIBRARY
# VULKAN_FOUND

if (WIN32)
    find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h HINTS
        "$ENV{VULKAN_SDK}/Include"
        "$ENV{VK_SDK_PATH}/Include")
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        find_library(VULKAN_LIBRARY NAMES vulkan-1 HINTS
            "$ENV{VULKAN_SDK}/Lib"
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VK_SDK_PATH}/Bin")
        find_library(VULKAN_STATIC_LIBRARY NAMES vkstatic.1 HINTS
            "$ENV{VULKAN_SDK}/Lib"
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VK_SDK_PATH}/Bin")
    else()
        find_library(VULKAN_LIBRARY NAMES vulkan-1 HINTS
            "$ENV{VULKAN_SDK}/Lib32"
            "$ENV{VULKAN_SDK}/Bin32"
            "$ENV{VK_SDK_PATH}/Bin32")
        find_library(VULKAN_STATIC_LIBRARY NAMES vkstatic.1 HINTS
            "$ENV{VULKAN_SDK}/Lib32"
            "$ENV{VULKAN_SDK}/Bin32"
            "$ENV{VK_SDK_PATH}/Bin32")
    endif()
elseif (APPLE)
    find_library(VULKAN_LIBRARY vulkan.1 HINTS
        "$ENV{VULKAN_SDK}/macOS/lib")
    find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h HINTS
        "$ENV{VULKAN_SDK}/macOS/include")
else()
    find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h HINTS
        "$ENV{VULKAN_SDK}/include")
    find_library(VULKAN_LIBRARY NAMES vulkan HINTS
        "$ENV{VULKAN_SDK}/lib")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vulkan DEFAULT_MSG VULKAN_LIBRARY VULKAN_INCLUDE_DIR)

mark_as_advanced(VULKAN_INCLUDE_DIR VULKAN_LIBRARY VULKAN_STATIC_LIBRARY)

