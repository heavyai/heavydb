#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Search roots used by both Vulkan loader and GLVND vendor layouts.
_NVIDIA_GRAPHICS_SEARCH_ROOTS=(
  /etc/xdg
  /etc
  /usr/local/share
  /usr/share
)

# Required: NVIDIA Vulkan ICD manifest.
# Sets NVIDIA_VK_ICD_PATH on success; clears it and returns 1 if not found.
# Note that Vulkan does not require any particular manifest filename, but
# NVIDIA implementations consistently use either "nvidia_icd.json" or
# "nvidia_icd.<arch>.json".
find_nvidia_vk_icd_path() {
  local root arch="${1:-$(uname -m)}"
  NVIDIA_VK_ICD_PATH=
  for root in "${_NVIDIA_GRAPHICS_SEARCH_ROOTS[@]}"; do
    if [[ -f "$root/vulkan/icd.d/nvidia_icd.json" ]]; then
      NVIDIA_VK_ICD_PATH="$root/vulkan/icd.d/nvidia_icd.json"
      return 0
    elif [[ -f "$root/vulkan/icd.d/nvidia_icd.${arch}.json" ]]; then
      NVIDIA_VK_ICD_PATH="$root/vulkan/icd.d/nvidia_icd.${arch}.json"
      return 0
    fi
  done
  return 1
}

# Optional: NVIDIA EGL vendor manifest (GLVND).
# Sets NVIDIA_EGL_VENDOR_PATH on success; clears it and returns 1 if not found.
# Note that glvnd does not require any particular manifest filename here, but
# NVIDIA implementations consistently use "10_nvidia.json".
find_nvidia_egl_vendor_path() {
  local root
  NVIDIA_EGL_VENDOR_PATH=
  for root in "${_NVIDIA_GRAPHICS_SEARCH_ROOTS[@]}"; do
    if [[ -f "$root/glvnd/egl_vendor.d/10_nvidia.json" ]]; then
      NVIDIA_EGL_VENDOR_PATH="$root/glvnd/egl_vendor.d/10_nvidia.json"
      return 0
    fi
  done
  return 1
}

# Apply env vars when manifests exist.
# Returns 1 only when the required Vulkan ICD manifest is missing.
export_nvidia_graphics_env() {
  if ! find_nvidia_vk_icd_path; then
    return 1
  fi
  export VK_ICD_FILENAMES="$NVIDIA_VK_ICD_PATH"

  if find_nvidia_egl_vendor_path; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_PATH"
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
  fi
  return 0
}
