/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/get_nvidia_compute_capability.h"

#include <iostream>

int main(int argc, char** argv) {
  std::vector<size_t> capabilities;

  try {
    capabilities = get_nvidia_compute_capability();
  } catch (const std::exception& e) {
    std::cerr << "get_nvidia_compute_capability failed: " << e.what();
    return 1;
  } catch (...) {
    std::cerr << "get_nvidia_compute_capability failed";
    return 2;
  }

  for (auto capability : capabilities) {
    std::cout << capability << std::endl;
    break;  // TODO(sy): need to add an --all flag
  }

  return 0;
}
