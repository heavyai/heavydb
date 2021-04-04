// Copyright (c) 2021 OmniSci, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
