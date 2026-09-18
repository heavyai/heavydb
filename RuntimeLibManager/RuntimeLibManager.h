/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

class RuntimeLibManager {
 public:
  static void loadRuntimeLibs(const std::string& torch_lib_path = std::string());
  static void loadTestRuntimeLibs();
  static bool is_libtorch_loaded();

 private:
  static bool is_libtorch_loaded_;
};