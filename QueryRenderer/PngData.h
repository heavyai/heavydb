/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace QueryRenderer {

class PngData {
 public:
  PngData() = default;
  explicit PngData(int width, int height);
  explicit PngData(int width,
                   int height,
                   const std::vector<std::byte>& pixels,
                   int compression_level = -1);

  void writeToFile(const std::string& filename);

  bool isValid() const;
  [[nodiscard]] std::string&& acquireString();

  // Check if compressionLevel is a valid value, throw otherwise
  // Current valid range is [0...9] (0=no compression, 9=max compression)
  // -1 = use default compression
  // TODO: std::optional
  static void validateCompressionLevel(int compression_level);

 private:
  void create(int width,
              int height,
              const std::vector<std::byte>& pixels,
              int compression_level);

  std::string png_data_;
};

}  // namespace QueryRenderer
