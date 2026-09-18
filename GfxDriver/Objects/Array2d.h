/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GFXDRIVER_OBJECTS_ARRAY2D_H_
#define GFXDRIVER_OBJECTS_ARRAY2D_H_

#include <cstring>
#include <memory>
#include "../RenderError.h"

namespace gfx {

namespace Objects {

/**
 * Simple 2d array of arithmetic types used to store and copy pixel data.
 * It is used exclusively by HitTestBuffers.
 */
template <typename T>
class Array2d {
 public:
  static_assert(std::is_arithmetic<T>::value, "Must be an arithmetic type");

  Array2d(uint32_t width, uint32_t height)
      : width_(0), height_(0), data_(nullptr), rows_() {
    _initialize(width, height);
  }

  uint32_t getWidth() const { return width_; }
  uint32_t getHeight() const { return height_; }

  void resize(uint32_t width, uint32_t height) { _initialize(width, height); }
  void copyFromPixelCenter(const Array2d<T>& src,
                           uint32_t srcCenterX,
                           uint32_t srcCenterY,
                           uint32_t radius) {
    CHECK_EQ(width_, height_);
    CHECK_EQ(width_, radius * 2 + 1);
    int src_start_x = static_cast<int>(srcCenterX) - static_cast<int>(radius);
    int src_start_y = static_cast<int>(srcCenterY) - static_cast<int>(radius);
    int src_end_x = std::min(srcCenterX + radius, src.getWidth() - 1);
    int src_end_y = std::min(srcCenterY + radius, src.getHeight() - 1);

    int dest_start_x = (src_start_x < 0) ? -src_start_x : 0;
    int dest_start_y = (src_start_y < 0) ? -src_start_y : 0;

    src_start_x = std::max(src_start_x, 0);
    src_start_y = std::max(src_start_y, 0);

    // TODO(croot): should we worry about resetting to 0 to handle the case of any
    // equivalent off-screen pixels needing to be set to 0? Currently this works ok
    // because this method is always called on newly initialized data, but this could
    // introduce a bug if there were any set operations applied to the data before calling
    // this method.
    for (int sy = src_start_y, dy = dest_start_y; sy <= src_end_y; ++sy, ++dy) {
      std::copy(&src.rows_[sy][src_start_x],
                &src.rows_[sy][src_end_x + 1],
                &rows_[dy][dest_start_x]);
    }
  }

  T get(uint32_t x, uint32_t y) {
    RUNTIME_EX_ASSERT(x < width_,
                      "Invalid x index: " + std::to_string(x) + ". It must be < " +
                          std::to_string(width_) + ".");
    RUNTIME_EX_ASSERT(y < height_,
                      "Invalid y index: " + std::to_string(y) + ". It must be < " +
                          std::to_string(height_) + ".");
    return rows_[y][x];
  }

  T getOrZero(int32_t x, int32_t y) {
    if (x < 0 || x >= static_cast<int32_t>(width_) || y < 0 ||
        y >= static_cast<int32_t>(height_)) {
      return static_cast<T>(0);
    }
    return rows_[static_cast<uint32_t>(y)][static_cast<uint32_t>(x)];
  }

  T* operator[](uint32_t row) {
    RUNTIME_EX_ASSERT(row < height_,
                      "Invalid height index: " + std::to_string(row) +
                          ". Height of 2d array is " + std::to_string(height_) + ".");
    return rows_[row];
  }

  const T* operator[](uint32_t row) const {
    RUNTIME_EX_ASSERT(row < height_,
                      "Invalid height index: " + std::to_string(row) +
                          ". Height of 2d array is " + std::to_string(height_) + ".");
    return rows_[row];
  }

  T* getDataPtr() {
    RUNTIME_EX_ASSERT(data_, "The 2d array is empty. Cannot retrieve data.");
    return data_.get();
  }

  const T* getDataPtr() const {
    RUNTIME_EX_ASSERT(data_, "The 2d array is empty. Cannot retrieve data.");
    return data_.get();
  }

  operator std::string() const {
    std::string rtn = "[";
    if (height_) {
      for (uint32_t j = height_ - 1; j >= 0; --j) {
        for (uint32_t i = 0; i < width_; ++i) {
          rtn += (i > 0 ? ", " : "");
          rtn += std::to_string(operator[](j)[i]);
        }
        if (j > 0) {
          rtn += "\n";
        }
      }
    }
    rtn += "]";

    return rtn;
  }

  void resetToDefault() { std::memset(data_.get(), 0, width_ * height_ * sizeof(T)); }

 private:
  uint32_t width_;
  uint32_t height_;

  std::shared_ptr<T> data_;
  std::vector<T*> rows_;

  void _initialize(uint32_t newWidth, uint32_t newHeight) {
    if (newWidth != width_ || newHeight != height_) {
      size_t num_elems = newHeight * newWidth;

      newWidth = (num_elems > 0 ? newWidth : 0);
      newHeight = (num_elems > 0 ? newHeight : 0);

      if (num_elems) {
        data_.reset(new T[num_elems], std::default_delete<T[]>());
        auto raw_data = data_.get();
        rows_.resize(newHeight);

        std::memset(data_.get(), 0, num_elems * sizeof(T));

        for (size_t i = 0; i < newHeight; ++i) {
          rows_[i] = &raw_data[i * newWidth];
        }
      } else {
        data_.reset();
        rows_.resize(0);
      }

      width_ = newWidth;
      height_ = newHeight;
    }
  }
};

}  // namespace Objects

}  // namespace gfx

#endif  // GFXDRIVER_OBJECTS_ARRAY2D_H_
