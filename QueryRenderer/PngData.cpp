/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/PngData.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

#include <png.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"

namespace QueryRenderer {

static void write_png_data(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::string* png_data = reinterpret_cast<std::string*>(png_get_io_ptr(png_ptr));
  size_t current_size = png_data->size();
  png_data->resize(current_size + length);
  std::memcpy(png_data->data() + current_size, data, length);
}

static void flush_png_data(png_structp) {}

void PngData::validateCompressionLevel(int compression_level) {
  RUNTIME_EX_ASSERT(compression_level >= -1 && compression_level <= 9,
                    "Invalid png compression level " + std::to_string(compression_level) +
                        ". It must be a value between 0 (no zlib compression) to 9 (most "
                        "zlib compression), or -1 (use default).");
}

PngData::PngData(int width, int height) {
  std::vector<std::byte> empty_image{static_cast<size_t>(width * height * 4),
                                     static_cast<std::byte>(0)};
  RENDER_LOG_SCOPE() << "w: " << width << "  h: " << height;
  create(width, height, empty_image, -1);
}

PngData::PngData(int width,
                 int height,
                 const std::vector<std::byte>& pixels,
                 int compression_level) {
  RENDER_LOG_SCOPE() << "w: " << width << "  h: " << height;
  create(width, height, pixels, compression_level);
}

void PngData::create(int width,
                     int height,
                     const std::vector<std::byte>& pixels,
                     int compression_level) {
  RUNTIME_EX_ASSERT(!pixels.empty() && (width > 0) && (height > 0),
                    "Cannot create PngData(). The pixels are empty.");

  validateCompressionLevel(compression_level);

  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  assert(png_ptr != nullptr);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr != nullptr);

  // TODO(croot) - rather than going the setjmp route, you can enable the
  // PNG_SETJMP_NOT_SUPPORTED compiler flag which would result in asserts
  // when libpng errors, according to its docs.
  // if (setjmp(png_jmpbuf(png_ptr))) {
  //   std::cerr << "Got a libpng error" << std::endl;
  //   // png_destroy_info_struct(png_ptr, &info_ptr);
  //   png_destroy_write_struct(&png_ptr, &info_ptr);
  //   assert(false);
  // }

  png_set_write_fn(png_ptr, &png_data_, write_png_data, flush_png_data);

  // set filtering?
  png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_UP);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_AVG);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_PAETH);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);

  // set filter weights/preferences? I can't seem to get this
  // to make a difference
  // double weights[3] = {2.0, 1.5, 1.1};
  // double costs[PNG_FILTER_VALUE_LAST] = {2.0, 2.0, 1.0, 2.0, 2.0};
  // png_set_filter_heuristics(png_ptr, PNG_FILTER_HEURISTIC_WEIGHTED, 3, weights, costs);

  // set zlib compression level
  // if (compression_level >= 0) {
  //  png_set_compression_level(png_ptr, compression_level);
  //}
  png_set_compression_level(png_ptr, compression_level);

  // other zlib params?
  // png_set_compression_mem_level(png_ptr, 8);
  // png_set_compression_strategy(png_ptr, PNG_Z_DEFAULT_STRATEGY);
  // png_set_compression_window_bits(png_ptr, 15);
  // png_set_compression_method(png_ptr, 8);
  // png_set_compression_buffer_size(png_ptr, 8192);

  // skip the 8 bytes signature?
  // png_set_sig_bytes(png_ptr, 8);

  int interlace_type =
      PNG_INTERLACE_NONE;  // or PNG_INTERLACE_ADAM7 if we ever want interlacing
  png_set_IHDR(png_ptr,
               info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               interlace_type,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  // write out the PNG header info (everything up to first IDAT)
  png_write_info(png_ptr, info_ptr);

  // make sure < 8-bit images are packed into pixels as tightly as possible - only
  // necessary for palette images, which we're not doing yet png_set_packing(png_ptr);

  const std::byte* raw_bytes = pixels.data();
  std::vector<png_byte*> row_pointers(height);

  for (int j = 0; j < height; ++j) {
    // invert j -- input pixel rows go bottom up, where pngs are
    // defined top-down.
    row_pointers[j] = const_cast<png_byte*>(
        reinterpret_cast<const png_byte*>(&raw_bytes[(height - j - 1) * width * 4]));
  }

  png_write_image(png_ptr, row_pointers.data());

  // can alternatively write per-row, but this didn't
  // seem to make a difference. I thought that perhaps
  // this could be parallelized, but png_write_row() doesn't
  // appear to be a fixed-function call.
  // for (j = 0; j < height; ++j) {
  //   png_write_row(png_ptr, row_pointers[j]);
  // }

  png_write_end(png_ptr, info_ptr);

  png_destroy_write_struct(&png_ptr, &info_ptr);
}

void PngData::writeToFile(const std::string& filename) {
  RUNTIME_EX_ASSERT(!png_data_.empty(),
                    "Cannot write file " + filename + ". The pixels are empty.");
  std::ofstream png_file(filename, std::ios::binary);
  png_file.write(reinterpret_cast<const char*>(png_data_.data()), png_data_.size());
  png_file.close();
}

bool PngData::isValid() const {
  return !png_data_.empty();
}

std::string&& PngData::acquireString() {
  return std::move(png_data_);
}

}  // namespace QueryRenderer
