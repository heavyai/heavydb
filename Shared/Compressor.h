/*
 * Copyright 2018 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file		Compressor.h
 * @brief		singleton class to handle concurrancy and state for blosc library.
 *A C++ wraper over a pure C library.
 *
 * Copyright (c) 2018 OmniSci, Inc.  All rights reserved.
 **/

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef COMPRESSOR_H
#define COMPRESSOR_H

class CompressionFailedError : public std::runtime_error {
 public:
  CompressionFailedError() : std::runtime_error("Compression Failed") {}
  CompressionFailedError(const std::string& e) : std::runtime_error(e) {}
};

class BloscCompressor {
 public:
  static BloscCompressor* getCompressor();

  // Compression algorithm takes extra space for scratch work, it uses the output
  // buffer to do the scratch work. We have to provide the compressor extra 10%
  // space for it.
  // https://github.com/Blosc/c-blosc/blob/c7792d6153eaf3d3d86eb33a28e9c613d2337040/blosc/blosclz.h#L28
  inline size_t getScratchSpaceSize(const size_t len) const { return len * 1.1; }

  // requires a compressed buffer at least as large as uncompressed buffer.
  // use 0 to always force compression for min_compressor_bytes.
  int64_t compress(const uint8_t* buffer,
                   const size_t buffer_size,
                   uint8_t* compressed_buffer,
                   const size_t compressed_buffer_size,
                   const size_t min_compressor_bytes);
  std::string compress(const std::string& buffer);

  size_t decompress(const uint8_t* compressed_buffer,
                    uint8_t* decompressed_buffer,
                    const size_t decompressed_size);
  std::string decompress(const std::string& buffer, const size_t decompressed_size);

  size_t compressOrMemcpy(const uint8_t* input_buffer,
                          uint8_t* output_buffer,
                          const size_t uncompressed_size,
                          const size_t min_compressor_bytes);

  bool decompressOrMemcpy(const uint8_t* compressed_buffer,
                          const size_t compressed_buffer_size,
                          uint8_t* decompressed_buffer,
                          const size_t decompressed_size);

  void getBloscBufferSizes(const uint8_t* data_ptr,
                           size_t* num_bytes_compressed,
                           size_t* num_bytes_uncompressed,
                           size_t* block_size);

  int setThreads(size_t num_threads);

  int setCompressor(std::string& compressor);

  ~BloscCompressor();

 private:
  BloscCompressor();
  std::mutex compressor_lock;
  static BloscCompressor* instance;
};

#endif
