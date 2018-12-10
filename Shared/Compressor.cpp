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
 * @file		Compressor.cpp
 * @brief		singleton class to handle concurrancy and state for blosc library.
 *A C++ wraper over a pure C library.
 *
 * Copyright (c) 2018 OmniSci, Inc.  All rights reserved.
 **/

#include "Compressor.h"
#include <blosc.h>
#include <glog/logging.h>
#include <string>
#include <thread>

// we only compress data if the payload size is greater than 512 MB
size_t g_compression_limit_bytes{512 * 1024 * 1024};

BloscCompressor::BloscCompressor() {
  std::lock_guard<std::mutex> compressor_lock_(compressor_lock);
  blosc_init();
  // We use maximum number of threads here since with tests we found that compression
  // speed gets lear scalling with corresponding to the number of threads being used.

  blosc_set_nthreads(std::thread::hardware_concurrency());

  // We chosse faster compressor, accepting slightly lower compression ratio
  // https://lz4.github.io/lz4/

  blosc_set_compressor(BLOSC_LZ4HC_COMPNAME);
}

BloscCompressor::~BloscCompressor() {
  std::lock_guard<std::mutex> compressor_lock_(compressor_lock);
  blosc_destroy();
}

int64_t BloscCompressor::compress(
    const uint8_t* buffer,
    const size_t buffer_size,
    uint8_t* compressed_buffer,
    const size_t compressed_buffer_size,
    const size_t min_compressor_bytes = g_compression_limit_bytes) {
  if (buffer_size < min_compressor_bytes && min_compressor_bytes != 0) {
    return 0;
  }
  std::lock_guard<std::mutex> compressor_lock_(compressor_lock);
  const auto compressed_len = blosc_compress(5,
                                             1,
                                             sizeof(unsigned char),
                                             buffer_size,
                                             buffer,
                                             &compressed_buffer[0],
                                             compressed_buffer_size);

  if (compressed_len <= 0) {
    // something went wrong. blosc retrun codes simply don't provide enough information
    // for us to decide what.

    throw CompressionFailedError(std::string("failed to compress result set of length ") +
                                 std::to_string(buffer_size));
  }
  // we need to tell the other endpoint the size of the acctual data so it can
  // decide whether it should decompress data or not. So we pass the original
  // data length. and only send the compressed result if the output of the
  // compressed result is smaller than the original
  return compressed_len;
}

std::string BloscCompressor::compress(const std::string& buffer) {
  const auto buffer_size = buffer.size();
  std::vector<uint8_t> compressed_buffer(getScratchSpaceSize(buffer_size));
  try {
    const size_t compressed_len = compress((uint8_t*)buffer.c_str(),
                                           buffer_size,
                                           &compressed_buffer[0],
                                           getScratchSpaceSize(buffer_size));
    if (compressed_len > 0 && compressed_len < buffer_size) {
      // we need to tell the other endpoint the size of the acctual data so it can
      // decide whether it should decompress data or not. So we pass the original
      // data length. and only send the compressed result if the output of the
      // compressed result is smaller than the original
      compressed_buffer.resize(compressed_len);
      return {compressed_buffer.begin(), compressed_buffer.end()};
    }
  } catch (const CompressionFailedError& e) {
  }

  return buffer;
}

size_t BloscCompressor::decompress(const uint8_t* compressed_buffer,
                                   uint8_t* decompressed_buffer,
                                   const size_t decompressed_size) {
  size_t decompressed_buf_len, compressed_buf_len, block_size, decompressed_len = 0;
  getBloscBufferSizes(
      &compressed_buffer[0], &compressed_buf_len, &decompressed_buf_len, &block_size);
  // check compressed buffer is a blosc compressed buffer.
  if (compressed_buf_len > 0 && decompressed_size == decompressed_buf_len) {
    std::lock_guard<std::mutex> compressor_lock_(compressor_lock);
    decompressed_len =
        blosc_decompress(&compressed_buffer[0], decompressed_buffer, decompressed_size);
  }

  if (decompressed_len == 0) {
    throw CompressionFailedError(
        std::string("failed to decompress buffer for compressed size: ") +
        std::to_string(compressed_buf_len));
  }
  if (decompressed_len != decompressed_size) {
    throw CompressionFailedError(
        std::string("decompression buffer size mismatch. Decompressed buffer length: ") +
        std::to_string(decompressed_len));
  }
  return decompressed_len;
}

std::string BloscCompressor::decompress(const std::string& buffer,
                                        const size_t decompressed_size) {
  std::vector<uint8_t> decompressed_buffer(decompressed_size);
  if (buffer.size() == decompressed_size) {
    return buffer;
  }
  try {
    decompress(
        (uint8_t*)&buffer[0], (uint8_t*)&decompressed_buffer[0], decompressed_size);
    return {decompressed_buffer.begin(), decompressed_buffer.end()};
  } catch (const CompressionFailedError& e) {
  }
  return buffer;
}

size_t BloscCompressor::compressOrMemcpy(const uint8_t* input_buffer,
                                         uint8_t* output_buffer,
                                         size_t uncompressed_size,
                                         const size_t min_compressor_bytes) {
  try {
    const auto compressed_size = compress(input_buffer,
                                          uncompressed_size,
                                          output_buffer,
                                          uncompressed_size,
                                          min_compressor_bytes);
    if (compressed_size > 0) {
      return compressed_size;
    }
  } catch (const CompressionFailedError& e) {
    // catch exceptions from blosc
    // we copy regardless what happens in compressor
    if (uncompressed_size > min_compressor_bytes) {
      LOG(WARNING) << "Compressor failed for byte size of " << uncompressed_size;
    }
  }
  memcpy(output_buffer, input_buffer, uncompressed_size);
  return uncompressed_size;
}

bool BloscCompressor::decompressOrMemcpy(const uint8_t* compressed_buffer,
                                         const size_t compressed_size,
                                         uint8_t* decompressed_buffer,
                                         const size_t decompressed_size) {
  try {
    decompress(compressed_buffer, decompressed_buffer, decompressed_size);
    return true;
  } catch (const CompressionFailedError& e) {
    // we will memcpy if we find that the buffer is not compressed

    if (compressed_size > decompressed_size) {
      throw std::runtime_error(
          "compressed buffer size is greater than decompressed buffer size.");
    }
  }
  memcpy(decompressed_buffer, compressed_buffer, decompressed_size);
  return false;
}

void BloscCompressor::getBloscBufferSizes(const uint8_t* data_ptr,
                                          size_t* num_bytes_compressed,
                                          size_t* num_bytes_uncompressed,
                                          size_t* block_size) {
  blosc_cbuffer_sizes(data_ptr, num_bytes_uncompressed, num_bytes_compressed, block_size);
}

BloscCompressor* BloscCompressor::instance = NULL;

BloscCompressor* BloscCompressor::getCompressor() {
  static std::mutex compressor_singleton_lock;
  std::lock_guard<std::mutex> singleton_lock(compressor_singleton_lock);

  if (instance == NULL) {
    instance = new BloscCompressor();
  }

  return instance;
}

int BloscCompressor::setThreads(size_t num_threads) {
  std::lock_guard<std::mutex> compressor_lock_(compressor_lock);
  return blosc_set_nthreads(num_threads);
}

int BloscCompressor::setCompressor(std::string& compressor_name) {
  std::lock_guard<std::mutex> compressor_lock_(compressor_lock);
  // Blosc is resilent enough to detect that the comprressor that was provided to it was
  // supported or not. If the compressor is invalid or not supported it will simply keep
  // current compressor.
  return blosc_set_compressor(compressor_name.c_str());
}
