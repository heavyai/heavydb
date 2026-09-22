/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file        FileBuffer.cpp
 * @brief
 *
 */

#include "DataMgr/FileMgr/FileBuffer.h"
#include <lz4.h>
#include <snappy.h>
#ifdef HAVE_NVCOMP_GDEFLATE
#include <nvcomp/native/gdeflate_cpu.h>
#endif
#ifdef HAVE_NVCOMP_BITCOMP
#include <nvcomp/native/bitcomp.h>
#endif
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include "DataMgr/FileMgr/FileMgr.h"
#include "Shared/File.h"
#include "Shared/checked_alloc.h"
#include "Shared/scope.h"
#include "Shared/threading.h"

using namespace std;

namespace File_Namespace {

bool g_enable_native_storage_compression{false};
bool g_enable_file_buffer_metadata_sidecar_only{false};
std::string g_native_storage_compression_codec{"snappy"};
size_t g_native_storage_compression_frame_size{64 * 1024};
int g_native_storage_compression_gdeflate_level{1};

FileBuffer::FileBuffer(FileMgr* fm,
                       const size_t pageSize,
                       const ChunkKey& chunkKey,
                       const size_t initialSize)
    : AbstractBuffer(fm->getDeviceId())
    , fm_(fm)
    , metadataPageSize_(fm_->getMetadataPageSize())
    , metadataPages_(metadataPageSize_)
    , pageSize_(pageSize)
    , chunkKey_(chunkKey) {
  // Create a new FileBuffer
  CHECK(fm_);
  setBufferHeaderSize();
  CHECK_GT(pageSize_, reservedHeaderSize_);
  pageDataSize_ = pageSize_ - reservedHeaderSize_;
  //@todo reintroduce initialSize - need to develop easy way of
  // differentiating these pre-allocated pages from "written-to" pages
  /*
  if (initalSize > 0) {
      // should expand to initialSize bytes
      size_t initialNumPages = (initalSize + pageSize_ -1) / pageSize_;
      int32_t epoch = fm_->epoch();
      for (size_t pageNum = 0; pageNum < initialNumPages; ++pageNum) {
          Page page = addNewMultiPage(epoch);
          writeHeader(page,pageNum,epoch);
      }
  }
  */
}

FileBuffer::FileBuffer(FileMgr* fm,
                       const size_t pageSize,
                       const ChunkKey& chunkKey,
                       const SQLTypeInfo sqlType,
                       const size_t initialSize)
    : AbstractBuffer(fm->getDeviceId(), sqlType)
    , fm_(fm)
    , metadataPageSize_(fm->getMetadataPageSize())
    , metadataPages_(metadataPageSize_)
    , pageSize_(pageSize)
    , chunkKey_(chunkKey) {
  CHECK(fm_);
  setBufferHeaderSize();
  pageDataSize_ = pageSize_ - reservedHeaderSize_;
}

FileBuffer::FileBuffer(FileMgr* fm,
                       /* const size_t pageSize,*/ const ChunkKey& chunkKey,
                       const std::vector<HeaderInfo>::const_iterator& headerStartIt,
                       const std::vector<HeaderInfo>::const_iterator& headerEndIt,
                       const std::vector<int8_t>* metadataPayload)
    : AbstractBuffer(fm->getDeviceId())
    , fm_(fm)
    , metadataPageSize_(fm->getMetadataPageSize())
    , metadataPages_(metadataPageSize_)
    , pageSize_(0)
    , chunkKey_(chunkKey) {
  // We are being assigned an existing FileBuffer on disk

  CHECK(fm_);
  setBufferHeaderSize();
  int32_t lastPageId = -1;
  int32_t curPageId = 0;
  for (auto vecIt = headerStartIt; vecIt != headerEndIt; ++vecIt) {
    curPageId = vecIt->pageId;

    // We only want to read last metadata page
    if (curPageId == -1) {  // stats page
      metadataPages_.push(vecIt->page, vecIt->versionEpoch);
    } else {
      if (curPageId != lastPageId) {
        // protect from bad data on disk, and give diagnostics
        if (fm->failOnReadError()) {
          if (curPageId != lastPageId + 1) {
            LOG(FATAL) << "Failure reading DB file " << show_chunk(chunkKey)
                       << " Current page " << curPageId << " last page " << lastPageId
                       << " epoch " << vecIt->versionEpoch;
          }
        }
        if (lastPageId == -1) {  // If we are on first real page
          initMetadataAndPageDataSize(metadataPayload);
        }
        MultiPage multiPage(pageSize_);
        multiPages_.push_back(multiPage);
        lastPageId = curPageId;
      }
      multiPages_.back().push(vecIt->page, vecIt->versionEpoch);
    }
  }
  if (curPageId == -1) {  // meaning there was only a metadata page
    initMetadataAndPageDataSize(metadataPayload);
  }
}

FileBuffer::~FileBuffer() {
  // need to free pages
  // NOP
}

void FileBuffer::reserve(const size_t numBytes) {
  if (isStorageCompressed()) {
    // Compressed payloads are replaced atomically after recompression. Reserving by
    // logical size would allocate unrelated raw pages and corrupt the physical layout
    // if a later compression or write step failed.
    return;
  }
  size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
  size_t numCurrentPages = multiPages_.size();
  auto epoch = getFileMgrEpoch();
  for (size_t pageNum = numCurrentPages; pageNum < numPagesRequested; ++pageNum) {
    Page page = addNewMultiPage(epoch);
    writeHeader(page, pageNum, epoch);
  }
}

namespace {
constexpr uint32_t kNativeStorageCompressionMagic{0x48435331};  // "HCS1"
constexpr uint32_t kNativeStorageCompressionVersion{1};
constexpr uint32_t kNativeStorageCompressionNone{0};
constexpr uint32_t kNativeStorageCompressionLz4{1};
constexpr uint32_t kNativeStorageCompressionSnappy{2};
constexpr uint32_t kNativeStorageCompressionGdeflate{3};
constexpr uint32_t kNativeStorageCompressionBitcompSparse{4};
constexpr uint32_t kNativeStorageCompressionBitcompDefault{5};
constexpr size_t kNativeStorageCompressionSnappyMaxFrameSize{1ULL << 24};
constexpr size_t kNativeStorageCompressionGdeflateMaxFrameSize{1ULL << 16};
constexpr size_t kNativeStorageCompressionBitcompMaxFrameSize{1ULL << 24};
constexpr size_t kNativeStorageCompressionHeaderBytes{sizeof(uint32_t) * 3 +
                                                      sizeof(uint64_t) * 4};

std::string normalize_native_storage_compression_codec_name(const std::string& codec) {
  auto lower_codec = codec;
  std::transform(
      lower_codec.begin(), lower_codec.end(), lower_codec.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  return lower_codec;
}

bool is_adaptive_native_storage_compression_codec(const std::string& codec) {
  const auto normalized_codec = normalize_native_storage_compression_codec_name(codec);
  return normalized_codec == "adaptive" || normalized_codec == "auto";
}

template <typename T>
void write_pod(FILE* f, const T value) {
  CHECK_EQ(fwrite(reinterpret_cast<const int8_t*>(&value), sizeof(T), 1, f), size_t(1));
}

template <typename T>
bool read_pod(FILE* f, T& value) {
  return fread(reinterpret_cast<int8_t*>(&value), sizeof(T), 1, f) == 1;
}

uint32_t native_storage_compression_codec_from_name(const std::string& codec) {
  const auto lower_codec = normalize_native_storage_compression_codec_name(codec);
  if (lower_codec == "none" || lower_codec.empty()) {
    return kNativeStorageCompressionNone;
  }
  if (lower_codec == "lz4") {
    return kNativeStorageCompressionLz4;
  }
  if (lower_codec == "snappy") {
    return kNativeStorageCompressionSnappy;
  }
  if (lower_codec == "gdeflate") {
#ifdef HAVE_NVCOMP_GDEFLATE
    return kNativeStorageCompressionGdeflate;
#else
    throw std::invalid_argument(
        "Native GDeflate storage is unavailable because this build does not include "
        "the nvCOMP CPU codec library");
#endif
  }
  if (lower_codec == "bitcomp" || lower_codec == "bitcomp-sparse") {
#ifdef HAVE_NVCOMP_BITCOMP
    return kNativeStorageCompressionBitcompSparse;
#else
    throw std::invalid_argument(
        "Native Bitcomp storage is unavailable because this build does not include "
        "nvCOMP Bitcomp support");
#endif
  }
  if (lower_codec == "bitcomp-default") {
#ifdef HAVE_NVCOMP_BITCOMP
    return kNativeStorageCompressionBitcompDefault;
#else
    throw std::invalid_argument(
        "Native Bitcomp storage is unavailable because this build does not include "
        "nvCOMP Bitcomp support");
#endif
  }
  throw std::invalid_argument("Unsupported native storage compression codec: " + codec);
}

std::string native_storage_compression_codec_name(const uint32_t codec) {
  switch (codec) {
    case kNativeStorageCompressionNone:
      return "none";
    case kNativeStorageCompressionLz4:
      return "lz4";
    case kNativeStorageCompressionSnappy:
      return "snappy";
    case kNativeStorageCompressionGdeflate:
      return "gdeflate";
    case kNativeStorageCompressionBitcompSparse:
      return "bitcomp-sparse";
    case kNativeStorageCompressionBitcompDefault:
      return "bitcomp-default";
    default:
      throw std::invalid_argument("Unsupported native storage compression codec id: " +
                                  std::to_string(codec));
  }
}

bool native_storage_compression_codec_is_supported(const uint32_t codec) {
  if (codec == kNativeStorageCompressionLz4 || codec == kNativeStorageCompressionSnappy) {
    return true;
  }
#ifdef HAVE_NVCOMP_GDEFLATE
  if (codec == kNativeStorageCompressionGdeflate) {
    return true;
  }
#endif
#ifdef HAVE_NVCOMP_BITCOMP
  if (codec == kNativeStorageCompressionBitcompSparse ||
      codec == kNativeStorageCompressionBitcompDefault) {
    return true;
  }
#endif
  return false;
}

bool native_storage_compression_frame_size_is_supported(const uint32_t codec,
                                                        const size_t frame_size) {
  if (codec == kNativeStorageCompressionLz4) {
    return frame_size <= static_cast<size_t>(LZ4_MAX_INPUT_SIZE);
  }
  if (codec == kNativeStorageCompressionSnappy) {
    return frame_size <= kNativeStorageCompressionSnappyMaxFrameSize;
  }
  if (codec == kNativeStorageCompressionGdeflate) {
    return frame_size <= kNativeStorageCompressionGdeflateMaxFrameSize;
  }
  if (codec == kNativeStorageCompressionBitcompSparse ||
      codec == kNativeStorageCompressionBitcompDefault) {
    return frame_size <= kNativeStorageCompressionBitcompMaxFrameSize;
  }
  return false;
}

#ifdef HAVE_NVCOMP_BITCOMP
struct BitcompPlanDeleter {
  void operator()(bitcompContext* plan) const noexcept {
    if (plan) {
      const auto status = bitcompDestroyPlan(plan);
      if (status != BITCOMP_SUCCESS) {
        LOG(ERROR) << "Failed to destroy native storage Bitcomp plan: status="
                   << static_cast<int>(status);
      }
    }
  }
};

using BitcompPlan = std::unique_ptr<bitcompContext, BitcompPlanDeleter>;

bitcompDataType_t native_storage_bitcomp_type(const SQLTypeInfo& sql_type,
                                              const size_t logical_size,
                                              const size_t frame_size) {
  const auto width = sql_type.get_size();
  if (width <= 0 || logical_size % static_cast<size_t>(width) != 0 ||
      frame_size % static_cast<size_t>(width) != 0) {
    return BITCOMP_UNSIGNED_8BIT;
  }
  switch (width) {
    case 1:
      return BITCOMP_UNSIGNED_8BIT;
    case 2:
      return BITCOMP_UNSIGNED_16BIT;
    case 4:
      return BITCOMP_UNSIGNED_32BIT;
    case 8:
      return BITCOMP_UNSIGNED_64BIT;
    default:
      return BITCOMP_UNSIGNED_8BIT;
  }
}

BitcompPlan make_native_storage_bitcomp_plan(const size_t uncompressed_size,
                                             const bitcompDataType_t data_type,
                                             const bitcompAlgorithm_t algorithm) {
  bitcompHandle_t raw_plan{};
  const auto status = bitcompCreatePlan(
      &raw_plan, uncompressed_size, data_type, BITCOMP_LOSSLESS, algorithm);
  if (status != BITCOMP_SUCCESS) {
    throw std::runtime_error("Failed to create native storage Bitcomp plan: status=" +
                             std::to_string(static_cast<int>(status)));
  }
  return BitcompPlan{raw_plan};
}
#endif

size_t checked_size_add(const size_t lhs,
                        const size_t rhs,
                        const char* const description) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(description);
  }
  return lhs + rhs;
}

size_t checked_size_multiply(const size_t lhs,
                             const size_t rhs,
                             const char* const description) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(description);
  }
  return lhs * rhs;
}

size_t checked_page_offset(const size_t page_num,
                           const size_t page_size,
                           const size_t within_page_offset,
                           const char* const description) {
  return checked_size_add(checked_size_multiply(page_num, page_size, description),
                          within_page_offset,
                          description);
}

size_t ceil_divide(const size_t value,
                   const size_t divisor,
                   const char* const description) {
  if (divisor == 0) {
    throw std::invalid_argument(description);
  }
  return value / divisor + (value % divisor != 0);
}

size_t native_storage_compression_metadata_size(const size_t frame_count) {
  return checked_size_add(
      kNativeStorageCompressionHeaderBytes,
      checked_size_multiply(frame_count,
                            sizeof(uint64_t),
                            "Native storage compression metadata size overflow"),
      "Native storage compression metadata size overflow");
}

struct NativeStorageCompressionFrames {
  std::vector<int8_t> payload;
  std::vector<size_t> frame_sizes;
};

NativeStorageCompressionFrames compress_native_storage_cpu_frames(
    const int8_t* const src,
    const size_t num_bytes,
    const size_t frame_size,
    const uint32_t codec,
    const ChunkKey& chunk_key) {
  CHECK(src);
  CHECK_GT(num_bytes, size_t(0));
  CHECK_GT(frame_size, size_t(0));
  CHECK(codec == kNativeStorageCompressionLz4 ||
        codec == kNativeStorageCompressionSnappy);

  const auto frame_count =
      ceil_divide(num_bytes, frame_size, "Invalid native compression frame size");
  size_t frame_capacity{0};
  if (codec == kNativeStorageCompressionLz4) {
    CHECK_LE(frame_size, static_cast<size_t>(LZ4_MAX_INPUT_SIZE));
    const auto lz4_capacity = LZ4_compressBound(static_cast<int>(frame_size));
    CHECK_GT(lz4_capacity, 0);
    frame_capacity = static_cast<size_t>(lz4_capacity);
  } else {
    frame_capacity = snappy::MaxCompressedLength(frame_size);
  }

  NativeStorageCompressionFrames compressed_frames;
  compressed_frames.payload.resize(checked_size_multiply(
      frame_count, frame_capacity, "Compressed payload size overflow"));
  compressed_frames.frame_sizes.resize(frame_count);

  const auto compress_frames = [&](const threading::blocked_range<size_t>& range) {
    for (auto frame_idx = range.begin(); frame_idx < range.end(); ++frame_idx) {
      const auto input_offset = checked_size_multiply(
          frame_idx, frame_size, "Native compression input offset overflow");
      const auto frame_uncompressed_size = std::min(frame_size, num_bytes - input_offset);
      CHECK(native_storage_compression_frame_size_is_supported(codec,
                                                               frame_uncompressed_size));
      auto* const output = compressed_frames.payload.data() + frame_idx * frame_capacity;
      size_t compressed_bytes{0};
      if (codec == kNativeStorageCompressionLz4) {
        const auto lz4_compressed_bytes =
            LZ4_compress_default(reinterpret_cast<const char*>(src + input_offset),
                                 reinterpret_cast<char*>(output),
                                 static_cast<int>(frame_uncompressed_size),
                                 static_cast<int>(frame_capacity));
        CHECK_GT(lz4_compressed_bytes, 0)
            << "Failed to compress native storage LZ4 frame for chunk "
            << show_chunk(chunk_key);
        compressed_bytes = static_cast<size_t>(lz4_compressed_bytes);
      } else {
        snappy::RawCompress(reinterpret_cast<const char*>(src + input_offset),
                            frame_uncompressed_size,
                            reinterpret_cast<char*>(output),
                            &compressed_bytes);
        CHECK_GT(compressed_bytes, size_t(0))
            << "Failed to compress native storage Snappy frame for chunk "
            << show_chunk(chunk_key);
      }
      CHECK_LE(compressed_bytes, frame_capacity);
      compressed_frames.frame_sizes[frame_idx] = compressed_bytes;
    }
  };

  constexpr size_t kParallelCompressionMinBytes = 1U << 20;
  const threading::blocked_range<size_t> frame_range(0, frame_count);
  if (num_bytes >= kParallelCompressionMinBytes && frame_count > 1) {
    threading::parallel_for(frame_range, compress_frames);
  } else {
    compress_frames(frame_range);
  }

  size_t compacted_size{0};
  for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
    const auto frame_output_offset = frame_idx * frame_capacity;
    const auto compressed_bytes = compressed_frames.frame_sizes[frame_idx];
    if (compacted_size != frame_output_offset) {
      std::memmove(compressed_frames.payload.data() + compacted_size,
                   compressed_frames.payload.data() + frame_output_offset,
                   compressed_bytes);
    }
    compacted_size = checked_size_add(
        compacted_size, compressed_bytes, "Compressed payload size overflow");
  }
  compressed_frames.payload.resize(compacted_size);
  return compressed_frames;
}

#ifdef HAVE_NVCOMP_BITCOMP
struct NativeStorageCompressionCandidate {
  uint32_t codec;
  std::vector<int8_t> payload;
  std::vector<size_t> frame_sizes;
};

NativeStorageCompressionCandidate compress_native_storage_candidate(
    const int8_t* const src,
    const size_t num_bytes,
    const size_t frame_size,
    const uint32_t codec,
    const SQLTypeInfo& sql_type,
    const ChunkKey& chunk_key) {
  CHECK(src);
  CHECK_GT(num_bytes, size_t(0));
  CHECK_GT(frame_size, size_t(0));
  CHECK(codec == kNativeStorageCompressionSnappy ||
        codec == kNativeStorageCompressionBitcompDefault);

  NativeStorageCompressionCandidate candidate{codec};
  if (codec == kNativeStorageCompressionSnappy) {
    auto compressed_frames =
        compress_native_storage_cpu_frames(src, num_bytes, frame_size, codec, chunk_key);
    candidate.payload = std::move(compressed_frames.payload);
    candidate.frame_sizes = std::move(compressed_frames.frame_sizes);
    return candidate;
  }

  candidate.frame_sizes.reserve(num_bytes / frame_size + (num_bytes % frame_size != 0));
  size_t input_offset{0};
  const auto bitcomp_data_type =
      native_storage_bitcomp_type(sql_type, num_bytes, frame_size);
  std::map<size_t, BitcompPlan> plans;
  while (input_offset < num_bytes) {
    const auto frame_uncompressed_size = std::min(frame_size, num_bytes - input_offset);
    auto plan_it = plans.find(frame_uncompressed_size);
    if (plan_it == plans.end()) {
      plan_it = plans
                    .emplace(frame_uncompressed_size,
                             make_native_storage_bitcomp_plan(frame_uncompressed_size,
                                                              bitcomp_data_type,
                                                              BITCOMP_DEFAULT_ALGO))
                    .first;
    }
    std::vector<int8_t> frame_output(bitcompMaxBuflen(frame_uncompressed_size));
    const auto compression_status = bitcompHostCompressLossless(
        plan_it->second.get(), src + input_offset, frame_output.data());
    if (compression_status != BITCOMP_SUCCESS) {
      throw std::runtime_error(
          "Failed to compress native storage Bitcomp frame for chunk " +
          show_chunk(chunk_key) +
          ": status=" + std::to_string(static_cast<int>(compression_status)));
    }
    size_t frame_compressed_size{};
    const auto size_status =
        bitcompGetCompressedSize(frame_output.data(), &frame_compressed_size);
    if (size_status != BITCOMP_SUCCESS || frame_compressed_size == 0 ||
        frame_compressed_size > frame_output.size()) {
      throw std::runtime_error(
          "Invalid compressed native storage Bitcomp frame for chunk " +
          show_chunk(chunk_key) +
          ": status=" + std::to_string(static_cast<int>(size_status)) +
          " compressed_bytes=" + std::to_string(frame_compressed_size) +
          " capacity=" + std::to_string(frame_output.size()));
    }
    const auto padding = (alignof(uint64_t) - frame_compressed_size % alignof(uint64_t)) %
                         alignof(uint64_t);
    const auto stored_bytes = checked_size_add(
        frame_compressed_size, padding, "Compressed Bitcomp frame size overflow");
    const auto output_offset = candidate.payload.size();
    candidate.payload.resize(
        checked_size_add(
            output_offset, stored_bytes, "Compressed Bitcomp payload size overflow"),
        0);
    std::memcpy(candidate.payload.data() + output_offset,
                frame_output.data(),
                frame_compressed_size);
    candidate.frame_sizes.push_back(stored_bytes);
    input_offset += frame_uncompressed_size;
  }
  return candidate;
}
#endif

size_t calculate_buffer_header_size(size_t chunk_size) {
  // Additional 3 * sizeof(int32_t) is for headerSize, pageId, and versionEpoch
  size_t header_size = checked_size_multiply(
      checked_size_add(chunk_size, 3, "FileBuffer header size overflow"),
      sizeof(int32_t),
      "FileBuffer header size overflow");
  size_t header_mod = header_size % FileBuffer::kHeaderBufferOffset;
  if (header_mod > 0) {
    header_size = checked_size_add(header_size,
                                   FileBuffer::kHeaderBufferOffset - header_mod,
                                   "FileBuffer header size overflow");
  }
  return header_size;
}
}  // namespace

NativeStorageCompressionConfig configured_native_storage_compression() {
  return NativeStorageCompressionConfig{g_enable_native_storage_compression,
                                        g_native_storage_compression_codec,
                                        g_native_storage_compression_frame_size,
                                        g_native_storage_compression_gdeflate_level};
}

void validate_native_storage_compression_config(
    const NativeStorageCompressionConfig& config) {
  if (is_adaptive_native_storage_compression_codec(config.codec)) {
#ifdef HAVE_NVCOMP_BITCOMP
    if (config.frame_size == 0 ||
        !native_storage_compression_frame_size_is_supported(
            kNativeStorageCompressionSnappy, config.frame_size) ||
        !native_storage_compression_frame_size_is_supported(
            kNativeStorageCompressionBitcompDefault, config.frame_size)) {
      throw std::invalid_argument(
          "Unsupported native storage compression frame size for adaptive codec: " +
          std::to_string(config.frame_size));
    }
    return;
#else
    throw std::invalid_argument(
        "Adaptive native storage compression requires nvCOMP Bitcomp support");
#endif
  }
  const auto codec = native_storage_compression_codec_from_name(config.codec);
  if (codec == kNativeStorageCompressionNone) {
    return;
  }
  if (config.frame_size == 0 ||
      !native_storage_compression_frame_size_is_supported(codec, config.frame_size)) {
    throw std::invalid_argument(
        "Unsupported native storage compression frame size for "
        "codec '" +
        config.codec + "': " + std::to_string(config.frame_size));
  }
  if (config.enabled && codec == kNativeStorageCompressionGdeflate) {
    if (!g_enable_file_buffer_metadata_sidecar_only) {
      throw std::invalid_argument(
          "Native GDeflate storage requires "
          "--enable-file-buffer-metadata-sidecar-only=true");
    }
    if (config.gdeflate_level < 0 || config.gdeflate_level > 12) {
      throw std::invalid_argument(
          "Native GDeflate compression level must be between 0 and 12");
    }
  }
}

void FileBuffer::setBufferHeaderSize() {
  reservedHeaderSize_ = calculate_buffer_header_size(chunkKey_.size());
}

bool FileBuffer::isStorageCompressed() const {
  return storageCompressionCodec_ != kNativeStorageCompressionNone;
}

bool FileBuffer::isLz4StorageCompressed() const {
  return storageCompressionCodec_ == kNativeStorageCompressionLz4;
}

bool FileBuffer::isSnappyStorageCompressed() const {
  return storageCompressionCodec_ == kNativeStorageCompressionSnappy;
}

bool FileBuffer::isGdeflateStorageCompressed() const {
  return storageCompressionCodec_ == kNativeStorageCompressionGdeflate;
}

bool FileBuffer::isBitcompStorageCompressed() const {
  return storageCompressionCodec_ == kNativeStorageCompressionBitcompSparse ||
         storageCompressionCodec_ == kNativeStorageCompressionBitcompDefault;
}

bool FileBuffer::isBitcompSparseStorageCompressed() const {
  return storageCompressionCodec_ == kNativeStorageCompressionBitcompSparse;
}

size_t FileBuffer::storageBitcompElementWidth() const {
  CHECK(isBitcompStorageCompressed());
  const auto width = sql_type_.get_size();
  if (width <= 0 || size_ % static_cast<size_t>(width) != 0 ||
      storageCompressionFrameSize_ % static_cast<size_t>(width) != 0) {
    return 1;
  }
  switch (width) {
    case 1:
    case 2:
    case 4:
    case 8:
      return static_cast<size_t>(width);
    default:
      return 1;
  }
}

size_t FileBuffer::storageCompressedSize() const {
  return compressedSize_;
}

size_t FileBuffer::storageCompressionFrameSize() const {
  return storageCompressionFrameSize_;
}

const std::vector<size_t>& FileBuffer::storageCompressedFrameSizes() const {
  return compressedFrameSizes_;
}

StorageRewriteStats FileBuffer::rewriteStoragePayload(
    const int32_t epoch,
    const NativeStorageCompressionConfig& compression_config) {
  StorageRewriteStats stats;
  stats.chunks_seen = 1;
  stats.logical_bytes = size_;
  stats.old_physical_bytes = isStorageCompressed() ? compressedSize_ : size_;
  if (isStorageCompressed()) {
    stats.chunks_already_compressed = 1;
  }

  std::vector<int8_t> logical_payload;
  if (size_ > 0) {
    logical_payload.resize(size_);
    read(logical_payload.data(), size_, 0, CPU_LEVEL);
  }

  replacePhysicalPayload(logical_payload.empty() ? nullptr : logical_payload.data(),
                         size_,
                         epoch,
                         compression_config);
  pendingStorageCompressionConfig_.reset();
  setUpdated();

  stats.chunks_rewritten = 1;
  stats.new_physical_bytes = isStorageCompressed() ? compressedSize_ : size_;
  return stats;
}

void FileBuffer::clearStorageCompressionMetadata() {
  storageCompressionCodec_ = kNativeStorageCompressionNone;
  storageCompressionFrameSize_ = 0;
  compressedSize_ = 0;
  compressedFrameSizes_.clear();
}

size_t FileBuffer::getMinPageSize() {
  constexpr size_t max_chunk_size{5};
  return calculate_buffer_header_size(max_chunk_size) + 1;
}

void FileBuffer::freePage(const Page& page) {
  freePage(page, false);
}

void FileBuffer::freePage(const Page& page, const bool isRolloff) {
  FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
  CHECK(fileInfo);
  fileInfo->freePage(page.pageNum, isRolloff, getFileMgrEpoch());
}

size_t FileBuffer::freeMetadataPages() {
  size_t num_pages_freed = metadataPages_.pageVersions.size();
  for (auto metaPageIt = metadataPages_.pageVersions.begin();
       metaPageIt != metadataPages_.pageVersions.end();
       ++metaPageIt) {
    freePage(metaPageIt->page, false /* isRolloff */);
  }
  while (metadataPages_.pageVersions.size() > 0) {
    metadataPages_.pop();
  }
  return num_pages_freed;
}

size_t FileBuffer::freeChunkPages() {
  size_t num_pages_freed = multiPages_.size();
  for (auto multiPageIt = multiPages_.begin(); multiPageIt != multiPages_.end();
       ++multiPageIt) {
    for (auto pageIt = multiPageIt->pageVersions.begin();
         pageIt != multiPageIt->pageVersions.end();
         ++pageIt) {
      freePage(pageIt->page, false /* isRolloff */);
    }
  }
  multiPages_.clear();
  return num_pages_freed;
}

size_t FileBuffer::freePages() {
  return freeMetadataPages() + freeChunkPages();
}

void FileBuffer::freePagesBeforeEpochForMultiPage(MultiPage& multiPage,
                                                  const int32_t targetEpoch,
                                                  const int32_t currentEpoch) {
  std::vector<EpochedPage> epochedPagesToFree =
      multiPage.freePagesBeforeEpoch(targetEpoch, currentEpoch);
  for (const auto& epochedPageToFree : epochedPagesToFree) {
    freePage(epochedPageToFree.page, true /* isRolloff */);
  }
}

void FileBuffer::freePagesBeforeEpoch(const int32_t targetEpoch) {
  // This method is only safe to be called within a checkpoint, after the sync and epoch
  // increment where a failure at any point32_t in the process would lead to a safe
  // rollback
  auto currentEpoch = getFileMgrEpoch();
  CHECK_LE(targetEpoch, currentEpoch);
  freePagesBeforeEpochForMultiPage(metadataPages_, targetEpoch, currentEpoch);
  for (auto& multiPage : multiPages_) {
    freePagesBeforeEpochForMultiPage(multiPage, targetEpoch, currentEpoch);
  }
  while (!multiPages_.empty() && multiPages_.back().pageVersions.empty()) {
    multiPages_.pop_back();
  }

  // Check if all buffer pages can be freed
  if (size_ == 0) {
    size_t max_historical_buffer_size{0};
    for (auto& epoch_page : metadataPages_.pageVersions) {
      // Create buffer that is used to get the buffer size at the epoch version
      FileBuffer buffer{fm_, pageSize_, chunkKey_};
      buffer.readMetadata(epoch_page.page);
      max_historical_buffer_size = std::max(max_historical_buffer_size, buffer.size());
    }

    // Free all chunk pages, if none of the old chunk versions has any data
    if (max_historical_buffer_size == 0) {
      freeChunkPages();
    }
  }
}

struct readThreadDS {
  FileMgr* t_fm;       // ptr to FileMgr
  size_t t_startPage;  // start page for the thread
  size_t t_endPage;    // last page for the thread
  int8_t* t_curPtr;    // pointer to the current location of the target for the thread
  size_t t_bytesLeft;  // number of bytes to be read in the thread
  size_t t_startPageOffset;  // offset - used for the first page of the buffer
  bool t_isFirstPage;        // true - for first page of the buffer, false - otherwise
  const std::vector<MultiPage>* multiPages;  // page map owned by FileBuffer
};

static size_t readForThread(FileBuffer* fileBuffer, const readThreadDS threadDS) {
  size_t startPage = threadDS.t_startPage;  // start reading at startPage, including it
  size_t endPage = threadDS.t_endPage;      // stop reading at endPage, not including it
  int8_t* curPtr = threadDS.t_curPtr;
  size_t bytesLeft = threadDS.t_bytesLeft;
  size_t totalBytesRead = 0;
  bool isFirstPage = threadDS.t_isFirstPage;

  // Traverse the logical pages
  for (size_t pageNum = startPage; pageNum < endPage; ++pageNum) {
    CHECK(threadDS.multiPages);
    CHECK((*threadDS.multiPages)[pageNum].pageSize == fileBuffer->pageSize());
    Page page = (*threadDS.multiPages)[pageNum].current().page;

    FileInfo* fileInfo = threadDS.t_fm->getFileInfoForFileId(page.fileId);
    CHECK(fileInfo);

    // Read the page into the destination (dst) buffer at its
    // current (cur) location
    size_t bytesRead = 0;
    if (isFirstPage) {
      bytesRead = fileInfo->read(
          checked_page_offset(page.pageNum,
                              fileBuffer->pageSize(),
                              checked_size_add(threadDS.t_startPageOffset,
                                               fileBuffer->reservedHeaderSize(),
                                               "FileBuffer read offset overflow"),
                              "FileBuffer read offset overflow"),
          min(fileBuffer->pageDataSize() - threadDS.t_startPageOffset, bytesLeft),
          curPtr);
      isFirstPage = false;
    } else {
      bytesRead = fileInfo->read(checked_page_offset(page.pageNum,
                                                     fileBuffer->pageSize(),
                                                     fileBuffer->reservedHeaderSize(),
                                                     "FileBuffer read offset overflow"),
                                 min(fileBuffer->pageDataSize(), bytesLeft),
                                 curPtr);
    }
    curPtr += bytesRead;
    bytesLeft -= bytesRead;
    totalBytesRead += bytesRead;
  }
  CHECK(bytesLeft == 0);

  return (totalBytesRead);
}

std::vector<FileBufferReadSpan> FileBuffer::getReadSpans(const size_t requested_num_bytes,
                                                         const size_t offset) const {
  CHECK(!isStorageCompressed())
      << "Compressed native storage chunks do not expose uncompressed read spans";
  return getPhysicalReadSpans(requested_num_bytes, offset, size_);
}

std::vector<FileBufferReadSpan> FileBuffer::getCompressedReadSpans() const {
  CHECK(isStorageCompressed());
  CHECK_GT(compressedSize_, size_t(0));
  return getPhysicalReadSpans(compressedSize_, 0, compressedSize_);
}

std::vector<FileBufferReadSpan> FileBuffer::getPhysicalReadSpans(
    const size_t requested_num_bytes,
    const size_t offset,
    const size_t physical_size) const {
  CHECK_LE(offset, physical_size) << "Requested physical span offset out of bounds";
  const size_t num_bytes =
      requested_num_bytes == 0 ? physical_size - offset : requested_num_bytes;
  CHECK_LE(num_bytes, physical_size - offset) << "Requested physical span out of bounds";
  if (num_bytes == 0) {
    return {};
  }

  const size_t start_page = offset / pageDataSize_;
  const size_t start_page_offset = offset % pageDataSize_;
  const size_t first_page_capacity = pageDataSize_ - start_page_offset;
  const size_t remaining_bytes =
      num_bytes > first_page_capacity ? num_bytes - first_page_capacity : size_t(0);
  const size_t num_pages_to_read =
      1 + remaining_bytes / pageDataSize_ + (remaining_bytes % pageDataSize_ != 0);
  CHECK_LE(start_page, multiPages_.size()) << "Requested page out of bounds";
  CHECK_LE(num_pages_to_read, multiPages_.size() - start_page)
      << "Requested page out of bounds";

  std::vector<FileBufferReadSpan> spans;
  spans.reserve(num_pages_to_read);

  size_t destination_offset = 0;
  size_t bytes_left = num_bytes;
  for (size_t logical_page = start_page; bytes_left > 0; ++logical_page) {
    const bool is_first_page = logical_page == start_page;
    const size_t page_payload_offset = is_first_page ? start_page_offset : size_t(0);
    const size_t bytes_this_page =
        std::min(pageDataSize_ - page_payload_offset, bytes_left);

    const auto& multi_page = multiPages_[logical_page];
    CHECK_EQ(multi_page.pageSize, pageSize_);
    Page page = multi_page.current().page;
    auto file_info = fm_->getFileInfoForFileId(page.fileId);
    CHECK(file_info);

    FileBufferReadSpan next_span;
    next_span.file_info = file_info;
    next_span.file_offset =
        checked_page_offset(page.pageNum,
                            pageSize_,
                            checked_size_add(reservedHeaderSize_,
                                             page_payload_offset,
                                             "FileBuffer read-span offset overflow"),
                            "FileBuffer read-span offset overflow");
    next_span.destination_offset = destination_offset;
    next_span.width_bytes = bytes_this_page;
    next_span.height = 1;
    next_span.source_pitch = pageSize_;
    next_span.destination_pitch = pageDataSize_;

    const bool full_payload_page =
        page_payload_offset == 0 && bytes_this_page == pageDataSize_;
    if (full_payload_page && !spans.empty()) {
      auto& previous_span = spans.back();
      const bool previous_is_full_payload_run =
          previous_span.width_bytes == pageDataSize_ &&
          previous_span.source_pitch == pageSize_ &&
          previous_span.destination_pitch == pageDataSize_;
      if (previous_is_full_payload_run && previous_span.file_info == file_info &&
          previous_span.file_offset + previous_span.source_pitch * previous_span.height ==
              next_span.file_offset &&
          previous_span.destination_offset +
                  previous_span.destination_pitch * previous_span.height ==
              next_span.destination_offset) {
        ++previous_span.height;
      } else {
        spans.push_back(next_span);
      }
    } else {
      spans.push_back(next_span);
    }

    destination_offset += bytes_this_page;
    bytes_left -= bytes_this_page;
  }
  CHECK_EQ(destination_offset, num_bytes);
  return spans;
}

void FileBuffer::read(int8_t* const dst,
                      const size_t numBytes,
                      const size_t offset,
                      const MemoryLevel dstBufferType,
                      const int32_t deviceId) {
  if (dstBufferType != CPU_LEVEL) {
    LOG(FATAL) << "Unsupported Buffer type";
  }
  readWithReaderThreads(dst, numBytes, offset, fm_->getNumReaderThreads());
}

void FileBuffer::readWithReaderThreads(int8_t* const dst,
                                       const size_t numBytes,
                                       const size_t offset,
                                       const size_t numReaderThreads) {
  CHECK_LE(offset, size_) << "Requested FileBuffer read offset out of bounds";
  const size_t logical_num_bytes = numBytes == 0 ? size_ - offset : numBytes;
  CHECK_LE(logical_num_bytes, size_ - offset)
      << "Requested FileBuffer read range out of bounds";
  if (isStorageCompressed()) {
    readCompressedWithReaderThreads(dst, logical_num_bytes, offset, numReaderThreads);
    return;
  }
  readPhysicalWithReaderThreads(dst, logical_num_bytes, offset, numReaderThreads);
}

void FileBuffer::readCompressedPayloadWithReaderThreads(int8_t* const dst,
                                                        const size_t numReaderThreads) {
  CHECK(isStorageCompressed());
  CHECK_GT(compressedSize_, size_t(0));
  readPhysicalWithReaderThreads(
      dst, compressedSize_, 0, std::max<size_t>(numReaderThreads, size_t(1)));
}

void FileBuffer::readPhysicalWithReaderThreads(int8_t* const dst,
                                               const size_t numBytes,
                                               const size_t offset,
                                               const size_t numReaderThreads) {
  if (numBytes == 0) {
    return;
  }
  CHECK(dst);

  // variable declarations
  size_t startPage = offset / pageDataSize_;
  size_t startPageOffset = offset % pageDataSize_;
  const size_t firstPageCapacity = pageDataSize_ - startPageOffset;
  const size_t remainingBytes =
      numBytes > firstPageCapacity ? numBytes - firstPageCapacity : size_t(0);
  size_t numPagesToRead =
      1 + remainingBytes / pageDataSize_ + (remainingBytes % pageDataSize_ != 0);
  /*
  if (startPage + numPagesToRead > multiPages_.size()) {
      cout << "Start page: " << startPage << endl;
      cout << "Num pages to read: " << numPagesToRead << endl;
      cout << "Num multipages: " << multiPages_.size() << endl;
      cout << "Offset: " << offset << endl;
      cout << "Num bytes: " << numBytes << endl;
  }
  */

  CHECK_LE(startPage, multiPages_.size()) << "Requested page out of bounds";
  CHECK_LE(numPagesToRead, multiPages_.size() - startPage)
      << "Requested page out of bounds";

  size_t numPagesPerThread = 0;
  size_t numBytesCurrent = numBytes;  // total number of bytes still to be read
  size_t bytesRead = 0;               // total number of bytes already being read
  size_t bytesLeftForThread = 0;      // number of bytes to be read in the thread
  size_t numExtraPages = 0;  // extra pages to be assigned one per thread as needed
  CHECK_GT(numReaderThreads, size_t(0));
  size_t numThreads = numReaderThreads;
  std::vector<readThreadDS>
      threadDSArr;  // array of threadDS, needed to avoid racing conditions

  if (numPagesToRead > numThreads) {
    numPagesPerThread = numPagesToRead / numThreads;
    numExtraPages = numPagesToRead - (numThreads * numPagesPerThread);
  } else {
    numThreads = numPagesToRead;
    numPagesPerThread = 1;
  }

  /* set threadDS for the first thread */
  readThreadDS threadDS;
  threadDS.t_fm = fm_;
  threadDS.t_startPage = offset / pageDataSize_;
  if (numExtraPages > 0) {
    threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread + 1;
    numExtraPages--;
  } else {
    threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread;
  }
  threadDS.t_curPtr = dst;
  threadDS.t_startPageOffset = offset % pageDataSize_;
  threadDS.t_isFirstPage = true;

  bytesLeftForThread = min(((threadDS.t_endPage - threadDS.t_startPage) * pageDataSize_ -
                            threadDS.t_startPageOffset),
                           numBytesCurrent);
  threadDS.t_bytesLeft = bytesLeftForThread;
  threadDS.multiPages = &multiPages_;

  if (numThreads == 1) {
    bytesRead += readForThread(this, threadDS);
  } else {
    std::vector<std::future<size_t>> threads;
    threads.reserve(numThreads);
    threadDSArr.reserve(numThreads);

    for (size_t i = 0; i < numThreads; i++) {
      threadDSArr.push_back(threadDS);
      threads.push_back(
          std::async(std::launch::async, readForThread, this, threadDSArr[i]));

      // calculate elements of threadDS
      threadDS.t_fm = fm_;
      threadDS.t_isFirstPage = false;
      threadDS.t_curPtr += bytesLeftForThread;
      threadDS.t_startPage +=
          threadDS.t_endPage -
          threadDS.t_startPage;  // based on # of pages read on previous iteration
      if (numExtraPages > 0) {
        threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread + 1;
        numExtraPages--;
      } else {
        threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread;
      }
      numBytesCurrent -= bytesLeftForThread;
      bytesLeftForThread = min(
          ((threadDS.t_endPage - threadDS.t_startPage) * pageDataSize_), numBytesCurrent);
      threadDS.t_bytesLeft = bytesLeftForThread;
    }

    for (auto& p : threads) {
      p.wait();
    }
    for (auto& p : threads) {
      bytesRead += p.get();
    }
  }
  CHECK(bytesRead == numBytes);
}

void FileBuffer::readCompressedWithReaderThreads(int8_t* const dst,
                                                 const size_t numBytes,
                                                 const size_t offset,
                                                 const size_t numReaderThreads) {
  const auto compression_error = [this](const std::string& detail) {
    return std::runtime_error(detail + " for chunk " + show_chunk(chunkKey_));
  };
  if (!isStorageCompressed()) {
    throw std::logic_error("Compressed read requested for an uncompressed FileBuffer.");
  }
  if (offset > size_ || numBytes > size_ - offset) {
    throw std::out_of_range("Requested compressed chunk logical range is out of bounds.");
  }
  if (!native_storage_compression_codec_is_supported(storageCompressionCodec_)) {
    throw compression_error("Unsupported native storage compression codec");
  }
  size_t expected_compressed_size = 0;
  for (const auto frame_size : compressedFrameSizes_) {
    expected_compressed_size =
        checked_size_add(expected_compressed_size,
                         frame_size,
                         "Native storage compressed frame size overflow");
  }
  if (compressedSize_ == 0 || compressedSize_ != expected_compressed_size ||
      storageCompressionFrameSize_ == 0) {
    throw compression_error("Invalid native storage compression metadata");
  }

  if (numBytes == 0) {
    return;
  }
  CHECK(dst);

  const auto logical_end = offset + numBytes;
  const auto first_frame_idx = offset / storageCompressionFrameSize_;
  const auto last_frame_idx = (logical_end - 1) / storageCompressionFrameSize_;
  CHECK_LT(last_frame_idx, compressedFrameSizes_.size());

  size_t compressed_read_offset = 0;
  for (size_t frame_idx = 0; frame_idx < first_frame_idx; ++frame_idx) {
    compressed_read_offset += compressedFrameSizes_[frame_idx];
  }
  size_t compressed_read_size = 0;
  for (size_t frame_idx = first_frame_idx; frame_idx <= last_frame_idx; ++frame_idx) {
    compressed_read_size += compressedFrameSizes_[frame_idx];
  }
  std::vector<int8_t> compressed(compressed_read_size);
  readPhysicalWithReaderThreads(compressed.data(),
                                compressed.size(),
                                compressed_read_offset,
                                std::max<size_t>(numReaderThreads, 1));

  std::vector<int8_t> frame_buffer;
#ifdef HAVE_NVCOMP_BITCOMP
  std::map<size_t, BitcompPlan> bitcomp_plans;
  const auto bitcomp_data_type =
      native_storage_bitcomp_type(sql_type_, size_, storageCompressionFrameSize_);
#endif
  size_t selected_compressed_offset = 0;
  for (size_t frame_idx = first_frame_idx; frame_idx <= last_frame_idx; ++frame_idx) {
    const auto frame_compressed_size = compressedFrameSizes_[frame_idx];
    const auto frame_logical_offset = frame_idx * storageCompressionFrameSize_;
    if (frame_logical_offset >= size_ || selected_compressed_offset > compressed.size() ||
        frame_compressed_size > compressed.size() - selected_compressed_offset) {
      throw compression_error("Native storage compression frame is out of bounds");
    }
    const auto frame_uncompressed_size =
        std::min(storageCompressionFrameSize_, size_ - frame_logical_offset);
    if (frame_uncompressed_size == 0 ||
        frame_compressed_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw compression_error("Invalid native storage compression frame size");
    }
    const auto frame_logical_end = frame_logical_offset + frame_uncompressed_size;
    const auto overlap_begin = std::max(offset, frame_logical_offset);
    const auto overlap_end = std::min(logical_end, frame_logical_end);
    CHECK_LT(overlap_begin, overlap_end);
    const bool reads_full_frame =
        overlap_begin == frame_logical_offset && overlap_end == frame_logical_end;
    if (!reads_full_frame) {
      frame_buffer.resize(frame_uncompressed_size);
    }
    auto* const frame_output =
        reads_full_frame ? dst + (frame_logical_offset - offset) : frame_buffer.data();
    if (storageCompressionCodec_ == kNativeStorageCompressionLz4) {
      if (frame_uncompressed_size > static_cast<size_t>(LZ4_MAX_INPUT_SIZE)) {
        throw compression_error("Native storage LZ4 frame is too large");
      }
      const auto decompressed_bytes = LZ4_decompress_safe(
          reinterpret_cast<const char*>(compressed.data() + selected_compressed_offset),
          reinterpret_cast<char*>(frame_output),
          static_cast<int>(frame_compressed_size),
          static_cast<int>(frame_uncompressed_size));
      if (decompressed_bytes != static_cast<int>(frame_uncompressed_size)) {
        throw compression_error("Failed to decompress native storage LZ4 frame");
      }
    } else if (storageCompressionCodec_ == kNativeStorageCompressionSnappy) {
      size_t expected_frame_uncompressed_size = 0;
      if (!snappy::GetUncompressedLength(
              reinterpret_cast<const char*>(compressed.data() +
                                            selected_compressed_offset),
              frame_compressed_size,
              &expected_frame_uncompressed_size) ||
          expected_frame_uncompressed_size != frame_uncompressed_size) {
        throw compression_error("Invalid native storage Snappy frame length");
      }
      if (!snappy::RawUncompress(reinterpret_cast<const char*>(
                                     compressed.data() + selected_compressed_offset),
                                 frame_compressed_size,
                                 reinterpret_cast<char*>(frame_output))) {
        throw compression_error("Failed to decompress native storage Snappy frame");
      }
    } else if (storageCompressionCodec_ == kNativeStorageCompressionGdeflate) {
#ifdef HAVE_NVCOMP_GDEFLATE
      const void* input_ptr = compressed.data() + selected_compressed_offset;
      const size_t input_size = frame_compressed_size;
      void* output_ptr = frame_output;
      size_t output_capacity = frame_uncompressed_size;
      size_t output_size = 0;
      try {
        gdeflate::decompressCPU(
            &input_ptr, &input_size, 1, &output_ptr, &output_capacity, &output_size);
      } catch (const std::exception& error) {
        throw compression_error("Failed to decompress native storage GDeflate frame: " +
                                std::string(error.what()));
      }
      if (output_size != frame_uncompressed_size) {
        throw compression_error("Invalid native storage GDeflate frame length");
      }
#else
      throw compression_error("Native GDeflate storage is unavailable in this build");
#endif
    } else if (storageCompressionCodec_ == kNativeStorageCompressionBitcompSparse ||
               storageCompressionCodec_ == kNativeStorageCompressionBitcompDefault) {
#ifdef HAVE_NVCOMP_BITCOMP
      const auto* const frame_input = compressed.data() + selected_compressed_offset;
      if (reinterpret_cast<uintptr_t>(frame_input) % alignof(uint64_t) != 0) {
        throw compression_error("Native storage Bitcomp frame is misaligned");
      }
      size_t encoded_uncompressed_size{};
      const auto size_status =
          bitcompGetUncompressedSize(frame_input, &encoded_uncompressed_size);
      if (size_status != BITCOMP_SUCCESS ||
          encoded_uncompressed_size != frame_uncompressed_size) {
        throw compression_error("Invalid native storage Bitcomp frame length");
      }
      size_t encoded_compressed_size{};
      const auto compressed_size_status =
          bitcompGetCompressedSize(frame_input, &encoded_compressed_size);
      const auto encoded_padding =
          (alignof(uint64_t) - encoded_compressed_size % alignof(uint64_t)) %
          alignof(uint64_t);
      if (compressed_size_status != BITCOMP_SUCCESS || encoded_compressed_size == 0 ||
          encoded_compressed_size > frame_compressed_size ||
          encoded_padding > frame_compressed_size - encoded_compressed_size ||
          encoded_compressed_size + encoded_padding != frame_compressed_size) {
        throw compression_error("Invalid native storage Bitcomp frame padding");
      }
      auto plan_it = bitcomp_plans.find(frame_uncompressed_size);
      if (plan_it == bitcomp_plans.end()) {
        const auto algorithm =
            storageCompressionCodec_ == kNativeStorageCompressionBitcompSparse
                ? BITCOMP_SPARSE_ALGO
                : BITCOMP_DEFAULT_ALGO;
        plan_it = bitcomp_plans
                      .emplace(frame_uncompressed_size,
                               make_native_storage_bitcomp_plan(
                                   frame_uncompressed_size, bitcomp_data_type, algorithm))
                      .first;
      }
      const auto status =
          bitcompHostUncompress(plan_it->second.get(), frame_input, frame_output);
      if (status != BITCOMP_SUCCESS) {
        throw compression_error(
            "Failed to decompress native storage Bitcomp frame: status=" +
            std::to_string(static_cast<int>(status)));
      }
#else
      throw compression_error("Native Bitcomp storage is unavailable in this build");
#endif
    } else {
      throw compression_error("Unsupported native storage compression codec");
    }
    if (!reads_full_frame) {
      std::memcpy(dst + (overlap_begin - offset),
                  frame_buffer.data() + (overlap_begin - frame_logical_offset),
                  overlap_end - overlap_begin);
    }
    selected_compressed_offset += frame_compressed_size;
  }
  CHECK_EQ(selected_compressed_offset, compressed.size());
}

void FileBuffer::copyPage(Page& srcPage,
                          Page& destPage,
                          const size_t numBytes,
                          const size_t offset) {
  CHECK_LE(offset, pageDataSize_);
  CHECK_LE(numBytes, pageDataSize_ - offset);
  FileInfo* srcFileInfo = fm_->getFileInfoForFileId(srcPage.fileId);
  FileInfo* destFileInfo = fm_->getFileInfoForFileId(destPage.fileId);

  int8_t* buffer = reinterpret_cast<int8_t*>(checked_malloc(numBytes));
  ScopeGuard guard = [&buffer] { free(buffer); };
  size_t bytesRead = srcFileInfo->read(
      checked_page_offset(
          srcPage.pageNum,
          pageSize_,
          checked_size_add(
              offset, reservedHeaderSize_, "FileBuffer source page offset overflow"),
          "FileBuffer source page offset overflow"),
      numBytes,
      buffer);
  CHECK(bytesRead == numBytes);
  size_t bytesWritten = destFileInfo->write(
      checked_page_offset(
          destPage.pageNum,
          pageSize_,
          checked_size_add(
              offset, reservedHeaderSize_, "FileBuffer destination page offset overflow"),
          "FileBuffer destination page offset overflow"),
      numBytes,
      buffer);
  CHECK(bytesWritten == numBytes);
}

Page FileBuffer::addNewMultiPage(const int32_t epoch) {
  Page page = fm_->requestFreePage(pageSize_, false);
  MultiPage multiPage(pageSize_);
  multiPage.push(page, epoch);
  multiPages_.emplace_back(multiPage);
  return page;
}

void FileBuffer::writeHeader(Page& page,
                             const int32_t pageId,
                             const int32_t epoch,
                             const bool writeMetadata) {
  int32_t intHeaderSize = chunkKey_.size() + 3;  // does not include chunkSize
  vector<int32_t> header(intHeaderSize);
  // in addition to chunkkey we need size of header, pageId, version
  header[0] =
      (intHeaderSize - 1) * sizeof(int32_t);  // don't need to include size of headerSize
                                              // value - sizeof(size_t) is for chunkSize
  std::copy(chunkKey_.begin(), chunkKey_.end(), header.begin() + 1);
  header[intHeaderSize - 2] = pageId;
  header[intHeaderSize - 1] = epoch;
  FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
  size_t pageSize = writeMetadata ? metadataPageSize_ : pageSize_;
  fileInfo->write(
      checked_page_offset(page.pageNum, pageSize, 0, "FileBuffer header offset overflow"),
      (intHeaderSize) * sizeof(int32_t),
      (int8_t*)&header[0]);
}

void FileBuffer::readMetadata(const Page& page) {
  if (reservedHeaderSize_ > metadataPageSize_) {
    throw std::runtime_error("FileBuffer metadata header exceeds metadata page for " +
                             show_chunk(chunkKey_));
  }
  std::vector<int8_t> payload(metadataPageSize_ - reservedHeaderSize_);
  auto* file_info = fm_->getFileInfoForFileId(page.fileId);
  CHECK(file_info);
  const auto payload_offset = checked_size_add(
      checked_size_multiply(
          page.pageNum, metadataPageSize_, "FileBuffer metadata offset overflow"),
      reservedHeaderSize_,
      "FileBuffer metadata offset overflow");
  CHECK_EQ(file_info->read(payload_offset, payload.size(), payload.data()),
           payload.size());
  readMetadataPayload(payload);
}

void FileBuffer::readMetadataPayload(const std::vector<int8_t>& payload) {
  if (payload.empty()) {
    throw std::runtime_error("Cannot read empty FileBuffer metadata payload for " +
                             show_chunk(chunkKey_));
  }
  FILE* f = fmemopen(const_cast<int8_t*>(payload.data()), payload.size(), "rb");
  if (!f) {
    throw std::runtime_error("Could not open FileBuffer metadata payload for " +
                             show_chunk(chunkKey_));
  }
  ScopeGuard file_guard = [&] { fclose(f); };
  readMetadataPayload(f, payload.size());
}

void FileBuffer::readMetadataPayload(FILE* f, const size_t payload_capacity) {
  if (!f) {
    throw std::invalid_argument("Cannot read FileBuffer metadata from a null stream.");
  }
  const auto metadata_error = [this](const std::string& detail) {
    return std::runtime_error(detail + " for " + fm_->describeSelf());
  };
  const auto payload_start = ftell(f);
  if (payload_start < 0) {
    throw metadata_error("Could not determine FileBuffer metadata position");
  }
  pendingStorageCompressionConfig_.reset();
  clearStorageCompressionMetadata();
  if (fread((int8_t*)&pageSize_, sizeof(size_t), 1, f) != size_t(1)) {
    throw metadata_error("Truncated FileBuffer page size metadata");
  }
  if (fread((int8_t*)&size_, sizeof(size_t), 1, f) != size_t(1)) {
    throw metadata_error("Truncated FileBuffer size metadata");
  }
  vector<int32_t> typeData(
      NUM_METADATA);  // assumes we will encode hasEncoder, bufferType,
                      // encodingType, encodingBits all as int
  if (fread((int8_t*)&(typeData[0]), sizeof(int32_t), typeData.size(), f) !=
      typeData.size()) {
    throw metadata_error("Truncated FileBuffer type metadata");
  }
  int32_t disk_version = typeData[0];
  auto system_version = Encoder::metadata_version_;
  if (disk_version < Encoder::MetadataVersion::kBase) {
    throw metadata_error("Encountered invalid metadata version " +
                         std::to_string(disk_version));
  }
  if (disk_version > system_version) {
    throw metadata_error("Encountered unsupported metadata version " +
                         std::to_string(disk_version) + ". Current version is " +
                         std::to_string(system_version));
  }
  bool has_encoder = static_cast<bool>(typeData[1]);
  if (has_encoder) {
    sql_type_.set_type(static_cast<SQLTypes>(typeData[2]));
    sql_type_.set_subtype(static_cast<SQLTypes>(typeData[3]));
    sql_type_.set_dimension(typeData[4]);
    sql_type_.set_scale(typeData[5]);
    sql_type_.set_notnull(static_cast<bool>(typeData[6]));
    sql_type_.set_compression(static_cast<EncodingType>(typeData[7]));
    sql_type_.set_comp_param(typeData[8]);
    sql_type_.set_size(typeData[9]);
    initEncoder(sql_type_);
    encoder_->readMetadata(f, disk_version);
  }

  uint32_t compression_magic = 0;
  if (!read_pod(f, compression_magic)) {
    if (disk_version >= Encoder::MetadataVersion::kNativeStorageCompression) {
      throw metadata_error("Missing native storage compression metadata");
    }
    return;
  }
  if (disk_version < Encoder::MetadataVersion::kNativeStorageCompression &&
      compression_magic != kNativeStorageCompressionMagic) {
    return;
  }
  if (compression_magic == 0) {
    // Early versions of the native-compression branch advanced the metadata version
    // for every chunk and wrote a zero marker for uncompressed payloads. Continue to
    // read those chunks, while writing all new uncompressed metadata in kRaster format.
    return;
  }
  if (compression_magic != kNativeStorageCompressionMagic) {
    throw metadata_error("Invalid native storage compression metadata");
  }

  uint32_t compression_version = 0;
  uint32_t compression_codec = kNativeStorageCompressionNone;
  uint64_t logical_size = 0;
  uint64_t compressed_size = 0;
  uint64_t frame_uncompressed_size = 0;
  uint64_t frame_count = 0;
  if (!(read_pod(f, compression_version) && read_pod(f, compression_codec) &&
        read_pod(f, logical_size) && read_pod(f, compressed_size) &&
        read_pod(f, frame_uncompressed_size) && read_pod(f, frame_count))) {
    throw metadata_error("Truncated native storage compression metadata");
  }
  if (compression_version != kNativeStorageCompressionVersion) {
    throw metadata_error("Unsupported native storage compression metadata version " +
                         std::to_string(compression_version));
  }
  if (compression_codec == kNativeStorageCompressionNone) {
    clearStorageCompressionMetadata();
    return;
  }
  if (!native_storage_compression_codec_is_supported(compression_codec)) {
    throw metadata_error("Unsupported native storage compression codec " +
                         std::to_string(compression_codec));
  }
  if (logical_size != static_cast<uint64_t>(size_)) {
    throw metadata_error(
        "Native storage compression logical size does not match chunk metadata");
  }
  if (logical_size == 0 || compressed_size == 0 || compressed_size >= logical_size ||
      frame_uncompressed_size == 0 || frame_count == 0) {
    throw metadata_error("Invalid native storage compression sizes");
  }
  if (compressed_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      frame_uncompressed_size >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      frame_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw metadata_error("Native storage compression metadata exceeds host limits");
  }
  const auto frame_size = static_cast<size_t>(frame_uncompressed_size);
  if (!native_storage_compression_frame_size_is_supported(compression_codec,
                                                          frame_size)) {
    throw metadata_error("Unsupported native storage compression frame size");
  }
  const auto expected_frame_count = size_ / frame_size + (size_ % frame_size != 0);
  if (frame_count != static_cast<uint64_t>(expected_frame_count)) {
    throw metadata_error(
        "Native storage compression frame count does not match logical size");
  }

  const auto frame_table_start = ftell(f);
  if (frame_table_start < payload_start) {
    throw metadata_error("Invalid native storage compression frame table position");
  }
  const auto metadata_bytes_used = static_cast<size_t>(frame_table_start - payload_start);
  if (metadata_bytes_used > payload_capacity) {
    throw metadata_error("Native storage compression header exceeds metadata payload");
  }
  const auto max_frame_count =
      (payload_capacity - metadata_bytes_used) / sizeof(uint64_t);
  if (frame_count > static_cast<uint64_t>(max_frame_count)) {
    throw metadata_error(
        "Native storage compression frame table exceeds metadata payload");
  }

  std::vector<uint64_t> disk_frame_sizes(static_cast<size_t>(frame_count));
  if (fread(disk_frame_sizes.data(), sizeof(uint64_t), disk_frame_sizes.size(), f) !=
      disk_frame_sizes.size()) {
    throw metadata_error("Truncated native storage compression frame table");
  }

  std::vector<size_t> compressed_frame_sizes;
  compressed_frame_sizes.reserve(disk_frame_sizes.size());
  uint64_t compressed_size_sum = 0;
  size_t logical_offset = 0;
  for (const auto frame_size_on_disk : disk_frame_sizes) {
    if (frame_size_on_disk == 0 || compressed_size_sum > compressed_size ||
        frame_size_on_disk > compressed_size - compressed_size_sum) {
      throw metadata_error(
          "Native storage compression frame sizes exceed compressed size");
    }
    const auto logical_frame_size = std::min(frame_size, size_ - logical_offset);
    uint64_t max_compressed_frame_size = 0;
    if (compression_codec == kNativeStorageCompressionLz4) {
      const auto compression_bound =
          LZ4_compressBound(static_cast<int>(logical_frame_size));
      if (compression_bound <= 0) {
        throw metadata_error("Invalid native LZ4 frame size");
      }
      max_compressed_frame_size = static_cast<uint64_t>(compression_bound);
    } else if (compression_codec == kNativeStorageCompressionGdeflate) {
#ifdef HAVE_NVCOMP_GDEFLATE
      size_t compression_bound = 0;
      try {
        gdeflate::compressCPUGetMaxOutputChunkSize(logical_frame_size,
                                                   &compression_bound);
      } catch (const std::exception& error) {
        throw metadata_error("Invalid native GDeflate frame size: " +
                             std::string(error.what()));
      }
      max_compressed_frame_size = compression_bound;
#else
      throw metadata_error("Native GDeflate storage is unavailable in this build");
#endif
    } else if (compression_codec == kNativeStorageCompressionBitcompSparse ||
               compression_codec == kNativeStorageCompressionBitcompDefault) {
#ifdef HAVE_NVCOMP_BITCOMP
      max_compressed_frame_size = bitcompMaxBuflen(logical_frame_size);
      if (frame_size_on_disk % alignof(uint64_t) != 0) {
        throw metadata_error("Native storage Bitcomp frame size is misaligned");
      }
#else
      throw metadata_error("Native Bitcomp storage is unavailable in this build");
#endif
    } else {
      max_compressed_frame_size =
          static_cast<uint64_t>(snappy::MaxCompressedLength(logical_frame_size));
    }
    if (frame_size_on_disk > max_compressed_frame_size) {
      throw metadata_error("Native storage compression frame exceeds codec bound");
    }
    compressed_frame_sizes.push_back(static_cast<size_t>(frame_size_on_disk));
    compressed_size_sum += frame_size_on_disk;
    logical_offset += logical_frame_size;
  }
  if (compressed_size_sum != compressed_size) {
    throw metadata_error(
        "Native storage compression frame sizes do not sum to compressed size");
  }
  if (logical_offset != size_) {
    throw metadata_error(
        "Native storage compression frames do not cover the logical chunk");
  }

  storageCompressionCodec_ = compression_codec;
  compressedSize_ = static_cast<size_t>(compressed_size);
  storageCompressionFrameSize_ = frame_size;
  compressedFrameSizes_ = std::move(compressed_frame_sizes);
}

void FileBuffer::writeMetadataBasePayload(FILE* f, const int32_t metadata_version) const {
  CHECK_GE(metadata_version, Encoder::MetadataVersion::kBase);
  CHECK_LE(metadata_version, Encoder::metadata_version_);
  CHECK_EQ(fwrite((int8_t*)&pageSize_, sizeof(size_t), 1, f), size_t(1));
  CHECK_EQ(fwrite((int8_t*)&size_, sizeof(size_t), 1, f), size_t(1));
  vector<int32_t> typeData(
      NUM_METADATA);  // assumes we will encode hasEncoder, bufferType,
                      // encodingType, encodingBits all as int32_t
  typeData[0] = metadata_version;
  typeData[1] = static_cast<int32_t>(hasEncoder());
  if (hasEncoder()) {
    typeData[2] = static_cast<int32_t>(sql_type_.get_type());
    typeData[3] = static_cast<int32_t>(sql_type_.get_subtype());
    typeData[4] = sql_type_.get_dimension();
    typeData[5] = sql_type_.get_scale();
    typeData[6] = static_cast<int32_t>(sql_type_.get_notnull());
    typeData[7] = static_cast<int32_t>(sql_type_.get_compression());
    typeData[8] = sql_type_.get_comp_param();
    typeData[9] = sql_type_.get_size();
  }
  CHECK_EQ(fwrite((int8_t*)&(typeData[0]), sizeof(int32_t), typeData.size(), f),
           typeData.size());
  if (hasEncoder()) {  // redundant
    encoder_->writeMetadata(f);
  }
}

void FileBuffer::writeMetadataPayload(FILE* f) const {
  const auto payload_start = ftell(f);
  CHECK_GE(payload_start, 0);
  CHECK_LE(reservedHeaderSize_, metadataPageSize_);
  const auto payload_capacity = canWriteMetadataSidecarOnly()
                                    ? FileMgr::MAX_SIDECAR_METADATA_PAYLOAD_SIZE
                                    : metadataPageSize_ - reservedHeaderSize_;
  writeMetadataBasePayload(f,
                           isStorageCompressed()
                               ? Encoder::MetadataVersion::kNativeStorageCompression
                               : Encoder::MetadataVersion::kRaster);
  if (isStorageCompressed()) {
    CHECK(native_storage_compression_codec_is_supported(storageCompressionCodec_));
    CHECK_GT(storageCompressionFrameSize_, size_t(0));
    CHECK_GT(compressedSize_, size_t(0));
    CHECK(!compressedFrameSizes_.empty());
    const auto current_pos = ftell(f);
    CHECK_GE(current_pos, payload_start);
    const auto bytes_used = static_cast<size_t>(current_pos - payload_start);
    CHECK_LE(bytes_used, payload_capacity)
        << "Native storage compression metadata header exceeds payload capacity";
    const auto bytes_left = payload_capacity - bytes_used;
    CHECK_LE(native_storage_compression_metadata_size(compressedFrameSizes_.size()),
             bytes_left)
        << "Native storage compression metadata exceeds payload capacity";

    write_pod(f, kNativeStorageCompressionMagic);
    write_pod(f, kNativeStorageCompressionVersion);
    write_pod(f, storageCompressionCodec_);
    write_pod(f, static_cast<uint64_t>(size_));
    write_pod(f, static_cast<uint64_t>(compressedSize_));
    write_pod(f, static_cast<uint64_t>(storageCompressionFrameSize_));
    write_pod(f, static_cast<uint64_t>(compressedFrameSizes_.size()));
    for (const auto frame_size : compressedFrameSizes_) {
      write_pod(f, static_cast<uint64_t>(frame_size));
    }
  }
  const auto payload_end = ftell(f);
  CHECK_GE(payload_end, payload_start);
  CHECK_LE(static_cast<size_t>(payload_end - payload_start), payload_capacity)
      << "FileBuffer metadata exceeds payload capacity";
}

std::vector<int8_t> FileBuffer::serializeMetadataPayload() const {
  char* metadata_buffer = nullptr;
  size_t metadata_size = 0;
  FILE* f = open_memstream(&metadata_buffer, &metadata_size);
  CHECK(f);
  ScopeGuard stream_guard = [&] {
    fclose(f);
    free(metadata_buffer);
  };

  writeMetadataPayload(f);
  CHECK_EQ(fflush(f), 0);
  CHECK(metadata_buffer);
  return std::vector<int8_t>(metadata_buffer, metadata_buffer + metadata_size);
}

void FileBuffer::writeMetadata(const int32_t epoch) {
  if (shouldWriteMetadataSidecarOnly()) {
    freeMetadataPages();
    metadataSidecarOnly_ = true;
    return;
  }

  // Right now stats page is size_ (in bytes), bufferType, encodingType,
  // encodingDataType, numElements
  Page page = fm_->requestFreePage(metadataPageSize_, true);
  writeHeader(page, -1, epoch, true);
  auto payload = serializeMetadataPayload();
  CHECK_LE(reservedHeaderSize_, metadataPageSize_);
  const auto payload_capacity = metadataPageSize_ - reservedHeaderSize_;
  if (!isStorageCompressed()) {
    // kRaster readers ignore page padding. Clearing its first word prevents a reused
    // metadata page from retaining the magic of the legacy experimental layout, where
    // compressed metadata followed a kRaster encoder payload without a version bump.
    // A payload with less than one word of padding cannot contain that magic.
    const uint32_t empty_compression_marker{0};
    const auto* marker = reinterpret_cast<const int8_t*>(&empty_compression_marker);
    if (payload.size() <= payload_capacity &&
        sizeof(empty_compression_marker) <= payload_capacity - payload.size()) {
      payload.insert(payload.end(), marker, marker + sizeof(empty_compression_marker));
    }
  }
  CHECK_LE(payload.size(), payload_capacity)
      << "FileBuffer metadata payload does not fit in metadata page";
  auto* file_info = fm_->getFileInfoForFileId(page.fileId);
  CHECK(file_info);
  const auto payload_offset = checked_size_add(
      checked_size_multiply(
          page.pageNum, metadataPageSize_, "FileBuffer metadata offset overflow"),
      reservedHeaderSize_,
      "FileBuffer metadata offset overflow");
  CHECK_EQ(file_info->write(payload_offset, payload.size(), payload.data()),
           payload.size());
  metadataPages_.push(page, epoch);
  metadataSidecarOnly_ = false;
}

bool FileBuffer::canWriteMetadataSidecarOnly() const {
  if (!fm_->hasFileMgrKey()) {
    return false;
  }

  // Metadata pages carry rollback-versioned metadata. The sidecar-only path is
  // current-version metadata, so keep physical metadata pages for rollback tables.
  if (fm_->maxRollbackEpochs() != 0) {
    return false;
  }

  // Once a chunk has been written sidecar-only, disabling the creation flag must not
  // discard the only durable copy of its metadata. A rewrite that enables rollback
  // above deliberately converts it back to a physical metadata page.
  return metadataSidecarOnly_ || g_enable_file_buffer_metadata_sidecar_only;
}

bool FileBuffer::shouldWriteMetadataSidecarOnly() const {
  // A zero-page chunk would be invisible to openFiles() without a metadata page. A
  // replacement payload may still plan for sidecar-only metadata before its data pages
  // have been installed.
  return !multiPages_.empty() && canWriteMetadataSidecarOnly();
}

bool FileBuffer::shouldWriteCompressedPayload(
    const size_t numBytes,
    const NativeStorageCompressionConfig& compression_config) const {
  if (!compression_config.enabled || numBytes == 0) {
    return false;
  }
  const auto adaptive_codec =
      is_adaptive_native_storage_compression_codec(compression_config.codec);
  const auto codec =
      adaptive_codec
          ? kNativeStorageCompressionSnappy
          : native_storage_compression_codec_from_name(compression_config.codec);
  if (codec == kNativeStorageCompressionNone) {
    return false;
  }
  validate_native_storage_compression_config(compression_config);
  if (codec == kNativeStorageCompressionGdeflate && !canWriteMetadataSidecarOnly()) {
    throw std::runtime_error(
        "Native GDeflate storage requires sidecar-only metadata and zero rollback "
        "epochs for " +
        fm_->describeSelf());
  }
  return true;
}

size_t FileBuffer::nativeStorageCompressionMetadataBytesLeft(
    const size_t payload_capacity) const {
  char* metadata_buffer = nullptr;
  size_t metadata_size = 0;
  FILE* f = open_memstream(&metadata_buffer, &metadata_size);
  CHECK(f);
  ScopeGuard stream_guard = [&] {
    fclose(f);
    free(metadata_buffer);
  };

  writeMetadataBasePayload(f, Encoder::MetadataVersion::kNativeStorageCompression);
  CHECK_EQ(fflush(f), 0);
  if (metadata_size >= payload_capacity) {
    return 0;
  }
  return payload_capacity - metadata_size;
}

void FileBuffer::writePhysicalPayload(const int8_t* src,
                                      const size_t numBytes,
                                      const int32_t epoch) {
  CHECK(multiPages_.empty()) << "writePhysicalPayload expects a new physical payload";
  if (numBytes == 0) {
    return;
  }
  CHECK(src);
  size_t bytes_left = numBytes;
  const int8_t* cur_ptr = src;
  const size_t num_pages_to_write =
      ceil_divide(numBytes, pageDataSize_, "Invalid FileBuffer page size");
  for (size_t page_num = 0; page_num < num_pages_to_write; ++page_num) {
    Page page = addNewMultiPage(epoch);
    writeHeader(page, page_num, epoch);
    CHECK(page.fileId >= 0);
    auto file_info = fm_->getFileInfoForFileId(page.fileId);
    const size_t bytes_this_page = std::min(pageDataSize_, bytes_left);
    const size_t bytes_written =
        file_info->write(checked_page_offset(page.pageNum,
                                             pageSize_,
                                             reservedHeaderSize_,
                                             "FileBuffer payload offset overflow"),
                         bytes_this_page,
                         cur_ptr);
    CHECK_EQ(bytes_written, bytes_this_page);
    cur_ptr += bytes_written;
    bytes_left -= bytes_written;
  }
  CHECK_EQ(bytes_left, size_t(0));
}

void FileBuffer::replacePhysicalPages(const int8_t* src,
                                      const size_t numBytes,
                                      const int32_t epoch) {
  std::vector<MultiPage> previous_pages;
  previous_pages.swap(multiPages_);
  try {
    writePhysicalPayload(src, numBytes, epoch);
  } catch (...) {
    // Replacement pages were never made authoritative. Retire any partial allocation
    // and restore the complete previous page set for continued in-process use.
    freeChunkPages();
    previous_pages.swap(multiPages_);
    throw;
  }

  std::vector<MultiPage> replacement_pages;
  replacement_pages.swap(multiPages_);

  if (fm_->maxRollbackEpochs() != 0) {
    // Whole-payload replacement must preserve the same rollback invariant as an
    // ordinary page update. Versions from the current, uncheckpointed epoch can be
    // discarded, but older versions remain addressable until FileMgr rolls them off.
    for (auto& previous_page : previous_pages) {
      while (!previous_page.pageVersions.empty() &&
             previous_page.pageVersions.back().epoch >= epoch) {
        freePage(previous_page.pageVersions.back().page);
        previous_page.pageVersions.pop_back();
      }
    }
    while (!previous_pages.empty() && previous_pages.back().pageVersions.empty()) {
      previous_pages.pop_back();
    }

    const auto shared_page_count =
        std::min(previous_pages.size(), replacement_pages.size());
    for (size_t page_idx = 0; page_idx < shared_page_count; ++page_idx) {
      auto replacement_versions = std::move(replacement_pages[page_idx].pageVersions);
      replacement_pages[page_idx].pageVersions =
          std::move(previous_pages[page_idx].pageVersions);
      for (const auto& replacement_version : replacement_versions) {
        replacement_pages[page_idx].push(replacement_version.page,
                                         replacement_version.epoch);
      }
    }

    // A compressed representation can use fewer physical pages. Retain any older
    // trailing page versions for rollback; current reads are bounded by compressedSize_.
    for (size_t page_idx = replacement_pages.size(); page_idx < previous_pages.size();
         ++page_idx) {
      replacement_pages.emplace_back(pageSize_);
      replacement_pages.back().pageVersions =
          std::move(previous_pages[page_idx].pageVersions);
    }
  } else {
    previous_pages.swap(multiPages_);
    freeChunkPages();
  }
  replacement_pages.swap(multiPages_);
}

void FileBuffer::replacePhysicalPayload(
    const int8_t* src,
    const size_t numBytes,
    const int32_t epoch,
    const NativeStorageCompressionConfig& compression_config) {
  if (shouldWriteCompressedPayload(numBytes, compression_config) &&
      writeCompressedPayload(src, numBytes, epoch, compression_config)) {
    return;
  }

  // Prepare the complete replacement before retiring the old pages. This keeps the
  // live FileBuffer usable after an allocation or write failure and prevents free-page
  // reuse from overwriting the rollback image during replacement.
  replacePhysicalPages(src, numBytes, epoch);
  clearStorageCompressionMetadata();
  size_ = numBytes;
}

bool FileBuffer::writeCompressedPayload(
    const int8_t* src,
    const size_t numBytes,
    const int32_t epoch,
    const NativeStorageCompressionConfig& compression_config) {
  CHECK(shouldWriteCompressedPayload(numBytes, compression_config));
  const auto adaptive_codec =
      is_adaptive_native_storage_compression_codec(compression_config.codec);
  auto codec = adaptive_codec
                   ? kNativeStorageCompressionSnappy
                   : native_storage_compression_codec_from_name(compression_config.codec);
  CHECK(native_storage_compression_codec_is_supported(codec));
  CHECK_LE(reservedHeaderSize_, metadataPageSize_);
  const auto metadata_payload_capacity = canWriteMetadataSidecarOnly()
                                             ? FileMgr::MAX_SIDECAR_METADATA_PAYLOAD_SIZE
                                             : metadataPageSize_ - reservedHeaderSize_;
  const auto metadata_bytes_left =
      nativeStorageCompressionMetadataBytesLeft(metadata_payload_capacity);
  if (metadata_bytes_left <= kNativeStorageCompressionHeaderBytes + sizeof(uint64_t)) {
    VLOG(1) << "Native storage compression metadata would not fit for chunk "
            << show_chunk(chunkKey_) << "; writing uncompressed payload";
    return false;
  }
  const size_t max_frame_count =
      (metadata_bytes_left - kNativeStorageCompressionHeaderBytes) / sizeof(uint64_t);
  CHECK_GT(max_frame_count, size_t(0));

  size_t frame_size = compression_config.frame_size;
  size_t frame_count =
      ceil_divide(numBytes, frame_size, "Invalid native compression frame size");
  if (frame_count > max_frame_count) {
    frame_size =
        ceil_divide(numBytes, max_frame_count, "Invalid native compression frame count");
    const bool may_use_bitcomp = adaptive_codec ||
                                 codec == kNativeStorageCompressionBitcompSparse ||
                                 codec == kNativeStorageCompressionBitcompDefault;
    const auto value_width = sql_type_.get_size();
    if (may_use_bitcomp && value_width > 1 &&
        numBytes % static_cast<size_t>(value_width) == 0) {
      frame_size =
          checked_size_multiply(ceil_divide(frame_size,
                                            static_cast<size_t>(value_width),
                                            "Invalid native compression value width"),
                                static_cast<size_t>(value_width),
                                "Native compression frame size overflow");
    }
    if (!native_storage_compression_frame_size_is_supported(codec, frame_size) ||
        (adaptive_codec && !native_storage_compression_frame_size_is_supported(
                               kNativeStorageCompressionBitcompDefault, frame_size))) {
      return false;
    }
    frame_count =
        ceil_divide(numBytes, frame_size, "Invalid native compression frame size");
    CHECK_LE(frame_count, max_frame_count);
  }

  std::vector<int8_t> compressed;
  std::vector<size_t> frame_sizes;
  frame_sizes.reserve(frame_count);
  size_t input_offset = 0;
  if (adaptive_codec) {
#ifdef HAVE_NVCOMP_BITCOMP
    auto snappy_candidate = compress_native_storage_candidate(
        src, numBytes, frame_size, kNativeStorageCompressionSnappy, sql_type_, chunkKey_);
    auto bitcomp_candidate =
        compress_native_storage_candidate(src,
                                          numBytes,
                                          frame_size,
                                          kNativeStorageCompressionBitcompDefault,
                                          sql_type_,
                                          chunkKey_);
    auto* selected_candidate =
        bitcomp_candidate.payload.size() < snappy_candidate.payload.size()
            ? &bitcomp_candidate
            : &snappy_candidate;
    if (selected_candidate->payload.size() >= numBytes) {
      return false;
    }
    codec = selected_candidate->codec;
    compressed = std::move(selected_candidate->payload);
    frame_sizes = std::move(selected_candidate->frame_sizes);
    input_offset = numBytes;
#else
    UNREACHABLE();
#endif
  } else if (codec == kNativeStorageCompressionGdeflate) {
#ifdef HAVE_NVCOMP_GDEFLATE
    size_t compression_bound = 0;
    gdeflate::compressCPUGetMaxOutputChunkSize(frame_size, &compression_bound);
    CHECK_GT(compression_bound, size_t(0));
    compressed.resize(checked_size_multiply(
        frame_count, compression_bound, "Compressed GDeflate payload size overflow"));
    std::vector<const void*> input_ptrs(frame_count);
    std::vector<size_t> input_sizes(frame_count);
    std::vector<void*> output_ptrs(frame_count);
    frame_sizes.assign(frame_count, compression_bound);
    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
      const auto frame_uncompressed_size = std::min(frame_size, numBytes - input_offset);
      input_ptrs[frame_idx] = src + input_offset;
      input_sizes[frame_idx] = frame_uncompressed_size;
      output_ptrs[frame_idx] = compressed.data() + frame_idx * compression_bound;
      input_offset += frame_uncompressed_size;
    }
    gdeflate::compressCPU(input_ptrs.data(),
                          input_sizes.data(),
                          frame_size,
                          frame_count,
                          output_ptrs.data(),
                          frame_sizes.data(),
                          compression_config.gdeflate_level);

    size_t compacted_size = 0;
    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
      CHECK_GT(frame_sizes[frame_idx], size_t(0));
      CHECK_LE(frame_sizes[frame_idx], compression_bound);
      const auto source_offset = frame_idx * compression_bound;
      if (compacted_size != source_offset) {
        std::memmove(compressed.data() + compacted_size,
                     compressed.data() + source_offset,
                     frame_sizes[frame_idx]);
      }
      compacted_size = checked_size_add(compacted_size,
                                        frame_sizes[frame_idx],
                                        "Compressed GDeflate payload size overflow");
    }
    compressed.resize(compacted_size);
#else
    UNREACHABLE();
#endif
  } else if (codec == kNativeStorageCompressionBitcompSparse ||
             codec == kNativeStorageCompressionBitcompDefault) {
#ifdef HAVE_NVCOMP_BITCOMP
    const auto bitcomp_data_type =
        native_storage_bitcomp_type(sql_type_, numBytes, frame_size);
    const auto bitcomp_algorithm = codec == kNativeStorageCompressionBitcompSparse
                                       ? BITCOMP_SPARSE_ALGO
                                       : BITCOMP_DEFAULT_ALGO;
    std::map<size_t, BitcompPlan> plans;
    while (input_offset < numBytes) {
      const auto frame_uncompressed_size = std::min(frame_size, numBytes - input_offset);
      auto plan_it = plans.find(frame_uncompressed_size);
      if (plan_it == plans.end()) {
        plan_it = plans
                      .emplace(frame_uncompressed_size,
                               make_native_storage_bitcomp_plan(frame_uncompressed_size,
                                                                bitcomp_data_type,
                                                                bitcomp_algorithm))
                      .first;
      }
      std::vector<int8_t> frame_output(bitcompMaxBuflen(frame_uncompressed_size));
      const auto compression_status = bitcompHostCompressLossless(
          plan_it->second.get(), src + input_offset, frame_output.data());
      if (compression_status != BITCOMP_SUCCESS) {
        throw std::runtime_error(
            "Failed to compress native storage Bitcomp frame for chunk " +
            show_chunk(chunkKey_) +
            ": status=" + std::to_string(static_cast<int>(compression_status)));
      }
      size_t compressed_bytes{};
      const auto size_status =
          bitcompGetCompressedSize(frame_output.data(), &compressed_bytes);
      if (size_status != BITCOMP_SUCCESS || compressed_bytes == 0 ||
          compressed_bytes > frame_output.size()) {
        throw std::runtime_error(
            "Invalid compressed native storage Bitcomp frame for chunk " +
            show_chunk(chunkKey_) +
            ": status=" + std::to_string(static_cast<int>(size_status)) +
            " compressed_bytes=" + std::to_string(compressed_bytes) +
            " capacity=" + std::to_string(frame_output.size()));
      }
      const auto padding =
          (alignof(uint64_t) - compressed_bytes % alignof(uint64_t)) % alignof(uint64_t);
      const auto stored_bytes = checked_size_add(
          compressed_bytes, padding, "Compressed Bitcomp frame size overflow");
      const auto output_offset = compressed.size();
      compressed.resize(
          checked_size_add(
              output_offset, stored_bytes, "Compressed Bitcomp payload size overflow"),
          0);
      std::memcpy(
          compressed.data() + output_offset, frame_output.data(), compressed_bytes);
      frame_sizes.push_back(stored_bytes);
      input_offset += frame_uncompressed_size;
    }
#else
    UNREACHABLE();
#endif
  } else {
    auto compressed_frames =
        compress_native_storage_cpu_frames(src, numBytes, frame_size, codec, chunkKey_);
    compressed = std::move(compressed_frames.payload);
    frame_sizes = std::move(compressed_frames.frame_sizes);
    input_offset = numBytes;
  }
  CHECK_EQ(input_offset, numBytes);
  CHECK_EQ(frame_sizes.size(), frame_count);

  if (compressed.size() >= numBytes) {
    return false;
  }

  replacePhysicalPages(compressed.data(), compressed.size(), epoch);
  size_ = numBytes;
  storageCompressionCodec_ = codec;
  storageCompressionFrameSize_ = frame_size;
  compressedSize_ = compressed.size();
  compressedFrameSizes_ = std::move(frame_sizes);
  return true;
}

void FileBuffer::appendToCompressedPayload(const int8_t* src,
                                           const size_t numBytes,
                                           const int32_t epoch) {
  CHECK(isStorageCompressed());
  if (numBytes == 0) {
    return;
  }
  const auto new_size =
      checked_size_add(size_, numBytes, "Compressed FileBuffer append size overflow");
  auto compression_config = currentStorageCompressionConfig();
  std::vector<int8_t> full_image(new_size);
  readCompressedWithReaderThreads(full_image.data(), size_, 0, 1);
  std::memcpy(full_image.data() + size_, src, numBytes);
  replacePhysicalPages(full_image.data(), full_image.size(), epoch);
  clearStorageCompressionMetadata();
  size_ = full_image.size();
  pendingStorageCompressionConfig_.emplace(std::move(compression_config));
}

void FileBuffer::writeToCompressedPayload(const int8_t* src,
                                          const size_t numBytes,
                                          const size_t offset,
                                          const int32_t epoch) {
  CHECK(isStorageCompressed());
  if (numBytes == 0) {
    return;
  }
  const size_t write_end =
      checked_size_add(offset, numBytes, "Compressed FileBuffer write range overflow");
  const size_t new_size = std::max(size_, write_end);
  auto compression_config = currentStorageCompressionConfig();
  std::vector<int8_t> full_image(new_size, 0);
  readCompressedWithReaderThreads(full_image.data(), size_, 0, 1);
  std::memcpy(full_image.data() + offset, src, numBytes);
  replacePhysicalPages(full_image.data(), full_image.size(), epoch);
  clearStorageCompressionMetadata();
  size_ = full_image.size();
  pendingStorageCompressionConfig_.emplace(std::move(compression_config));
}

void FileBuffer::finalizePendingStorageCompression(const int32_t epoch) {
  if (!pendingStorageCompressionConfig_) {
    return;
  }
  CHECK(!isStorageCompressed());
  CHECK(isDirty());

  std::vector<int8_t> logical_payload(size_);
  if (!logical_payload.empty()) {
    readPhysicalWithReaderThreads(
        logical_payload.data(), logical_payload.size(), 0, fm_->getNumReaderThreads());
    writeCompressedPayload(logical_payload.data(),
                           logical_payload.size(),
                           epoch,
                           *pendingStorageCompressionConfig_);
  }
  pendingStorageCompressionConfig_.reset();
}

NativeStorageCompressionConfig FileBuffer::currentStorageCompressionConfig() const {
  CHECK(isStorageCompressed());
  return NativeStorageCompressionConfig{
      true,
      native_storage_compression_codec_name(storageCompressionCodec_),
      storageCompressionFrameSize_,
      g_native_storage_compression_gdeflate_level};
}

void FileBuffer::append(int8_t* src,
                        const size_t numBytes,
                        const MemoryLevel srcBufferType,
                        const int32_t deviceId) {
  CHECK(srcBufferType == CPU_LEVEL) << "Unsupported Buffer type";
  if (numBytes > std::numeric_limits<size_t>::max() - size_) {
    throw std::overflow_error("FileBuffer append size overflow");
  }
  setAppended();

  auto epoch = getFileMgrEpoch();
  if (isStorageCompressed()) {
    appendToCompressedPayload(src, numBytes, epoch);
    return;
  }
  if (size_ == 0) {
    const auto compression_config = configured_native_storage_compression();
    if (shouldWriteCompressedPayload(numBytes, compression_config)) {
      replacePhysicalPayload(src, numBytes, epoch, compression_config);
      return;
    }
  }

  size_t startPage = size_ / pageDataSize_;
  size_t startPageOffset = size_ % pageDataSize_;
  const size_t firstPageCapacity = pageDataSize_ - startPageOffset;
  const size_t remainingBytes =
      numBytes > firstPageCapacity ? numBytes - firstPageCapacity : size_t(0);
  const size_t numPagesToWrite =
      numBytes == 0
          ? 0
          : 1 + remainingBytes / pageDataSize_ + (remainingBytes % pageDataSize_ != 0);
  size_t bytesLeft = numBytes;
  int8_t* curPtr = src;  // a pointer to the current location in dst being written to
  size_t initialNumPages = multiPages_.size();
  size_ = size_ + numBytes;
  for (size_t pageNum = startPage; pageNum < startPage + numPagesToWrite; ++pageNum) {
    Page page;
    if (pageNum >= initialNumPages) {
      page = addNewMultiPage(epoch);
      writeHeader(page, pageNum, epoch);
    } else {
      // we already have a new page at current
      // epoch for this page - just grab this page
      page = multiPages_[pageNum].current().page;
    }
    CHECK(page.fileId >= 0);  // make sure page was initialized
    FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
    size_t bytesWritten;
    if (pageNum == startPage) {
      bytesWritten = fileInfo->write(
          checked_page_offset(page.pageNum,
                              pageSize_,
                              checked_size_add(startPageOffset,
                                               reservedHeaderSize_,
                                               "FileBuffer append offset overflow"),
                              "FileBuffer append offset overflow"),
          min(pageDataSize_ - startPageOffset, bytesLeft),
          curPtr);
    } else {
      bytesWritten =
          fileInfo->write(checked_page_offset(page.pageNum,
                                              pageSize_,
                                              reservedHeaderSize_,
                                              "FileBuffer append offset overflow"),
                          min(pageDataSize_, bytesLeft),
                          curPtr);
    }
    curPtr += bytesWritten;
    bytesLeft -= bytesWritten;
  }
  CHECK(bytesLeft == 0);
}

void FileBuffer::write(int8_t* src,
                       const size_t numBytes,
                       const size_t offset,
                       const MemoryLevel srcBufferType,
                       const int32_t deviceId) {
  CHECK(srcBufferType == CPU_LEVEL) << "Unsupported Buffer type";
  if (numBytes > std::numeric_limits<size_t>::max() - offset) {
    throw std::overflow_error("FileBuffer write range overflow");
  }
  const size_t write_end = offset + numBytes;

  auto epoch = getFileMgrEpoch();

  bool tempIsAppended = false;
  setDirty();
  if (offset < size_) {
    setUpdated();
  }
  if (write_end > size_) {
    tempIsAppended = true;  // because is_appended_ could have already been true - to
                            // avoid rewriting header
    setAppended();
  }

  if (isStorageCompressed()) {
    writeToCompressedPayload(src, numBytes, offset, epoch);
    return;
  }
  if (offset == 0 && size_ == 0) {
    const auto compression_config = configured_native_storage_compression();
    if (shouldWriteCompressedPayload(numBytes, compression_config)) {
      replacePhysicalPayload(src, numBytes, epoch, compression_config);
      return;
    }
  }

  if (tempIsAppended) {
    size_ = write_end;
  }

  size_t startPage = offset / pageDataSize_;
  size_t startPageOffset = offset % pageDataSize_;
  const size_t firstPageCapacity = pageDataSize_ - startPageOffset;
  const size_t remainingBytes =
      numBytes > firstPageCapacity ? numBytes - firstPageCapacity : size_t(0);
  const size_t numPagesToWrite =
      numBytes == 0
          ? 0
          : 1 + remainingBytes / pageDataSize_ + (remainingBytes % pageDataSize_ != 0);
  size_t bytesLeft = numBytes;
  int8_t* curPtr = src;  // a pointer to the current location in dst being written to
  size_t initialNumPages = multiPages_.size();

  if (startPage >
      initialNumPages) {  // means there is a gap we need to allocate pages for
    for (size_t pageNum = initialNumPages; pageNum < startPage; ++pageNum) {
      Page page = addNewMultiPage(epoch);
      writeHeader(page, pageNum, epoch);
    }
  }
  for (size_t pageNum = startPage; pageNum < startPage + numPagesToWrite; ++pageNum) {
    Page page;
    if (pageNum >= initialNumPages) {
      page = addNewMultiPage(epoch);
      writeHeader(page, pageNum, epoch);
    } else if (multiPages_[pageNum].current().epoch <
               epoch) {  // need to create new page b/c this current one lags epoch and we
                         // can't overwrite it also need to copy if we are on first or
                         // last page
      Page lastPage = multiPages_[pageNum].current().page;
      page = fm_->requestFreePage(pageSize_, false);
      multiPages_[pageNum].push(page, epoch);
      if (pageNum == startPage && startPageOffset > 0) {
        // copyPage takes care of header offset so don't worry
        // about it
        copyPage(lastPage, page, startPageOffset, 0);
      }
      if (pageNum == (startPage + numPagesToWrite - 1) &&
          bytesLeft > 0) {  // bytesLeft should always > 0
        copyPage(lastPage,
                 page,
                 pageDataSize_ - bytesLeft,
                 bytesLeft);  // these would be empty if we're appending but we won't
                              // worry about it right now
      }
      writeHeader(page, pageNum, epoch);
    } else {
      // we already have a new page at current
      // epoch for this page - just grab this page
      page = multiPages_[pageNum].current().page;
    }
    CHECK(page.fileId >= 0);  // make sure page was initialized
    FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
    size_t bytesWritten;
    if (pageNum == startPage) {
      bytesWritten = fileInfo->write(
          checked_page_offset(page.pageNum,
                              pageSize_,
                              checked_size_add(startPageOffset,
                                               reservedHeaderSize_,
                                               "FileBuffer write offset overflow"),
                              "FileBuffer write offset overflow"),
          min(pageDataSize_ - startPageOffset, bytesLeft),
          curPtr);
    } else {
      bytesWritten =
          fileInfo->write(checked_page_offset(page.pageNum,
                                              pageSize_,
                                              reservedHeaderSize_,
                                              "FileBuffer write offset overflow"),
                          min(pageDataSize_, bytesLeft),
                          curPtr);
    }
    curPtr += bytesWritten;
    bytesLeft -= bytesWritten;
    if (tempIsAppended && pageNum == startPage + numPagesToWrite - 1) {  // if last page
      //@todo below can lead to undefined - we're overwriting num
      // bytes valid at checkpoint
      writeHeader(page, 0, multiPages_[0].current().epoch, true);
    }
  }
  CHECK(bytesLeft == 0);
}

int32_t FileBuffer::getFileMgrEpoch() {
  auto [db_id, tb_id] = get_table_prefix(chunkKey_);
  return fm_->epoch(db_id, tb_id);
}

std::string FileBuffer::dump() const {
  std::stringstream ss;
  ss << "chunk_key = " << show_chunk(chunkKey_) << "\n";
  ss << "has_encoder = " << (hasEncoder() ? "true\n" : "false\n");
  ss << "size_ = " << size_ << "\n";
  return ss.str();
}

void FileBuffer::initMetadataAndPageDataSize(const std::vector<int8_t>* metadataPayload) {
  if (metadataPayload) {
    readMetadataPayload(*metadataPayload);
    metadataSidecarOnly_ = metadataPages_.pageVersions.empty();
  } else {
    if (metadataPages_.pageVersions.empty()) {
      throw std::runtime_error("Missing durable FileBuffer metadata for chunk " +
                               show_chunk(chunkKey_));
    }
    CHECK(metadataPages_.current().page.fileId != -1);  // was initialized
    readMetadata(metadataPages_.current().page);
    metadataSidecarOnly_ = false;
  }
  if (pageSize_ <= reservedHeaderSize_) {
    throw std::runtime_error("Invalid FileBuffer page size for chunk " +
                             show_chunk(chunkKey_));
  }
  pageDataSize_ = pageSize_ - reservedHeaderSize_;
}

bool FileBuffer::isMissingPages() const {
  // Detect the case where a page is missing by comparing the amount of pages read
  // with the metadata size.
  const auto physical_size = isStorageCompressed() ? compressedSize_ : size();
  const auto expected_page_count =
      ceil_divide(physical_size, pageDataSize_, "Invalid FileBuffer page size");
  if (expected_page_count > multiPages_.size()) {
    return true;
  }
  return std::any_of(multiPages_.begin(),
                     multiPages_.begin() + expected_page_count,
                     [](const auto& page) { return page.pageVersions.empty(); });
}

size_t FileBuffer::numChunkPages() const {
  size_t total_size = 0;
  for (const auto& multi_page : multiPages_) {
    total_size = checked_size_add(
        total_size, multi_page.pageVersions.size(), "FileBuffer page count overflow");
  }
  return total_size;
}
}  // namespace File_Namespace
