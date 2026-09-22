/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file		FileBuffer.h
 * @brief
 *
 */

#pragma once

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/FileMgr/Page.h"

#include <cstdio>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

#include "Logger/Logger.h"

using namespace Data_Namespace;

#define NUM_METADATA 10

namespace File_Namespace {

extern bool g_enable_native_storage_compression;
extern bool g_enable_file_buffer_metadata_sidecar_only;
extern std::string g_native_storage_compression_codec;
extern size_t g_native_storage_compression_frame_size;
extern int g_native_storage_compression_gdeflate_level;

struct NativeStorageCompressionConfig {
  bool enabled{false};
  std::string codec{"none"};
  size_t frame_size{0};
  int gdeflate_level{1};
};

NativeStorageCompressionConfig configured_native_storage_compression();
void validate_native_storage_compression_config(
    const NativeStorageCompressionConfig& config);

// forward declarations
class FileMgr;
class CachingFileMgr;
struct FileInfo;

struct FileBufferReadSpan {
  FileInfo* file_info{nullptr};
  size_t file_offset{0};
  size_t destination_offset{0};
  size_t width_bytes{0};
  size_t height{1};
  size_t source_pitch{0};
  size_t destination_pitch{0};

  size_t bytes() const {
    CHECK(height == 0 || width_bytes <= std::numeric_limits<size_t>::max() / height)
        << "FileBuffer read span size overflow";
    return width_bytes * height;
  }
};

struct StorageRewriteStats {
  size_t chunks_seen{0};
  size_t chunks_rewritten{0};
  size_t chunks_already_compressed{0};
  size_t logical_bytes{0};
  size_t old_physical_bytes{0};
  size_t new_physical_bytes{0};
};

/**
 * @class   FileBuffer
 * @brief   Represents/provides access to contiguous data stored in the file system.
 *
 * The FileBuffer consists of logical pages, which can map to any identically-sized
 * page in any file of the underlying file system. A page's metadata (file and page
 * number) are stored in MultiPage objects, and each MultiPage includes page
 * metadata for multiple versions of the same page.
 *
 * Note that a "Chunk" is brought into a FileBuffer by the FileMgr.
 *
 * Note(s): Forbid Copying Idiom 4.1
 */
class FileBuffer : public AbstractBuffer {
  friend class FileMgr;
  friend class CachingFileMgr;

 public:
  /**
   * @brief Constructs a FileBuffer object.
   */
  FileBuffer(FileMgr* fm,
             const size_t pageSize,
             const ChunkKey& chunkKey,
             const size_t initialSize = 0);

  FileBuffer(FileMgr* fm,
             const size_t pageSize,
             const ChunkKey& chunkKey,
             const SQLTypeInfo sqlType,
             const size_t initialSize = 0);

  FileBuffer(FileMgr* fm,
             /* const size_t pageSize,*/ const ChunkKey& chunkKey,
             const std::vector<HeaderInfo>::const_iterator& headerStartIt,
             const std::vector<HeaderInfo>::const_iterator& headerEndIt,
             const std::vector<int8_t>* metadataPayload = nullptr);

  /// Destructor
  ~FileBuffer() override;

  Page addNewMultiPage(const int32_t epoch);

  void reserve(const size_t numBytes) override;

  size_t freeMetadataPages();
  size_t freeChunkPages();
  size_t freePages();
  void freePagesBeforeEpoch(const int32_t targetEpoch);

  void read(int8_t* const dst,
            const size_t numBytes = 0,
            const size_t offset = 0,
            const MemoryLevel dstMemoryLevel = CPU_LEVEL,
            const int32_t deviceId = -1) override;
  void readWithReaderThreads(int8_t* const dst,
                             const size_t numBytes,
                             const size_t offset,
                             const size_t numReaderThreads);
  void readCompressedPayloadWithReaderThreads(int8_t* const dst,
                                              const size_t numReaderThreads);
  std::vector<FileBufferReadSpan> getReadSpans(const size_t numBytes,
                                               const size_t offset) const;
  std::vector<FileBufferReadSpan> getCompressedReadSpans() const;
  bool isStorageCompressed() const;
  bool isLz4StorageCompressed() const;
  bool isSnappyStorageCompressed() const;
  bool isGdeflateStorageCompressed() const;
  bool isBitcompStorageCompressed() const;
  bool isBitcompSparseStorageCompressed() const;
  size_t storageBitcompElementWidth() const;
  size_t storageCompressedSize() const;
  size_t storageCompressionFrameSize() const;
  const std::vector<size_t>& storageCompressedFrameSizes() const;
  StorageRewriteStats rewriteStoragePayload(
      const int32_t epoch,
      const NativeStorageCompressionConfig& compression_config);

  /**
   * @brief Writes the contents of source (src) into new versions of the affected logical
   * pages.
   *
   * This method will write the contents of source (src) into new version of the affected
   * logical pages. New pages are only appended if the value of epoch (in FileMgr)
   *
   */
  void write(int8_t* src,
             const size_t numBytes,
             const size_t offset = 0,
             const MemoryLevel srcMemoryLevel = CPU_LEVEL,
             const int32_t deviceId = -1) override;

  void append(int8_t* src,
              const size_t numBytes,
              const MemoryLevel srcMemoryLevel = CPU_LEVEL,
              const int32_t deviceId = -1) override;
  void copyPage(Page& srcPage,
                Page& destPage,
                const size_t numBytes,
                const size_t offset = 0);
  inline Data_Namespace::MemoryLevel getType() const override { return DISK_LEVEL; }

  /// Not implemented for FileMgr -- throws a runtime_error
  int8_t* getMemoryPtr() override {
    LOG(FATAL) << "Operation not supported.";
    return nullptr;  // satisfy return-type warning
  }

  /// Returns the number of pages in the FileBuffer.
  inline size_t pageCount() const override { return multiPages_.size(); }

  /// Returns whether or not a buffer has data pages.  It is possible for a buffer to
  /// represent metadata (have a size and encode) but not contain actual data.
  inline bool hasDataPages() const { return pageCount() > 0; }

  /// Returns the size in bytes of each page in the FileBuffer.
  inline size_t pageSize() const override { return pageSize_; }

  /// Returns the size in bytes of the data portion of each page in the FileBuffer.
  inline virtual size_t pageDataSize() const { return pageDataSize_; }

  /// Returns the size in bytes of the reserved header portion of each page in the
  /// FileBuffer.
  inline virtual size_t reservedHeaderSize() const { return reservedHeaderSize_; }

  /// Returns vector of MultiPages in the FileBuffer.
  inline virtual std::vector<MultiPage> getMultiPage() const { return multiPages_; }
  inline MultiPage getMetadataPage() const { return metadataPages_; }

  /// Returns the total number of bytes allocated for the FileBuffer.
  inline size_t reservedSize() const override { return multiPages_.size() * pageSize_; }

  /// Returns the total number of used bytes in the FileBuffer.
  // inline virtual size_t used() const {

  inline size_t numMetadataPages() const { return metadataPages_.pageVersions.size(); };

  bool isMissingPages() const;
  size_t numChunkPages() const;
  std::string dump() const;

  static size_t getMinPageSize();

  // Used for testing
  void freePage(const Page& page);

  static constexpr size_t kHeaderBufferOffset{32};

 private:
  // FileBuffer(const FileBuffer&);      // private copy constructor
  // FileBuffer& operator=(const FileBuffer&); // private overloaded assignment operator

  /// Write header writes header at top of page in format
  // headerSize(numBytes), ChunkKey, pageId, version epoch
  // void writeHeader(Page &page, const int32_t pageId, const int32_t epoch, const bool
  // writeSize = false);
  void writeHeader(Page& page,
                   const int32_t pageId,
                   const int32_t epoch,
                   const bool writeMetadata = false);
  void writeMetadata(const int32_t epoch);
  void readMetadata(const Page& page);
  void writeMetadataBasePayload(FILE* f, int32_t metadata_version) const;
  void writeMetadataPayload(FILE* f) const;
  void readMetadataPayload(FILE* f, size_t payload_capacity);
  void readMetadataPayload(const std::vector<int8_t>& payload);
  std::vector<int8_t> serializeMetadataPayload() const;
  void setBufferHeaderSize();
  void clearStorageCompressionMetadata();
  std::vector<FileBufferReadSpan> getPhysicalReadSpans(const size_t requestedNumBytes,
                                                       const size_t offset,
                                                       const size_t physicalSize) const;
  void readPhysicalWithReaderThreads(int8_t* const dst,
                                     const size_t numBytes,
                                     const size_t offset,
                                     const size_t numReaderThreads);
  void readCompressedWithReaderThreads(int8_t* const dst,
                                       const size_t numBytes,
                                       const size_t offset,
                                       const size_t numReaderThreads);
  void writePhysicalPayload(const int8_t* src,
                            const size_t numBytes,
                            const int32_t epoch);
  void replacePhysicalPages(const int8_t* src,
                            const size_t numBytes,
                            const int32_t epoch);
  void replacePhysicalPayload(const int8_t* src,
                              const size_t numBytes,
                              const int32_t epoch,
                              const NativeStorageCompressionConfig& compression_config);
  void appendToCompressedPayload(const int8_t* src,
                                 const size_t numBytes,
                                 const int32_t epoch);
  void writeToCompressedPayload(const int8_t* src,
                                const size_t numBytes,
                                const size_t offset,
                                const int32_t epoch);
  void finalizePendingStorageCompression(const int32_t epoch);
  bool shouldWriteCompressedPayload(
      const size_t numBytes,
      const NativeStorageCompressionConfig& compression_config) const;
  bool canWriteMetadataSidecarOnly() const;
  bool shouldWriteMetadataSidecarOnly() const;
  bool usesMetadataSidecarOnly() const { return metadataSidecarOnly_; }
  size_t nativeStorageCompressionMetadataBytesLeft(size_t payload_capacity) const;
  bool writeCompressedPayload(const int8_t* src,
                              const size_t numBytes,
                              const int32_t epoch,
                              const NativeStorageCompressionConfig& compression_config);
  NativeStorageCompressionConfig currentStorageCompressionConfig() const;

  void freePage(const Page& page, const bool isRolloff);
  void freePagesBeforeEpochForMultiPage(MultiPage& multiPage,
                                        const int32_t targetEpoch,
                                        const int32_t currentEpoch);
  void initMetadataAndPageDataSize(const std::vector<int8_t>* metadataPayload = nullptr);
  int32_t getFileMgrEpoch();

  FileMgr* fm_;  // a reference to FileMgr is needed for writing to new pages in available
                 // files
  // pageSize_ is non-const because it can be read from a metadata file to create a
  // non-standard page size. metadataPageSize_ is const because we need to know what the
  // metadata page size is in order to know which files are metadata files (and therefore
  // determine page size).  This is set earlier in FileMgr construction.
  const size_t metadataPageSize_;
  MultiPage metadataPages_;
  std::vector<MultiPage> multiPages_;
  size_t pageSize_;
  size_t pageDataSize_;
  size_t reservedHeaderSize_;  // lets make this a constant now for simplicity - 128 bytes
  ChunkKey chunkKey_;
  uint32_t storageCompressionCodec_{0};
  size_t storageCompressionFrameSize_{0};
  size_t compressedSize_{0};
  std::vector<size_t> compressedFrameSizes_;
  std::optional<NativeStorageCompressionConfig> pendingStorageCompressionConfig_;
  bool metadataSidecarOnly_{false};
};

}  // namespace File_Namespace
