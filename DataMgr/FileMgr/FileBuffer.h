/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @file		FileBuffer.h
 * @author		Steven Stewart <steve@map-d.com>
 * @author		Todd Mostak <todd@map-d.com>
 */

#ifndef DATAMGR_MEMORY_FILE_FILEBUFFER_H
#define DATAMGR_MEMORY_FILE_FILEBUFFER_H

#include "../AbstractBuffer.h"
#include "Page.h"

#include <iostream>
#include <stdexcept>

using namespace Data_Namespace;

#define NUM_METADATA 10
#define METADATA_VERSION 0

namespace File_Namespace {

class FileMgr;  // forward declaration

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

 public:
  /**
   * @brief Constructs a FileBuffer object.
   */
  FileBuffer(FileMgr* fm, const size_t pageSize, const ChunkKey& chunkKey, const size_t initialSize = 0);

  FileBuffer(FileMgr* fm,
             const size_t pageSize,
             const ChunkKey& chunkKey,
             const SQLTypeInfo sqlType,
             const size_t initialSize = 0);

  FileBuffer(FileMgr* fm,
             /* const size_t pageSize,*/ const ChunkKey& chunkKey,
             const std::vector<HeaderInfo>::const_iterator& headerStartIt,
             const std::vector<HeaderInfo>::const_iterator& headerEndIt);

  /// Destructor
  virtual ~FileBuffer();

  Page addNewMultiPage(const int epoch);

  void reserve(const size_t numBytes);

  void freePages();

  virtual void read(int8_t* const dst,
                    const size_t numBytes = 0,
                    const size_t offset = 0,
                    const MemoryLevel dstMemoryLevel = CPU_LEVEL,
                    const int deviceId = -1);

  /**
   * @brief Writes the contents of source (src) into new versions of the affected logical pages.
   *
   * This method will write the contents of source (src) into new version of the affected
   * logical pages. New pages are only appended if the value of epoch (in FileMgr)
   *
   */
  virtual void write(int8_t* src,
                     const size_t numBytes,
                     const size_t offset = 0,
                     const MemoryLevel srcMemoryLevel = CPU_LEVEL,
                     const int deviceId = -1);

  virtual void append(int8_t* src,
                      const size_t numBytes,
                      const MemoryLevel srcMemoryLevel = CPU_LEVEL,
                      const int deviceId = -1);
  void copyPage(Page& srcPage, Page& destPage, const size_t numBytes, const size_t offset = 0);
  virtual inline Data_Namespace::MemoryLevel getType() const { return DISK_LEVEL; }

  /// Not implemented for FileMgr -- throws a runtime_error
  virtual int8_t* getMemoryPtr() { LOG(FATAL) << "Operation not supported."; }

  /// Returns the number of pages in the FileBuffer.
  inline virtual size_t pageCount() const { return multiPages_.size(); }

  /// Returns the size in bytes of each page in the FileBuffer.
  inline virtual size_t pageSize() const { return pageSize_; }

  /// Returns the size in bytes of the data portion of each page in the FileBuffer.
  inline virtual size_t pageDataSize() const { return pageDataSize_; }

  /// Returns the size in bytes of the reserved header portion of each page in the FileBuffer.
  inline virtual size_t reservedHeaderSize() const { return reservedHeaderSize_; }

  /// Returns vector of MultiPages in the FileBuffer.
  inline virtual std::vector<MultiPage> getMultiPage() const { return multiPages_; }

  inline virtual size_t size() const { return size_; }

  /// Returns the total number of bytes allocated for the FileBuffer.
  inline virtual size_t reservedSize() const { return multiPages_.size() * pageSize_; }

  /// Returns the total number of used bytes in the FileBuffer.
  // inline virtual size_t used() const {

  /// Returns whether or not the FileBuffer has been modified since the last flush/checkpoint.
  virtual bool isDirty() const { return isDirty_; }

 private:
  // FileBuffer(const FileBuffer&);      // private copy constructor
  // FileBuffer& operator=(const FileBuffer&); // private overloaded assignment operator

  /// Write header writes header at top of page in format
  // headerSize(numBytes), ChunkKey, pageId, version epoch
  // void writeHeader(Page &page, const int pageId, const int epoch, const bool writeSize = false);
  void writeHeader(Page& page, const int pageId, const int epoch, const bool writeMetadata = false);
  void writeMetadata(const int epoch);
  void readMetadata(const Page& page);
  void calcHeaderBuffer();

  FileMgr* fm_;  // a reference to FileMgr is needed for writing to new pages in available files
  static size_t headerBufferOffset_;
  MultiPage metadataPages_;
  std::vector<MultiPage> multiPages_;
  size_t pageSize_;
  size_t pageDataSize_;
  size_t reservedHeaderSize_;  // lets make this a constant now for simplicity - 128 bytes
  ChunkKey chunkKey_;
};

}  // File_Namespace

#endif  // DATAMGR_MEMORY_FILE_FILEBUFFER_H
