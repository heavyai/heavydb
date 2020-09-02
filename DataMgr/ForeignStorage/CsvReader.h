/*
 * Copyright 2020 OmniSci, Inc.
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

#pragma once

#include <boost/filesystem.hpp>

#include "Archive/PosixFileArchive.h"
#include "ImportExport/CopyParams.h"
#include "Shared/mapd_glob.h"
namespace foreign_storage {

// Archive reader for csv archives
// Supports an initial full scan with calls to read()
// When the entire Csv object has been read isScanFinished() returns true
// Previously read data can then be re-read with readRegion()
class CsvReader {
 public:
  CsvReader() = default;
  virtual ~CsvReader() = default;

  /**
   * Read up to max_size bytes from archive into buffer starting
   * starting from the end of the last read
   *
   * @param buffer - buffer to load into
   * @param max_size - maximum number of bytes to read into the buffer
   * @return number of bytes actually read
   */
  virtual size_t read(void* buffer, size_t max_size) = 0;

  /**
   * @return true if the entire CSV has been read
   */
  virtual bool isScanFinished() = 0;

  /**
   * Read up to max_size bytes from archive, starting at given offset
   * isScanFinished() must return true to use readRegion
   *
   * @param buffer - buffer to load into
   * @param offset - starting point into the archive to read
   * @param size - maximum number of bytes to read into the buffer
   * @return number of bytes actually read
   */
  virtual size_t readRegion(void* buffer, size_t offset, size_t size) = 0;

  /**
   * @param size - variable passed by reference is updated with the size of this CSV
   * object
   * @return if size is known
   */
  virtual bool getSize(size_t& size) = 0;
};

// Single uncompressed file, that supports FSEEK for faster random access
class SingleFileReader : public CsvReader {
 public:
  SingleFileReader(const std::string& file_path,
                   const import_export::CopyParams& copy_params);
  ~SingleFileReader() override { fclose(file_); }

  // Delete copy assignment to prevent copying resource pointer
  SingleFileReader(const SingleFileReader&) = delete;
  SingleFileReader& operator=(const SingleFileReader&) = delete;

  size_t read(void* buffer, size_t max_size) override {
    size_t bytes_read = fread(buffer, 1, max_size, file_);
    if (!scan_finished_) {
      scan_finished_ = feof(file_);
    }
    return bytes_read;
  }

  size_t readRegion(void* buffer, size_t offset, size_t size) override {
    CHECK(isScanFinished());
    if (fseek(file_, offset + header_offset_, SEEK_SET) != 0) {
      throw std::runtime_error{"An error occurred when attempting to read offset " +
                               std::to_string(offset) + " in file: \"" + file_path_ +
                               "\". " + strerror(errno)};
    }
    return fread(buffer, 1, size, file_);
  }

  bool isScanFinished() override { return scan_finished_; }

  bool getSize(size_t& size) override {
    size = file_size_;
    return true;
  }

 private:
  std::FILE* file_;
  size_t file_size_;
  std::string file_path_;
  // We've reached the end of the file
  bool scan_finished_;

  // Size of the CSV header in bytes
  size_t header_offset_;
};

// Single archive, does not support random access
class CompressedFileReader : public CsvReader {
 public:
  CompressedFileReader(const std::string& file_path,
                       const import_export::CopyParams& copy_params);

  size_t read(void* buffer, size_t max_size) override;

  size_t readRegion(void* buffer, size_t offset, size_t size) override;

  bool isScanFinished() override { return scan_finished_; }

  bool getSize(size_t& size) override { return false; }

 private:
  /**
   * Reopen file and reset back to the beginning
   */
  void resetArchive();

  /**
   * Go to next archive entry/header with valid data
   */
  void nextEntry();

  /**
   * Skip Header of CSV file
   */
  void skipHeader();

  /**
   * Get the next block from the current archive file
   */
  void fetchBlock();

  /**
   * Skip forward N bytes without reading the data
   * @param n_bytes - number of bytes to skip
   */
  void skipBytes(size_t n_bytes);

  std::string file_path_;
  import_export::CopyParams copy_params_;

  std::unique_ptr<Archive> arch_;
  // Pointer to current uncompressed block from the archive
  const void* current_block_;
  // Number of chars remaining in the current block
  size_t block_chars_remaining_;
  // Overall number of bytes read in the archive (minus headers)
  size_t current_offset_;
  // We've reached the end of the last file
  bool scan_finished_;
};

// Combines several archives into single object
class MultiFileReader : public CsvReader {
 public:
  MultiFileReader();
  bool getSize(size_t& size) override {
    if (size_known_) {
      size = total_size_;
    }
    return size_known_;
  }
  size_t read(void* buffer, size_t max_size) override;

  size_t readRegion(void* buffer, size_t offset, size_t size) override;

  bool isScanFinished() override { return (current_index_ >= files_.size()); }

 protected:
  std::vector<std::unique_ptr<CsvReader>> files_;
  // Total file size if known
  size_t total_size_;
  // If total file size is known
  bool size_known_;

 private:
  /**
   * @param byte_offset byte offset into the fileset from the initial scan
   * @return the file index for a given byte offset
   */
  size_t offsetToIndex(size_t byte_offset);

  // Size of each file + all previous files
  std::vector<size_t> cumulative_sizes_;
  // Current file being read
  size_t current_index_;
  // Overall number of bytes read in the directory (minus headers)
  size_t current_offset_;
};

// Single file or directory with multiple files
class LocalMultiFileReader : public MultiFileReader {
 public:
  LocalMultiFileReader(const std::string& file_path,
                       const import_export::CopyParams& copy_params);
};

}  // namespace foreign_storage
