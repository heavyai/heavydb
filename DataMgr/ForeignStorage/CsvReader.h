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

namespace foreign_storage {

// Archive reader for csv archives
// Supports an initial full scan with calls to read()
// When the entire Csv object has been read isScanFinished() returns true
// Previously read data can then be re-read with readRegion()
class CsvReader {
 public:
  CsvReader(const std::string& file_path, const import_export::CopyParams& copy_params)
      : copy_params_(copy_params), file_path_(file_path){};
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
   * @return size of the CSV remaining to be read
   * */
  virtual size_t getRemainingSize() = 0;

  /**
   * @return if remaining size is known
   */
  virtual bool isRemainingSizeKnown() = 0;
  /**
   * Rescan the target files
   * Throws an exception if the rescan fails (ie files are not in a valid appended state
   * or not supported)
   * @param file_offset - where to resume the scan from (end of the last row) as
   *  not all of the bytes may have been consumed by the upstream compoennet

   */
  virtual void checkForMoreRows(size_t file_offset) {
    throw std::runtime_error{"APPEND mode not yet supported for this table."};
  }

 protected:
  import_export::CopyParams copy_params_;
  std::string file_path_;
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

    total_bytes_read_ += bytes_read;
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

  size_t getRemainingSize() override { return data_size_ - total_bytes_read_; }

  bool isRemainingSizeKnown() override { return true; };
  void checkForMoreRows(size_t file_offset) override;

 private:
  std::FILE* file_;
  // Size of CSV data in file
  size_t data_size_;
  // We've reached the end of the file
  bool scan_finished_;

  // Size of the CSV header in bytes
  size_t header_offset_;

  size_t total_bytes_read_;
};

class ArchiveWrapper {
 public:
  ArchiveWrapper(const std::string& file_path)
      : current_block_(nullptr)
      , block_chars_remaining_(0)
      , current_entry_(-1)
      , file_path_(file_path) {
    resetArchive();
  }

  /**
   * Skip to entry in archive
   */
  void skipToEntry(int entry_number);

  // Go to next consecutive entry in archive
  bool nextEntry();

  bool currentEntryFinished() const { return (block_chars_remaining_ == 0); }

  bool currentEntryDataAvailable() const { return block_chars_remaining_; }

  // Consume given amount of data from current block, copying into dest_buffer if set
  void consumeDataFromCurrentEntry(size_t size, char* dest_buffer = nullptr);

  // Return the next char from the buffer without consuming it
  char peekNextChar();

  int getCurrentEntryIndex() const { return current_entry_; }

  // Reset archive, start reading again from the first entry
  void resetArchive();

  std::string entryName() { return arch_->entryName(); }

 private:
  /**
   * Get the next block from the current archive file
   */
  void fetchBlock();

  std::unique_ptr<Archive> arch_;
  // Pointer to current uncompressed block from the archive
  const void* current_block_;
  // Number of chars remaining in the current block
  size_t block_chars_remaining_;
  // Index of entry of the archive file
  int current_entry_;

  std::string file_path_;
};

// Single archive, does not support random access
class CompressedFileReader : public CsvReader {
 public:
  CompressedFileReader(const std::string& file_path,
                       const import_export::CopyParams& copy_params);

  size_t read(void* buffer, size_t max_size) override;

  size_t readRegion(void* buffer, size_t offset, size_t size) override;

  bool isScanFinished() override { return scan_finished_; }

  size_t getRemainingSize() override { return 0; }

  bool isRemainingSizeKnown() override { return false; };

  void checkForMoreRows(size_t file_offset) override;

 private:
  /**
   * Go to next archive entry/header with valid data
   */
  void nextEntry();

  /**
   * Skip Header of CSV file
   */
  void skipHeader();

  /**
   * Skip forward N bytes in current entry without reading the data
   * @param n_bytes - number of bytes to skip
   */
  void skipBytes(size_t n_bytes);

  // Read bytes in current entry adjusting for EOF
  size_t readInternal(void* buffer, size_t read_size, size_t buffer_size);

  ArchiveWrapper archive_;

  // Are we doing initial scan or an append
  bool initial_scan_;
  // We've reached the end of the last file
  bool scan_finished_;

  // Overall number of bytes read in the archive (minus headers)
  size_t current_offset_;

  // Index of current entry in order they appear in
  // cumulative_sizes_/sourcenames_/archive_entry_index_
  int current_index_;

  // Size of each file + all previous files
  std::vector<size_t> cumulative_sizes_;
  // Names of the file in the archive
  std::vector<std::string> sourcenames_;
  // Index of the entry in the archive
  // Order can change during append operation
  std::vector<int> archive_entry_index_;
};

// Combines several archives into single object
class MultiFileReader : public CsvReader {
 public:
  MultiFileReader(const std::string& file_path,
                  const import_export::CopyParams& copy_params);

  size_t getRemainingSize() override;

  bool isRemainingSizeKnown() override;

  size_t read(void* buffer, size_t max_size) override;

  size_t readRegion(void* buffer, size_t offset, size_t size) override;

  bool isScanFinished() override { return (current_index_ >= files_.size()); }

 protected:
  std::vector<std::unique_ptr<CsvReader>> files_;
  std::set<std::string> file_locations_;

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

  void checkForMoreRows(size_t file_offset) override;

 private:
  void insertFile(std::string location);
};

}  // namespace foreign_storage
