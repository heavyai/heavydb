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

#include "DataMgr/ForeignStorage/CsvReader.h"
namespace foreign_storage {

SingleFileReader::SingleFileReader(const std::string& file_path,
                                   const import_export::CopyParams& copy_params)
    : file_path_(file_path), scan_finished_(false), header_offset_(0) {
  file_ = fopen(file_path.c_str(), "rb");
  if (!file_) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path + "\". " + strerror(errno)};
  }

  // Skip header and record offset
  if (copy_params.has_header != import_export::ImportHeaderRow::NO_HEADER) {
    std::ifstream file{file_path};
    CHECK(file.good());
    std::string line;
    std::getline(file, line, copy_params.line_delim);
    file.close();
    header_offset_ = line.size() + 1;
  }
  fseek(file_, 0, SEEK_END);
  file_size_ = ftell(file_) - header_offset_;

  if (fseek(file_, header_offset_, SEEK_SET) != 0) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path + "\". " + strerror(errno)};
  };
}

CompressedFileReader::CompressedFileReader(const std::string& file_path,
                                           const import_export::CopyParams& copy_params)
    : file_path_(file_path)
    , copy_params_(copy_params)
    , current_block_(nullptr)
    , block_chars_remaining_(0)
    , current_offset_(0)
    , scan_finished_(false) {
  resetArchive();
}

size_t CompressedFileReader::read(void* buffer, size_t max_size) {
  size_t remaining_size = max_size;
  char* dest = static_cast<char*>(buffer);

  while (remaining_size > 0 && block_chars_remaining_ > 0) {
    size_t copy_size = (block_chars_remaining_ < remaining_size) ? block_chars_remaining_
                                                                 : remaining_size;
    memcpy(dest, current_block_, copy_size);
    block_chars_remaining_ -= copy_size;
    remaining_size -= copy_size;
    dest += copy_size;
    current_block_ = static_cast<const char*>(current_block_) + copy_size;
    // Keep fetching blocks until buffer is full or we reach the end of this file
    if (block_chars_remaining_ == 0) {
      fetchBlock();
    }
  }
  // Open the next file if this is done
  if (block_chars_remaining_ == 0) {
    nextEntry();
  }

  current_offset_ += (max_size - remaining_size);
  return max_size - remaining_size;
}

size_t CompressedFileReader::readRegion(void* buffer, size_t offset, size_t size) {
  CHECK(isScanFinished());
  if (offset < current_offset_) {
    // Need to restart from the beginning
    resetArchive();
  }
  skipBytes(offset - current_offset_);
  return read(buffer, size);
}

/**
 * Reopen file and reset back to the beginning
 */
void CompressedFileReader::resetArchive() {
  arch_.reset(new PosixFileArchive(file_path_, false));
  block_chars_remaining_ = 0;
  current_offset_ = 0;
  nextEntry();
}

/**
 * Go to next archive entry/header with valid data
 */
void CompressedFileReader::nextEntry() {
  block_chars_remaining_ = 0;
  while (block_chars_remaining_ == 0) {
    // read headers until one has data
    if (arch_.get()->read_next_header()) {
      fetchBlock();
      skipHeader();
    } else {
      scan_finished_ = true;
      return;
    }
  }
}

/**
 * Skip Header of CSV file
 */
void CompressedFileReader::skipHeader() {
  if (copy_params_.has_header != import_export::ImportHeaderRow::NO_HEADER) {
    while (block_chars_remaining_) {
      block_chars_remaining_--;
      if (static_cast<const char*>(current_block_)[0] == copy_params_.line_delim) {
        current_block_ = static_cast<const char*>(current_block_) + 1;
        break;
      }
      current_block_ = static_cast<const char*>(current_block_) + 1;
      if (block_chars_remaining_ == 0) {
        fetchBlock();
      }
    }
  }
}

/**
 * Get the next block from the current archive file
 */
void CompressedFileReader::fetchBlock() {
  int64_t offset;
  auto ok =
      arch_.get()->read_data_block(&current_block_, &block_chars_remaining_, &offset);
  if (!ok) {
    block_chars_remaining_ = 0;
  }
}

/**
 * Skip forward N bytes without reading the data
 * @param n_bytes - number of bytes to skip
 */
void CompressedFileReader::skipBytes(size_t n_bytes) {
  current_offset_ += n_bytes;
  while (n_bytes > 0) {
    // Keep fetching blocks/entries until we've gone through N bytes
    if (block_chars_remaining_ <= n_bytes) {
      n_bytes -= block_chars_remaining_;
      block_chars_remaining_ = 0;
      // Fetch more data
      fetchBlock();
    } else {
      block_chars_remaining_ -= n_bytes;
      current_block_ = static_cast<const char*>(current_block_) + n_bytes;
      n_bytes = 0;
    }
    if (block_chars_remaining_ == 0) {
      nextEntry();
    }
    if (block_chars_remaining_ == 0) {
      // We've reached the end of the archive
      throw std::runtime_error{"Invalid offset into archive"};
    }
  }
}
MultiFileReader::MultiFileReader()
    : current_index_(0), current_offset_(0), total_size_(0), size_known_(true) {}

LocalMultiFileReader::LocalMultiFileReader(const std::string& file_path,
                                           const import_export::CopyParams& copy_params) {
  std::vector<std::string> file_locations;
  if (boost::filesystem::is_directory(file_path)) {
    // Find all files in this directory
    for (boost::filesystem::recursive_directory_iterator it(file_path), eit; it != eit;
         ++it) {
      if (!boost::filesystem::is_directory(it->path())) {
        file_locations.push_back(it->path().string());
      }
    }
  } else {
    file_locations.push_back(file_path);
  }

  const std::vector<std::string> compressed_exts = {
      ".zip", ".gz", ".tar", ".rar", ".bz2", ".7z", ".tgz"};
  const std::vector<std::string> uncompressed_exts = {"", ".csv", ".tsv", ".txt"};
  files_.reserve(file_locations.size());
  for (const auto& location : file_locations) {
    if (std::find(compressed_exts.begin(),
                  compressed_exts.end(),
                  boost::filesystem::extension(location)) != compressed_exts.end()) {
      files_.emplace_back(std::make_unique<CompressedFileReader>(location, copy_params));
    } else if (std::find(uncompressed_exts.begin(),
                         uncompressed_exts.end(),
                         boost::filesystem::extension(location)) !=
               uncompressed_exts.end()) {
      files_.emplace_back(std::make_unique<SingleFileReader>(location, copy_params));
    } else {
      throw std::runtime_error{"Invalid extention for file \"" + location + "\"."};
    }
    if (files_.back()->isScanFinished()) {
      // remove any initially empty files
      files_.pop_back();
      continue;
    }
    size_t new_size;
    size_known_ = size_known_ && files_.back()->getSize(new_size);
    if (size_known_) {
      total_size_ += new_size;
    }
  }
}

size_t MultiFileReader::read(void* buffer, size_t max_size) {
  if (isScanFinished()) {
    return 0;
  }

  size_t bytes_read = files_[current_index_].get()->read(buffer, max_size);
  current_offset_ += bytes_read;
  if (current_index_ < files_.size() && files_[current_index_].get()->isScanFinished()) {
    cumulative_sizes_.push_back(current_offset_);
    current_index_++;
  }
  return bytes_read;
}

size_t MultiFileReader::readRegion(void* buffer, size_t offset, size_t size) {
  CHECK(isScanFinished());
  // Get file index
  auto index = offsetToIndex(offset);
  // Get offset into this file
  size_t base = 0;
  if (index > 0) {
    base = cumulative_sizes_[index - 1];
  }
  size_t bytes_read = files_[index].get()->readRegion(buffer, offset - base, size);
  return bytes_read;
}

/**
 * @param byte_offset byte offset into the fileset from the initial scan
 * @return the file index for a given byte offset
 */
size_t MultiFileReader::offsetToIndex(size_t byte_offset) {
  auto iterator =
      std::upper_bound(cumulative_sizes_.begin(), cumulative_sizes_.end(), byte_offset);
  if (iterator == cumulative_sizes_.end()) {
    throw std::runtime_error{"Invalid offset into archive"};
  }
  return iterator - cumulative_sizes_.begin();
}

}  // namespace foreign_storage
