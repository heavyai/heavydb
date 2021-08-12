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

#include "DataMgr/ForeignStorage/FileReader.h"
#include "ForeignStorageException.h"
#include "FsiJsonUtils.h"

namespace foreign_storage {

namespace {
/**
 * Adds an end of line character (specified by the line_delim parameter) to provided
 * buffer, if this is the last read buffer and if the buffer does not already end with an
 * end of line character. This allows for appropriate parsing by the
 * file_buffer_parser utility functions, which expect the end of rows to be indicated
 * by end of line characters in the buffer.
 * Also removes extra EOL that may be inserted at the EOF that will not be present if the
 * file is appended to
 */
void adjust_eof(size_t& read_size,
                const size_t buffer_size,
                char* buffer,
                const char line_delim) {
  if (read_size == 0 || buffer[read_size - 1] != line_delim) {
    CHECK(buffer_size > read_size);
    static_cast<char*>(buffer)[read_size] = line_delim;
    read_size++;
  } else if (read_size > 1 && buffer[read_size - 2] == line_delim) {
    // Extra newline may have been due to the file encoding
    // and may disappear during an append
    read_size--;
  }
}

/**
 * @param cumulative_sizes Size of each file + all previous files
 * @param byte_offset byte offset into the fileset from the initial scan
 * @return the file index for a given byte offset
 */
size_t offset_to_index(const std::vector<size_t>& cumulative_sizes, size_t byte_offset) {
  auto iterator =
      std::upper_bound(cumulative_sizes.begin(), cumulative_sizes.end(), byte_offset);
  if (iterator == cumulative_sizes.end()) {
    throw std::runtime_error{"Invalid offset into cumulative_sizes"};
  }
  return iterator - cumulative_sizes.begin();
}

size_t get_data_size(size_t file_size, size_t header_size) {
  // Add 1 byte for possible need to insert a newline
  return file_size - header_size + 1;
}

}  // namespace

SingleFileReader::SingleFileReader(const std::string& file_path,
                                   const import_export::CopyParams& copy_params)
    : FileReader(file_path, copy_params)
    , scan_finished_(false)
    , header_offset_(0)
    , total_bytes_read_(0) {
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

  data_size_ = get_data_size(ftell(file_), header_offset_);

  if (fseek(file_, static_cast<long int>(header_offset_), SEEK_SET) != 0) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path + "\". " + strerror(errno)};
  };
}

SingleFileReader::SingleFileReader(const std::string& file_path,
                                   const import_export::CopyParams& copy_params,
                                   const rapidjson::Value& value)
    : FileReader(file_path, copy_params)
    , scan_finished_(true)
    , header_offset_(0)
    , total_bytes_read_(0) {
  file_ = fopen(file_path.c_str(), "rb");
  if (!file_) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path + "\". " + strerror(errno)};
  }
  json_utils::get_value_from_object(value, header_offset_, "header_offset");
  json_utils::get_value_from_object(value, total_bytes_read_, "total_bytes_read");
  json_utils::get_value_from_object(value, data_size_, "data_size");
}

void SingleFileReader::serialize(rapidjson::Value& value,
                                 rapidjson::Document::AllocatorType& allocator) const {
  CHECK(scan_finished_);
  json_utils::add_value_to_object(value, header_offset_, "header_offset", allocator);
  json_utils::add_value_to_object(
      value, total_bytes_read_, "total_bytes_read", allocator);
  json_utils::add_value_to_object(value, data_size_, "data_size", allocator);
};

void SingleFileReader::checkForMoreRows(size_t file_offset,
                                        const ForeignServer* server_options,
                                        const UserMapping* user_mapping) {
  CHECK(isScanFinished());
  // Re-open file and check if there is any new data in it
  fclose(file_);
  file_ = fopen(file_path_.c_str(), "rb");
  if (!file_) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path_ + "\". " + strerror(errno)};
  }
  fseek(file_, 0, SEEK_END);
  size_t new_file_size = ftell(file_);
  size_t new_data_size = get_data_size(new_file_size, header_offset_);
  if (new_data_size < data_size_) {
    throw_removed_row_error(file_path_);
  }
  if (fseek(file_, static_cast<long int>(file_offset + header_offset_), SEEK_SET) != 0) {
    throw std::runtime_error{"An error occurred when attempting to read offset " +
                             std::to_string(file_offset + header_offset_) +
                             " in file: \"" + file_path_ + "\". " + strerror(errno)};
  }
  if (new_data_size > data_size_) {
    scan_finished_ = false;
    total_bytes_read_ = file_offset;
    data_size_ = new_data_size;
  }
}

/**
 * Skip to entry in archive
 */
void ArchiveWrapper::skipToEntry(int entry_number) {
  if (current_entry_ >= entry_number) {
    resetArchive();
  }
  while (current_entry_ < entry_number) {
    if (arch_.get()->read_next_header()) {
      current_entry_++;
    } else {
      throw std::runtime_error{"Invalid archive entry"};
    }
  }
  fetchBlock();
}

// Go to next consecutive entry
bool ArchiveWrapper::nextEntry() {
  bool success = arch_.get()->read_next_header();
  if (success) {
    current_entry_++;
    fetchBlock();
  }
  return success;
}

void ArchiveWrapper::consumeDataFromCurrentEntry(size_t size, char* dest_buffer) {
  CHECK(size <= block_chars_remaining_);
  block_chars_remaining_ -= size;
  if (dest_buffer != nullptr) {
    memcpy(dest_buffer, current_block_, size);
  }
  current_block_ = static_cast<const char*>(current_block_) + size;
  if (block_chars_remaining_ == 0) {
    fetchBlock();
  }
}

char ArchiveWrapper::ArchiveWrapper::peekNextChar() {
  CHECK(block_chars_remaining_ > 0);
  return static_cast<const char*>(current_block_)[0];
}

void ArchiveWrapper::resetArchive() {
  arch_.reset(new PosixFileArchive(file_path_, false));
  block_chars_remaining_ = 0;
  // We will increment to 0 when reading first entry
  current_entry_ = -1;
}

void ArchiveWrapper::fetchBlock() {
  int64_t offset;
  auto ok =
      arch_.get()->read_data_block(&current_block_, &block_chars_remaining_, &offset);
  if (!ok) {
    block_chars_remaining_ = 0;
  }
}

CompressedFileReader::CompressedFileReader(const std::string& file_path,
                                           const import_export::CopyParams& copy_params)
    : FileReader(file_path, copy_params)
    , archive_(file_path)
    , initial_scan_(true)
    , scan_finished_(false)
    , current_offset_(0)
    , current_index_(-1) {
  // Initialize first entry
  nextEntry();
}

CompressedFileReader::CompressedFileReader(const std::string& file_path,
                                           const import_export::CopyParams& copy_params,
                                           const rapidjson::Value& value)
    : CompressedFileReader(file_path, copy_params) {
  scan_finished_ = true;
  initial_scan_ = false;
  sourcenames_.clear();
  archive_entry_index_.clear();
  cumulative_sizes_.clear();
  json_utils::get_value_from_object(value, sourcenames_, "sourcenames");
  json_utils::get_value_from_object(value, cumulative_sizes_, "cumulative_sizes");
  json_utils::get_value_from_object(value, archive_entry_index_, "archive_entry_index");
}

size_t CompressedFileReader::readInternal(void* buffer,
                                          size_t read_size,
                                          size_t buffer_size) {
  size_t remaining_size = read_size;
  char* dest = static_cast<char*>(buffer);
  while (remaining_size > 0 && !archive_.currentEntryFinished()) {
    size_t copy_size = (archive_.currentEntryDataAvailable() < remaining_size)
                           ? archive_.currentEntryDataAvailable()
                           : remaining_size;
    // copy data into dest
    archive_.consumeDataFromCurrentEntry(copy_size, dest);
    remaining_size -= copy_size;
    dest += copy_size;
  }
  size_t bytes_read = read_size - remaining_size;
  if (archive_.currentEntryFinished() && (bytes_read < read_size)) {
    adjust_eof(
        bytes_read, buffer_size, static_cast<char*>(buffer), copy_params_.line_delim);
    current_offset_ += bytes_read;
    nextEntry();
  } else {
    current_offset_ += bytes_read;
  }
  return bytes_read;
}

size_t CompressedFileReader::read(void* buffer, size_t max_size) {
  // Leave one extra char in case we need to insert a delimiter
  size_t bytes_read = readInternal(buffer, max_size - 1, max_size);
  return bytes_read;
}

size_t CompressedFileReader::readRegion(void* buffer, size_t offset, size_t size) {
  CHECK(isScanFinished());

  // Determine where in the archive we are
  size_t index = offset_to_index(cumulative_sizes_, offset);
  CHECK(archive_entry_index_.size() > index);
  auto archive_entry = archive_entry_index_[index];
  current_index_ = static_cast<int>(index);

  // If we are in the wrong entry or too far in the right one skip to the correct entry
  if (archive_entry != archive_.getCurrentEntryIndex() ||
      (archive_entry == archive_.getCurrentEntryIndex() && offset < current_offset_)) {
    archive_.skipToEntry(archive_entry);
    skipHeader();
    current_offset_ = 0;
    if (index > 0) {
      current_offset_ = cumulative_sizes_[index - 1];
    }
  }
  skipBytes(offset - current_offset_);
  return readInternal(buffer, size, size);
}

/**
 * Go to next archive entry/header with valid data
 */
void CompressedFileReader::nextEntry() {
  do {
    // Go to the next index
    current_index_++;
    if (static_cast<int>(cumulative_sizes_.size()) < current_index_) {
      cumulative_sizes_.push_back(current_offset_);
    }
    if (!initial_scan_) {
      // Entry # in the archive is known and might not be the next one in the file
      if (static_cast<int>(archive_entry_index_.size()) > current_index_) {
        archive_.skipToEntry(archive_entry_index_[current_index_]);
        skipHeader();
      } else {
        scan_finished_ = true;
        return;
      }
    } else {
      // Read next header in archive and save the sourcename
      if (archive_.nextEntry()) {
        // read headers until one has data
        CHECK(sourcenames_.size() == archive_entry_index_.size());
        sourcenames_.emplace_back(archive_.entryName());
        archive_entry_index_.emplace_back(archive_.getCurrentEntryIndex());
        skipHeader();
      } else {
        scan_finished_ = true;
        initial_scan_ = false;
        return;
      }
    }
  } while (archive_.currentEntryFinished());
}

/**
 * Skip file header
 */
void CompressedFileReader::skipHeader() {
  if (copy_params_.has_header != import_export::ImportHeaderRow::NO_HEADER) {
    while (!archive_.currentEntryFinished()) {
      if (archive_.peekNextChar() == copy_params_.line_delim) {
        archive_.consumeDataFromCurrentEntry(1);
        break;
      }
      archive_.consumeDataFromCurrentEntry(1);
    }
  }
}

/**
 * Skip forward N bytes without reading the data in current entry
 * @param n_bytes - number of bytes to skip
 */
void CompressedFileReader::skipBytes(size_t n_bytes) {
  current_offset_ += n_bytes;
  while (n_bytes > 0) {
    if (!archive_.currentEntryDataAvailable()) {
      // We've reached the end of the entry
      return;
    }
    // Keep fetching blocks/entries until we've gone through N bytes
    if (archive_.currentEntryDataAvailable() <= n_bytes) {
      n_bytes -= archive_.currentEntryDataAvailable();
      archive_.consumeDataFromCurrentEntry(archive_.currentEntryDataAvailable());
    } else {
      archive_.consumeDataFromCurrentEntry(n_bytes);
      n_bytes = 0;
    }
  }
}

void CompressedFileReader::checkForMoreRows(size_t file_offset,
                                            const ForeignServer* server_options,
                                            const UserMapping* user_mapping) {
  CHECK(initial_scan_ == false);
  size_t initial_entries = archive_entry_index_.size();

  // Reset all entry indexes for existing items
  for (size_t index = 0; index < archive_entry_index_.size(); index++) {
    archive_entry_index_[index] = -1;
  }

  // Read headers and determine location of existing and new files
  int entry_number = 0;
  archive_.resetArchive();
  while (archive_.nextEntry()) {
    auto it = find(sourcenames_.begin(), sourcenames_.end(), archive_.entryName());
    if (it != sourcenames_.end()) {
      // Record new index of already read file
      auto index = it - sourcenames_.begin();
      archive_entry_index_[index] = entry_number;
    } else {
      // Append new source file
      sourcenames_.emplace_back(archive_.entryName());
      archive_entry_index_.emplace_back(entry_number);
    }
    entry_number++;
  }

  // Error if we are missing a file from a previous scan
  for (size_t index = 0; index < archive_entry_index_.size(); index++) {
    if (archive_entry_index_[index] == -1) {
      throw std::runtime_error{
          "Foreign table refreshed with APPEND mode missing archive entry \"" +
          sourcenames_[index] + "\" from file \"" +
          boost::filesystem::path(file_path_).filename().string() + "\"."};
    }
  }

  archive_.resetArchive();
  if (initial_entries < archive_entry_index_.size()) {
    // We found more files
    current_index_ = static_cast<int>(initial_entries) - 1;
    current_offset_ = cumulative_sizes_[current_index_];
    // iterate through new entries until we get one with data
    do {
      nextEntry();
    } while (archive_.currentEntryFinished() &&
             current_index_ < static_cast<int>(archive_entry_index_.size()));

    if (archive_.currentEntryDataAvailable()) {
      scan_finished_ = false;
    }
  } else {
    // No new files but this may be an archive of a single file
    // Check if we only have one file and check if it has more data
    // May have still have multiple entries with some empty that are ignored
    // like directories
    size_t last_size = 0;
    size_t file_index = -1;
    size_t num_file_entries = 0;
    for (size_t index = 0; index < cumulative_sizes_.size(); index++) {
      if (cumulative_sizes_[index] > last_size) {
        file_index = index;
        num_file_entries++;
        last_size = cumulative_sizes_[index];
      }
    }
    if (num_file_entries == 1) {
      current_index_ = static_cast<int>(file_index);
      current_offset_ = 0;
      size_t last_eof = cumulative_sizes_[file_index];

      // reset cumulative_sizes_ with initial zero sizes
      auto old_cumulative_sizes = std::move(cumulative_sizes_);
      cumulative_sizes_ = {};
      for (size_t zero_index = 0; zero_index < file_index; zero_index++) {
        cumulative_sizes_.emplace_back(0);
      }

      // Go to Index of file and read to where we left off
      archive_.skipToEntry(archive_entry_index_[file_index]);
      skipHeader();
      skipBytes(last_eof);
      if (!archive_.currentEntryFinished()) {
        scan_finished_ = false;
      } else {
        // There was no new data, so put back the old data structure
        cumulative_sizes_ = std::move(old_cumulative_sizes);
      }
    }
  }
};

void CompressedFileReader::serialize(
    rapidjson::Value& value,
    rapidjson::Document::AllocatorType& allocator) const {
  // Should be done initial scan
  CHECK(scan_finished_);
  CHECK(!initial_scan_);

  json_utils::add_value_to_object(value, sourcenames_, "sourcenames", allocator);
  json_utils::add_value_to_object(
      value, cumulative_sizes_, "cumulative_sizes", allocator);
  json_utils::add_value_to_object(
      value, archive_entry_index_, "archive_entry_index", allocator);
};

MultiFileReader::MultiFileReader(const std::string& file_path,
                                 const import_export::CopyParams& copy_params)
    : FileReader(file_path, copy_params), current_index_(0), current_offset_(0) {}

MultiFileReader::MultiFileReader(const std::string& file_path,
                                 const import_export::CopyParams& copy_params,
                                 const rapidjson::Value& value)
    : FileReader(file_path, copy_params), current_index_(0), current_offset_(0) {
  json_utils::get_value_from_object(value, file_locations_, "file_locations");
  json_utils::get_value_from_object(value, cumulative_sizes_, "cumulative_sizes");
  json_utils::get_value_from_object(value, current_offset_, "current_offset");
  json_utils::get_value_from_object(value, current_index_, "current_index");

  // Validate files_metadata here, but objects will be recreated by child class
  CHECK(value.HasMember("files_metadata"));
  CHECK(value["files_metadata"].IsArray());
  CHECK(file_locations_.size() == value["files_metadata"].GetArray().Size());
}

void MultiFileReader::serialize(rapidjson::Value& value,
                                rapidjson::Document::AllocatorType& allocator) const {
  json_utils::add_value_to_object(value, file_locations_, "file_locations", allocator);
  json_utils::add_value_to_object(
      value, cumulative_sizes_, "cumulative_sizes", allocator);
  json_utils::add_value_to_object(value, current_offset_, "current_offset", allocator);
  json_utils::add_value_to_object(value, current_index_, "current_index", allocator);

  // Serialize metadata from all files
  rapidjson::Value files_metadata(rapidjson::kArrayType);
  for (size_t index = 0; index < files_.size(); index++) {
    rapidjson::Value file_metadata(rapidjson::kObjectType);
    files_[index]->serialize(file_metadata, allocator);
    files_metadata.PushBack(file_metadata, allocator);
  }
  value.AddMember("files_metadata", files_metadata, allocator);
};

size_t MultiFileReader::getRemainingSize() {
  size_t total_size = 0;
  for (size_t index = current_index_; index < files_.size(); index++) {
    total_size += files_[index]->getRemainingSize();
  }
  return total_size;
}

bool MultiFileReader::isRemainingSizeKnown() {
  bool size_known = true;
  for (size_t index = current_index_; index < files_.size(); index++) {
    size_known = size_known && files_[index]->isRemainingSizeKnown();
  }
  return size_known;
};

LocalMultiFileReader::LocalMultiFileReader(const std::string& file_path,
                                           const import_export::CopyParams& copy_params)
    : MultiFileReader(file_path, copy_params) {
  if (!boost::filesystem::exists(file_path)) {
    throw_file_not_found_error(file_path);
  }
  std::set<std::string> file_locations;
  if (boost::filesystem::is_directory(file_path)) {
    // Find all files in this directory
    for (boost::filesystem::recursive_directory_iterator
             it(file_path, boost::filesystem::symlink_option::recurse),
         eit;
         it != eit;
         ++it) {
      if (!boost::filesystem::is_directory(it->path())) {
        file_locations.insert(it->path().string());
      }
    }
  } else {
    file_locations.insert(file_path);
  }
  for (const auto& location : file_locations) {
    insertFile(location);
  }
}

namespace {
bool is_compressed_file(const std::string& location) {
  const std::vector<std::string> compressed_exts = {
      ".zip", ".gz", ".tar", ".rar", ".bz2", ".7z", ".tgz"};
  const std::vector<std::string> uncompressed_exts = {"", ".csv", ".tsv", ".txt"};
  if (std::find(compressed_exts.begin(),
                compressed_exts.end(),
                boost::filesystem::extension(location)) != compressed_exts.end()) {
    return true;
  } else if (std::find(uncompressed_exts.begin(),
                       uncompressed_exts.end(),
                       boost::filesystem::extension(location)) !=
             uncompressed_exts.end()) {
    return false;
  } else {
    throw std::runtime_error{"Invalid extention for file \"" + location + "\"."};
  }
}
}  // namespace

LocalMultiFileReader::LocalMultiFileReader(const std::string& file_path,
                                           const import_export::CopyParams& copy_params,
                                           const rapidjson::Value& value)
    : MultiFileReader(file_path, copy_params, value) {
  // Constructs file from files_metadata
  for (size_t index = 0; index < file_locations_.size(); index++) {
    if (is_compressed_file(file_locations_[index])) {
      files_.emplace_back(std::make_unique<CompressedFileReader>(
          file_locations_[index],
          copy_params_,
          value["files_metadata"].GetArray()[index]));
    } else {
      files_.emplace_back(
          std::make_unique<SingleFileReader>(file_locations_[index],
                                             copy_params_,
                                             value["files_metadata"].GetArray()[index]));
    }
  }
}

void LocalMultiFileReader::insertFile(std::string location) {
  if (is_compressed_file(location)) {
    files_.emplace_back(std::make_unique<CompressedFileReader>(location, copy_params_));
  } else {
    files_.emplace_back(std::make_unique<SingleFileReader>(location, copy_params_));
  }
  if (files_.back()->isScanFinished()) {
    // skip any initially empty files
    files_.pop_back();
  } else {
    file_locations_.push_back(location);
  }
}

void LocalMultiFileReader::checkForMoreRows(size_t file_offset,
                                            const ForeignServer* server_options,
                                            const UserMapping* user_mapping) {
  // Look for new files
  std::set<std::string> new_locations;
  CHECK(isScanFinished());
  CHECK(file_offset == current_offset_);
  if (boost::filesystem::is_directory(file_path_)) {
    // Find all files in this directory
    std::set<std::string> all_file_paths;
    for (boost::filesystem::recursive_directory_iterator
             it(file_path_, boost::filesystem::symlink_option::recurse),
         eit;
         it != eit;
         ++it) {
      bool new_file =
          std::find(file_locations_.begin(), file_locations_.end(), it->path()) ==
          file_locations_.end();
      if (!boost::filesystem::is_directory(it->path()) && new_file) {
        new_locations.insert(it->path().string());
      }
      all_file_paths.emplace(it->path().string());
    }

    for (const auto& file_path : file_locations_) {
      if (all_file_paths.find(file_path) == all_file_paths.end()) {
        throw_removed_file_error(file_path);
      }
    }
  }
  if (new_locations.size() > 0) {
    for (const auto& location : new_locations) {
      insertFile(location);
    }
  } else if (files_.size() == 1) {
    // Single file, check if it has new data
    files_[0].get()->checkForMoreRows(file_offset);
    if (!files_[0].get()->isScanFinished()) {
      current_index_ = 0;
      cumulative_sizes_ = {};
    }
  }
}

size_t MultiFileReader::read(void* buffer, size_t max_size) {
  if (isScanFinished()) {
    return 0;
  }
  // Leave one extra char in case we need to insert a delimiter
  size_t bytes_read = files_[current_index_].get()->read(buffer, max_size - 1);
  if (files_[current_index_].get()->isScanFinished()) {
    adjust_eof(bytes_read, max_size, static_cast<char*>(buffer), copy_params_.line_delim);
  }
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
  auto index = offset_to_index(cumulative_sizes_, offset);
  // Get offset into this file
  size_t base = 0;
  if (index > 0) {
    base = cumulative_sizes_[index - 1];
  }

  size_t read_size = size;
  if (offset + size == cumulative_sizes_[index]) {
    // Skip the last byte as it may have been an inserted delimiter
    read_size--;
  }
  size_t bytes_read = files_[index].get()->readRegion(buffer, offset - base, read_size);

  if (offset + size == cumulative_sizes_[index]) {
    // Re-insert delimiter
    static_cast<char*>(buffer)[size - 1] = copy_params_.line_delim;
    bytes_read++;
  }

  return bytes_read;
}

}  // namespace foreign_storage
