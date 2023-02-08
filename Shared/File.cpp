/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file    File.cpp
 * @brief   Implementation of helper methods for File I/O.
 *
 */

#include "Shared/File.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Logger/Logger.h"
#include "OSDependent/heavyai_fs.h"

#include <boost/filesystem.hpp>

bool g_read_only{false};

namespace File_Namespace {

std::string get_data_file_path(const std::string& base_path,
                               int file_id,
                               size_t page_size) {
  return base_path + "/" + std::to_string(file_id) + "." + std::to_string(page_size) +
         std::string(DATA_FILE_EXT);  // DATA_FILE_EXT has preceding "."
}

std::string get_legacy_data_file_path(const std::string& new_data_file_path) {
  auto legacy_path = boost::filesystem::canonical(new_data_file_path);
  legacy_path.replace_extension(kLegacyDataFileExtension);
  return legacy_path.string();
}

std::pair<FILE*, std::string> create(const std::string& basePath,
                                     const int fileId,
                                     const size_t pageSize,
                                     const size_t numPages) {
  auto path = get_data_file_path(basePath, fileId, pageSize);
  if (numPages < 1 || pageSize < 1) {
    LOG(FATAL) << "Error trying to create file '" << path
               << "', Number of pages and page size must be positive integers. numPages "
               << numPages << " pageSize " << pageSize;
  }
  FILE* f = heavyai::fopen(path.c_str(), "w+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to create file '" << path
               << "', the error was: " << std::strerror(errno);
  }
  fseek(f, static_cast<long>((pageSize * numPages) - 1), SEEK_SET);
  fputc(EOF, f);
  fseek(f, 0, SEEK_SET);  // rewind
  if (fileSize(f) != pageSize * numPages) {
    LOG(FATAL) << "Error trying to create file '" << path << "', file size "
               << fileSize(f) << " does not equal pageSize * numPages "
               << pageSize * numPages;
  }
  boost::filesystem::create_symlink(boost::filesystem::canonical(path).filename(),
                                    get_legacy_data_file_path(path));
  return {f, path};
}

FILE* create(const std::string& full_path, const size_t requested_file_size) {
  FILE* f = heavyai::fopen(full_path.c_str(), "w+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to create file '" << full_path
               << "', the error was:  " << std::strerror(errno);
  }
  fseek(f, static_cast<long>(requested_file_size - 1), SEEK_SET);
  fputc(EOF, f);
  fseek(f, 0, SEEK_SET);  // rewind
  if (fileSize(f) != requested_file_size) {
    LOG(FATAL) << "Error trying to create file '" << full_path << "', file size "
               << fileSize(f) << " does not equal requested_file_size "
               << requested_file_size;
  }
  return f;
}

FILE* open(int file_id) {
  std::string s(std::to_string(file_id) + std::string(DATA_FILE_EXT));
  return open(s);
}

FILE* open(const std::string& path) {
  FILE* f = heavyai::fopen(path.c_str(), "r+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to open file '" << path
               << "', the errno was: " << std::strerror(errno);
  }
  return f;
}

void close(FILE* f) {
  CHECK(f);
  CHECK_EQ(fflush(f), 0);
  CHECK_EQ(fclose(f), 0);
}

bool removeFile(const std::string& base_path, const std::string& filename) {
  const std::string file_path = base_path + filename;
  return remove(file_path.c_str()) == 0;
}

size_t read(FILE* f,
            const size_t offset,
            const size_t size,
            int8_t* buf,
            const std::string& file_path) {
  // read "size" bytes from the offset location in the file into the buffer
  CHECK_EQ(fseek(f, static_cast<long>(offset), SEEK_SET), 0);
  size_t bytesRead = fread(buf, sizeof(int8_t), size, f);
  auto expected_bytes_read = sizeof(int8_t) * size;
  CHECK_EQ(bytesRead, expected_bytes_read)
      << "Unexpected number of bytes read from file: " << file_path
      << ". Expected bytes read: " << expected_bytes_read
      << ", actual bytes read: " << bytesRead << ", offset: " << offset
      << ", file stream error set: " << (std::ferror(f) ? "true" : "false")
      << ", EOF reached: " << (std::feof(f) ? "true" : "false");
  return bytesRead;
}

size_t write(FILE* f, const size_t offset, const size_t size, const int8_t* buf) {
  // write size bytes from the buffer to the offset location in the file
  if (fseek(f, static_cast<long>(offset), SEEK_SET) != 0) {
    LOG(FATAL)
        << "Error trying to write to file (during positioning seek) the error was: "
        << std::strerror(errno);
  }
  size_t bytesWritten = fwrite(buf, sizeof(int8_t), size, f);
  if (bytesWritten != sizeof(int8_t) * size) {
    LOG(FATAL) << "Error trying to write to file (during fwrite) the error was: "
               << std::strerror(errno);
  }
  return bytesWritten;
}

size_t append(FILE* f, const size_t size, const int8_t* buf) {
  return write(f, fileSize(f), size, buf);
}

size_t readPage(FILE* f,
                const size_t pageSize,
                const size_t pageNum,
                int8_t* buf,
                const std::string& file_path) {
  return read(f, pageNum * pageSize, pageSize, buf, file_path);
}

size_t readPartialPage(FILE* f,
                       const size_t pageSize,
                       const size_t offset,
                       const size_t readSize,
                       const size_t pageNum,
                       int8_t* buf,
                       const std::string& file_path) {
  return read(f, pageNum * pageSize + offset, readSize, buf, file_path);
}

size_t writePage(FILE* f, const size_t pageSize, const size_t pageNum, int8_t* buf) {
  return write(f, pageNum * pageSize, pageSize, buf);
}

size_t writePartialPage(FILE* f,
                        const size_t pageSize,
                        const size_t offset,
                        const size_t writeSize,
                        const size_t pageNum,
                        int8_t* buf) {
  return write(f, pageNum * pageSize + offset, writeSize, buf);
}

size_t appendPage(FILE* f, const size_t pageSize, int8_t* buf) {
  return write(f, fileSize(f), pageSize, buf);
}

/// @todo There may be an issue casting to size_t from long.
size_t fileSize(FILE* f) {
  fseek(f, 0, SEEK_END);
  size_t size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  return size;
}

// this is a helper function to rename existing directories
// allowing for an async process to actually remove the physical directries
// and subfolders and files later
// it is required due to the large amount of time it can take to delete
// physical files from large disks
void renameForDelete(const std::string directoryName) {
  boost::system::error_code ec;
  boost::filesystem::path directoryPath(directoryName);
  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

  if (boost::filesystem::exists(directoryPath) &&
      boost::filesystem::is_directory(directoryPath)) {
    boost::filesystem::path newDirectoryPath(directoryName + "_" +
                                             std::to_string(ms.count()) + "_DELETE_ME");
    boost::filesystem::rename(directoryPath, newDirectoryPath, ec);

#ifdef _WIN32
    // On Windows we sometimes fail to rename a directory with System: 5 error
    // code (access denied). An attempt to stop in debugger and look for opened
    // handles for some of directory content shows no opened handles and actually
    // allows renaming to execute successfully. It's not clear why, but a short
    // pause allows to rename directory successfully. Until reasons are known,
    // use this retry loop as a workaround.
    int tries = 10;
    while (ec.value() != boost::system::errc::success && tries) {
      LOG(ERROR) << "Failed to rename directory " << directoryPath << " error was " << ec
                 << " (" << tries << " attempts left)";
      std::this_thread::sleep_for(std::chrono::milliseconds(100 / tries));
      tries--;
      boost::filesystem::rename(directoryPath, newDirectoryPath, ec);
    }
#endif

    if (ec.value() == boost::system::errc::success) {
      std::thread th([newDirectoryPath]() {
        boost::system::error_code ec;
        boost::filesystem::remove_all(newDirectoryPath, ec);
        // We dont check error on remove here as we cant log the
        // issue fromdetached thrad, its not safe to LOG from here
        // This is under investigation as clang detects TSAN issue data race
        // the main system wide file_delete_thread will clean up any missed files
      });
      // let it run free so we can return
      // if it fails the file_delete_thread in DBHandler will clean up
      th.detach();

      return;
    }

    LOG(FATAL) << "Failed to rename file " << directoryName << " to "
               << directoryName + "_" + std::to_string(ms.count()) + "_DELETE_ME  Error: "
               << ec;
  }
}

}  // namespace File_Namespace

// Still temporary location but avoids the link errors in the new distributed branch.
// See the comment file_delete.h

#include <atomic>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <thread>

void file_delete(std::atomic<bool>& program_is_running,
                 const unsigned int wait_interval_seconds,
                 const std::string base_path) {
  const auto wait_duration = std::chrono::seconds(wait_interval_seconds);
  const boost::filesystem::path path(base_path);
  while (program_is_running) {
    using vec = std::vector<boost::filesystem::path>;  // store paths,
    vec v;
    boost::system::error_code ec;

    // copy vector from iterator as was getting weird random errors if
    // removed direct from iterator
    copy(boost::filesystem::directory_iterator(path),
         boost::filesystem::directory_iterator(),
         back_inserter(v));
    for (vec::const_iterator it(v.begin()); it != v.end(); ++it) {
      std::string object_name(it->string());

      if (boost::algorithm::ends_with(object_name, "DELETE_ME")) {
        LOG(INFO) << " removing object " << object_name;
        boost::filesystem::remove_all(*it, ec);
        if (ec.value() != boost::system::errc::success) {
          LOG(ERROR) << "Failed to remove object " << object_name << " error was " << ec;
        }
      }
    }

    std::this_thread::sleep_for(wait_duration);
  }
}
