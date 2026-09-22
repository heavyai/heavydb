/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    File.cpp
 * @brief   Implementation of helper methods for File I/O.
 *
 */

#include "Shared/File.h"
#include "Shared/file_delete.h"

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Shared/heavyai_fs.h"

#include "Logger/Logger.h"

#include <boost/filesystem.hpp>

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

namespace {

size_t checked_page_offset(const size_t page_size,
                           const size_t page_num,
                           const size_t within_page_offset = 0) {
  if (page_size != 0 && page_num > std::numeric_limits<size_t>::max() / page_size) {
    throw std::overflow_error("File page offset multiplication overflow");
  }
  const auto page_offset = page_num * page_size;
  if (within_page_offset > std::numeric_limits<size_t>::max() - page_offset) {
    throw std::overflow_error("File page offset addition overflow");
  }
  return page_offset + within_page_offset;
}

template <typename Syscall>
size_t positional_io_exact(FILE* f,
                           const size_t offset,
                           const size_t size,
                           int8_t* buf,
                           const std::string& file_path,
                           const char* operation,
                           Syscall syscall) {
  if (!f) {
    throw std::invalid_argument("Cannot " + std::string(operation) +
                                " through a null file stream.");
  }
  const auto max_file_offset = static_cast<uintmax_t>(std::numeric_limits<off_t>::max());
  if (static_cast<uintmax_t>(offset) > max_file_offset ||
      static_cast<uintmax_t>(size) > max_file_offset - offset) {
    throw std::overflow_error(
        "File " + std::string(operation) +
        " range exceeds the platform file-offset limit: " + file_path);
  }

  int fd = fileno(f);
  if (fd == -1) {
    LOG(FATAL) << "Error obtaining file descriptor for " << operation
               << " from file: " << file_path << ", error was: " << std::strerror(errno);
  }

  size_t total_bytes = 0;
  while (total_bytes < size) {
    const auto remaining_bytes = size - total_bytes;
    const auto request_bytes = std::min(
        remaining_bytes, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const auto request_offset = offset + total_bytes;
    const auto rv =
        syscall(fd, buf + total_bytes, request_bytes, static_cast<off_t>(request_offset));
    if (rv == -1) {
      if (errno == EINTR) {
        continue;
      }
      LOG(FATAL) << "Error trying to " << operation << " file: " << file_path
                 << ", offset: " << request_offset << ", size: " << request_bytes
                 << ", error was: " << std::strerror(errno);
    }
    if (rv == 0) {
      LOG(FATAL) << "Unexpected EOF while trying to " << operation
                 << " file: " << file_path << ", offset: " << request_offset
                 << ", size: " << request_bytes << ", total requested: " << size
                 << ", total completed: " << total_bytes;
    }
    total_bytes += static_cast<size_t>(rv);
  }

  return total_bytes;
}

}  // namespace

size_t read(FILE* f,
            const size_t offset,
            const size_t size,
            int8_t* buf,
            const std::string& file_path) {
  return positional_io_exact(
      f,
      offset,
      size,
      buf,
      file_path,
      "read",
      [](int fd, int8_t* dst, size_t request_bytes, off_t request_offset) {
        return ::pread(fd, dst, request_bytes, request_offset);
      });
}

size_t write(FILE* f, const size_t offset, const size_t size, const int8_t* buf) {
  return positional_io_exact(
      f,
      offset,
      size,
      const_cast<int8_t*>(buf),
      "<unknown>",
      "write",
      [](int fd, int8_t* src, size_t request_bytes, off_t request_offset) {
        return ::pwrite(fd, src, request_bytes, request_offset);
      });
}

size_t append(FILE* f, const size_t size, const int8_t* buf) {
  return write(f, fileSize(f), size, buf);
}

size_t readPage(FILE* f,
                const size_t pageSize,
                const size_t pageNum,
                int8_t* buf,
                const std::string& file_path) {
  return read(f, checked_page_offset(pageSize, pageNum), pageSize, buf, file_path);
}

size_t readPartialPage(FILE* f,
                       const size_t pageSize,
                       const size_t offset,
                       const size_t readSize,
                       const size_t pageNum,
                       int8_t* buf,
                       const std::string& file_path) {
  return read(
      f, checked_page_offset(pageSize, pageNum, offset), readSize, buf, file_path);
}

size_t writePage(FILE* f, const size_t pageSize, const size_t pageNum, int8_t* buf) {
  return write(f, checked_page_offset(pageSize, pageNum), pageSize, buf);
}

size_t writePartialPage(FILE* f,
                        const size_t pageSize,
                        const size_t offset,
                        const size_t writeSize,
                        const size_t pageNum,
                        int8_t* buf) {
  return write(f, checked_page_offset(pageSize, pageNum, offset), writeSize, buf);
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

// file_delete() implementation lives here; see Shared/file_delete.h.

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <thread>

void file_delete(std::atomic<bool>& program_is_running,
                 const unsigned int wait_interval_seconds,
                 const std::string base_path) {
  const auto wait_duration = std::chrono::seconds(wait_interval_seconds);
  constexpr auto shutdown_poll_interval = std::chrono::milliseconds(100);
  const boost::filesystem::path path(base_path);
  boost::system::error_code last_scan_error;
  while (program_is_running) {
    using vec = std::vector<boost::filesystem::path>;  // store paths,
    vec v;
    boost::system::error_code ec;

    // Snapshot entries before deleting them; removing entries while iterating caused
    // intermittent traversal errors.
    boost::filesystem::directory_iterator file_it(path, ec);
    const boost::filesystem::directory_iterator end_it;
    while (!ec && file_it != end_it) {
      v.emplace_back(file_it->path());
      file_it.increment(ec);
    }
    if (ec) {
      if (ec != last_scan_error) {
        LOG(ERROR) << "Failed to scan deleted-file directory " << path << ": "
                   << ec.message();
      }
      last_scan_error = ec;
    } else if (last_scan_error) {
      LOG(INFO) << "Resumed scanning deleted-file directory " << path;
      last_scan_error.clear();
    }
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

    const auto wake_time = std::chrono::steady_clock::now() + wait_duration;
    while (program_is_running) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= wake_time) {
        break;
      }
      const auto remaining = wake_time - now;
      std::this_thread::sleep_for(
          remaining < shutdown_poll_interval ? remaining : shutdown_poll_interval);
    }
  }
}
