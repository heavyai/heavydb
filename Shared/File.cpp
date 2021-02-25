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
 * @file    File.cpp
 * @author  Steven Stewart <steve@map-d.com>
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
#include "OSDependent/omnisci_fs.h"

bool g_read_only{false};

namespace File_Namespace {

FILE* create(const std::string& basePath,
             const int fileId,
             const size_t pageSize,
             const size_t numPages) {
  std::string path(basePath + "/" + std::to_string(fileId) + "." +
                   std::to_string(pageSize) +
                   std::string(MAPD_FILE_EXT));  // MAPD_FILE_EXT has preceding "."
  if (numPages < 1 || pageSize < 1) {
    LOG(FATAL) << "Error trying to create file '" << path
               << "', Number of pages and page size must be positive integers. numPages "
               << numPages << " pageSize " << pageSize;
  }
  FILE* f = omnisci::fopen(path.c_str(), "w+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to create file '" << path
               << "', the error was: " << std::strerror(errno);
    ;
  }
  fseek(f, static_cast<long>((pageSize * numPages) - 1), SEEK_SET);
  fputc(EOF, f);
  fseek(f, 0, SEEK_SET);  // rewind
  if (fileSize(f) != pageSize * numPages) {
    LOG(FATAL) << "Error trying to create file '" << path << "', file size "
               << fileSize(f) << " does not equal pageSize * numPages "
               << pageSize * numPages;
  }

  return f;
}

FILE* create(const std::string& fullPath, const size_t requestedFileSize) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to create file '" << fullPath
               << "', not allowed read only ";
  }
  FILE* f = omnisci::fopen(fullPath.c_str(), "w+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to create file '" << fullPath
               << "', the error was:  " << std::strerror(errno);
    ;
  }
  fseek(f, static_cast<long>(requestedFileSize - 1), SEEK_SET);
  fputc(EOF, f);
  fseek(f, 0, SEEK_SET);  // rewind
  if (fileSize(f) != requestedFileSize) {
    LOG(FATAL) << "Error trying to create file '" << fullPath << "', file size "
               << fileSize(f) << " does not equal requestedFileSize "
               << requestedFileSize;
  }
  return f;
}

FILE* open(int fileId) {
  std::string s(std::to_string(fileId) + std::string(MAPD_FILE_EXT));
  FILE* f = omnisci::fopen(
      s.c_str(), g_read_only ? "rb" : "r+b");  // opens existing file for updates
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to open file '" << s
               << "', the error was: " << std::strerror(errno);
  }
  return f;
}

FILE* open(const std::string& path) {
  FILE* f = omnisci::fopen(
      path.c_str(), g_read_only ? "rb" : "r+b");  // opens existing file for updates
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

bool removeFile(const std::string basePath, const std::string filename) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to remove file '" << filename << "', running readonly";
  }
  const std::string filePath = basePath + filename;
  return remove(filePath.c_str()) == 0;
}

size_t read(FILE* f, const size_t offset, const size_t size, int8_t* buf) {
  // read "size" bytes from the offset location in the file into the buffer
  CHECK_EQ(fseek(f, static_cast<long>(offset), SEEK_SET), 0);
  size_t bytesRead = fread(buf, sizeof(int8_t), size, f);
  CHECK_EQ(bytesRead, sizeof(int8_t) * size);
  return bytesRead;
}

size_t write(FILE* f, const size_t offset, const size_t size, int8_t* buf) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to write file '" << f << "', running readonly";
  }
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

size_t append(FILE* f, const size_t size, int8_t* buf) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to append file '" << f << "', running readonly";
  }
  return write(f, fileSize(f), size, buf);
}

size_t readPage(FILE* f, const size_t pageSize, const size_t pageNum, int8_t* buf) {
  return read(f, pageNum * pageSize, pageSize, buf);
}

size_t readPartialPage(FILE* f,
                       const size_t pageSize,
                       const size_t offset,
                       const size_t readSize,
                       const size_t pageNum,
                       int8_t* buf) {
  return read(f, pageNum * pageSize + offset, readSize, buf);
}

size_t writePage(FILE* f, const size_t pageSize, const size_t pageNum, int8_t* buf) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to writePage file '" << f << "', running readonly";
  }
  return write(f, pageNum * pageSize, pageSize, buf);
}

size_t writePartialPage(FILE* f,
                        const size_t pageSize,
                        const size_t offset,
                        const size_t writeSize,
                        const size_t pageNum,
                        int8_t* buf) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to writePartialPage file '" << f << "', running readonly";
  }
  return write(f, pageNum * pageSize + offset, writeSize, buf);
}

size_t appendPage(FILE* f, const size_t pageSize, int8_t* buf) {
  if (g_read_only) {
    LOG(FATAL) << "Error trying to appendPage file '" << f << "', running readonly";
  }
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
