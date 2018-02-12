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
#include <iostream>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include "File.h"
#include <glog/logging.h>

namespace File_Namespace {

FILE* create(const std::string& basePath, const int fileId, const size_t pageSize, const size_t numPages) {
  std::string path(basePath + std::to_string(fileId) + "." + std::to_string(pageSize) +
                   std::string(MAPD_FILE_EXT));  // MAPD_FILE_EXT has preceding "."
  if (numPages < 1 || pageSize < 1) {
    LOG(FATAL) << "Error trying to create file '" << path
               << "', Number of pages and page size must be positive integers. numPages " << numPages << " pageSize "
               << pageSize;
  }
  FILE* f = fopen(path.c_str(), "w+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to create file '" << path << "', the errno is " << errno;
  }
  fseek(f, (pageSize * numPages) - 1, SEEK_SET);
  fputc(EOF, f);
  fseek(f, 0, SEEK_SET);  // rewind
  if (fileSize(f) != pageSize * numPages) {
    LOG(FATAL) << "Error trying to create file '" << path << "', file size " << fileSize(f)
               << " does not equal pageSize * numPages " << pageSize * numPages;
  }

  return f;
}

FILE* create(const std::string& fullPath, const size_t requestedFileSize) {
  FILE* f = fopen(fullPath.c_str(), "w+b");
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to create file '" << fullPath << "', the errno is " << errno;
  }
  fseek(f, requestedFileSize - 1, SEEK_SET);
  fputc(EOF, f);
  fseek(f, 0, SEEK_SET);  // rewind
  if (fileSize(f) != requestedFileSize) {
    LOG(FATAL) << "Error trying to create file '" << fullPath << "', file size " << fileSize(f)
               << " does not equal requestedFileSize " << requestedFileSize;
  }
  return f;
}

FILE* open(int fileId) {
  std::string s(std::to_string(fileId) + std::string(MAPD_FILE_EXT));
  FILE* f = fopen(s.c_str(), "r+b");  // opens existing file for updates
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to open file '" << s << "', the errno is " << errno;
  }
  return f;
}

FILE* open(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r+b");  // opens existing file for updates
  if (f == nullptr) {
    LOG(FATAL) << "Error trying to open file '" << path << "', the errno is " << errno;
  }
  return f;
}

void close(FILE* f) {
  CHECK(f);
  CHECK_EQ(fflush(f), 0);
  CHECK_EQ(fclose(f), 0);
}

bool removeFile(const std::string basePath, const std::string filename) {
  const std::string filePath = basePath + filename;
  return remove(filePath.c_str()) == 0;
}

size_t read(FILE* f, const size_t offset, const size_t size, int8_t* buf) {
  // read "size" bytes from the offset location in the file into the buffer
  fseek(f, offset, SEEK_SET);
  size_t bytesRead = fread(buf, sizeof(int8_t), size, f);
  CHECK_EQ(bytesRead, sizeof(int8_t) * size);
  return bytesRead;
}

size_t write(FILE* f, const size_t offset, const size_t size, int8_t* buf) {
  // write size bytes from the buffer to the offset location in the file
  fseek(f, offset, SEEK_SET);
  size_t bytesWritten = fwrite(buf, sizeof(int8_t), size, f);
  CHECK_EQ(bytesWritten, sizeof(int8_t) * size);
  fflush(f);  // needed?
  return bytesWritten;
}

size_t append(FILE* f, const size_t size, int8_t* buf) {
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

}  // File_Namespace
