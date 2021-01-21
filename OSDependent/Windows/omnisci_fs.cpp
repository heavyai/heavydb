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

#include "OSDependent/omnisci_fs.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <fcntl.h>
#include <io.h>

#include <windows.h>

#include <memoryapi.h>

#include "Logger/Logger.h"

namespace omnisci {

size_t file_size(const int fd) {
  struct _stat64i32 buf;
  const auto err = _fstat64i32(fd, &buf);
  CHECK_EQ(0, err);
  return buf.st_size;
}

void* checked_mmap(const int fd, const size_t sz) {
  auto handle = _get_osfhandle(fd);
  HANDLE map_handle =
      CreateFileMapping(reinterpret_cast<HANDLE>(handle), NULL, PAGE_READWRITE, 0, 0, 0);
  CHECK(map_handle);
  auto map_ptr = MapViewOfFile(map_handle, FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, sz);
  CHECK(map_ptr);
  CHECK(CloseHandle(map_handle) != 0);
  return map_ptr;
}

void checked_munmap(void* addr, size_t length) {
  CHECK(UnmapViewOfFile(addr) != 0);
}

int msync(void* addr, size_t length, bool async) {
  auto err = FlushViewOfFile(addr, length);
  return err != 0 ? 0 : -1;
}

int fsync(int fd) {
  // TODO: FlushFileBuffers
  auto file = _fdopen(fd, "a+");
  return fflush(file);
}

int open(const char* path, int flags, int mode) {
  return _open(path, flags, mode);
}

void close(const int fd) {
  _close(fd);
}

::FILE* fopen(const char* filename, const char* mode) {
  FILE* f;
  auto err = fopen_s(&f, filename, mode);
  // Handle 'too many open files' error
  if (err == EMFILE) {
    auto max_handles = _getmaxstdio();
    if (max_handles < 8192) {
      auto res = _setmaxstdio(8192);
      if (res < 0) {
        LOG(FATAL) << "Cannot increase maximum number of open files";
      }
      err = fopen_s(&f, filename, mode);
    }
  }
  CHECK(!err);
  return f;
}

int get_page_size() {
  return 4096;  // TODO: reasonable guess for now
}

}  // namespace omnisci
