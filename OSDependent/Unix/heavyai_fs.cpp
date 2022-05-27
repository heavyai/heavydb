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

#include "OSDependent/heavyai_fs.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "Logger/Logger.h"

namespace heavyai {

int get_page_size() {
  return getpagesize();
}

size_t file_size(const int fd) {
  struct stat buf;
  int err = fstat(fd, &buf);
  CHECK_EQ(0, err);
  return buf.st_size;
}

void* checked_mmap(const int fd, const size_t sz) {
  auto ptr = mmap(nullptr, sz, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
  CHECK(ptr != reinterpret_cast<void*>(-1));
#ifdef __linux__
#ifdef MADV_HUGEPAGE
  madvise(ptr, sz, MADV_RANDOM | MADV_WILLNEED | MADV_HUGEPAGE);
#else
  madvise(ptr, sz, MADV_RANDOM | MADV_WILLNEED);
#endif
#endif
  return ptr;
}

void checked_munmap(void* addr, size_t length) {
  CHECK_EQ(0, munmap(addr, length));
}

int msync(void* addr, size_t length, bool async) {
  // TODO: support MS_INVALIDATE?
  return ::msync(addr, length, async ? MS_ASYNC : MS_SYNC);
}

int fsync(int fd) {
  return ::fsync(fd);
}

int open(const char* path, int flags, int mode) {
  return ::open(path, flags, mode);
}

void close(const int fd) {
  ::close(fd);
}

::FILE* fopen(const char* filename, const char* mode) {
  return ::fopen(filename, mode);
}

::FILE* popen(const char* command, const char* type) {
  return ::popen(command, type);
}

int32_t pclose(::FILE* fh) {
  return ::pclose(fh);
}

int32_t ftruncate(const int32_t fd, int64_t length) {
  return ::ftruncate(fd, length);
}

int safe_open(const char* path, int flags, mode_t mode) noexcept {
  for (int ret;;) {
    ret = ::open(path, flags, mode);
    if (ret == -1 && errno == EINTR) {  // interrupted by signal
      continue;
    }
    return ret;
  }
  UNREACHABLE();
}

int safe_close(int fd) noexcept {
  for (int ret;;) {
    ret = ::close(fd);
    if (ret == -1 && errno == EINTR) {  // interrupted by signal
      continue;
    }
    return ret;
  }
  UNREACHABLE();
}

int safe_fcntl(int fd, int cmd, struct flock* fl) noexcept {
  for (int ret;;) {
    ret = ::fcntl(fd, cmd, fl);
    if (ret == -1 && errno == EINTR) {  // interrupted by signal
      continue;
    }
    return ret;
  }
  UNREACHABLE();
}

ssize_t safe_read(const int fd, void* buffer, const size_t buffer_size) noexcept {
  for (ssize_t ret, sz = 0;;) {
    ret = ::read(fd, &static_cast<char*>(buffer)[sz], buffer_size - sz);
    if (ret == -1) {
      if (errno == EINTR) {  // interrupted by signal
        continue;
      }
      return -1;
    }
    if (ret == 0) {  // EOF
      return sz;
    }
    sz += ret;
    if (sz == static_cast<ssize_t>(buffer_size)) {
      return sz;
    }
    // either an EOF is coming or interrupted by signal
  }
  UNREACHABLE();
}

ssize_t safe_write(const int fd, const void* buffer, const size_t buffer_size) noexcept {
  for (ssize_t ret, sz = 0;;) {
    ret = ::write(fd, &static_cast<char const*>(buffer)[sz], buffer_size - sz);
    if (ret == -1) {
      if (errno == EINTR) {  // interrupted by signal
        continue;
      }
      return -1;
    }
    sz += ret;
    if (sz == static_cast<ssize_t>(buffer_size)) {
      return sz;
    }
    // either an error is coming (such as disk full) or interrupted by signal
  }
  UNREACHABLE();
}

int32_t safe_ftruncate(const int32_t fd, int64_t length) noexcept {
  for (int ret;;) {
    ret = ::ftruncate(fd, length);
    if (ret == -1 && errno == EINTR) {  // interrupted by signal
      continue;
    }
    return ret;
  }
  UNREACHABLE();
}

}  // namespace heavyai
