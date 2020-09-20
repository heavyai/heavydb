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

#include <fcntl.h>
#include <io.h>

namespace omnisci {

size_t file_size(const int fd) {
  return 0;
}

void* checked_mmap(const int fd, const size_t sz) {
  return nullptr;
}

void checked_munmap(void* addr, size_t length) {
  return;
}

int msync(void* addr, size_t length, bool async) {
  return 0;
}

int fsync(int fd) {
  return 0;
}

int open(const char* path, int flags, int mode) {
  return _open(path, flags, mode);
}

void close(const int fd) {
  _close(fd);
}

}  // namespace omnisci
