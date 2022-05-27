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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // not _WIN32

namespace heavyai {

size_t file_size(const int fd);

void* checked_mmap(const int fd, const size_t sz);

void checked_munmap(void* addr, size_t length);

int msync(void* addr, size_t length, bool async);

int fsync(int fd);

int open(const char* path, int flags, int mode);

void close(const int fd);

::FILE* fopen(const char* filename, const char* mode);

::FILE* popen(const char* command, const char* type);

int32_t pclose(::FILE* fh);

int get_page_size();

int32_t ftruncate(const int32_t fd, int64_t length);

#ifndef _WIN32
// Signal-safe versions of low-level posix functions. Won't fail w/EINTR.
int safe_open(const char* path, int flags, mode_t mode) noexcept;
int safe_close(int fd) noexcept;
int safe_fcntl(int fd, int cmd, struct flock* fl) noexcept;
ssize_t safe_read(const int fd, void* buffer, const size_t buffer_size) noexcept;
ssize_t safe_write(const int fd, const void* buffer, const size_t buffer_size) noexcept;
int32_t safe_ftruncate(const int32_t fd, int64_t length) noexcept;
#endif  // not _WIN32

}  // namespace heavyai
