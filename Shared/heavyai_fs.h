/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

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

// Signal-safe versions of low-level posix functions. Won't fail w/EINTR.
int safe_open(const char* path, int flags, mode_t mode) noexcept;
int safe_close(int fd) noexcept;
int safe_fcntl(int fd, int cmd, struct flock* fl) noexcept;
ssize_t safe_read(const int fd, void* buffer, const size_t buffer_size) noexcept;
ssize_t safe_write(const int fd, const void* buffer, const size_t buffer_size) noexcept;
int32_t safe_ftruncate(const int32_t fd, int64_t length) noexcept;

}  // namespace heavyai
