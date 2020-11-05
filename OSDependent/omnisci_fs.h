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

#pragma once

#include <stddef.h>
#include <stdio.h>

namespace omnisci {

size_t file_size(const int fd);

void* checked_mmap(const int fd, const size_t sz);

void checked_munmap(void* addr, size_t length);

int msync(void* addr, size_t length, bool async);

int fsync(int fd);

int open(const char* path, int flags, int mode);

void close(const int fd);

::FILE* fopen(const char* filename, const char* mode);

int get_page_size();

}  // namespace omnisci
