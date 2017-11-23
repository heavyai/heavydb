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

#ifndef ARCHIVE_POSIXFILEARCHIVE_H_
#define ARCHIVE_POSIXFILEARCHIVE_H_

#include <stdio.h>

#include "Archive.h"

// this is the archive class for files hosted locally or remotely with
// POSIX compliant file name. !! 7z files work only with this class !!
class PosixFileArchive : public Archive {
  const size_t buf_size = (1 << 20);

 public:
  PosixFileArchive(const std::string url, const bool plain_text) : Archive(url, plain_text) {
    // some well-known file.exts imply plain text
    if (!this->plain_text)
      this->plain_text =
          boost::filesystem::extension(url_part(5)) == ".csv" || boost::filesystem::extension(url_part(5)) == ".tsv" ||
          boost::filesystem::extension(url_part(5)) == ".txt" || boost::filesystem::extension(url_part(5)) == "";

    if (this->plain_text)
      buf = new char[buf_size];

    init_for_read();
  }

  virtual ~PosixFileArchive() {
    if (fp)
      fclose(fp);
    if (buf)
      delete[] buf;
  }

  virtual void init_for_read() {
    auto file_path = url_part(5);
    if (plain_text) {
      if (nullptr == (fp = fopen(file_path.c_str(), "r")))
        throw std::runtime_error(std::string("fopen(") + file_path + "): " + strerror(errno));
    } else {
      if (ARCHIVE_OK != archive_read_open_filename(ar, file_path.c_str(), 1 << 16))
        throw std::runtime_error(std::string("fopen(") + file_path + "): " + strerror(errno));
    }
  }

  virtual bool read_next_header() {
    if (plain_text)
      return !feof(fp);
    else
      return Archive::read_next_header();
  }

  virtual bool read_data_block(const void** buff, size_t* size, int64_t* offset) {
    if (plain_text) {
      size_t nread;
      if (0 >= (nread = fread(buf, 1, buf_size, fp)))
        return false;
      *buff = buf;
      *size = nread;
      *offset = ftell(fp);
      return true;
    } else
      return Archive::read_data_block(buff, size, offset);
  }

 private:
  char* buf = nullptr;
  FILE* fp = nullptr;
};

#endif /* ARCHIVE_POSIXFILEARCHIVE_H_ */
