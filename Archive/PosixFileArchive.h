/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARCHIVE_POSIXFILEARCHIVE_H_
#define ARCHIVE_POSIXFILEARCHIVE_H_

#include <cstdio>

#include "Archive.h"

// archive read buffer size, configurable for unit test.
extern size_t g_archive_read_buf_size;

// this is the archive class for files hosted locally or remotely with
// POSIX compliant file name. !! 7z files work only with this class !!
class PosixFileArchive : public Archive {
 public:
  PosixFileArchive(const std::string url, const bool plain_text)
      : Archive(url, plain_text) {
    // some well-known file.exts imply plain text
    if (!this->plain_text) {
      auto const ext = boost::filesystem::path(url_part(5)).extension();
      this->plain_text = ext == ".csv" || ext == ".tsv" || ext == ".txt" || ext == "";
    }

    if (this->plain_text) {
      buf = new char[g_archive_read_buf_size];
    }

    init_for_read();
  }

  ~PosixFileArchive() override {
    if (fp) {
      fclose(fp);
    }
    if (buf) {
      delete[] buf;
    }
  }

  void init_for_read() override {
    auto file_path = url_part(5);
    if (plain_text) {
      if (nullptr == (fp = fopen(file_path.c_str(), "r"))) {
        throw std::runtime_error(std::string("fopen(") + file_path +
                                 "): " + strerror(errno));
      }
    } else {
      if (ARCHIVE_OK != archive_read_open_filename(ar, file_path.c_str(), 1 << 16)) {
        throw std::runtime_error(std::string("fopen(") + file_path +
                                 "): " + strerror(errno));
      }
    }
  }

  bool read_next_header() override {
    if (plain_text) {
      return !feof(fp);
    } else {
      return Archive::read_next_header();
    }
  }

  bool read_data_block(const void** buff, size_t* size, int64_t* offset) override {
    if (plain_text) {
      size_t nread;
      if (0 >= (nread = fread(buf, 1, g_archive_read_buf_size, fp))) {
        return false;
      }
      *buff = buf;
      *size = nread;
      *offset = ftell(fp);
      return true;
    } else {
      // need original (compressed) offset for row estimation of compressed files
      auto ret = Archive::read_data_block(buff, size, offset);
      *offset = Archive::get_position_compressed();
      return ret;
    }
  }

 private:
  char* buf = nullptr;
  FILE* fp = nullptr;
};

#endif /* ARCHIVE_POSIXFILEARCHIVE_H_ */
