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

#ifndef ARCHIVE_ARCHIVE_H_
#define ARCHIVE_ARCHIVE_H_

#include <regex>
#include <string>

#include <archive.h>
#include <archive_entry.h>

// this is the base class from which all archives that represent files sources
// hosted on native/netwrok filesystems, AWS S3, HDFS, HTTP URL, FTP URL, ...
// etc are derived.
class Archive {
 public:
  Archive(const std::string url, const bool plain_text)
      : url(url), plain_text(plain_text) {
    parse_url(url, url_parts);

    if (0 == (ar = archive_read_new()))
      throw std::runtime_error(std::string("archive_read_new failed!"));

//!! LIBARCHIVE_ENABLE_ALL may trigger exception "detect_column_types error: libarchive
//! error: Missing type keyword in
//! mtree specification"
//!! on ridiculously simple raw data
//#define LIBARCHIVE_ENABLE_ALL
#ifdef LIBARCHIVE_ENABLE_ALL
    // this increases ~800kb code size
    archive_read_support_format_all(ar);
    archive_read_support_filter_all(ar);
    archive_read_support_format_raw(ar);
#else
    // list supported formats to bypass the mtree exception
    archive_read_support_format_ar(ar);
    archive_read_support_format_cpio(ar);
    archive_read_support_format_empty(ar);
    archive_read_support_format_lha(ar);
    archive_read_support_format_tar(ar);
    archive_read_support_format_xar(ar);
    archive_read_support_format_7zip(ar);
    archive_read_support_format_cab(ar);
    archive_read_support_format_rar(ar);
    archive_read_support_format_iso9660(ar);
    archive_read_support_format_zip(ar);

    archive_read_support_filter_bzip2(ar);
    archive_read_support_filter_compress(ar);
    archive_read_support_filter_gzip(ar);
    archive_read_support_filter_lzip(ar);
    archive_read_support_filter_lzma(ar);
    archive_read_support_filter_xz(ar);
    archive_read_support_filter_uu(ar);
    archive_read_support_filter_rpm(ar);
    archive_read_support_filter_lrzip(ar);
    archive_read_support_filter_lzop(ar);
    archive_read_support_filter_grzip(ar);
#endif
    // libarchive assumes archive formats, so without this bzip2 and gzip won't work!
    // see related issue at https://github.com/libarchive/libarchive/issues/586
    archive_read_support_format_raw(ar);
  }

  virtual ~Archive() {
    if (ar)
      archive_read_close(ar);
    if (ar)
      archive_read_free(ar);
    ar = 0;
  }

  virtual std::string archive_error(int err) {
    auto cstr = archive_error_string(ar);
    return std::string("libarchive error: ") +
           (cstr ? std::string(cstr) : std::to_string(err));
  }

  virtual bool read_next_header() {
    int rc;
    archive_entry* entry;
    switch (rc = archive_read_next_header(ar, &entry)) {
      case ARCHIVE_EOF:
        return false;  // signal caller end of stream
      case ARCHIVE_OK:
        return true;
    }
    throw std::runtime_error(archive_error(rc));
  }

  virtual bool read_data_block(const void** buff, size_t* size, int64_t* offset) {
    int rc;
    switch (rc = archive_read_data_block(ar, buff, size, offset)) {
      case ARCHIVE_EOF:
        return false;  // signal caller end of stream
      case ARCHIVE_OK:
        return true;
    }
    throw std::runtime_error(archive_error(rc));
  }

  virtual int64_t get_position_compressed() const { return archive_filter_bytes(ar, -1); }

  /*  !!!
      7z files can't work with streaming model. Only local 7z files work.
      That is, use archive_read_open_filename for 7z and any posix file.

      Any non-7z data source in generic url format has two options:
      1) stream data to a local named pipe file
      2) a) specify url in COPY FROM
         b) define url-specific Archive class
         c) customize init_for_read() which uses archive_read_open

   */
  virtual int open() { return ARCHIVE_OK; }              // nop
  virtual int close() { return ARCHIVE_OK; }             // nop
  virtual ssize_t read(const void** buff) { return 0; }  // nop

  virtual void init_for_read() {
    // set libarchive callbacks
    archive_read_open(ar, this, Archive::open, Archive::read, Archive::close);
  }

  // these methods are callback for libarchive
  static ssize_t read(struct archive* a, void* client_data, const void** buff) {
    return ((Archive*)client_data)->read(buff);
  }

  static int open(struct archive* a, void* client_data) {
    return ((Archive*)client_data)->open();
  }

  static int close(struct archive* a, void* client_data) {
    return ((Archive*)client_data)->close();
  }

  static void parse_url(const std::string url, std::map<int, std::string>& url_parts) {
    /*
      input example: http://localhost.com/path\?hue\=br\#cool
      output should be:
                    0: http://localhost.com/path?hue=br#cool
                    1: http:
                    2: http
                    3: //localhost.com
                    4: localhost.com
                    5: /path
                    6: ?hue=br
                    7: hue=br
                    8: #cool
                    9: cool
     */
    std::smatch sm;
    std::regex url_regex(R"(^(([^:\/?#]+):)?(//([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?)",
                         std::regex::extended);
    if (!std::regex_match(url, sm, url_regex))
      throw std::runtime_error(std::string("malformed url: ") + url);

    // sm is only an iterator over local 'url'
    // so have to copy out matched parts
    for (size_t i = 0; i < sm.size(); i++)
      url_parts[i] = sm[i].str();
  }

  const std::string url_part(const int i) { return url_parts[i]; }

 protected:
  std::string url;
  std::map<int, std::string> url_parts;
  archive* ar = 0;
  bool plain_text;
};

#endif /* ARCHIVE_ARCHIVE_H_ */
