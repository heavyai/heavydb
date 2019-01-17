/*
 * Copyright 2019 OmniSci, Inc.
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
 * @file    file_delete.h
 * @author  michael@omnisci.com>
 * @brief   shared utility for mapd_server and string dictionary server to remove old
 * files
 *
 */
#include <atomic>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <thread>

#ifndef FILE_DELETE_H
#define FILE_DELETE_H

// this is to clean up the deleted files
// this should be moved into the new stuff that kursat is working on when it is in place
void file_delete(std::atomic<bool>& program_is_running,
                 const unsigned int wait_interval_seconds,
                 const std::string base_path) {
  const auto wait_duration = std::chrono::seconds(wait_interval_seconds);
  const boost::filesystem::path path(base_path);
  while (program_is_running) {
    typedef std::vector<boost::filesystem::path> vec;  // store paths,
    vec v;
    boost::system::error_code ec;

    // copy vector from iterator as was getting weird random errors if
    // removed direct from iterator
    copy(boost::filesystem::directory_iterator(path),
         boost::filesystem::directory_iterator(),
         back_inserter(v));
    for (vec::const_iterator it(v.begin()); it != v.end(); ++it) {
      std::string object_name(it->string());

      if (boost::algorithm::ends_with(object_name, "DELETE_ME")) {
        LOG(INFO) << " removing object " << object_name;
        boost::filesystem::remove_all(*it, ec);
        if (ec.value() != boost::system::errc::success) {
          LOG(ERROR) << "Failed to remove object " << object_name << " error was " << ec;
        }
      }
    }

    std::this_thread::sleep_for(wait_duration);
  }
}

#endif  // FILE_DELETE_H
