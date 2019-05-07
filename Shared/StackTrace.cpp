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

#include <Shared/StackTrace.h>

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED 1

#include <boost/algorithm/string.hpp>
#include <boost/stacktrace.hpp>

std::string getCurrentStackTrace(uint32_t num_frames_to_skip,
                                 const char* stop_at_this_frame,
                                 bool skip_void_and_stl_frames) {
  std::string stack_trace;

  uint32_t frame_skip_count = num_frames_to_skip;

  // get the entire stacktrace
  auto st = boost::stacktrace::stacktrace();

  // process frames
  for (auto& frame : st) {
    // skip frame?
    if (frame_skip_count > 0) {
      frame_skip_count--;
      continue;
    }

    // get function name for this frame
    std::string frame_string = frame.name();

    // trim to plain function or template name
    size_t open_paren_or_angle = frame_string.find_first_of("(<");
    if (open_paren_or_angle != std::string::npos) {
      frame_string.erase(open_paren_or_angle, std::string::npos);
    }

    // skip stuff that we usually don't want
    if (skip_void_and_stl_frames) {
      if (boost::istarts_with(frame_string, "void")) {
        continue;
      }
      if (boost::istarts_with(frame_string, "std::")) {
        continue;
      }
    }

    // stop when we hit a particular function?
    if (stop_at_this_frame) {
      if (boost::starts_with(frame_string, stop_at_this_frame)) {
        break;
      }
    }

    // stop at main anyway
    if (boost::starts_with(frame_string, "main")) {
      break;
    }

    // add file and line? (if we can get them)
    if (frame.source_file().size()) {
      frame_string +=
          " (" + frame.source_file() + ":" + std::to_string(frame.source_line()) + ")";
    }

    // add to string
    stack_trace += frame_string + std::string("\n");
  }

  // done
  return stack_trace;
}
