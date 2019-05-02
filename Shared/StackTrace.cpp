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

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>

#include <cxxabi.h>
#include <execinfo.h>

std::string getCurrentStackTrace(const char* stop_at_this_frame,
                                 bool skip_void_and_stl_frames,
                                 bool omit_current_frame) {
  auto AutoFree = [](auto* p) { free(p); };

  // capture up to this many frames
  static const int kSTACK_FRAMES_TO_CAPTURE = 50;

  // capture frames
  void* stack_frames[kSTACK_FRAMES_TO_CAPTURE];
  size_t num_stack_frames = backtrace(stack_frames, kSTACK_FRAMES_TO_CAPTURE);
  std::unique_ptr<char*, decltype(AutoFree)> stack_frame_cstrs(
      backtrace_symbols(stack_frames, num_stack_frames), AutoFree);

  // prepare to build string
  std::string stack_trace;

  // always skip first entry
  // optionally skip second entry (the current frame)
  size_t start_frame = omit_current_frame ? 2 : 1;
  for (size_t i = start_frame; i < num_stack_frames; i++) {
    std::string stack_frame_string(stack_frame_cstrs.get()[i]);

    // pull out the part in parentheses
    size_t open_paren = stack_frame_string.find_first_of('(');
    CHECK(open_paren != std::string::npos);
    size_t close_paren = stack_frame_string.find_last_of(')');
    CHECK(close_paren != std::string::npos);
    std::string mangled_fn_string =
        stack_frame_string.substr(open_paren + 1, close_paren - open_paren - 1);

    // drop the "+offset" part at the end
    size_t plus = mangled_fn_string.find_last_of('+');
    CHECK(plus != std::string::npos);
    mangled_fn_string.erase(plus, std::string::npos);

    // unmangle what's left
    int status = -1;
    std::unique_ptr<char, decltype(AutoFree)> unmangled_fn_cstr(
        abi::__cxa_demangle(mangled_fn_string.c_str(), 0, 0, &status), AutoFree);
    if (status != 0) {
      // unmangle failed, skip
      continue;
    }
    std::string unmangled_fn_string(unmangled_fn_cstr.get());

    // inspect what we have
    if (unmangled_fn_string.size()) {
      // trim to plain function or template name
      size_t open_paren_or_angle = unmangled_fn_string.find_first_of("(<");
      unmangled_fn_string.erase(open_paren_or_angle, std::string::npos);

      if (skip_void_and_stl_frames) {
        if (boost::istarts_with(unmangled_fn_string, "void")) {
          continue;
        }
        if (boost::istarts_with(unmangled_fn_string, "std::")) {
          continue;
        }
      }

      if (stop_at_this_frame) {
        // stop when we hit this
        if (boost::starts_with(unmangled_fn_string, stop_at_this_frame)) {
          break;
        }
      }

      // add to string
      stack_trace += unmangled_fn_string + std::string("\n");
    }
  }

  // done
  return stack_trace;
}
