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

#ifndef _MEASURE_H_
#define _MEASURE_H_

#include <chrono>
#include <iomanip>
#include <sstream>
#include "Logger.h"

extern bool g_enable_debug_timer;

template <typename TimeT = std::chrono::milliseconds>
struct measure {
  template <typename F, typename... Args>
  static typename TimeT::rep execution(F func, Args&&... args) {
    auto start = std::chrono::steady_clock::now();
    func(std::forward<Args>(args)...);
    auto duration =
        std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);
    return duration.count();
  }
};

template <typename Type = std::chrono::steady_clock::time_point>
Type timer_start() {
  return std::chrono::steady_clock::now();
}

template <typename Type = std::chrono::steady_clock::time_point,
          typename TypeR = std::chrono::milliseconds>
typename TypeR::rep timer_stop(Type clock_begin) {
  auto duration =
      std::chrono::duration_cast<TypeR>(std::chrono::steady_clock::now() - clock_begin);
  return duration.count();
}

template <typename Type = std::chrono::steady_clock::time_point,
          typename TypeR = std::chrono::milliseconds>
std::string timer_lap(Type clock_begin, Type& clock_last) {
  auto now = std::chrono::steady_clock::now();
  auto overall_duration = (now - clock_begin);
  auto since_last_duration = (now - clock_last);
  auto overall = std::chrono::duration_cast<TypeR>(overall_duration);
  auto since_last = std::chrono::duration_cast<TypeR>(since_last_duration);
  clock_last = now;
  // std::string ret(overall.count() + " elapsed " + since_last.count());
  std::ostringstream oss;
  oss << overall.count() << " - " << since_last.count();
  return oss.str();
}

struct InjectTimer {
  InjectTimer(std::string const& description, int const& lineNum, std::string const& func)
      : description_(description), lineNum_(lineNum), func_(func) {
    if (g_enable_debug_timer) {
      start_ = timer_start();
      LOG(INFO) << "Timer start " << std::setfill(' ') << std::setw(35) << description_
                << " " << std::setw(35) << func_ << ":" << std::setw(5) << lineNum_;
    }
  }

  ~InjectTimer() {
    if (g_enable_debug_timer) {
      LOG(INFO) << "Timer end   " << std::setfill(' ') << std::setw(35) << description_
                << " " << std::setw(35) << func_ << ":" << std::setw(5) << lineNum_
                << " elapsed " << timer_stop(start_) << " ms";
    }
  }

  std::string description_;
  int lineNum_;
  std::string func_;

  std::chrono::steady_clock::time_point start_;
};
#define INJECT_TIMER(DESC) InjectTimer DESC(#DESC, __LINE__, __FUNCTION__)

template <typename Fn, Fn fn, typename... Args>
typename std::result_of<Fn(Args...)>::type time_wrap(Args... args) {
  InjectTimer t("test", 1, "test");
  return fn(std::forward<Args>(args)...);
}
#define TIME_WRAP(FUNC) time_wrap<decltype(&FUNC), &FUNC>

#endif  // _MEASURE_H_
