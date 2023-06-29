/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#ifndef __CUDACC__

#include <assert.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class CpuTimer {
  using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

 public:
  enum class TimerResolution { kSeconds, kMilliseconds, kMicroseconds, kNanoseconds };

  CpuTimer(const std::string& label) : timer_label_(label), nest_level_(0) {
    initialize();
  }
  CpuTimer(const std::string& label, const int64_t nest_level)
      : timer_label_(label), nest_level_(nest_level) {
    initialize();
  }

  ~CpuTimer() {
    if (nest_level_ == 0) {
      print_timings();
    }
  }

  void start_event_timer(const std::string& event_label) {
    start_event_timer_impl(event_label, false);
  }

  std::shared_ptr<CpuTimer> start_nested_event_timer(const std::string& event_label) {
    auto& event_timer = start_event_timer_impl(event_label, true);
    return event_timer.nested_event_timer;
  }

  void end_event_timer() {
    if (!event_timers_.empty()) {
      event_timers_.back().finalize();
    }
  }

 private:
  struct EventTimer {
    std::string event_label;
    time_point start_time;
    time_point end_time;
    std::shared_ptr<CpuTimer> nested_event_timer{nullptr};
    bool is_finished{false};

    EventTimer(const std::string& event_label,
               const size_t nest_level,
               const bool make_nested)
        : event_label(event_label)
        , start_time(std::chrono::high_resolution_clock::now()) {
      if (make_nested) {
        const std::string nested_event_label{event_label + " sub-steps"};
        nested_event_timer =
            std::make_shared<CpuTimer>(nested_event_label, nest_level + 1);
      }
    }

    time_point finalize() {
      if (!is_finished) {
        if (nested_event_timer != nullptr) {
          nested_event_timer->finalize();
        }
        end_time = std::chrono::high_resolution_clock::now();
        is_finished = true;
      }
      return end_time;
    }
  };

  EventTimer& start_event_timer_impl(const std::string& event_label,
                                     const bool make_nested) {
    if (!event_timers_.empty()) {
      event_timers_.back().finalize();
    }
    event_timers_.emplace_back(EventTimer(event_label, nest_level_, make_nested));
    return event_timers_.back();
  }

  void initialize() { timers_start_ = std::chrono::high_resolution_clock::now(); }

  void finalize() {
    timers_end_ = std::chrono::high_resolution_clock::now();
    if (!event_timers_.empty()) {
      event_timers_.back().finalize();
    }
  }

  void print_timings() {
    if (nest_level_ == 0) {
      finalize();
    }
    const std::string header{timer_label_ + " Timings"};
    const std::string header_underline{std::string(header.size(), '=')};
    const std::string nest_spaces{std::string(nest_level_ * 2, ' ')};
    std::cout << std::endl
              << nest_spaces << header << std::endl
              << nest_spaces << header_underline << std::endl;

    const size_t timings_left_margin{50};

    for (auto& event_timer : event_timers_) {
      const int64_t ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     event_timer.end_time - event_timer.start_time)
                                     .count();
      const int64_t total_ms_elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(event_timer.end_time -
                                                                timers_start_)
              .count();
      const size_t label_width{event_timer.event_label.size()};
      const size_t margin_calc{
          label_width < timings_left_margin ? timings_left_margin - label_width : 0};

      std::cout << nest_spaces << event_timer.event_label << ": "
                << std::setw(margin_calc) << std::fixed << std::setprecision(4)
                << ms_elapsed << " ms elapsed, " << total_ms_elapsed << " ms total"
                << std::endl;

      if (event_timer.nested_event_timer != nullptr) {
        event_timer.nested_event_timer->print_timings();
        std::cout << std::endl;
      }
    }
    const int64_t total_ms_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(timers_end_ - timers_start_)
            .count();
    std::cout << nest_spaces << timer_label_ << " total elapsed: " << std::fixed
              << std::setprecision(4) << total_ms_elapsed << " ms" << std::endl;
  }

  const std::string timer_label_;
  const size_t nest_level_;
  time_point timers_start_;
  time_point timers_end_;
  std::vector<EventTimer> event_timers_;
};

#endif  // #ifndef __CUDACC__