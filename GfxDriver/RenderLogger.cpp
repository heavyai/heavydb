/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/RenderLogger.h"

#include <iomanip>
#include <iostream>
#include <limits>

#include <boost/filesystem.hpp>

#include "Logger/Logger.h"
#include "Shared/nvtx_helpers.h"

namespace render_logger {

static constexpr bool log_scope_closure = false;
static constexpr bool inject_debug_timer = false;

int g_scope_depth{0};
int g_max_scope_depth{std::numeric_limits<int>::max()};
Sink g_sink{Sink::kStdOut};

// Return last component of path
std::string filename(char const* path) {
  return boost::filesystem::path(path).filename().string();
}

void newline() {
  switch (g_sink) {
    case Sink::kNull:
      break;
    case Sink::kStdOut:
      std::cout << std::endl;
      break;
    case Sink::kStdErr:
      std::cerr << std::endl;
      break;
    case Sink::kLogInfo:
      LOG(INFO) << std::endl;
      break;
  }
}

void flush(std::ostringstream& os) {
  if (os.rdbuf()->in_avail() == 0) {
    return;
  }
  os << std::endl;
  switch (g_sink) {
    case Sink::kNull:
      break;
    case Sink::kStdOut:
      std::cout << os.str();
      break;
    case Sink::kStdErr:
      std::cerr << os.str();
      break;
    case Sink::kLogInfo:
      LOG(INFO) << os.str();
      break;
  }
  os.str("");
  os.clear();
}

Closure::~Closure() {
  g_scope_depth--;
  if constexpr (log_scope_closure) {
    RENDER_LOG() << name_ << " END";
  }
  if (g_scope_depth == 0 && g_max_scope_depth > 1) {
    // add an extra newline when exiting root scope
    newline();
  }
  nvtx_helpers::heavyai_range_pop();
}

Logger::Logger() : os_{std::make_unique<std::ostringstream>()} {
  *os_ << std::boolalpha;  // default to true/false for bools
}

Logger::Logger(const char* file, int line, const char* name, ClosureUqPtr& closure)
    : os_{std::make_unique<std::ostringstream>()} {
  CHECK(name != nullptr) << "Scoped RenderLogger requires a name";
  *os_ << std::boolalpha;  // default to true/false for bools
  indent() << name << "  " << filename(file) << ':' << line << " ";
  closure = std::make_unique<Closure>();
  if constexpr (log_scope_closure) {
    closure->name_ = name;
  }
  if constexpr (inject_debug_timer) {
    closure->debug_timer_ =
        std::make_unique<logger::DebugTimer>(logger::INFO, file, line, name);
  }
  g_scope_depth++;
  nvtx_helpers::heavyai_range_push(nvtx_helpers::Category::kRenderLogger, name, file);
}

Logger::Logger(const char* file,
               int line,
               const char* name,
               ClosureUqPtr& closure,
               const std::set<uint32_t>& gpu_set)
    : Logger(file, line, name, closure) {
  *os_ << "- " << format_gpuid_set(gpu_set);
}

Logger::Logger(const char* file,
               int line,
               const char* name,
               ClosureUqPtr& closure,
               const std::vector<uint32_t>& gpu_vector)
    : Logger(file, line, name, closure) {
  *os_ << "- " << format_gpuid_vector(gpu_vector);
}

Logger::Logger(const char* file,
               int line,
               const char* name,
               ClosureUqPtr& closure,
               LogCallback callback)
    : Logger(file, line, name, closure) {
  callback(*os_);
}

Logger::Logger(const char* file,
               int line,
               const char* name,
               ClosureUqPtr& closure,
               uint32_t gpu_id)
    : Logger(file, line, name, closure) {
  *os_ << "- on gpu: " << gpu_id << " ";
}

Logger::~Logger() {
  flush(*os_);
}

std::ostream& Logger::get() {
  return *os_;
}

std::ostream& Logger::indent() {
  if (g_scope_depth > 0) {
    *os_ << std::left << std::setw(g_scope_depth * 2) << ' ';
  }
  return *os_;
}

int Logger::getDepth() {
  return g_scope_depth;
}
int& Logger::getMaxDepth() {
  return g_max_scope_depth;
}
Sink& Logger::getSink() {
  return g_sink;
}

}  // namespace render_logger
