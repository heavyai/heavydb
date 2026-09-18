/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMMANDHISTORYFILE_H
#define COMMANDHISTORYFILE_H

#include <iostream>
#include <string>
#include <utility>

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

inline constexpr char const* const getDefaultHistoryFilename() {
  return ".heavysql_history";
}

class DefaultEnvResolver {
 public:
  uid_t getuid() const { return ::getuid(); }

  auto const* getpwuid(uid_t uid) const { return ::getpwuid(uid); }

  const char* getpwdir(uid_t uid) const {
    auto* p = getpwuid(uid);
    if (p) {
      return p->pw_dir;
    }
    return nullptr;
  }

  auto const* getenv(char const* env_var_name) const {
    return ::secure_getenv(env_var_name);
  }
};

std::string getHomeDirectory() {
  if (auto const* home = ::secure_getenv("HOME")) {
    return home;
  }
  DefaultEnvResolver r;
  return r.getpwdir(r.getuid());
}

template <typename ENV_RESOLVER>
class CommandHistoryFileImpl : private ENV_RESOLVER {
 public:
  CommandHistoryFileImpl() : CommandHistoryFileImpl(resolveCommandFile()) {}
  explicit CommandHistoryFileImpl(std::string command_file_name)
      : command_file_name_(command_file_name) {}

  operator char const*() const { return command_file_name_.c_str(); }

  template <typename ER>
  friend inline std::istream& operator>>(std::istream& i,
                                         CommandHistoryFileImpl<ER>& cmd_file);
  template <typename ER>
  friend inline std::ostream& operator<<(std::ostream& o,
                                         CommandHistoryFileImpl<ER> const& cmd_file);

 private:
  const char* resolveHomeDirectory() const {
    auto* home_env = this->getenv("HOME");
    if (home_env == nullptr) {
      return ENV_RESOLVER::getpwdir(ENV_RESOLVER::getuid());
    }
    return home_env;
  }

  std::string const resolveCommandFile() const {
    auto home_dir = resolveHomeDirectory();
    if (home_dir == nullptr) {  // Just use default command history file name in current
                                // dir in this scenario
      return std::string(getDefaultHistoryFilename());
    }
    return std::string(home_dir) + '/' + getDefaultHistoryFilename();
  }

 private:
  std::string command_file_name_;
};

template <typename ENV_RESOLVER>
inline std::ostream& operator<<(std::ostream& o,
                                CommandHistoryFileImpl<ENV_RESOLVER> const& cmd_file) {
  o << cmd_file.command_file_name_;
  return o;
}

template <typename ENV_RESOLVER>
inline std::istream& operator>>(std::istream& i,
                                CommandHistoryFileImpl<ENV_RESOLVER>& cmd_file) {
  i >> cmd_file.command_file_name_;
  return i;
}

using CommandHistoryFile = CommandHistoryFileImpl<DefaultEnvResolver>;

#endif
