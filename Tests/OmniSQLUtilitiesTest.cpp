/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../SQLFrontend/CommandHistoryFile.h"
#include "gtest/gtest.h"

#include <boost/program_options.hpp>
#include <cstring>
#include <fstream>
#include <string>

#include <unistd.h>

#include <type_traits>
// Mocks

using GetEnvRetType = decltype(DefaultEnvResolver().getenv(""));
using GetPWUIDRetType = decltype(DefaultEnvResolver().getpwuid(0));

namespace {

std::string makeUnitTestHomeDirectory() {
  return std::string("/tmp/heavydb_command_history_home_") + std::to_string(::getuid()) +
         "_" + std::to_string(::getpid());
}

const char* unitTestHomeDirectory() {
  static const std::string path = makeUnitTestHomeDirectory();
  return path.c_str();
}

std::string defaultHistoryPathFor(const char* home_dir) {
  return std::string(home_dir) + '/' + getDefaultHistoryFilename();
}

std::string defaultResolvedHistoryPath() {
  DefaultEnvResolver resolver;
  if (auto* home_env = resolver.getenv("HOME")) {
    return defaultHistoryPathFor(home_env);
  }
  if (auto* pw_dir = resolver.getpwdir(resolver.getuid())) {
    return defaultHistoryPathFor(pw_dir);
  }
  return getDefaultHistoryFilename();
}

class DefaultUnitTestResolver {
 public:
  template <typename... ARGS>
  GetEnvRetType getenv(ARGS&&...) const {
    return nullptr;
  }
  template <typename... ARGS>
  GetPWUIDRetType getpwuid(ARGS&&...) const {
    return nullptr;
  }
  template <typename... ARGS>
  const char* getpwdir(ARGS&&...) const {
    return nullptr;
  }
  auto getuid() const { return ::getuid(); }
};

class NoHomeNoPWEntResolver : public DefaultUnitTestResolver {};

class NoHomePWEntResolver : public DefaultUnitTestResolver {
 public:
  template <typename... ARGS>
  const char* getpwdir(ARGS&&...) const {
    return unitTestHomeDirectory();
  }
};

class HomeResolver : public DefaultUnitTestResolver {
 public:
  template <typename... ARGS>
  GetEnvRetType getenv(ARGS&&...) const {
    return unitTestHomeDirectory();
  }
  template <typename... ARGS>
  GetPWUIDRetType getpwuid(ARGS&&...) const {
    throw std::runtime_error("Unexpected getpwuid() invocation.");
  }
  template <typename... ARGS>
  const char* getpwdir(ARGS&&...) const {
    throw std::runtime_error("Unexpected getpwdir() invocation.");
  }
};

}  // namespace

// Mock-base class equivalents of CommandHistoryFile
using CommandHistoryFile_NoHomeNoPWEnt = CommandHistoryFileImpl<NoHomeNoPWEntResolver>;
using CommandHistoryFile_NoHomePWEnt = CommandHistoryFileImpl<NoHomePWEntResolver>;
using CommandHistoryFile_Home = CommandHistoryFileImpl<HomeResolver>;

// Tests
TEST(CommandHistoryFile, NoHomeEnv) {
  CommandHistoryFile_NoHomeNoPWEnt cmd_file;
  ASSERT_EQ(std::string(getDefaultHistoryFilename()), std::string(cmd_file));

  CommandHistoryFile_NoHomePWEnt cmd_file2;
  ASSERT_EQ(defaultHistoryPathFor(unitTestHomeDirectory()), std::string(cmd_file2));
}

TEST(CommandHistoryFile, HomeEnv) {
  CommandHistoryFile_Home cmd_file;
  ASSERT_EQ(defaultHistoryPathFor(unitTestHomeDirectory()), std::string(cmd_file));
}

TEST(CommandHistoryFile, Basic) {
  CommandHistoryFile cmd_file;
  ASSERT_EQ(defaultResolvedHistoryPath(), std::string(cmd_file));
}

TEST(CommandHistoryFile, Custom) {
  CommandHistoryFile cmd_file("mutley.txt");
  ASSERT_EQ(std::string("mutley.txt"), std::string(cmd_file));
}

TEST(CommandHistoryFile, BoostProgramOptionsCompatibility_DefaultOption) {
  namespace po = boost::program_options;
  po::options_description desc("Options");
  CommandHistoryFile cmd_file;

  int fake_argc = 1;
  char const* fake_argv[] = {"lulz"};

  desc.add_options()(
      "history", po::value<CommandHistoryFile>(&cmd_file), "History filename");
  po::variables_map vm;
  po::store(po::command_line_parser(fake_argc, fake_argv).options(desc).run(), vm);
  po::notify(vm);

  ASSERT_EQ(defaultResolvedHistoryPath(), std::string(cmd_file));
}

TEST(CommandHistoryFile, BoostProgramOptionsCompatibility_SetOption) {
  namespace po = boost::program_options;
  po::options_description desc("Options");
  CommandHistoryFile cmd_file;

  int fake_argc = 2;
  char const* fake_argv[] = {"lulz", "--history=dudley_dawson.txt"};

  desc.add_options()(
      "history", po::value<CommandHistoryFile>(&cmd_file), "History filename");
  po::variables_map vm;
  po::store(po::command_line_parser(fake_argc, fake_argv).options(desc).run(), vm);
  po::notify(vm);

  ASSERT_EQ(std::string("dudley_dawson.txt"), std::string(cmd_file));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
