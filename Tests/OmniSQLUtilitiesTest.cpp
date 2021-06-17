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

#include "../SQLFrontend/CommandHistoryFile.h"
#include "gtest/gtest.h"

#include <boost/program_options.hpp>
#include <cstring>
#include <fstream>

#include <type_traits>
// Mocks

using GetEnvRetType = decltype(DefaultEnvResolver().getenv(""));
#ifndef _WIN32
using GetPWUIDRetType = decltype(DefaultEnvResolver().getpwuid(0));
#endif

class DefaultUnitTestResolver {
 public:
  template <typename... ARGS>
  GetEnvRetType getenv(ARGS&&...) const {
    return nullptr;
  }
#ifndef _WIN32
  template <typename... ARGS>
  GetPWUIDRetType getpwuid(ARGS&&...) const {
    return nullptr;
  }
#endif
  template <typename... ARGS>
  const char* getpwdir(ARGS&&...) const {
    return nullptr;
  }
  auto getuid() const { return ::getuid(); }
};

class NoHomeNoPWEntResolver : public DefaultUnitTestResolver {};

class NoHomePWEntResolver : public DefaultEnvResolver {
 public:
  template <typename... ARGS>
  GetEnvRetType getenv(ARGS&&...) const {
    return nullptr;
  }
};

class HomeResolver : public DefaultEnvResolver {
 public:
  template <typename... ARGS>
  GetEnvRetType getenv(ARGS&&... args) const {
    return DefaultEnvResolver::getenv(std::forward<ARGS>(args)...);
  }
#ifndef _WIN32
  template <typename... ARGS>
  GetPWUIDRetType getpwuid(ARGS&&...) const {
    throw std::runtime_error("Unexpected getpwuid() invocation.");
  }
#endif
  template <typename... ARGS>
  const char* getpwdir(ARGS&&...) const {
    throw std::runtime_error("Unexpected getpwdir() invocation.");
  }
};

// Mock-base class equivalents of CommandHistoryFile
using CommandHistoryFile_NoHomeNoPWEnt = CommandHistoryFileImpl<NoHomeNoPWEntResolver>;
using CommandHistoryFile_NoHomePWEnt = CommandHistoryFileImpl<NoHomePWEntResolver>;
using CommandHistoryFile_Home = CommandHistoryFileImpl<HomeResolver>;

// Tests
TEST(CommandHistoryFile, NoHomeEnv) {
  CommandHistoryFile_NoHomeNoPWEnt cmd_file;
  ASSERT_EQ(std::string(getDefaultHistoryFilename()), std::string(cmd_file));

  CommandHistoryFile_NoHomePWEnt cmd_file2;
  ASSERT_EQ(getHomeDirectory() + '/' + std::string(getDefaultHistoryFilename()),
            std::string(cmd_file2));
}

TEST(CommandHistoryFile, HomeEnv) {
  CommandHistoryFile_Home cmd_file;
  ASSERT_EQ(getHomeDirectory() + '/' + std::string(getDefaultHistoryFilename()),
            std::string(cmd_file));
}

TEST(CommandHistoryFile, Basic) {
  CommandHistoryFile cmd_file;
  ASSERT_EQ(getHomeDirectory() + '/' + std::string(getDefaultHistoryFilename()),
            std::string(cmd_file));
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

  ASSERT_EQ(getHomeDirectory() + '/' + std::string(getDefaultHistoryFilename()),
            std::string(cmd_file));
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
