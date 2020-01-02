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
 * @file CommandLineTest.cpp
 * @brief Test suite for executables, scripts, their respecitve flags,
 * and other functionality invoked from the command line.
 */

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>

#include "Shared/Logger.h"
#include "TestHelpers.h"

namespace bp = boost::process;
namespace bf = boost::filesystem;
using path = bf::path;

namespace {
bool find_file(const path& root, const path& fileName, path& result) {
  bool found = false;
  const bf::recursive_directory_iterator begin(root), end;
  const auto it = std::find_if(begin, end, [&fileName](const bf::directory_entry& e) {
    return e.path().filename() == fileName;
  });

  if (it != end) {
    result = it->path();
    found = true;
  }

  return found;
}
}  // namespace

// Class to test the initdb executable.
class InitDBTest : public testing::Test {
 private:
  path initdb_;
  const std::string temp_dir_ = "temp_data";
  const std::string nonexistant_dir_ = "temp_data2";

 protected:
  void SetUp() override {
    ASSERT_FALSE(bf::exists(temp_dir_));
    ASSERT_FALSE(bf::exists(nonexistant_dir_));
    bf::create_directory(temp_dir_);
    ASSERT_TRUE(find_file(bf::relative(path("../")), "initdb", initdb_));
  }
  void TearDown() override { bf::remove_all(temp_dir_); }

 public:
  path get_executable() const { return initdb_; }
  const std::string get_temp_dir() const { return temp_dir_; }
  const std::string get_nonexistant_dir() const { return nonexistant_dir_; }
};

// Contains the a testcase that involves running an executable.
class CommandLineTestcase {
 private:
  path executable_;
  int expected_return_code_;
  std::string flags, expected_std_out_, expected_std_err_;
  std::string std_out_line_, std_err_line_, std_out_string_ = "", std_err_string_ = "";
  bp::ipstream std_out_pipe_, std_err_pipe_;

  // Runs the testcase and evalutates return code, stdErr, and stdOut.
  void evaluate() {
    int returnCode = bp::system(executable_.string() + " " + flags,
                                bp::std_out > std_out_pipe_,
                                bp::std_err > std_err_pipe_);
    while (std::getline(std_out_pipe_, std_out_line_)) {
      std_out_string_ += std_out_line_;
    }
    while (std::getline(std_err_pipe_, std_err_line_)) {
      std_err_string_ += std_err_line_;
    }
    // Since we are using raw strings, prune out any newlines.
    boost::erase_all(expected_std_out_, "\n");
    boost::erase_all(expected_std_err_, "\n");

    ASSERT_EQ(returnCode, expected_return_code_);
    ASSERT_EQ(std_out_string_, expected_std_out_);
    ASSERT_EQ(std_err_string_, expected_std_err_);
  }

 public:
  CommandLineTestcase(path e, std::string f, bool rc, std::string so, std::string se)
      : executable_(e)
      , expected_return_code_(rc)
      , flags(f)
      , expected_std_out_(so)
      , expected_std_err_(se) {
    evaluate();
  }
};

// No data directory specified.
TEST_F(InitDBTest, NoDataFlag) {
  CommandLineTestcase(get_executable(),
                      "",
                      1,
                      "",
                      "Usage Error: the option '--data' is required but missing");
}
// Help flag.
TEST_F(InitDBTest, Help) {
  CommandLineTestcase(get_executable(),
                      "-h",
                      0,
                      R"(Options:
  -h [ --help ]                         Print help messages 
  --data arg                            Directory path to OmniSci catalogs
  -f [ --force ]                        Force overwriting of existing OmniSci 
                                        instance
  --skip-geo                            Skip inserting sample geo data
  --enable-thrift-logs [=arg(=1)] (=0)  Enable writing messages directly from 
                                        thrift to stdout/stderr.

Logging:
  --log-directory arg (="mapd_log")     Logging directory. May be relative to 
                                        data directory, or absolute.
  --log-file-name arg (=initdb.{SEVERITY}.%Y%m%d-%H%M%S.log)
                                        Log file name relative to 
                                        log-directory.
  --log-symlink arg (=initdb.{SEVERITY})
                                        Symlink to active log.
  --log-severity arg (=INFO)            Log to file severity level: INFO 
                                        WARNING ERROR FATAL
  --log-severity-clog arg (=ERROR)      Log to console severity level: INFO 
                                        WARNING ERROR FATAL
  --log-channels arg                    Log channel debug info: IR PTX
  --log-auto-flush arg (=1)             Flush logging buffer to file after each
                                        message.
  --log-max-files arg (=100)            Maximum number of log files to keep.
  --log-min-free-space arg (=20971520)  Minimum number of bytes left on device 
                                        before oldest log files are deleted.
  --log-rotate-daily arg (=1)           Start new log files at midnight.
  --log-rotation-size arg (=10485760)   Maximum file size in bytes before new 
                                        log files are started.)",
                      "");
}
// Base case - empty directory to init.
TEST_F(InitDBTest, EmptyDir) {
  CommandLineTestcase(get_executable(), get_temp_dir(), 0, "", "");
}
// Blocked by existing database.
TEST_F(InitDBTest, AlreadyInit) {
  CommandLineTestcase(get_executable(), get_temp_dir(), 0, "", "");
  CommandLineTestcase(get_executable(),
                      get_temp_dir(),
                      1,
                      "",
                      "OmniSci catalogs already initialized at " + get_temp_dir() +
                          ". Use -f to force reinitialization.");
}
// Override existing database.
TEST_F(InitDBTest, Force) {
  CommandLineTestcase(get_executable(), get_temp_dir(), 0, "", "");
  CommandLineTestcase(get_executable(), get_temp_dir() + " -f", 0, "", "");
}
// Data directory does not exist.
TEST_F(InitDBTest, MissingDir) {
  CommandLineTestcase(get_executable(),
                      get_nonexistant_dir(),
                      1,
                      "",
                      "Catalog basepath " + get_nonexistant_dir() + " does not exist.");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
