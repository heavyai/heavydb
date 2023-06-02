#include "RuntimeLibManager.h"
#include "Logger/Logger.h"
#include "OSDependent/heavyai_path.h"

#include <boost/dll/shared_library.hpp>

#include <map>
#include <memory>
#include <unordered_map>

boost::filesystem::path get_runtime_test_lib_tfs_path() {
  boost::filesystem::path runtime_test_lib_tfs_path{heavyai::get_root_abs_path()};
  runtime_test_lib_tfs_path /= "QueryEngine";
  runtime_test_lib_tfs_path /= "TableFunctions";
  runtime_test_lib_tfs_path /= "RuntimeLibTestFunctions";
  runtime_test_lib_tfs_path /= "RuntimeLibTestTableFunctions";

  return runtime_test_lib_tfs_path;
}

boost::filesystem::path get_runtime_test_lib_path() {
  boost::filesystem::path runtime_test_lib_path{heavyai::get_root_abs_path()};
  runtime_test_lib_path /= "QueryEngine";
  runtime_test_lib_path /= "TableFunctions";
  runtime_test_lib_path /= "RuntimeLibTestFunctions";
  runtime_test_lib_path /= "TestRuntimeLib";

  return runtime_test_lib_path;
}

void RuntimeLibManager::loadRuntimeLibs() {}

boost::dll::shared_library testLib;
boost::dll::shared_library testLibTFs;

void RuntimeLibManager::loadTestRuntimeLibs() {
  testLib = boost::dll::shared_library(get_runtime_test_lib_path(),
                                       boost::dll::load_mode::append_decorations |
                                           boost::dll::load_mode::rtld_global |
                                           boost::dll::load_mode::search_system_folders |
                                           boost::dll::load_mode::rtld_deepbind);

  if (!testLib.is_loaded()) {
    throw(std::runtime_error("Failed to load test runtime library!"));
  }

  testLibTFs = boost::dll::shared_library(
      get_runtime_test_lib_tfs_path(),
      boost::dll::load_mode::rtld_lazy | boost::dll::load_mode::search_system_folders |
          boost::dll::load_mode::rtld_global | boost::dll::load_mode::append_decorations |
          boost::dll::load_mode::rtld_deepbind);
  if (!testLibTFs.is_loaded()) {
    throw(std::runtime_error("Failed to load test runtime library table functions!!"));
  }

  if (!testLibTFs.has("init_table_functions")) {
    LOG(FATAL)
        << "Test runtime table function library has no init_table_functions symbol!";
  }
  auto initFunction = testLibTFs.get<void(void)>("init_table_functions");
  initFunction();
}