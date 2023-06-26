#ifdef HAVE_TORCH_TFS
#include <torch/version.h>
#endif

#include "Logger/Logger.h"
#include "OSDependent/heavyai_path.h"
#include "RuntimeLibManager.h"

#include <boost/dll/shared_library.hpp>

#include <map>
#include <memory>
#include <unordered_map>

boost::dll::shared_library libtorch;
boost::dll::shared_library libTorchTFs;

bool RuntimeLibManager::is_libtorch_loaded_;

boost::filesystem::path get_torch_table_functions_path() {
  boost::filesystem::path torch_table_functions_path{heavyai::get_root_abs_path()};
  torch_table_functions_path /= "QueryEngine";
  torch_table_functions_path /= "TableFunctions";
  torch_table_functions_path /= "SystemFunctions";
  torch_table_functions_path /= "os";
  torch_table_functions_path /= "Torch";
  torch_table_functions_path /= "TorchTableFunctions";

  return torch_table_functions_path;
}

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

void RuntimeLibManager::loadRuntimeLibs(const std::string& torch_lib_path) {
  is_libtorch_loaded_ = false;
#ifdef HAVE_TORCH_TFS
  // Third party libraries must be loaded with the RTLD_GLOBAL flag set, so their symbols
  // can be available for resolution by further dynamic loads
  // They must also be loaded using RTLD_DEEPBIND, in case they have statically linked
  // symbols from common libraries that conflict with symbols also statically linked with
  // heavydb. For instance, LibTorch also statically links with LLVM, so not using
  // RTLD_DEEPBIND will cause issues with LibTorch referring to heavy's LLVM symbols or
  // vice-versa.
  if (!torch_lib_path.empty()) {
    boost::dll::fs::path lib_path(torch_lib_path);
    libtorch = boost::dll::shared_library(
        lib_path,
        boost::dll::load_mode::rtld_global | boost::dll::load_mode::rtld_deepbind);
  } else {
    libtorch = boost::dll::shared_library("libtorch",
                                          boost::dll::load_mode::search_system_folders |
                                              boost::dll::load_mode::append_decorations |
                                              boost::dll::load_mode::rtld_global |
                                              boost::dll::load_mode::rtld_deepbind);
  }

  if (!libtorch.is_loaded()) {
    throw(std::runtime_error(
        "Failed to load runtime library LibTorch. Support for library is disabled!"));
  }

  LOG(WARNING) << "This HeavyDB was built against LibTorch version " << TORCH_VERSION
               << ". Using a different version of LibTorch may cause breaking behavior!";

  // Shared libraries containing table function implementations must also be loaded
  // using RTLD_GLOBAL, as the JIT-ed LLVM code needs to resolve the symbols at runtime.
  // They must also be loaded using RTLD_DEEPBIND, which makes sure name binding favors
  // symbols local to the library before considering global ones. This makes sure the
  // TableFunctionsFactory::init() calls within the library call its own init() function
  // rather than the server's main one.
  // TODO: This behavior may not be supported in Windows, we should test symbol
  // resolution in Windows platforms to make sure this works.
  boost::dll::fs::path torch_tfs_lib_path(get_torch_table_functions_path());
  libTorchTFs = boost::dll::shared_library(
      torch_tfs_lib_path,
      boost::dll::load_mode::rtld_lazy | boost::dll::load_mode::search_system_folders |
          boost::dll::load_mode::rtld_global | boost::dll::load_mode::append_decorations |
          boost::dll::load_mode::rtld_deepbind);
  if (!libTorchTFs.is_loaded()) {
    throw(
        std::runtime_error("Failed to load LibTorch table function module. LibTorch "
                           "table function support is disabled!"));
  }

  if (!libTorchTFs.has("init_table_functions")) {
    LOG(FATAL) << "Load-time table function module does not contain "
                  "'init_table_functions' symbol!";
  }

  // calls the module's TableFunctionsFactory::init() to register table functions
  auto initFunction = libTorchTFs.get<void(void)>("init_table_functions");
  initFunction();
  is_libtorch_loaded_ = true;
#endif
}

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

bool RuntimeLibManager::is_libtorch_loaded() {
  return is_libtorch_loaded_;
}