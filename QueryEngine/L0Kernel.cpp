#include "QueryEngine/L0Kernel.h"

#include "L0Mgr/L0Exception.h"
#include "L0Mgr/Utils.h"
#include "Logger/Logger.h"  // CHECK

#include <iostream>

#include <level_zero/ze_api.h>

L0BinResult spv_to_bin(const std::string& spv,
                       const unsigned block_size,
                       const l0::L0Manager* mgr) {
  CHECK(!spv.empty());
  CHECK(mgr);

  void* bin{nullptr};
  size_t binSize{0};

  auto driver = mgr->drivers()[0];
  auto device = driver->devices()[0];

  CHECK(driver);
  CHECK(device);

  std::ofstream out("complete.spv", std::ios::binary);
  out.write((char*)spv.data(), spv.size());

  auto module = device->create_module((uint8_t*)spv.data(), spv.size());
  std::cerr << "Module created" << std::endl;
  auto kernel = module->create_kernel("wrapper_scalar_expr", 1, 1, 1);

  L0_SAFE_CALL(zeModuleGetNativeBinary(module->handle(), &binSize, nullptr));

  uint8_t* pBinary = new uint8_t[binSize];
  L0_SAFE_CALL(zeModuleGetNativeBinary(module->handle(), &binSize, pBinary));
  bin = pBinary;

  // ~L0Kernel
  // ~L0Module
  return {bin, binSize};
}

L0DeviceCompilationContext::L0DeviceCompilationContext(const void* image,
                                                       const size_t image_size,
                                                       const std::string& kernel_name,
                                                       const int device_id,
                                                       unsigned int num_options,
                                                       void** option_vals)
    : device_id_(device_id) {}

L0DeviceCompilationContext::~L0DeviceCompilationContext() {}