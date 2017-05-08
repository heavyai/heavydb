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

#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include "cuda.h"
#include <glog/logging.h>

template <typename TimeT = std::chrono::milliseconds>
struct measure {
  template <typename F, typename... Args>
  static typename TimeT::rep execution(F func, Args&&... args) {
    auto start = std::chrono::steady_clock::now();
    func(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);
    return duration.count();
  }
};

void checkCudaErrors(CUresult err) {
  if (err != CUDA_SUCCESS) {
    std::cout << err << std::endl;
  }
  assert(err == CUDA_SUCCESS);
}

/// main - Program entry point
int main(int argc, char** argv) {
  CUdevice device;
  CUmodule cudaModule;
  CUcontext context;
  CUfunction function;

  // CUDA initialization
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGet(&device, 0));

  std::ifstream t("kernel.ptx");
  std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

  // Create driver context
  checkCudaErrors(cuCtxCreate(&context, 0, device));

  // Create module for object
  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));

  // Get kernel function
  checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

  int64_t N = 1000000000L;
  int8_t* byte_stream_col_0 = new int8_t[N];
  memset(byte_stream_col_0, 42, N);

  CUdeviceptr devBufferA;
  checkCudaErrors(cuMemAlloc(&devBufferA, sizeof(int8_t) * N));
  checkCudaErrors(cuMemcpyHtoD(devBufferA, byte_stream_col_0, sizeof(int8_t) * N));

  CUdeviceptr devBufferAA;
  checkCudaErrors(cuMemAlloc(&devBufferAA, sizeof(CUdeviceptr)));
  checkCudaErrors(cuMemcpyHtoD(devBufferAA, &devBufferA, sizeof(CUdeviceptr)));

  unsigned blockSizeX = 128;
  unsigned blockSizeY = 1;
  unsigned blockSizeZ = 1;
  unsigned gridSizeX = 128;
  unsigned gridSizeY = 1;
  unsigned gridSizeZ = 1;

  CUdeviceptr devBufferB;
  int64_t* result_vec = new int64_t[blockSizeX * gridSizeX * sizeof(int64_t)];
  checkCudaErrors(cuMemAlloc(&devBufferB, blockSizeX * gridSizeX * sizeof(int64_t)));

  CUdeviceptr devBufferN;
  int64_t row_count = N;
  checkCudaErrors(cuMemAlloc(&devBufferN, sizeof(int64_t)));
  checkCudaErrors(cuMemcpyHtoD(devBufferN, &row_count, sizeof(int64_t)));

  CUdeviceptr devBufferI;
  int64_t init_agg_val = 0;
  checkCudaErrors(cuMemAlloc(&devBufferI, sizeof(int64_t)));
  checkCudaErrors(cuMemcpyHtoD(devBufferI, &init_agg_val, sizeof(int64_t)));

  void* KernelParams[] = {&devBufferAA, &devBufferN, &devBufferI, &devBufferB};

  LOG(INFO) << measure<std::chrono::microseconds>::execution([&]() {
    checkCudaErrors(cuLaunchKernel(
        function, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, 0, NULL, KernelParams, NULL));
    checkCudaErrors(cuMemcpyDtoH(result_vec, devBufferB, blockSizeX * gridSizeX * sizeof(int64_t)));
  });

  int64_t result = 0;
  for (size_t i = 0; i < blockSizeX * gridSizeX; ++i) {
    result += result_vec[i];
  }
  std::cout << result << std::endl;

  delete[] result_vec;
  delete[] byte_stream_col_0;

  // Clean-up
  checkCudaErrors(cuMemFree(devBufferA));
  checkCudaErrors(cuMemFree(devBufferAA));
  checkCudaErrors(cuMemFree(devBufferB));
  checkCudaErrors(cuMemFree(devBufferN));
  checkCudaErrors(cuModuleUnload(cudaModule));
  checkCudaErrors(cuCtxDestroy(context));

  return 0;
}
