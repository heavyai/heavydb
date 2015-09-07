/*
 * Copyright (c) 2014 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <math.h>
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>
#include "nvvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// The full path to the libcudadevrt.a is determined by the build environment.
const char* _libCudaDevRt = "/usr/local/cuda-6.5/targets/x86_64-linux/lib/libcudadevrt.a";

const char* getLibCudaDevRtName(void) {
  // double check the library exists
  FILE* fh = fopen(_libCudaDevRt, "rb");

  if (fh == NULL) {
    fprintf(stderr, "Error reading file %s\n", _libCudaDevRt);
    exit(-1);
  }
  fclose(fh);

  return _libCudaDevRt;
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions

void __checkCudaErrors(CUresult err, const char* file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
            err,
            getCudaDrvErrorString(err),
            file,
            line);
    exit(-1);
  }
}

CUdevice cudaDeviceInit() {
  CUdevice cuDevice = 0;
  int deviceCount = 0;
  CUresult err = cuInit(0);
  char name[100];
  int major = 0, minor = 0;

  if (CUDA_SUCCESS == err)
    checkCudaErrors(cuDeviceGetCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
    exit(-1);
  }
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));
  cuDeviceGetName(name, 100, cuDevice);
  printf("Using CUDA Device [0]: %s\n", name);

  checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
  printf("compute capability = %d.%d\n", major, minor);
  if (major < 3) {
    fprintf(stderr, "Device 0 is not sm_30 or later\n");
    exit(-1);
  }
  return cuDevice;
}

CUresult initCUDA(CUcontext* phContext,
                  CUdevice* phDevice,
                  CUmodule* phModule,
                  CUfunction* phKernel,
                  const char* ptx,
                  const char* libCudaDevRtName) {
  CUlinkState linkState;
  void* cubin;
  size_t cubinSize;

  // Initialize
  *phDevice = cudaDeviceInit();
  // Create context on the device
  checkCudaErrors(cuCtxCreate(phContext, 0, *phDevice));
  // link ptx and the device library
  checkCudaErrors(cuLinkCreate(0, NULL, NULL, &linkState));
  checkCudaErrors(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx, strlen(ptx) + 1, 0, 0, 0, 0));
  checkCudaErrors(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, libCudaDevRtName, 0, NULL, NULL));
  checkCudaErrors(cuLinkComplete(linkState, &cubin, &cubinSize));
  checkCudaErrors(cuLinkDestroy(linkState));
  // load the linked binary
  checkCudaErrors(cuModuleLoadData(phModule, cubin));
  // Locate the kernel entry poin
  checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "kernel"));

  return CUDA_SUCCESS;
}

char* loadProgramSource(const char* filename, size_t* size) {
  struct stat statbuf;
  FILE* fh;
  char* source = NULL;
  *size = 0;
  fh = fopen(filename, "rb");
  if (fh) {
    stat(filename, &statbuf);
    source = (char*)malloc(statbuf.st_size + 1);
    if (source) {
      size_t bytes_read = fread(source, statbuf.st_size, 1, fh);
      if (bytes_read < static_cast<size_t>(statbuf.st_size)) {
        fprintf(stderr, "Error reading file %s\n", filename);
        exit(-1);
      }
      source[statbuf.st_size] = 0;
      *size = statbuf.st_size + 1;
    }
  } else {
    fprintf(stderr, "Error reading file %s\n", filename);
    exit(-1);
  }
  return source;
}

char* generatePTX(const char* ll, size_t size, const char* filename) {
  nvvmResult result;
  nvvmProgram program;
  size_t PTXSize;
  char* PTX = NULL;
  const char* options[] = {"-arch=compute_30"};

  result = nvvmCreateProgram(&program);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmCreateProgram: Failed\n");
    exit(-1);
  }

  result = nvvmAddModuleToProgram(program, ll, size, filename);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmAddModuleToProgram: Failed\n");
    exit(-1);
  }

  result = nvvmCompileProgram(program, 1, options);
  if (result != NVVM_SUCCESS) {
    char* Msg = NULL;
    size_t LogSize;
    fprintf(stderr, "nvvmCompileProgram: Failed\n");
    nvvmGetProgramLogSize(program, &LogSize);
    Msg = (char*)malloc(LogSize);
    nvvmGetProgramLog(program, Msg);
    fprintf(stderr, "%s\n", Msg);
    free(Msg);
    return nullptr;
  }

  result = nvvmGetCompiledResultSize(program, &PTXSize);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
    exit(-1);
  }

  PTX = (char*)malloc(PTXSize);
  result = nvvmGetCompiledResult(program, PTX);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmGetCompiledResult: Failed\n");
    free(PTX);
    exit(-1);
  }

  result = nvvmDestroyProgram(&program);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmDestroyProgram: Failed\n");
    free(PTX);
    exit(-1);
  }

  return PTX;
}

int main_sample(int argc, char** argv) {
  const unsigned int nThreads = 1;
  const unsigned int nBlocks = 1;

  CUcontext hContext = 0;
  CUdevice hDevice = 0;
  CUmodule hModule = 0;
  CUfunction hKernel = 0;
  char* ptx = NULL;
  const char* libCudaDevRtName = NULL;
  int depth = 0;

  // Get the ll from file
  size_t size = 0;
  // Kernel parameters
  void* params[] = {&depth};
#if BUILD_64_BIT
  const char* filename = "gpu64.ll";
#else
  const char* filename = "gpu32.ll";
#endif
  char* ll = loadProgramSource(filename, &size);
  fprintf(stdout, "NVVM IR ll file loaded\n");

  libCudaDevRtName = getLibCudaDevRtName();

  // Use libnvvm to generte PTX
  ptx = generatePTX(ll, size, filename);
  fprintf(stdout, "PTX generated:\n");
  fprintf(stdout, "%s\n", ptx);

  // Initialize the device and get a handle to the kernel
  checkCudaErrors(initCUDA(&hContext, &hDevice, &hModule, &hKernel, ptx, libCudaDevRtName));

  // Launch the kernel
  checkCudaErrors(cuLaunchKernel(hKernel, nBlocks, 1, 1, nThreads, 1, 1, 0, NULL, params, NULL));

  if (hModule) {
    checkCudaErrors(cuModuleUnload(hModule));
    hModule = 0;
  }
  if (hContext) {
    checkCudaErrors(cuCtxDestroy(hContext));
    hContext = 0;
  }

  free(ll);
  free(ptx);

  return 0;
}
