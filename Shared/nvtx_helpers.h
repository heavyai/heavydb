/*
 * Copyright (c) 2021 OmniSci, Inc.
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

#pragma once

#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#ifdef HAVE_CUDA
#include <nvtx3/nvToolsExtCuda.h>
#endif  // HAVE_CUDA
#else   // HAVE_NVTX
using nvtxDomainHandle_t = void*;
using nvtxRangeId_t = uint64_t;
#endif  // HAVE_NVTX

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA

namespace nvtx_helpers {

// Standard categories for the Omnisci domain
// Additional temporary categories can be named with name_domain_category().
// Permanent categories should be aded to the enum here and in nvtx_helpers.cpp
enum class Category { kNone, kDebugTimer, kQueryStateTimer, kRenderLogger };

#ifdef HAVE_NVTX
//
// Create / destroy domain and name categories
//
void init();
void shutdown();

//
// Omnisci domain
//

// Get the domain handle
// To add ranges to the domain that do not use the default categories, get the
// handle and use the domain_range_* functions. This allows using distinct colors or
// specialized categories

// NOTE: The domain will be null if NVTX profiling is not enabled by the nsys launcher.
// The various nvtxDomain* function appear to check for this case, but if the caller does
// extra work for NVTX it's a good idea to check if the domain pointer is null first
const nvtxDomainHandle_t get_omnisci_domain();

// Push and pop ranges
// Pass __FILE__ as the file param and it will be appended after name, or nullptr to
// ignore
void omnisci_range_push(Category c, const char* name, const char* file);
void omnisci_range_pop();

[[nodiscard]] nvtxRangeId_t omnisci_range_start(Category c, const char* name);
void omnisci_range_end(nvtxRangeId_t r);

void omnisci_set_mark(Category c, const char* message);

//
// Resource naming
//
// We need the OS thread ID, not the pthreads id for this method, so this is somewhat
// limiting since the only easy way to get the ID is by naming it from the thread itself
// TODO: use nvtxDomainResourceCreate to name a thread using a pthead identifier
void name_current_thread(const char* name);

//
// Name Cuda resources
//
// Naming Cuda resources seems iffy (it doesn't always work)
#ifdef HAVE_CUDA
inline void name_cuda_device(CUdevice device, const char* name) {
  nvtxNameCuDeviceA(device, name);
}
inline void name_cuda_context(CUcontext context, const char* name) {
  nvtxNameCuContextA(context, name);
}
inline void name_cuda_stream(CUstream stream, const char* name) {
  nvtxNameCuStreamA(stream, name);
}
inline void name_cuda_event(CUevent event, const char* name) {
  nvtxNameCuEventA(event, name);
}
#endif  // HAVE_CUDA

// Name a category index. If using a custom category as part of the Omnisci domain, you
// must ensure the ID does not conflict with the Category enum above
inline void name_category(uint32_t category, const char* name) {
  nvtxNameCategory(category, name);
}

inline void name_domain_category(nvtxDomainHandle_t domain,
                                 uint32_t category,
                                 const char* name) {
  nvtxDomainNameCategoryA(domain, category, name);
}

//
// Wrappers for custom timeline usage outside the Omnisci domain
//

// Create a basic event object
inline nvtxEventAttributes_t make_event_attrib(const char* name, uint32_t color) {
  nvtxEventAttributes_t event_attrib = {};
  event_attrib.version = NVTX_VERSION;
  event_attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  if (color != 0) {
    event_attrib.colorType = NVTX_COLOR_ARGB;
    event_attrib.color = color;
  }
  event_attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event_attrib.message.ascii = name;
  return event_attrib;
}

//
// Default domain marker and range support. These will be grouped under the NVTX domain
//

// Mark an instantaneous event
inline void set_mark(const char* message) {
  nvtxMark(message);
}

// Push a generic nested range begin (no domain or color)
inline void range_push(const char* name) {
  nvtxRangePushA(name);
}

// Push a nested range begin with attributes
inline void range_push(const char* name, const uint32_t category, const uint32_t color) {
  auto event_attrib = make_event_attrib(name, color);
  nvtxRangePushEx(&event_attrib);
}

// Pop a nested range
inline void range_pop() {
  nvtxRangePop();
}

// Start an unnested range with no attributes
[[nodiscard]] inline nvtxRangeId_t range_start(const char* name) {
  return nvtxRangeStartA(name);
}

// Start an unnested range with attributes
[[nodiscard]] inline nvtxRangeId_t range_start(const char* name,
                                               const uint32_t category,
                                               const uint32_t color) {
  auto event_attrib = make_event_attrib(name, color);
  return nvtxRangeStartEx(&event_attrib);
}

// End an unnested range
inline void range_end(const nvtxRangeId_t range) {
  nvtxRangeEnd(range);
}

//
// Custom domain support
//

// Create a custom domain use in domain calls
[[nodiscard]] inline nvtxDomainHandle_t create_domain(const char* name) {
  return nvtxDomainCreateA(name);
}

// Destroy a custom domain
inline void destroy_domain(nvtxDomainHandle_t domain) {
  nvtxDomainDestroy(domain);
}

// Set an instantaneous event in a custom domain
inline void set_domain_mark(nvtxDomainHandle_t domain, const char* name, uint32_t color) {
  auto event_attrib = make_event_attrib(name, color);
  nvtxDomainMarkEx(domain, &event_attrib);
}

// Push a nested domain range including attributes
inline void domain_range_push(nvtxDomainHandle_t domain,
                              const char* name,
                              uint32_t category,
                              uint32_t color) {
  auto event_attrib = make_event_attrib(name, color);
  event_attrib.category = category;
  nvtxDomainRangePushEx(domain, &event_attrib);
}

// Pop a nested range from a domain
inline void domain_range_pop(nvtxDomainHandle_t domain) {
  nvtxDomainRangePop(domain);
}

// Start an unnested range with attributes
[[nodiscard]] inline nvtxRangeId_t domain_range_start(nvtxDomainHandle_t domain,
                                                      const char* name,
                                                      const uint32_t category,
                                                      const uint32_t color) {
  auto event_attrib = make_event_attrib(name, color);
  return nvtxDomainRangeStartEx(domain, &event_attrib);
}

// End an unnested range
inline void domain_range_end(nvtxDomainHandle_t domain, nvtxRangeId_t range) {
  nvtxDomainRangeEnd(domain, range);
}

#else  // HAVE_NVTX

//
// Stub functions
// These should be optimized out in release builds
//
inline void init() {}
inline void shutdown() {}

inline void omnisci_range_push(Category, const char*, const char*) {}
inline void omnisci_range_pop() {}
inline nvtxRangeId_t omnisci_range_start(Category, const char*) {
  return nvtxRangeId_t{};
}
inline void omnisci_range_end(nvtxRangeId_t) {}
inline void omnisci_set_mark(Category, const char*) {}

inline void name_current_thread(const char*) {}
inline void name_category(uint32_t category, const char* name) {}
inline void name_domain_category(nvtxDomainHandle_t domain,
                                 uint32_t category,
                                 const char* name) {}

inline void set_mark(const char*) {}
inline void range_push(const char*) {}
inline void range_pop() {}
inline nvtxRangeId_t range_start(const char*) {
  return nvtxRangeId_t{};
}
inline nvtxRangeId_t range_start(const char*, uint32_t, uint32_t) {
  return nvtxRangeId_t{};
}
inline void range_end(nvtxRangeId_t) {}

inline nvtxDomainHandle_t create_domain(const char*) {
  return nullptr;
}
inline void destroy_domain(nvtxDomainHandle_t) {}
inline void set_domain_mark(nvtxDomainHandle_t, const char*, uint32_t) {}
inline void domain_range_push(nvtxDomainHandle_t, const char*, uint32_t, uint32_t) {}
inline void domain_range_pop(nvtxDomainHandle_t) {}

#ifdef HAVE_CUDA
inline void name_cuda_device(CUdevice, const char*) {}
inline void name_cuda_context(CUcontext, const char*) {}
inline void name_cuda_stream(CUstream, const char*) {}
inline void name_cuda_event(CUevent, const char*) {}
#endif  // HAVE_CUDA

#endif  // HAVE_NVTX

}  // namespace nvtx_helpers
