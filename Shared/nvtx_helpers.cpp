#include "Shared/nvtx_helpers.h"

#include <array>
#include <sstream>

#ifdef _WIN32
#include <processthreadsapi.h>
#else  // _WIN32
#include <sys/syscall.h>
#include <unistd.h>
#endif  // _WIN32

#include <boost/filesystem/path.hpp>

#include "Logger/Logger.h"

namespace nvtx_helpers {
struct CategoryInfo {
  uint32_t category;
  uint32_t color;
};

#define NUM_CATEGORIES 4u
static std::array<CategoryInfo, NUM_CATEGORIES> g_category_infos = {};
static nvtxDomainHandle_t g_omnisci_domain = nullptr;

// Control whether or not to set the event category when using the Omnisci domain
// Setting the category can clutter the timeline event names. Colors will still be
// used though so the visual organization remains
#define SET_OMNISCI_EVENT_CATEGORY 0

void init() {
  // The domain will be null if NVTX profiling is not enabled by the nsys launcher
  g_omnisci_domain = nvtxDomainCreateA("Omnisci");
  if (g_omnisci_domain) {
    auto create_category = [](Category c, const char* name, uint32_t color) {
      auto category_index = static_cast<uint32_t>(c);
      CHECK_LT(category_index, NUM_CATEGORIES);
      g_category_infos[category_index] = {category_index, color};
      nvtxDomainNameCategoryA(g_omnisci_domain, category_index, name);
    };
    // skip category 0 (none), as trying to modify it triggers errors during nsight
    // analysis
    create_category(Category::kDebugTimer, "DebugTimer", 0xFFB0E0E6);
    create_category(Category::kQueryStateTimer, "QueryStateTimer", 0xFF98FB98);
    create_category(Category::kRenderLogger, "RenderLogger", 0xFFEE8AA);
  }
}

void shutdown() {
  if (g_omnisci_domain) {
    nvtxDomainDestroy(g_omnisci_domain);
    g_omnisci_domain = nullptr;
  }
}

const nvtxDomainHandle_t get_omnisci_domain() {
  return g_omnisci_domain;
}

namespace {
inline nvtxEventAttributes_t make_omnisci_event(Category c, const char* name) {
  auto category_index = static_cast<uint32_t>(c);
  CHECK_LT(category_index, NUM_CATEGORIES);
  auto const& info = g_category_infos[category_index];
  nvtxEventAttributes_t event_attrib = {};
  event_attrib.version = NVTX_VERSION;
  event_attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event_attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event_attrib.message.ascii = name;
  if (info.color != 0) {
    event_attrib.colorType = NVTX_COLOR_ARGB;
    event_attrib.color = info.color;
  }
#if SET_OMNISCI_EVENT_CATEGORY
  event_attrib.category = info.category;
#endif
  return event_attrib;
}

// Return last component of path
std::string filename(char const* path) {
  return boost::filesystem::path(path).filename().string();
}
}  // namespace

void omnisci_range_push(Category c, const char* name, const char* file) {
  if (g_omnisci_domain) {
    if (file) {
      std::stringstream ss;
      ss << name;
      ss << " | " << filename(file);
      auto str = ss.str();
      auto event = make_omnisci_event(c, str.c_str());
      nvtxDomainRangePushEx(g_omnisci_domain, &event);
    } else {
      auto event = make_omnisci_event(c, name);
      nvtxDomainRangePushEx(g_omnisci_domain, &event);
    }
  }
}

void omnisci_range_pop() {
  if (g_omnisci_domain) {
    nvtxDomainRangePop(g_omnisci_domain);
  }
}

nvtxRangeId_t omnisci_range_start(Category c, const char* name) {
  if (g_omnisci_domain) {
    auto event = make_omnisci_event(c, name);
    return nvtxDomainRangeStartEx(g_omnisci_domain, &event);
  } else {
    return nvtxRangeId_t{};
  }
}

void omnisci_range_end(nvtxRangeId_t r) {
  if (g_omnisci_domain) {
    nvtxDomainRangeEnd(g_omnisci_domain, r);
  }
}

void omnisci_set_mark(Category c, const char* name) {
  if (g_omnisci_domain) {
    auto event = make_omnisci_event(c, name);
    nvtxDomainMarkEx(g_omnisci_domain, &event);
  }
}

void name_current_thread(const char* name) {
#ifdef _WIN32
  uint32_t thread_id = GetCurrentThreadId();
#else
#ifdef SYS_gettid
  uint32_t thread_id = static_cast<uint32_t>(syscall(SYS_gettid));
#else  // SYS_gettid
#error "SYS_gettid unavailable on this system"
#endif  // SYS_gettid
#endif  // _WIN32
  nvtxNameOsThreadA(thread_id, name);
}

}  // namespace nvtx_helpers
