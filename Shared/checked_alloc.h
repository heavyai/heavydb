#ifndef CHECKED_ALLOC_H
#define CHECKED_ALLOC_H

#include <cstdlib>
#include <string>

class OutOfHostMemory : public std::runtime_error {
 public:
  OutOfHostMemory(const size_t size)
      : std::runtime_error("Failed to allocate " + std::to_string(size) + " bytes of memory") {}
};

inline void* checked_malloc(const size_t size) {
  auto ptr = malloc(size);
  if (!ptr) {
    throw OutOfHostMemory(size);
  }
  return ptr;
}

inline void* checked_calloc(const size_t nmemb, const size_t size) {
  auto ptr = calloc(nmemb, size);
  if (!ptr) {
    throw OutOfHostMemory(nmemb * size);
  }
  return ptr;
}

#endif  // CHECKED_ALLOC_H
