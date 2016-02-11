#ifndef THREAD_COUNT_H
#define THREAD_COUNT_H

inline int cpu_threads() {
  // could use std::thread::hardware_concurrency(), but some
  // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
  // Play it POSIX.1 safe instead.
  return std::max(2 * sysconf(_SC_NPROCESSORS_CONF), 1L);
}

#endif  // THREAD_COUNT_H
