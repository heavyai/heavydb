#ifndef MAPD_SHARED_MUTEX
#define MAPD_SHARED_MUTEX

#ifdef HAS_SHARED_MUTEX
#include <shared_mutex>
#ifdef HAVE_FOLLY
#include <folly/SharedMutex.h>
typedef folly::SharedMutex mapd_shared_mutex;
#else
typedef std::shared_timed_mutex mapd_shared_mutex;
#endif  // HAVE_FOLLY
#define mapd_lock_guard std::lock_guard
#define mapd_unique_lock std::unique_lock
#define mapd_shared_lock std::shared_lock
#else
#include <boost/thread/shared_mutex.hpp>
#ifdef HAVE_FOLLY
#include <folly/SharedMutex.h>
typedef folly::SharedMutex mapd_shared_mutex;
#else
typedef boost::shared_mutex mapd_shared_mutex;
#endif  // HAVE_FOLLY
#define mapd_lock_guard boost::lock_guard
#define mapd_unique_lock boost::unique_lock
#define mapd_shared_lock boost::shared_lock
#endif  // HAS_SHARED_MUTEX

#endif  // MAPD_SHARED_MUTEX
