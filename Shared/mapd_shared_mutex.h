#ifndef MAPD_SHARED_MUTEX
#define MAPD_SHARED_MUTEX

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#ifdef HAS_SHARED_MUTEX
#if defined(__linux__) && GCC_VERSION < 50000
#include <cerrno>
#include <stdexcept>

#define CHECK_NO_ERROR(err) \
  {                         \
    if (err) {              \
      abort();              \
    }                       \
  }

class mapd_shared_mutex {
 public:
  mapd_shared_mutex() = default;
  ~mapd_shared_mutex() = default;

  mapd_shared_mutex(const mapd_shared_mutex&) = delete;
  mapd_shared_mutex& operator=(const mapd_shared_mutex&) = delete;

  void lock() {
    int err = pthread_rwlock_wrlock(&rw_lock_);
    if (err == EDEADLK) {
      throw std::runtime_error("Deadlock would occur");
    }
    CHECK_NO_ERROR(err);
  }

  bool try_lock() {
    int err = pthread_rwlock_trywrlock(&rw_lock_);
    if (err == EBUSY) {
      return false;
    }
    CHECK_NO_ERROR(err);
    return true;
  }

  void unlock() {
    int err = pthread_rwlock_unlock(&rw_lock_);
    CHECK_NO_ERROR(err);
  }

  void lock_shared() {
    int err = 0;
    do {
      err = pthread_rwlock_rdlock(&rw_lock_);
    } while (err == EAGAIN);
    if (err == EDEADLK) {
      throw std::runtime_error("Deadlock would occur");
    }
    CHECK_NO_ERROR(err);
  }

  bool try_lock_shared() {
    int err = pthread_rwlock_tryrdlock(&rw_lock_);
    if (err == EBUSY || err == EAGAIN) {
      return false;
    }
    CHECK_NO_ERROR(err);
    return true;
  }

  void unlock_shared() { unlock(); }

 private:
  pthread_rwlock_t rw_lock_ = PTHREAD_RWLOCK_INITIALIZER;
};

#undef CHECK_NO_ERROR
#else
#include <shared_mutex>
typedef std::shared_timed_mutex mapd_shared_mutex;
#endif  // defined(__linux__) && GCC_VERSION < 50000
#include <shared_mutex>
#define mapd_lock_guard std::lock_guard
#define mapd_unique_lock std::unique_lock
#define mapd_shared_lock std::shared_lock
#else
#include <boost/thread/shared_mutex.hpp>
typedef boost::shared_mutex mapd_shared_mutex;
#define mapd_lock_guard boost::lock_guard
#define mapd_unique_lock boost::unique_lock
#define mapd_shared_lock boost::shared_lock
#endif  // HAS_SHARED_MUTEX

#endif  // MAPD_SHARED_MUTEX
