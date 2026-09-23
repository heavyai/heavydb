/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/heavyai_locks.h"

namespace heavyai {

DistributedSharedMutex::~DistributedSharedMutex() {
  std::unique_lock lk(mutex_);
  // Wait for an extended lock to finish before destructing.
  if (auto th = std::move(extended_lock_thread_); th) {
    // Prevent a new extended lock after this one ends.
    ++ref_count_;
    is_extended_ = true;
    // Join.
    lk.unlock();
    th->join();
    lk.lock();
    --ref_count_;
  }
  // We shouldn't be destructing if locks still need the lockfile.
  CHECK(ref_count_ == 0);
}

void DistributedSharedMutex::lock() {
  // Strongly prioritize read locks over write locks, for now.
  // A write lock will be taken only when nobody else is waiting for a lock.
  // TODO(sy): Remove after we can use OFD locks.
  while (!takeLock(Mode::RW, Block::NO)) {
    using namespace std::literals;
    std::this_thread::sleep_for(50ms);
  }
}

bool DistributedSharedMutex::try_lock() {
  return takeLock(Mode::RW, Block::NO);
}

void DistributedSharedMutex::unlock() {
  releaseLock();
}

void DistributedSharedMutex::lock_shared() {
  takeLock(Mode::RO, Block::YES);
}

bool DistributedSharedMutex::try_lock_shared() {
  return takeLock(Mode::RO, Block::NO);
}

void DistributedSharedMutex::unlock_shared() {
  releaseLock();
}

// Convert an RO lock to an RW lock. Can deadlock! Don't use?
// NOTE(sy): Will throw an error for now. See convertLock() function.
void DistributedSharedMutex::convert_lock() {
  convertLock(Mode::RW, Block::YES);
}

bool DistributedSharedMutex::try_convert_lock() {
  return convertLock(Mode::RW, Block::NO);
}

// Convert an RW lock to an RO lock. Normally will succeed. OK to use.
void DistributedSharedMutex::convert_lock_shared() {
  convertLock(Mode::RO, Block::YES);
}

bool DistributedSharedMutex::try_convert_lock_shared() {
  return convertLock(Mode::RO, Block::NO);
}

std::string DistributedSharedMutex::get_lockfilename() const {
  return lockfilename_;
}

bool DistributedSharedMutex::takeLock(Mode mode, Block block) {
  CHECK(mode != Mode::UNDEFINED) << "locking mode not specified";
  if (callbacks_.pre_lock_callback) {
    callbacks_.pre_lock_callback(mode == Mode::RW);
  }
  bool flag = innerTakeLock(mode, block);
  if (flag && callbacks_.post_lock_callback) {
    // Only run callback if we successfully acquired lock.
    callbacks_.post_lock_callback(mode == Mode::RW);
  }
  return flag;
}

void DistributedSharedMutex::initializeRefCountIfNeeded() {
  if (thread_ref_count_.count(this) == 0) {
    thread_ref_count_[this] = 0;
  }
}

void DistributedSharedMutex::incrementRefCount() {
  ++ref_count_;
  ++thread_ref_count_[this];
}

void DistributedSharedMutex::decrementRefCount() {
  --ref_count_;
  if (thread_ref_count_.count(this) != 0) {
    --thread_ref_count_[this];
  }
}

bool DistributedSharedMutex::innerTakeLock(Mode const mode, Block const block) {
  // Critical section.
  auto timer = DEBUG_TIMER(__func__);
  std::unique_lock lk(mutex_);
  initializeRefCountIfNeeded();

  // The POSIX lockfile already has been taken.
  if (mode_ != Mode::UNDEFINED) {
    auto [take_a_new_lock, success] = innerTakeLockAfterWaiting(mode, block, lk);
    if (!take_a_new_lock) {
      // Returning here probably means we're sharing somebody's existing Mode::RO lock
      // or maybe we're reporting failure to acquire the lock due to Block::NO.
      return success;
    }
    // Fall through to take a new lock...
  }

  // Must open a file descriptor before the lockfile can be locked.
  CHECK(mode_ == Mode::UNDEFINED);
  mode_ = mode;
  if (g_read_only && mode_ == Mode::RW) {
    throw std::runtime_error("tried to take a read+write lock in read-only mode");
  }
  VLOG(2) << "taking [" << lockfilename_ << "] " << modeString(mode_) << " lock";
  fd_ = heavyai::safe_open(lockfilename_.c_str(), O_RDWR | O_CREAT, 0664);
  if (fd_ == -1) {
    throw std::runtime_error("failed to open lockfile: " + lockfilename_ + ": " +
                             std::string(strerror(errno)) + " (" + std::to_string(errno) +
                             ")");
  }

  // Take the lockfile. fcntl() is the most modern, most portable, posix locking.
  auto [fl, cmd] = prepareLockSettings(mode_, block);
  int ret = heavyai::safe_fcntl(fd_, cmd, &fl);
  if (block == Block::NO && ret == -1 &&
      (errno == EACCES || errno == EAGAIN)) {  // non-blocking, but locked by someone else
    if (heavyai::safe_close(fd_) == -1) {
      // although close() "failed", fd_ is now closed on most arches
      throw std::runtime_error("failed to close lockfile: " + lockfilename_ + ": " +
                               std::string(strerror(errno)) + " (" +
                               std::to_string(errno) + ")");
    }
    mode_ = Mode::UNDEFINED;
    return false;  // try_lock() or try_lock_shared() will fail
  }
  if (ret == -1) {
    auto errno0 = errno;
    heavyai::safe_close(fd_);
    throw std::runtime_error("failed to lock lockfile: " + lockfilename_ + ": " +
                             std::string(strerror(errno0)) + " (" +
                             std::to_string(errno0) + ")");
  }
  lock_start_time_ = std::chrono::steady_clock::now();

  // Read the contents of the lockfile.
  alignas(alignof(size_t)) char buffer[100];
  ssize_t sz = heavyai::safe_read(fd_, buffer, sizeof(buffer));
  if (sz <= -1) {
    auto errno0 = errno;
    heavyai::safe_close(fd_);
    throw std::runtime_error("failed to read lockfile: " + lockfilename_ + ": " +
                             std::string(strerror(errno0)) + " (" +
                             std::to_string(errno0) + ")");
  } else if (sz != 0 && sz != sizeof(size_t)) {
    heavyai::safe_close(fd_);
    throw std::runtime_error("failed to read lockfile: " + lockfilename_);
  }

  // If another process had a write lock since we last did, caches may need reloading.
  bool reload{false};
  if (sz != 0) {  // lockfile already existed
    CHECK_EQ(size_t(sz), sizeof(size_t));
    size_t version = shared::heavyai_ntohll(*reinterpret_cast<size_t*>(&buffer));
    reload = (version_ != version);
    if (reload) {
      VLOG(2) << "detected [" << lockfilename_ + "] version changed from " << version_
              << " to " << version;
    }
    version_ = version;
  } else {
    VLOG(2) << "initial [" << lockfilename_ + "] version " << version_;
  }

  // If we're taking a write lock, increment the version number in the lockfile.
  if (mode_ == Mode::RW) {
    lseek(fd_, 0, SEEK_SET);
    size_t version = shared::heavyai_htonll(++version_);
    ssize_t sz =
        heavyai::safe_write(fd_, static_cast<void const*>(&version), sizeof(version));
    if (sz == -1) {
      --version_;
      auto errno0 = errno;
      heavyai::safe_close(fd_);
      throw std::runtime_error("failed to write lockfile: " + lockfilename_ + ": " +
                               std::string(strerror(errno0)) + " (" +
                               std::to_string(errno0) + ")");
    }
    VLOG(2) << "incremented [" << lockfilename_ + "] version to " << version_;
  }

  // Success.
  incrementRefCount();

  // Notify the user that caches need to be reloaded.
  // The internal local mutex_ is recursive in case reload_cache_callback()
  // tries to lock this lockfile recursively, so that we won't deadlock.
  if (reload && callbacks_.reload_cache_callback) {
    callbacks_.reload_cache_callback(version_);
  }

  // Share a new RO lock with other threads that are waiting for an RO lock.
  if (mode_ == Mode::RO) {
    while (!thread_queue_.empty() && thread_queue_.front().first == Mode::RO) {
      auto [thread_mode, pr] = thread_queue_.front();
      thread_queue_.pop();
      pr->set_value(/*already_locked=*/true);
    }
  }

  VLOG(2) << "successfully took [" << lockfilename_ << "] " << modeString(mode_)
          << " lock";

  return true;
}  // innerTakeLock()

std::tuple<bool /*take_a_new_lock*/, bool /*success*/>
DistributedSharedMutex::innerTakeLockAfterWaiting(
    Mode const mode,
    Block const block,
    std::unique_lock<std::recursive_mutex>& lk) {
  // Handle recursive locks. Current thread already has a lock of some kind.
  if (thread_ref_count_[this]) {
    return {false, innerTakeLockRecursively(mode, block, lk)};
  }  // Handle recursive locks.

  // Share an RO lock without blocking.
  if (thread_queue_.empty() && mode_ == Mode::RO && mode == Mode::RO) {
    VLOG(2) << "sharing [" << lockfilename_ << "] " << modeString(mode_) << " lock";
    incrementRefCount();
    return {false, true};  // share an RO lock
  }

  // Block on the thread queue. Current thread will wait its turn for a lock.
  if (block == Block::NO) {
    return {false, false};  // try_lock() or try_lock_shared() will fail
  }
  VLOG(2) << "waiting for [" << lockfilename_ << "] " << modeString(mode) << " lock)";
  std::promise<bool> pr;
  thread_queue_.emplace(mode, &pr);
  lk.unlock();
  bool already_locked = pr.get_future().get();  // usually blocks
  if (already_locked) {
    // POSIX lockfile is locked.
    VLOG(2) << "done waiting for [" << lockfilename_ << "] (already locked) "
            << modeString(mode_) << " lock";
    incrementRefCount();
    return {false, true};  // share an RO lock
  } else {
    // POSIX lockfile isn't locked.
    VLOG(2) << "done waiting for [" << lockfilename_ << "] (about to be locked) "
            << modeString(mode) << " lock";
    // Other threads trying to race us here will get pushed onto the back of the queue
    // because mode_ is still set ***even though the POSIX lock has been released***.
    lk.lock();
    mode_ = Mode::UNDEFINED;
    return {true, {}};  // Falls through to take a new lock.
  }
}  // innerTakeLockAfterWaiting()

bool DistributedSharedMutex::innerTakeLockRecursively(
    Mode const mode,
    Block const block,
    std::unique_lock<std::recursive_mutex>& lk) {
  VLOG(2) << "recursive [" << lockfilename_ << "] " << modeString(mode) << " lock";
  if (mode == Mode::RO) {  // we want to read
    incrementRefCount();
    return true;                  // share this thread's existing RO or RW lock
  } else if (mode == Mode::RW) {  // we want to write
    if (mode_ == Mode::RW) {
      incrementRefCount();
      return true;  // this thread already has an RW lock
    } else if (mode_ == Mode::RO) {
      throw std::runtime_error(
          "won't automatically convert lock from read-only to read+write, call "
          "convert_lock() instead?");
    }
  }
  UNREACHABLE() << "unexpected locking mode";
  return false;
}  // innerTakeLockRecursively()

void DistributedSharedMutex::releaseLock() {
  auto mode = mode_;
  if (callbacks_.pre_unlock_callback) {
    callbacks_.pre_unlock_callback(mode == Mode::RW);
  }
  bool flag = innerReleaseLock();
  if (flag && callbacks_.post_unlock_callback) {
    // Only run callback if lock was ssuccessfully released (it could still be shared).
    callbacks_.post_unlock_callback(mode == Mode::RW);
  }
}  // releaseLock()

bool DistributedSharedMutex::innerReleaseLock() {
  // Critical section.
  auto timer = DEBUG_TIMER(__func__);
  std::unique_lock lk(mutex_);

  // See if it's really time to release this lock yet.
  // (Inside reload_cache_callback they might take this lock recursively.)
  maybeExtendLockDuration();
  if (ref_count_ > 1) {
    decrementRefCount();
    VLOG(2) << "decrementing [" << lockfilename_ << "] " << modeString(mode_) << " lock";
    return false;
  }
  CHECK_EQ(ref_count_, 1U);

  // Release the lockfile.
  VLOG(2) << "releasing [" << lockfilename_ << "] " << modeString(mode_) << " lock";
  auto [fl, cmd] = prepareUnlockSettings();
  int ret = heavyai::safe_fcntl(fd_, cmd, &fl);
  if (ret == -1) {
    // should never get here, but do our best now
    auto errno0 = errno;
    decrementRefCount();
    CHECK_EQ(ref_count_, 0U);
    is_extended_ = false;
    heavyai::safe_close(fd_);
    throw std::runtime_error("failed to unlock lockfile: " + lockfilename_ + ": " +
                             std::string(strerror(errno0)) + " (" +
                             std::to_string(errno0) + ")");
  }

  // Close the file descriptor that was required to lock the lockfile.
  if (heavyai::safe_close(fd_) == -1) {
    // although close() "failed", fd_ is now closed on most arches
    decrementRefCount();
    CHECK_EQ(ref_count_, 0U);
    is_extended_ = false;
    throw std::runtime_error("failed to close lockfile: " + lockfilename_ + ": " +
                             std::string(strerror(errno)) + " (" + std::to_string(errno) +
                             ")");
  }

  // Success.
  fd_ = -1;
  is_extended_ = false;
  decrementRefCount();
  CHECK_EQ(ref_count_, 0U);

  // Wake up the next thread waiting for a lock.
  if (!thread_queue_.empty()) {
    auto [thread_mode, pr] = thread_queue_.front();
    thread_queue_.pop();
    lk.unlock();
    // Wake up the thread that is first in line in the thread queue. Other threads
    // trying to race us here will get pushed onto the back of the queue because mode_
    // is still set ***even though the POSIX lock has been released***.
    pr->set_value(/*already_locked=*/false);
  } else {
    mode_ = Mode::UNDEFINED;
  }
  return true;
}  // innerReleaseLock()

void DistributedSharedMutex::maybeExtendLockDuration() {
  // At this point we hold a lock on mutex_.
  if (!is_extended_ && mode_ == Mode::RO && ref_count_ == 1 &&
      g_lockfile_lock_extension_milliseconds) {
    // At this point, if extended_lock_thread_ is not null, then it must be ready to
    // join, because its execution point can't be before its mutex_ critical section
    // where is_extended_ would be true, and it can't be inside its mutex_ critical
    // section when we're inside our own mutex_ critical section.
    if (auto th = std::move(extended_lock_thread_); th) {
      th->join();
    }
    // Only extend a lock for up to g_lockfile_lock_extension_milliseconds total length.
    // Beyond that length, the lock will be released normally so writers won't starve.
    auto sleep_until_time =
        lock_start_time_ +
        std::chrono::milliseconds(g_lockfile_lock_extension_milliseconds);
    if (std::chrono::steady_clock::now() >= sleep_until_time) {
      return;
    }
    is_extended_ = true;
    VLOG(2) << "extending [" << lockfilename_ << "] " << modeString(mode_) << " lock";
    // Create a thread to extend the lock.
    std::promise<void> pr;
    extended_lock_thread_ = std::make_unique<std::thread>([this, sleep_until_time, &pr] {
      // Here this thread is sharing the lock of the thread that launched it.
      incrementRefCount();
      extended_lock_ = std::make_unique<std::shared_lock<DistributedSharedMutex>>(
          *this, std::adopt_lock);
      pr.set_value();  // Tell the launching thread it can release it's lock now.
      std::this_thread::sleep_until(sleep_until_time);
      std::unique_lock lk(mutex_);
      // Finished extending the lock.
      // The extended_lock_ must be reset before the is_extended_ flag is cleared
      // because the shared_lock destructor called by reset() calls releaseLock()
      // which calls maybeExtendLockDuration() which checks the flag.
      extended_lock_.reset();
      // The is_extended_ flag won't be cleared here unless there are no other locks
      // being held on this mutex. If there are other locks being held, then the
      // flag will stay set until, elsewhere, ref_count_ drops to zero. This will
      // prevent another extended period from starting immediately and the lockfile
      // potentially being held forever.
      is_extended_ = ref_count_;
    });
    pr.get_future().get();  // Blocks until the launched thread is done using our lock.
  }
}  // maybeExtendLockDuration()

bool DistributedSharedMutex::convertLock(Mode const mode, Block const block) {
  if (mode == Mode::RW) {
    // NOTE(sy): POSIX locks allow two or more threads holding read locks to request an
    // upgrade to write locks without releasing the lock inbetween. That could work fine
    // for a single thread, but two or more threads will instantly deadlock, either
    // freezing forever or returning errno EDEADLK (I've seen both outcomes). Leaving
    // the code here in case there's some legitimate reason for using this feature but
    // will return an error for now.
    throw std::runtime_error("converting to a write lock not allowed");

    if (mode_ == Mode::RW) {
      return true;
    }

    // We already have a read lock and it needs to be converted to a write lock.
    // NOTE: Can deadlock if somebody else already is converting a read-only lock to a
    // read+write lock. Sometimes the filesystem will generate an EDEADLK "Resource
    // deadlock avoided" error, but that isn't guaranteed.
    VLOG(2) << "converting [" << lockfilename_ << "] read-only lock to read+write lock";
    auto [fl, cmd] = prepareLockSettings(Mode::RW, block);
    int ret = heavyai::safe_fcntl(fd_, cmd, &fl);
    if (block == Block::NO && ret == -1 &&
        (errno == EACCES || errno == EAGAIN)) {  // locked by someone else
      return false;
    }
    if (ret == -1) {
      throw std::runtime_error("failed to convert lock on lockfile: " + lockfilename_ +
                               ": " + std::string(strerror(errno)) + " (" +
                               std::to_string(errno) + ")");
    }
    mode_ = Mode::RW;
    is_extended_ = false;

    // Increment the version number in the lockfile.
    lseek(fd_, 0, SEEK_SET);
    std::stringstream ss;
    ss << ++version_ << "\n";
    std::string s = ss.str();
    ssize_t sz = heavyai::safe_write(fd_, s.c_str(), s.size());
    if (sz == -1) {
      auto errno0 = errno;
      heavyai::safe_close(fd_);
      throw std::runtime_error("failed to write lockfile: " + lockfilename_ + ": " +
                               std::string(strerror(errno0)) + " (" +
                               std::to_string(errno0) + ")");
    }
    return true;
  } else if (mode == Mode::RO) {
    if (mode_ == Mode::RO) {
      return true;
    }

    // We already have a write lock and it needs to be converted to a read lock.
    VLOG(2) << "converting [" << lockfilename_ << "] read+write lock to read-only lock)";
    auto [fl, cmd] = prepareLockSettings(Mode::RO, block);
    int ret = heavyai::safe_fcntl(fd_, cmd, &fl);
    if (ret == -1) {
      throw std::runtime_error("failed to convert lock on lockfile: " + lockfilename_ +
                               ": " + std::string(strerror(errno)) + " (" +
                               std::to_string(errno) + ")");
    }
    mode_ = Mode::RO;
    return true;
  }
  UNREACHABLE() << "unexpected locking mode";
  return false;
}  // convertLock()

std::pair<struct flock, int> DistributedSharedMutex::prepareLockSettings(
    Mode const mode,
    Block const block) {
  if (mode == Mode::UNDEFINED) {
    throw std::runtime_error("locking mode not specified");
  }
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_type = (mode == Mode::RO ? F_RDLCK : F_WRLCK);
  fl.l_whence = SEEK_SET;
  int cmd;
#ifdef __linux__
  // cmd = (block == Block::YES) ? F_OFD_SETLKW : F_OFD_SETLK;  // broken on centos
  cmd = (block == Block::YES) ? F_SETLKW : F_SETLK;
#else
  cmd = (block == Block::YES) ? F_SETLKW : F_SETLK;
#endif  // __linux__
  return {fl, cmd};
}

std::pair<struct flock, int> DistributedSharedMutex::prepareUnlockSettings() {
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_type = F_UNLCK;
  fl.l_whence = SEEK_SET;
  int cmd;
#ifdef __linux__
  // cmd = F_OFD_SETLK;  // TODO(sy): broken on centos
  cmd = F_SETLK;
#else
  cmd = F_SETLK;
#endif  // __linux__
  return {fl, cmd};
}

std::string DistributedSharedMutex::modeString(Mode const mode) {
  if (mode == Mode::UNDEFINED) {
    throw std::runtime_error("locking mode not specified");
  }
  using namespace std::literals;
  switch (mode) {
    case Mode::RO: {
      return "read-only"s;
    }
    case Mode::RW: {
      return "read+write"s;
    }
    default: {
      throw std::runtime_error("unexpected locking mode");
    }
  }
}

}  // namespace heavyai
