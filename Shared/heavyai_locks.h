/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////// EXAMPLE
//
// DistributedSharedMutex dmutex{"example.lockfile", [&](size_t version){
//   /* Called when we lock the mutex, but only when we need to invalidate
//    * any cached copies of the data protected by the lockfile. */
// }};
//
// std::unique_lock write_lock{dmutex};  // Standard read-write lock.
// std::shared_lock read_lock{dmutex};   // Standard read-only lock.
//
// Optionally, instead of passing in the callback, a Callbacks struct can be
// passed in, allowing a variety of powerful callbacks.

//////////// DESIGN
//
// The DistributedSharedMutex uses a tiny lockfile containing a version number.
// That version number gets incremented every time a process takes an exclusive
// (write) lock. Each process keeps track of the last version number seen, making
// it easy to detect when the data protected by the lockfile has changed.
//
// Expected to be NFSv4 compatible and Lustre 2.13 compatible.
//
// As an optimization, DistributedSharedMutex holds on to read locks for a brief period
// of time after the lock is released by the user. Testing shows that this optimization
// makes locking on a shared network filesystem (ex: NFS) nearly as fast as a local
// filesystem. (See: maybeExtendLockDuration(), g_lockfile_lock_extension_milliseconds)
//
// The lockfile actually uses standard POSIX locking, which is likely to save us a
// large amount of development and maintenance. However, unfortunately, we're stuck
// on an old version of the Linux kernel (HeavyDB releases are built on CentOS 7
// from 2014 using Linux kernel 3.10 from 2013) which was just before support for
// OFD locks were added to the kernel.
//
// The lack of OFD locks means we had to implement per-thread reference counting
// and thread queuing ourselves, which is tricky and error-prone. Hopefully we can
// remove half of the code in DistributedSharedMutex after we upgrade our build
// process to more recent kernels.
//
// If POSIX lockfiles ever turn out to be undesirable, we could create an alternate
// version of DistributedSharedMutex using our own simple lock server. In fact,
// writing such a server probably would have been easier than dealing with the
// missing OFD locks, but in the long run, POSIX lockfiles are expected to give us
// a better product with less maintenance and easier configuration (no separate
// lock server we would need to install).
//
// See also about OFD locks:
//   https://lwn.net/Articles/586904/
//   https://kernelnewbies.org/Linux_3.15#New_file_locking_scheme:_open_file_description_locks
//   https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock#Priority_policies
//   https://rfc1149.net/blog/2011/01/07/the-third-readers-writers-problem/

#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include "Logger/Logger.h"
#include "Shared/heavyai_fs.h"
#include "Shared/misc.h"

namespace heavyai {

// MutexInterface:
// An exclusive mutex compatible with std::unique_lock.
class MutexInterface {
 public:
  virtual void lock() = 0;
  virtual bool try_lock() = 0;
  virtual void unlock() = 0;

  virtual ~MutexInterface() {}
};

// SharedMutexInterface:
// A sharable mutex compatible with std::unique_lock and std::shared_lock.
class SharedMutexInterface : public MutexInterface {
 public:
  virtual void lock_shared() = 0;
  virtual bool try_lock_shared() = 0;
  virtual void unlock_shared() = 0;

  virtual ~SharedMutexInterface() {}
};

// DistributedSharedMutex:
// A recursive mutex based on a lockfile.
// Intended for locking between nodes in a cluster.
// NFSv4 compatible. Lustre 2.13 compatible.
// Can tell us if another writer had the lock since we last did.
// (Meaning that we now need to reload our caches.)
class DistributedSharedMutex final : public SharedMutexInterface {
 public:
  struct Callbacks {
    std::function<void(bool /*write*/)> pre_lock_callback;
    std::function<void(size_t /*version*/)> reload_cache_callback;
    std::function<void(bool /*write*/)> post_lock_callback;
    std::function<void(bool /*write*/)> pre_unlock_callback;
    std::function<void(bool /*write*/)> post_unlock_callback;
  };

  DistributedSharedMutex(std::filesystem::path lockfilename,
                         std::function<void(size_t)> reload_cache_callback = {})
      : lockfilename_(lockfilename)
      , callbacks_(Callbacks{{}, reload_cache_callback, {}, {}, {}}) {}

  DistributedSharedMutex(std::filesystem::path lockfilename, Callbacks callbacks)
      : lockfilename_(lockfilename), callbacks_(callbacks) {}

  ~DistributedSharedMutex();

  void lock() override;
  bool try_lock() override;
  void unlock() override;

  void lock_shared() override;
  bool try_lock_shared() override;
  void unlock_shared() override;

  // Convert an RO lock to an RW lock. Can deadlock! Don't use?
  // NOTE(sy): Will throw an error for now. See convertLock() function.
  virtual void convert_lock();
  virtual bool try_convert_lock();

  // Convert an RW lock to an RO lock. Normally will succeed. OK to use.
  virtual void convert_lock_shared();
  virtual bool try_convert_lock_shared();

  virtual std::string get_lockfilename() const;

 private:
  enum class Mode { UNDEFINED, RO /*read-only*/, RW /*read+write*/ };
  enum class Block { YES, NO };

  virtual bool takeLock(Mode mode, Block block);
  virtual void initializeRefCountIfNeeded();
  virtual void incrementRefCount();
  virtual void decrementRefCount();
  virtual bool innerTakeLock(Mode const mode, Block const block);
  virtual std::tuple<bool /*take_a_new_lock*/, bool /*success*/>
  innerTakeLockAfterWaiting(Mode const mode,
                            Block const block,
                            std::unique_lock<std::recursive_mutex>& lk);
  virtual bool innerTakeLockRecursively(Mode const mode,
                                        Block const block,
                                        std::unique_lock<std::recursive_mutex>& lk);
  virtual void releaseLock();
  virtual bool innerReleaseLock();
  virtual void maybeExtendLockDuration();
  virtual bool convertLock(Mode const mode, Block const block);
  static std::pair<struct flock, int> prepareLockSettings(Mode const mode,
                                                          Block const block);
  static std::pair<struct flock, int> prepareUnlockSettings();
  static std::string modeString(Mode const mode);

  const std::string lockfilename_;
  Callbacks callbacks_;
  std::recursive_mutex mutex_;
  std::queue<std::pair<Mode, std::promise<bool>*>> thread_queue_;
  std::invoke_result_t<decltype(std::chrono::steady_clock::now)> lock_start_time_;
  size_t ref_count_{0};
  int fd_{-1};
  size_t version_{0};
  Mode mode_{Mode::UNDEFINED};
  static thread_local inline std::unordered_map<DistributedSharedMutex*, size_t>
      thread_ref_count_;  // Ref count per thread is used for recursive locks.
  bool is_extended_{false};
  std::unique_ptr<std::shared_lock<DistributedSharedMutex>> extended_lock_;
  std::unique_ptr<std::thread> extended_lock_thread_;
};  // class DistributedSharedMutex

}  // namespace heavyai

inline bool g_read_only{false};
inline bool g_multi_instance{
    false};  // TODO(sy): set true after internal testing is complete
inline size_t g_lockfile_lock_extension_milliseconds{1000};
