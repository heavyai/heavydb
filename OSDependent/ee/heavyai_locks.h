/*
 * Copyright (C) 2022 HEAVY.AI, Inc.
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

#error "file only available with EE"
