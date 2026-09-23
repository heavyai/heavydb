/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RWLocks.h"
#include "Catalog.h"
#include "Shared/heavyai_locks.h"
#include "Shared/heavyai_shared_mutex.h"
#include "SysCatalog.h"

bool g_verbose_lock_logging{false};

namespace {
template <typename T>
constexpr char const* get_cat_type() {
  static_assert(std::is_same_v<T, Catalog_Namespace::Catalog> ||
                std::is_same_v<T, Catalog_Namespace::SysCatalog>);
  return std::is_same_v<T, Catalog_Namespace::Catalog> ? "Catalog" : "SysCatalog";
}
}  // namespace

namespace Catalog_Namespace {

template <typename T>
cat_lock<T>::cat_lock(const T* cat) : cat_(cat), holds_lock_(false) {}

template <typename T>
cat_lock<T>::~cat_lock() {}

template <typename T>
read_lock<T>::read_lock(const T* cat) : cat_lock<T>(cat) {
  lock();
}

template <typename T>
read_lock<T>::~read_lock() {
  unlock();
}

template <typename T>
void read_lock<T>::lock() {
  std::thread::id tid = std::this_thread::get_id();

  if (this->cat_->mutex_desc_.thread_holding_lock != tid && !T::thread_holds_read_lock) {
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Attempting to acquire Read lock for " << get_cat_type<T>();

    if (g_multi_instance) {
      CHECK(this->cat_->mutex_desc_.dist_mutex) << "Distributed mutex not initialized";
      dlock_ = heavyai::shared_lock<heavyai::DistributedSharedMutex>(
          *this->cat_->mutex_desc_.dist_mutex);
    } else {
      lock_ = heavyai::shared_lock<heavyai::shared_mutex>(
          this->cat_->mutex_desc_.local_mutex);
    }
    T::thread_holds_read_lock = true;
    this->holds_lock_ = true;
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Acquired Read lock for " << get_cat_type<T>();
  }
}

template <typename T>
void read_lock<T>::unlock() {
  if (this->holds_lock_) {
    T::thread_holds_read_lock = false;
    if (g_multi_instance) {
      dlock_.unlock();
    } else {
      lock_.unlock();
    }
    this->holds_lock_ = false;
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Released Read lock for " << get_cat_type<T>();
  }
}

template <typename T>
sqlite_lock<T>::sqlite_lock(const T* cat) : cat_lock<T>(cat), cat_read_lock_(this->cat_) {
  lock();
}

template <typename T>
sqlite_lock<T>::~sqlite_lock() {
  unlock();
}

template <typename T>
void sqlite_lock<T>::lock() {
  std::thread::id tid = std::this_thread::get_id();

  if (this->cat_->sqlite_mutex_desc_.thread_holding_lock != tid) {
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Attempting to acquire SQLite lock for " << get_cat_type<T>();
    if (g_multi_instance) {
      CHECK(this->cat_->mutex_desc_.dist_mutex) << "Distributed mutex not initialized";
      dlock_ = heavyai::unique_lock<heavyai::DistributedSharedMutex>(
          *this->cat_->sqlite_mutex_desc_.dist_mutex);
    } else {
      lock_ = heavyai::unique_lock<heavyai::shared_mutex>(
          this->cat_->sqlite_mutex_desc_.local_mutex);
    }
    this->cat_->sqlite_mutex_desc_.thread_holding_lock = tid;
    this->holds_lock_ = true;
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Acquired SQLite lock for " << get_cat_type<T>();
  }
}

template <typename T>
void sqlite_lock<T>::unlock() {
  if (this->holds_lock_) {
    std::thread::id no_thread;
    this->cat_->sqlite_mutex_desc_.thread_holding_lock = no_thread;
    if (g_multi_instance) {
      dlock_.unlock();
    } else {
      lock_.unlock();
    }
    cat_read_lock_.unlock();
    this->holds_lock_ = false;
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Released SQLite lock for " << get_cat_type<T>();
  }
}

template <typename T>
write_lock<T>::write_lock(const T* cat) : cat_lock<T>(cat) {
  lock();
}

template <typename T>
write_lock<T>::~write_lock() {
  unlock();
}

template <typename T>
void write_lock<T>::lock() {
  std::thread::id tid = std::this_thread::get_id();

  if (this->cat_->mutex_desc_.thread_holding_lock != tid) {
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Attempting to acquire Write lock for " << get_cat_type<T>();
    if (g_multi_instance) {
      CHECK(this->cat_->mutex_desc_.dist_mutex) << "Distributed mutex not initialized";
      dlock_ = heavyai::unique_lock<heavyai::DistributedSharedMutex>(
          *this->cat_->mutex_desc_.dist_mutex);
    } else {
      lock_ = heavyai::unique_lock<heavyai::shared_mutex>(
          this->cat_->mutex_desc_.local_mutex);
    }
    this->cat_->mutex_desc_.thread_holding_lock = tid;
    this->holds_lock_ = true;
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Acquired Write lock for " << get_cat_type<T>();
  }
}

template <typename T>
void write_lock<T>::unlock() {
  if (this->holds_lock_) {
    std::thread::id no_thread;
    this->cat_->mutex_desc_.thread_holding_lock = no_thread;
    if (g_multi_instance) {
      dlock_.unlock();
    } else {
      lock_.unlock();
    }
    this->holds_lock_ = false;
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Released Write lock for " << get_cat_type<T>();
  }
}

cat_init_lock::cat_init_lock(const SysCatalog* cat) : cat_lock<SysCatalog>(cat) {
  lock();
}

cat_init_lock::~cat_init_lock() {
  unlock();
}

void cat_init_lock::lock() {
  std::thread::id tid = std::this_thread::get_id();
  if (this->cat_->cat_init_mutex_desc_.thread_holding_lock != tid) {
    LOG_IF(INFO, g_verbose_lock_logging)
        << "Attempting to acquire Cat Init lock for SysCatalog";
    lock_ = heavyai::unique_lock<heavyai::shared_mutex>(
        this->cat_->cat_init_mutex_desc_.local_mutex);
    this->cat_->cat_init_mutex_desc_.thread_holding_lock = tid;
    this->holds_lock_ = true;
    LOG_IF(INFO, g_verbose_lock_logging) << "Acquired Cat Init lock for SysCatalog";
  }
}

void cat_init_lock::unlock() {
  if (holds_lock_) {
    std::thread::id no_thread;
    this->cat_->cat_init_mutex_desc_.thread_holding_lock = no_thread;
    lock_.unlock();
    this->holds_lock_ = false;
    LOG_IF(INFO, g_verbose_lock_logging) << "Released Cat Init lock for SysCatalog";
  }
}

template class read_lock<Catalog>;
template class read_lock<SysCatalog>;
template class write_lock<Catalog>;
template class write_lock<SysCatalog>;
template class sqlite_lock<Catalog>;
template class sqlite_lock<SysCatalog>;

}  // namespace Catalog_Namespace
