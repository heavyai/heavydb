/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <list>
#include <mutex>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index_container.hpp>

#include "QueryRenderer/Utils/TimeUtils.h"

namespace QueryRenderer {

// TODO(scb) - #2797 move this stuff into GfxDriver and allow ResourceManager
// to own the pools and eliminate all the smart pointer usage (at least shared/weak).
// ownership / lifetime control relationship:
// PerGpuData -> DeviceContext -> ResourceManager -> Pool -> Resources
template <typename T, size_t max_inactive_time, class... InitTypes>
class BaseQueryRsrcPool {
 public:
  BaseQueryRsrcPool() = default;
  virtual ~BaseQueryRsrcPool() {}

  std::weak_ptr<T> getInactiveRsrc(InitTypes... args) {
    std::lock_guard<std::mutex> pool_lock(pool_mtx_);

    std::shared_ptr<T> rtn;
    if (inactive_rsrc_queue_.size() == 0) {
      rtn = initializeRsrc(args...);
      CHECK(rtn.use_count() == 1);
      rsrc_map_.emplace(rtn);
    } else {
      std::weak_ptr<T> rsrc_wk_ptr;
      while ((rsrc_wk_ptr = inactive_rsrc_queue_.front()).expired()) {
        inactive_rsrc_queue_.pop_front();
      }
      inactive_rsrc_queue_.pop_front();
      rtn = rsrc_wk_ptr.lock();
      CHECK(rtn);
      updateRsrc(rtn, args...);
      CHECK(rsrc_wk_ptr.use_count() == 2);
    }

    flushInactiveQueue();
    return rtn;
  }

  void setRsrcInactive(std::weak_ptr<T>& rsrc_ptr) {
    // resource may never have been acquired in the first place
    if (rsrc_ptr.expired()) {
      return;
    }

    std::lock_guard<std::mutex> pool_lock(pool_mtx_);

    auto rsrc_sh_ptr = rsrc_ptr.lock();
    auto itr = rsrc_map_.find(rsrc_sh_ptr);
    CHECK(itr != rsrc_map_.end() && rsrc_ptr.use_count() == 2);

    inactivateRsrc(rsrc_sh_ptr.get());
    rsrc_map_.modify(itr, ChangeLastUsedTime());
    flushInactiveQueue();
    inactive_rsrc_queue_.push_back(rsrc_ptr);
  }

  void deleteRsrc(std::weak_ptr<T>& rsrc_ptr) {
    std::lock_guard<std::mutex> pool_lock(pool_mtx_);
    {
      auto rsrc_sh_ptr = rsrc_ptr.lock();
      CHECK(rsrc_sh_ptr);
      auto itr = rsrc_map_.find(rsrc_sh_ptr);
      CHECK(itr != rsrc_map_.end() && rsrc_ptr.use_count() == 2);
      rsrc_map_.erase(itr);
      deleteRsrc(rsrc_sh_ptr.get());
    }
    flushInactiveQueue();
  }

 protected:
  virtual std::shared_ptr<T> initializeRsrc(InitTypes... args) = 0;
  virtual void updateRsrc(std::shared_ptr<T>& rsrc_ptr, InitTypes... args) = 0;
  virtual void inactivateRsrc(T* inactivated_rsrc_ptr) noexcept {}
  virtual void deleteRsrc(const T* deleted_rsrc_ptr) {}

 private:
  void flushInactiveQueue() {
    // delete rsrcs that have been inactive for a while
    static const std::chrono::milliseconds max_rsrc_idle_time =
        std::chrono::milliseconds(max_inactive_time);
    std::chrono::milliseconds cutoff_time = getCurrentTimeMS() - max_rsrc_idle_time;

    while (inactive_rsrc_queue_.begin() != inactive_rsrc_queue_.end()) {
      auto ptr = inactive_rsrc_queue_.front();
      auto rsrc_sh_ptr = ptr.lock();
      if (rsrc_sh_ptr) {
        auto itr = rsrc_map_.find(rsrc_sh_ptr);
        CHECK(itr != rsrc_map_.end());

        if (itr->last_used_time >= cutoff_time) {
          break;
        }

        rsrc_map_.erase(itr);
        deleteRsrc(rsrc_sh_ptr.get());
      }
      inactive_rsrc_queue_.pop_front();
    }
  }

  struct RsrcContainer {
    std::shared_ptr<T> rsrc_ptr;
    std::chrono::milliseconds last_used_time;

    RsrcContainer(const std::shared_ptr<T>& resource_ptr) : rsrc_ptr(resource_ptr) {
      last_used_time = getCurrentTimeMS();
    }
  };

  struct ChangeLastUsedTime {
    ChangeLastUsedTime() : new_time_(getCurrentTimeMS()) {}
    void operator()(RsrcContainer& container) { container.last_used_time = new_time_; }

   private:
    std::chrono::milliseconds new_time_;
  };

  using RsrcMap = boost::multi_index_container<
      RsrcContainer,
      boost::multi_index::indexed_by<boost::multi_index::hashed_unique<
          boost::multi_index::
              member<RsrcContainer, std::shared_ptr<T>, &RsrcContainer::rsrc_ptr>>>>;

  RsrcMap rsrc_map_;
  std::list<std::weak_ptr<T>> inactive_rsrc_queue_;
  std::mutex pool_mtx_;
};

}  // namespace QueryRenderer
