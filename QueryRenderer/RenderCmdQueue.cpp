/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/RenderCmdQueue.h"

#include "Shared/nvtx_helpers.h"
#include "Shared/scope.h"

namespace QueryRenderer {

//
// RenderCmdQueue
//

RenderCmdQueue::RenderCmdQueue()
    : current_command_id_(-1), alive_(true), stopped_(false) {
  std::thread t1(&RenderCmdQueue::start, this);
  executor_thread_id_ = t1.get_id();
  t1.detach();
}

RenderCmdQueue::~RenderCmdQueue() {
  if (alive_) {
    stop();
  }
}

void RenderCmdQueue::submit(RenderFunc render_func, CleanupFunc clean_func) {
  if (std::this_thread::get_id() == executor_thread_id_) {
    ScopeGuard reset_renderer = [clean_func] {
      if (clean_func) {
        clean_func();
      }
    };
    render_func();
  } else {
    submitToQueue(render_func, clean_func);
  }
}

void RenderCmdQueue::stop() {
  if (alive_ || !stopped_) {
    alive_ = false;
    end_command_condv_.notify_all();
    start_command_condv_.notify_all();

    std::unique_lock<std::mutex> lock(stopping_mutex_);
    stopping_condv_.wait(lock, [this] { return stopped_; });
  }
}

void RenderCmdQueue::submitToQueue(RenderFunc render_func, CleanupFunc clean_func) {
  CHECK(!stopped_);
  CommandId cmd_id{-1};
  {
    std::lock_guard<std::mutex> lock(start_command_mutex_);
    cmd_id = ++current_command_id_;
    command_queue_.emplace_back(render_func, clean_func, cmd_id);
  }
  start_command_condv_.notify_one();

  std::unique_lock<std::mutex> lock(end_command_mutex_);
  end_command_condv_.wait(
      lock, [this, &cmd_id] { return finished_command_.id == cmd_id || !alive_; });
  if (alive_ && finished_command_.err) {
    std::rethrow_exception(finished_command_.err);
  }
}

void RenderCmdQueue::runNextRenderCommand() {
  std::unique_lock<std::mutex> lock(start_command_mutex_);
  start_command_condv_.wait(lock, [this] { return command_queue_.size() || !alive_; });
  if (alive_) {
    auto command = command_queue_.front();
    command_queue_.pop_front();
    logger::LocalIdsScopeGuard lisg = command.parent_thread_local_ids.setNewThreadId();
    try {
      ScopeGuard unset_renderer = [&command] {
        if (command.clean_func) {
          command.clean_func();
        }
      };
      command.render_func();
    } catch (...) {
      command.err = std::current_exception();
    }
    addFinishedCommand(command);
  }
}

void RenderCmdQueue::addFinishedCommand(const RenderCmdQueue::RenderCommand& command) {
  {
    std::lock_guard<std::mutex> lock(end_command_mutex_);
    finished_command_ = command;
  }
  end_command_condv_.notify_one();
}

void RenderCmdQueue::setStopped() {
  {
    std::lock_guard<std::mutex> lock(stopping_mutex_);
    stopped_ = true;
  }
  stopping_condv_.notify_one();
}

void RenderCmdQueue::start() {
  nvtx_helpers::name_current_thread("RenderCmdQueue");
  ScopeGuard thread_exit = [this] {
    if (alive_) {
      CHECK(false) << "Render thread exited abnormally";
      alive_ = false;
    }
    setStopped();
  };

  while (true) {
    runNextRenderCommand();
    if (!alive_) {
      break;
    }
  }
  alive_ = false;
}

}  // namespace QueryRenderer
