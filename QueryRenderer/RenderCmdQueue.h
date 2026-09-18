/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
#include <thread>

#include "Logger/Logger.h"

namespace QueryRenderer {

class RenderCmdQueue {
 public:
  using RenderFunc = std::function<void()>;
  using CleanupFunc = std::function<void()>;

  RenderCmdQueue();
  ~RenderCmdQueue();

  void submit(RenderFunc render_func, CleanupFunc clean_func = nullptr);
  void stop();

 private:
  using CommandId = int64_t;
  struct RenderCommand {
    RenderFunc render_func;
    CleanupFunc clean_func;
    CommandId id;
    std::exception_ptr err;
    logger::ThreadLocalIds parent_thread_local_ids;

    RenderCommand() : id{-1}, parent_thread_local_ids{logger::thread_local_ids()} {}
    RenderCommand(RenderFunc rf, CleanupFunc cf, const CommandId i)
        : render_func{rf}
        , clean_func{cf}
        , id{i}
        , parent_thread_local_ids{logger::thread_local_ids()} {}
  };

  std::list<RenderCommand> command_queue_;
  std::mutex start_command_mutex_;
  std::condition_variable start_command_condv_;

  CommandId current_command_id_;
  RenderCommand finished_command_;
  std::mutex end_command_mutex_;
  std::condition_variable end_command_condv_;

  std::mutex stopping_mutex_;
  std::condition_variable stopping_condv_;

  bool alive_;
  bool stopped_;

  std::thread::id executor_thread_id_;

  void submitToQueue(RenderFunc render_func, CleanupFunc clean_func);
  void runNextRenderCommand();
  void addFinishedCommand(const RenderCommand& command);
  void start();
  void setStopped();
};

}  // namespace QueryRenderer
