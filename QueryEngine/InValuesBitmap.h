/*
 * @file    InValuesBitmap.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_INVALUESBITMAP_H
#define QUERYENGINE_INVALUESBITMAP_H

#include "../DataMgr/DataMgr.h"

#include <llvm/IR/Value.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

class Executor;

class FailedToCreateBitmap : public std::runtime_error {
 public:
  FailedToCreateBitmap() : std::runtime_error("FailedToCreateBitmap") {}
};

class InValuesBitmap {
 public:
  InValuesBitmap(const std::vector<int64_t>& values,
                 const int64_t null_val,
                 const Data_Namespace::MemoryLevel memory_level,
                 const int device_count,
                 Data_Namespace::DataMgr* data_mgr);
  ~InValuesBitmap();

  llvm::Value* codegen(llvm::Value* needle, Executor* executor) const;

  bool isEmpty() const;

  bool hasNull() const;

 private:
  std::vector<int8_t*> bitsets_;
  bool rhs_has_null_;
  int64_t min_val_;
  int64_t max_val_;
  const int64_t null_val_;
  const Data_Namespace::MemoryLevel memory_level_;
  const int device_count_;
};

#endif  // QUERYENGINE_INVALUESBITMAP_H
