/*
 * @file    ThrustAllocator.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Bridge allocator for thrust that delegates to DataMgr methods.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef THRUSTALLOCATOR_H
#define THRUSTALLOCATOR_H

#include <unordered_map>

namespace Data_Namespace {
class DataMgr;
class AbstractBuffer;
};  // Data_Namespace

class ThrustAllocator {
 public:
  typedef int8_t value_type;
  ThrustAllocator(Data_Namespace::DataMgr* mgr, const int id) : data_mgr_(mgr), device_id_(id) {}

  int8_t* allocate(std::ptrdiff_t num_bytes);
  void deallocate(int8_t* ptr, size_t num_bytes);

 private:
  Data_Namespace::DataMgr* data_mgr_;
  const int device_id_;
  typedef std::unordered_map<int8_t*, Data_Namespace::AbstractBuffer*> PtrMapperType;
  PtrMapperType raw_to_ab_ptr_;
};

#endif /* THRUSTALLOCATOR_H */
