/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * File:   parameters.h
 * Author: michael
 *
 * Created on January 19, 2017, 4:31 PM
 */

#pragma once

#include <string>

struct SystemParameters {
  size_t cuda_block_size = 0;       // block size for the kernel execution
  size_t cuda_grid_size = 0;        // grid size for the kernel execution
  size_t calcite_max_mem = 1024;    // max memory for calcite jvm in MB
  int omnisci_server_port = 6274;   // default port omnisci_server runs on
  int calcite_port = 6279;          // default port for calcite server to run on
  std::string ha_group_id;          // name of the HA group this server is in
  std::string ha_unique_server_id;  // name of the HA unique id for this server
  std::string ha_brokers;           // name of the HA broker
  std::string ha_shared_data;       // name of shared data directory base
  bool is_decr_start_epoch;         // are we doing a start epoch decrement?
  size_t cpu_buffer_mem_bytes = 0;  // max size of memory reserved for CPU buffers [bytes]
  size_t gpu_buffer_mem_bytes = 0;  // max size of memory reserved for GPU buffers [bytes]
  size_t min_cpu_slab_size =
      1L << 28;  // min size of CPU buffer pool memory allocations [bytes], default=256MB
  size_t min_gpu_slab_size =
      1L << 28;  // min size of GPU buffer pool memory allocations [bytes], default=256MB
  size_t max_cpu_slab_size =
      1L << 32;  // max size of CPU buffer pool memory allocations [bytes], default=4GB
  size_t max_gpu_slab_size =
      1L << 31;  // max size of CPU buffer pool memory allocations [bytes], default=2GB
  double gpu_input_mem_limit = 0.9;  // Punt query to CPU if input mem exceeds % GPU mem
  std::string config_file = "";
  std::string ssl_cert_file = "";    // file path to server's certified PKI certificate
  std::string ssl_key_file = "";     // file path to server's' private PKI key
  std::string ssl_trust_store = "";  // file path to java jks version of ssl_key_fle
  std::string ssl_trust_password = "";
  std::string ssl_keystore = "";
  std::string ssl_keystore_password = "";  // pass phrae for java jks trust store.
  std::string ssl_trust_ca_file = "";
  bool ssl_transport_client_auth = false;
  bool aggregator = false;
  bool enable_calcite_view_optimize =
      true;  // allow calcite to optimize the relalgebra for a view query
  size_t calcite_timeout =
      5000;  // calcite send/receive timeout (connect timeout hard coded to 2s)
  int num_executors = 1;

  SystemParameters() : cuda_block_size(0), cuda_grid_size(0), calcite_max_mem(1024) {}
};
