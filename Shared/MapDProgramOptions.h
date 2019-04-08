/*
 * Copyright 2018 OmniSci Technologies, Inc.
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
 * @file    MapDProgramOptions.h
 * @author  Kursat Uvez <kursat.uvez@omnisci.com>
 *
 */

#ifndef MAPDPROGRAMOPTIONS_H
#define MAPDPROGRAMOPTIONS_H

#include <boost/program_options.hpp>
#include "Catalog/AuthMetadata.h"
#include "LeafHostInfo.h"

class MapDProgramOptions : public boost::program_options::options_description {
 public:
  MapDProgramOptions();

  int http_port = 6278;
  size_t reserved_gpu_mem = 1 << 27;
  std::string base_path;
  std::string config_file = {"mapd.conf"};
  std::string cluster_file = {"cluster.conf"};
  bool cpu_only = false;
  bool flush_log = true;
  bool verbose_logging = false;
  bool jit_debug = false;
  bool allow_multifrag = true;
  bool read_only = false;
  bool allow_loop_joins = false;
  bool enable_legacy_syntax = true;
  AuthMetadata authMetadata;

  MapDParameters mapd_parameters;
  bool enable_rendering = false;
  bool enable_watchdog = true;
  bool enable_dynamic_watchdog = false;
  unsigned dynamic_watchdog_time_limit = 10000;

  size_t render_mem_bytes = 500000000;
  size_t render_poly_cache_bytes = 300000000;
  int num_gpus = -1;  // Can be used to override number of gpus detected on system - -1
                      // means do not override
  int start_gpu = 0;
  size_t num_reader_threads = 0;         // number of threads used when loading data
  std::string db_query_file = {""};      // path to file containing warmup queries list
  bool enable_access_priv_check = true;  // enable DB objects access privileges checking
  int idle_session_duration =
      MINSPERHOUR;  // Inactive session tolerance in mins (60 mins)
  int max_session_duration =
      MINSPERMONTH;  // maximum session life in days (30 Days)
                     // (https://pages.nist.gov/800-63-3/sp800-63b.html#aal3reauth)

 private:
  void fillOptions(boost::program_options::options_description& desc);
  void fillAdvancedOptions(boost::program_options::options_description& desc_adv);

  boost::program_options::variables_map vm;

 public:
  std::vector<LeafHostInfo> db_leaves;
  std::vector<LeafHostInfo> string_leaves;

  bool parse_command_line(int argc, char** argv, int& return_code);
};

#endif  // MAPDPROGRAMOPTIONS_H
