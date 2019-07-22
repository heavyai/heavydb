/*
 * Copyright 2018 OmniSci, Inc.
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

#include "MapDServer.h"
#include "ThriftHandler/MapDHandler.h"

#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/transport/TSSLServerSocket.h>
#include <thrift/transport/TSSLSocket.h>
#include <thrift/transport/TServerSocket.h>

#include "MapDRelease.h"

#include "Shared/Logger.h"
#include "Shared/MapDParameters.h"
#include "Shared/file_delete.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/scope.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <csignal>
#include <sstream>
#include <thread>
#include <vector>

using namespace ::apache::thrift;
using namespace ::apache::thrift::concurrency;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::server;
using namespace ::apache::thrift::transport;

unsigned connect_timeout{20000};
unsigned recv_timeout{300000};
unsigned send_timeout{300000};

extern bool g_cache_string_hash;
extern size_t g_leaf_count;
extern bool g_skip_intermediate_count;
extern bool g_enable_bump_allocator;
extern size_t g_max_memory_allocation_size;
extern size_t g_min_memory_allocation_size;

bool g_enable_thrift_logs{false};

std::atomic<bool> g_running{true};
std::atomic<int> g_saw_signal{-1};

TableGenerations table_generations_from_thrift(
    const std::vector<TTableGeneration>& thrift_table_generations) {
  TableGenerations table_generations;
  for (const auto& thrift_table_generation : thrift_table_generations) {
    table_generations.setGeneration(
        thrift_table_generation.table_id,
        TableGeneration{static_cast<size_t>(thrift_table_generation.tuple_count),
                        static_cast<size_t>(thrift_table_generation.start_rowid)});
  }
  return table_generations;
}

mapd::shared_ptr<MapDHandler> g_warmup_handler =
    0;  // global "g_warmup_handler" needed to avoid circular dependency
        // between "MapDHandler" & function "run_warmup_queries"
mapd::shared_ptr<MapDHandler> g_mapd_handler = 0;
std::once_flag g_shutdown_once_flag;

void shutdown_handler() {
  if (g_mapd_handler) {
    std::call_once(g_shutdown_once_flag, []() { g_mapd_handler->shutdown(); });
  }
}

void register_signal_handler(int signum, void (*handler)(int)) {
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  if (handler != SIG_DFL && handler != SIG_IGN) {
    // block all signal deliveries while inside the signal handler
    sigfillset(&act.sa_mask);
  }
  act.sa_handler = handler;
  sigaction(signum, &act, NULL);
}

// Signal handler to set a global flag telling the server to exit.
// Do not call other functions inside this (or any) signal handler
// unless you really know what you are doing. See also:
//   man 7 signal-safety
//   man 7 signal
//   https://en.wikipedia.org/wiki/Reentrancy_(computing)
void omnisci_signal_handler(int signum) {
  // Record the signal number for logging during shutdown.
  // Only records the first signal if called more than once.
  int expected_signal{-1};
  if (!g_saw_signal.compare_exchange_strong(expected_signal, signum)) {
    return;  // this wasn't the first signal
  }

  // This point should never be reached more than once.

  // Tell heartbeat() to shutdown by unsetting the 'g_running' flag.
  // If 'g_running' is already false, this has no effect and the
  // shutdown is already in progress. If a core dump is in progress,
  // say if we CHECK'ed and Logger.cpp called abort(), this should be
  // harmless because heartbeat() will notice and skip calling exit().
  g_running = false;

  // Wait briefly to give heartbeat() a head start on the shutdown work.
  sleep(2);

  // Trigger whatever default action this signal would have done,
  // such as terminate the process or dump core.
  register_signal_handler(signum, SIG_DFL);
  kill(getpid(), signum);
}

void register_signal_handlers() {
  register_signal_handler(SIGINT, omnisci_signal_handler);
  register_signal_handler(SIGQUIT, omnisci_signal_handler);
  register_signal_handler(SIGHUP, omnisci_signal_handler);
  register_signal_handler(SIGTERM, omnisci_signal_handler);
  register_signal_handler(SIGSEGV, omnisci_signal_handler);
  register_signal_handler(SIGABRT, omnisci_signal_handler);
  // Thrift secure socket can cause problems with SIGPIPE
  register_signal_handler(SIGPIPE, SIG_IGN);
}

void start_server(TThreadedServer& server, const int port) {
  try {
    server.serve();
  } catch (std::exception& e) {
    LOG(ERROR) << "Exception: " << e.what() << ": port " << port << std::endl;
  }
}

void releaseWarmupSession(TSessionId& sessionId, std::ifstream& query_file) {
  query_file.close();
  if (sessionId != g_warmup_handler->getInvalidSessionId()) {
    g_warmup_handler->disconnect(sessionId);
  }
}

void run_warmup_queries(mapd::shared_ptr<MapDHandler> handler,
                        std::string base_path,
                        std::string query_file_path) {
  // run warmup queries to load cache if requested
  if (query_file_path.empty()) {
    return;
  }
  LOG(INFO) << "Running DB warmup with queries from " << query_file_path;
  try {
    g_warmup_handler = handler;
    std::string db_info;
    std::string user_keyword, user_name, db_name;
    std::ifstream query_file;
    Catalog_Namespace::UserMetadata user;
    Catalog_Namespace::DBMetadata db;
    TSessionId sessionId = g_warmup_handler->getInvalidSessionId();

    ScopeGuard session_guard = [&] { releaseWarmupSession(sessionId, query_file); };
    query_file.open(query_file_path);
    while (std::getline(query_file, db_info)) {
      if (db_info.length() == 0) {
        continue;
      }
      std::istringstream iss(db_info);
      iss >> user_keyword >> user_name >> db_name;
      if (user_keyword.compare(0, 4, "USER") == 0) {
        // connect to DB for given user_name/db_name with super_user_rights (without
        // password), & start session
        g_warmup_handler->super_user_rights_ = true;
        g_warmup_handler->connect(sessionId, user_name, "", db_name);
        g_warmup_handler->super_user_rights_ = false;

        // read and run one query at a time for the DB with the setup connection
        TQueryResult ret;
        std::string single_query;
        while (std::getline(query_file, single_query)) {
          if (single_query.length() == 0) {
            continue;
          }
          if (single_query.compare("}") == 0) {
            single_query.clear();
            break;
          }
          g_warmup_handler->sql_execute(ret, sessionId, single_query, true, "", -1, -1);
          single_query.clear();
        }

        // stop session and disconnect from the DB
        g_warmup_handler->disconnect(sessionId);
        sessionId = g_warmup_handler->getInvalidSessionId();
      } else {
        LOG(WARNING) << "\nSyntax error in the file: " << query_file_path.c_str()
                     << " Missing expected keyword USER. Following line will be ignored: "
                     << db_info.c_str() << std::endl;
      }
      db_info.clear();
    }
  } catch (...) {
    LOG(WARNING) << "Exception while executing warmup queries. "
                 << "Warmup may not be fully completed. Will proceed nevertheless."
                 << std::endl;
  }
}

namespace po = boost::program_options;

class MapDProgramOptions {
 public:
  MapDProgramOptions(char const* argv0) : log_options_(argv0) {
    fillOptions();
    fillAdvancedOptions();
  }
  int http_port = 6278;
  size_t reserved_gpu_mem = 1 << 27;
  std::string base_path;
  std::string config_file = {"mapd.conf"};
  std::string cluster_file = {"cluster.conf"};
  std::string license_path = {""};
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
  bool enable_spirv = false;
  bool enable_auto_clear_render_mem = false;
  int render_oom_retry_threshold = 0;  // in milliseconds
  size_t render_mem_bytes = 500000000;
  size_t render_poly_cache_bytes = 300000000;

  bool enable_watchdog = true;
  bool enable_dynamic_watchdog = false;
  unsigned dynamic_watchdog_time_limit = 10000;

  /**
   * Can be used to override the number of gpus detected on the system
   * -1 means do not override
   */
  int num_gpus = -1;
  int start_gpu = 0;
  /**
   * Number of threads used when loading data
   */
  size_t num_reader_threads = 0;
  /**
   * path to file containing warmup queries list
   */
  std::string db_query_file = {""};
  /**
   * Inactive session tolerance in mins (60 mins)
   */
  int idle_session_duration = kMinsPerHour;
  /**
   * Maximum session life in mins (43,200 mins == 30 Days)
   * (https://pages.nist.gov/800-63-3/sp800-63b.html#aal3reauth)
   */
  int max_session_duration = kMinsPerMonth;
  std::string udf_file_name = {""};

 private:
  void fillOptions();
  void fillAdvancedOptions();
  [[deprecated(
      "Glog was replaced by boost.log. See source comments for more info.")]] void
  temporarily_support_deprecated_log_options_201904();

  po::options_description help_desc;
  po::options_description developer_desc;
  logger::LogOptions log_options_;
  po::variables_map vm;

 public:
  std::vector<LeafHostInfo> db_leaves;
  std::vector<LeafHostInfo> string_leaves;

  boost::optional<int> parse_command_line(int argc, char const* const* argv);
};

void MapDProgramOptions::fillOptions() {
  help_desc.add_options()("help,h", "Show available options.");
  help_desc.add_options()(
      "allow-cpu-retry",
      po::value<bool>(&g_allow_cpu_retry)
          ->default_value(g_allow_cpu_retry)
          ->implicit_value(true),
      R"(Allow the queries which failed on GPU to retry on CPU, even when watchdog is enabled.)");
  help_desc.add_options()("allow-loop-joins",
                          po::value<bool>(&allow_loop_joins)
                              ->default_value(allow_loop_joins)
                              ->implicit_value(true),
                          "Enable loop joins.");
  help_desc.add_options()("bigint-count",
                          po::value<bool>(&g_bigint_count)
                              ->default_value(g_bigint_count)
                              ->implicit_value(false),
                          "Use 64-bit count.");
  help_desc.add_options()("calcite-max-mem",
                          po::value<size_t>(&mapd_parameters.calcite_max_mem)
                              ->default_value(mapd_parameters.calcite_max_mem),
                          "Max memory available to calcite JVM.");
  help_desc.add_options()("calcite-port",
                          po::value<int>(&mapd_parameters.calcite_port)
                              ->default_value(mapd_parameters.calcite_port),
                          "Calcite port number.");
  help_desc.add_options()("config",
                          po::value<std::string>(&config_file),
                          "Path to server configuration file.");
  help_desc.add_options()("cpu-buffer-mem-bytes",
                          po::value<size_t>(&mapd_parameters.cpu_buffer_mem_bytes)
                              ->default_value(mapd_parameters.cpu_buffer_mem_bytes),
                          "Size of memory reserved for CPU buffers, in bytes.");
  help_desc.add_options()(
      "cpu-only",
      po::value<bool>(&cpu_only)->default_value(cpu_only)->implicit_value(true),
      "Run on CPU only, even if GPUs are available.");
  help_desc.add_options()("cuda-block-size",
                          po::value<size_t>(&mapd_parameters.cuda_block_size)
                              ->default_value(mapd_parameters.cuda_block_size),
                          "Size of block to use on GPU.");
  help_desc.add_options()("cuda-grid-size",
                          po::value<size_t>(&mapd_parameters.cuda_grid_size)
                              ->default_value(mapd_parameters.cuda_grid_size),
                          "Size of grid to use on GPU.");
  help_desc.add_options()(
      "data",
      po::value<std::string>(&base_path)->required()->default_value("data"),
      "Directory path to OmniSci data storage (catalogs, raw data, log files, etc).");
  help_desc.add_options()("db-query-list",
                          po::value<std::string>(&db_query_file),
                          "Path to file containing OmniSci warmup queries.");
  help_desc.add_options()("dynamic-watchdog-time-limit",
                          po::value<unsigned>(&dynamic_watchdog_time_limit)
                              ->default_value(dynamic_watchdog_time_limit)
                              ->implicit_value(10000),
                          "Dynamic watchdog time limit, in milliseconds.");
  help_desc.add_options()("enable-debug-timer",
                          po::value<bool>(&g_enable_debug_timer)
                              ->default_value(g_enable_debug_timer)
                              ->implicit_value(true),
                          "Enable debug timer logging.");
  help_desc.add_options()("enable-dynamic-watchdog",
                          po::value<bool>(&enable_dynamic_watchdog)
                              ->default_value(enable_dynamic_watchdog)
                              ->implicit_value(true),
                          "Enable dynamic watchdog.");
  help_desc.add_options()("enable-filter-push-down",
                          po::value<bool>(&g_enable_filter_push_down)
                              ->default_value(g_enable_filter_push_down)
                              ->implicit_value(true),
                          "Enable filter push down through joins.");
  help_desc.add_options()("enable-overlaps-hashjoin",
                          po::value<bool>(&g_enable_overlaps_hashjoin)
                              ->default_value(g_enable_overlaps_hashjoin)
                              ->implicit_value(true),
                          "Enable the overlaps hash join framework allowing for range "
                          "join (e.g. spatial overlaps) computation using a hash table.");
  help_desc.add_options()(
      "enable-string-dict-hash-cache",
      po::value<bool>(&g_cache_string_hash)
          ->default_value(g_cache_string_hash)
          ->implicit_value(true),
      "Cache string hash values in the string dictionary server during import.");
  help_desc.add_options()(
      "enable-thrift-logs",
      po::value<bool>(&g_enable_thrift_logs)
          ->default_value(g_enable_thrift_logs)
          ->implicit_value(true),
      "Enable writing messages directly from thrift to stdout/stderr.");
  help_desc.add_options()("enable-watchdog",
                          po::value<bool>(&enable_watchdog)
                              ->default_value(enable_watchdog)
                              ->implicit_value(true),
                          "Enable watchdog.");
  help_desc.add_options()(
      "filter-push-down-low-frac",
      po::value<float>(&g_filter_push_down_low_frac)
          ->default_value(g_filter_push_down_low_frac)
          ->implicit_value(g_filter_push_down_low_frac),
      "Lower threshold for selectivity of filters that are pushed down.");
  help_desc.add_options()(
      "filter-push-down-high-frac",
      po::value<float>(&g_filter_push_down_high_frac)
          ->default_value(g_filter_push_down_high_frac)
          ->implicit_value(g_filter_push_down_high_frac),
      "Higher threshold for selectivity of filters that are pushed down.");
  help_desc.add_options()("filter-push-down-passing-row-ubound",
                          po::value<size_t>(&g_filter_push_down_passing_row_ubound)
                              ->default_value(g_filter_push_down_passing_row_ubound)
                              ->implicit_value(g_filter_push_down_passing_row_ubound),
                          "Upperbound on the number of rows that should pass the filter "
                          "if the selectivity is less than "
                          "the high fraction threshold.");
  help_desc.add_options()("from-table-reordering",
                          po::value<bool>(&g_from_table_reordering)
                              ->default_value(g_from_table_reordering)
                              ->implicit_value(true),
                          "Enable automatic table reordering in FROM clause.");
  help_desc.add_options()("gpu-buffer-mem-bytes",
                          po::value<size_t>(&mapd_parameters.gpu_buffer_mem_bytes)
                              ->default_value(mapd_parameters.gpu_buffer_mem_bytes),
                          "Size of memory reserved for GPU buffers, in bytes, per GPU.");
  help_desc.add_options()("gpu-input-mem-limit",
                          po::value<double>(&mapd_parameters.gpu_input_mem_limit)
                              ->default_value(mapd_parameters.gpu_input_mem_limit),
                          "Force query to CPU when input data memory usage exceeds this "
                          "percentage of available GPU memory.");
  help_desc.add_options()(
      "hll-precision-bits",
      po::value<int>(&g_hll_precision_bits)
          ->default_value(g_hll_precision_bits)
          ->implicit_value(g_hll_precision_bits),
      "Number of bits used from the hash value used to specify the bucket number.");
  help_desc.add_options()("http-port",
                          po::value<int>(&http_port)->default_value(http_port),
                          "HTTP port number.");
  help_desc.add_options()(
      "idle-session-duration",
      po::value<int>(&idle_session_duration)->default_value(idle_session_duration),
      "Maximum duration of idle session.");
  help_desc.add_options()("inner-join-fragment-skipping",
                          po::value<bool>(&g_inner_join_fragment_skipping)
                              ->default_value(g_inner_join_fragment_skipping)
                              ->implicit_value(true),
                          "Enable/disable inner join fragment skipping. This feature is "
                          "considered stable and is enabled by default. This "
                          "parameter will be removed in a future release.");
  help_desc.add_options()(
      "max-session-duration",
      po::value<int>(&max_session_duration)->default_value(max_session_duration),
      "Maximum duration of active session.");
  help_desc.add_options()(
      "null-div-by-zero",
      po::value<bool>(&g_null_div_by_zero)
          ->default_value(g_null_div_by_zero)
          ->implicit_value(true),
      "Return null on division by zero instead of throwing an exception.");
  help_desc.add_options()(
      "num-reader-threads",
      po::value<size_t>(&num_reader_threads)->default_value(num_reader_threads),
      "Number of reader threads to use.");
  help_desc.add_options()(
      "overlaps-max-table-size-bytes",
      po::value<size_t>(&g_overlaps_max_table_size_bytes)
          ->default_value(g_overlaps_max_table_size_bytes),
      "The maximum size in bytes of the hash table for an overlaps hash join.");
  help_desc.add_options()("port,p",
                          po::value<int>(&mapd_parameters.omnisci_server_port)
                              ->default_value(mapd_parameters.omnisci_server_port),
                          "TCP Port number.");
  help_desc.add_options()("num-gpus",
                          po::value<int>(&num_gpus)->default_value(num_gpus),
                          "Number of gpus to use.");
  help_desc.add_options()(
      "read-only",
      po::value<bool>(&read_only)->default_value(read_only)->implicit_value(true),
      "Enable read-only mode.");
  help_desc.add_options()(
      "res-gpu-mem",
      po::value<size_t>(&reserved_gpu_mem)->default_value(reserved_gpu_mem),
      "Reduces GPU memory available to the OmniSci allocator by this amount. Used for "
      "compiled code cache and ancillary GPU functions and other processes that may also "
      "be using the GPU concurrent with OmniSciDB.");
  help_desc.add_options()("start-gpu",
                          po::value<int>(&start_gpu)->default_value(start_gpu),
                          "First gpu to use.");
  help_desc.add_options()("trivial-loop-join-threshold",
                          po::value<unsigned>(&g_trivial_loop_join_threshold)
                              ->default_value(g_trivial_loop_join_threshold)
                              ->implicit_value(1000),
                          "The maximum number of rows in the inner table of a loop join "
                          "considered to be trivially small.");
  help_desc.add_options()("verbose",
                          po::value<bool>(&verbose_logging)
                              ->default_value(verbose_logging)
                              ->implicit_value(true),
                          "Write additional debug log messages to server logs.");
  help_desc.add_options()("version,v", "Print Version Number.");

  help_desc.add_options()(
      "flush-log",
      po::value<bool>(&flush_log)->default_value(flush_log)->implicit_value(true),
      R"(DEPRECATED - Immediately flush logs to disk. Set to false if this is a performance bottleneck.)"
      " Replaced by log-auto-flush.");
  help_desc.add(log_options_.get_options());
}

void MapDProgramOptions::fillAdvancedOptions() {
  developer_desc.add_options()("dev-options", "Print internal developer options.");
  developer_desc.add_options()(
      "enable-calcite-view-optimize",
      po::value<bool>(&mapd_parameters.enable_calcite_view_optimize)
          ->default_value(mapd_parameters.enable_calcite_view_optimize)
          ->implicit_value(true),
      "Enable additional calcite (query plan) optimizations when a view is part of the "
      "query.");
  developer_desc.add_options()(
      "enable-columnar-output",
      po::value<bool>(&g_enable_columnar_output)
          ->default_value(g_enable_columnar_output)
          ->implicit_value(true),
      "Enable columnar output for intermediate/final query steps.");
  developer_desc.add_options()("enable-legacy-syntax",
                               po::value<bool>(&enable_legacy_syntax)
                                   ->default_value(enable_legacy_syntax)
                                   ->implicit_value(true),
                               "Enable legacy syntax.");
  developer_desc.add_options()(
      "enable-multifrag",
      po::value<bool>(&allow_multifrag)
          ->default_value(allow_multifrag)
          ->implicit_value(true),
      "Enable execution over multiple fragments in a single round-trip to GPU.");
  developer_desc.add_options()(
      "enable-shared-mem-group-by",
      po::value<bool>(&g_enable_smem_group_by)
          ->default_value(g_enable_smem_group_by)
          ->implicit_value(true),
      "Enable using GPU shared memory for some GROUP BY queries.");
  developer_desc.add_options()("enable-window-functions",
                               po::value<bool>(&g_enable_window_functions)
                                   ->default_value(g_enable_window_functions)
                                   ->implicit_value(true),
                               "Enable experimental window function support.");
  developer_desc.add_options()(
      "jit-debug-ir",
      po::value<bool>(&jit_debug)->default_value(jit_debug)->implicit_value(true),
      "Enable runtime debugger support for the JIT. Note that this flag is incompatible "
      "with the `ENABLE_JIT_DEBUG` build flag. The generated code can be found at "
      "`/tmp/mapdquery`.");
  developer_desc.add_options()(
      "skip-intermediate-count",
      po::value<bool>(&g_skip_intermediate_count)
          ->default_value(g_skip_intermediate_count)
          ->implicit_value(true),
      "Skip pre-flight counts for intermediate projections with no filters.");
  developer_desc.add_options()(
      "strip-join-covered-quals",
      po::value<bool>(&g_strip_join_covered_quals)
          ->default_value(g_strip_join_covered_quals)
          ->implicit_value(true),
      "Remove quals from the filtered count if they are covered by a "
      "join condition (currently only ST_Contains).");
  developer_desc.add_options()(
      "max-output-projection-allocation-bytes",
      po::value<size_t>(&g_max_memory_allocation_size)
          ->default_value(g_max_memory_allocation_size),
      "Maximum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. Default is the maximum slab size (sizes greater "
      "than the maximum slab size have no affect). Requires bump allocator.");
  developer_desc.add_options()(
      "min-output-projection-allocation-bytes",
      po::value<size_t>(&g_min_memory_allocation_size)
          ->default_value(g_min_memory_allocation_size),
      "Minimum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. If an allocation of this size cannot be "
      "obtained, the query will be retried with different execution parameters and/or on "
      "CPU (if allow-cpu-retry is enabled). Requires bump allocator.");
  developer_desc.add_options()(
      "enable-bump-allocator",
      po::value<bool>(&g_enable_bump_allocator)
          ->default_value(g_enable_bump_allocator)
          ->implicit_value(true),
      "Enable the bump allocator for projection queries on GPU. The bump allocator will "
      "allocate a fixed size buffer for each query, track the number of rows passing the "
      "kernel during query execution, and copy back only the rows that passed the kernel "
      "to CPU after execution. When disabled, pre-flight count queries are used to size "
      "the output buffer for projection queries.");
  developer_desc.add_options()("ssl-cert",
                               po::value<std::string>(&mapd_parameters.ssl_cert_file)
                                   ->default_value(std::string("")),
                               "SSL Validated public certficate.");
  developer_desc.add_options()("ssl-private-key",
                               po::value<std::string>(&mapd_parameters.ssl_key_file)
                                   ->default_value(std::string("")),
                               "SSL private key file.");
  // Note ssl_trust_store is passed through to Calcite via mapd_parameters
  // todo(jack): add ensure ssl-trust-store exists if cert and private key in use
  developer_desc.add_options()("ssl-trust-store",
                               po::value<std::string>(&mapd_parameters.ssl_trust_store)
                                   ->default_value(std::string("")),
                               "SSL Validated public cert as a java trust store.");
  developer_desc.add_options()("ssl-trust-password",
                               po::value<std::string>(&mapd_parameters.ssl_trust_password)
                                   ->default_value(std::string("")),
                               "SSL java trust store password.");
  developer_desc.add_options()(
      "udf",
      po::value<std::string>(&udf_file_name),
      "Load user defined extension functions from this file at startup. The file is "
      "expected to be a C/C++ file with extension .cpp.");
};

namespace {

std::stringstream sanitize_config_file(std::ifstream& in) {
  // Strip the web section out of the config file so boost can validate program options
  std::stringstream ss;
  std::string line;
  while (std::getline(in, line)) {
    ss << line << "\n";
    if (line == "[web]") {
      break;
    }
  }
  return ss;
}

bool trim_and_check_file_exists(std::string& filename, const std::string desc) {
  if (!filename.empty()) {
    boost::algorithm::trim_if(filename, boost::is_any_of("\"'"));
    if (!boost::filesystem::exists(filename)) {
      std::cerr << desc << " " << filename << " does not exist." << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

boost::optional<int> MapDProgramOptions::parse_command_line(int argc,
                                                            char const* const* argv) {
  po::positional_options_description positional_options;
  positional_options.add("data", 1);

  po::options_description all_desc("All options");
  all_desc.add(help_desc).add(developer_desc);

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(all_desc)
                  .positional(positional_options)
                  .run(),
              vm);
    po::notify(vm);

    if (vm.count("config")) {
      std::ifstream settings_file(config_file);

      auto sanitized_settings = sanitize_config_file(settings_file);

      po::store(po::parse_config_file(sanitized_settings, all_desc, false), vm);
      po::notify(vm);
      settings_file.close();
    }

    if (!trim_and_check_file_exists(mapd_parameters.ssl_cert_file, "ssl cert file")) {
      return 1;
    }
    if (!trim_and_check_file_exists(mapd_parameters.ssl_trust_store, "ssl trust store")) {
      return 1;
    }
    if (!trim_and_check_file_exists(mapd_parameters.ssl_key_file, "ssl key file")) {
      return 1;
    }

    if (vm.count("help")) {
      std::cout << "Usage: omnisci_server <data directory path> [-p <port number>] "
                   "[--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << help_desc << std::endl;
      return 0;
    }
    if (vm.count("dev-options")) {
      std::cout << "Usage: omnisci_server <data directory path> [-p <port number>] "
                   "[--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << developer_desc << std::endl;
      return 0;
    }
    if (vm.count("version")) {
      std::cout << "OmniSci Version: " << MAPD_RELEASE << std::endl;
      return 0;
    }

    g_enable_watchdog = enable_watchdog;
    g_enable_dynamic_watchdog = enable_dynamic_watchdog;
    g_dynamic_watchdog_time_limit = dynamic_watchdog_time_limit;
  } catch (po::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (g_hll_precision_bits < 1 || g_hll_precision_bits > 16) {
    std::cerr << "hll-precision-bits must be between 1 and 16." << std::endl;
    return 1;
  }

  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  const auto data_path = boost::filesystem::path(base_path) / "mapd_data";
  if (!boost::filesystem::exists(data_path)) {
    std::cerr << "OmniSci data directory does not exist at '" << base_path
              << "'. Run initdb " << base_path << std::endl;
    return 1;
  }

  const auto lock_file = boost::filesystem::path(base_path) / "omnisci_server_pid.lck";
  auto pid = std::to_string(getpid());
  int pid_fd = open(lock_file.c_str(), O_RDWR | O_CREAT, 0644);
  if (pid_fd == -1) {
    auto err = std::string("Failed to open PID file ") + std::string(lock_file.c_str()) +
               std::string(". ") + strerror(errno) + ".";
    std::cerr << err << std::endl;
    return 1;
  }
  if (lockf(pid_fd, F_TLOCK, 0) == -1) {
    auto err = std::string("Another OmniSci Server is using data directory ") +
               base_path + std::string(".");
    std::cerr << err << std::endl;
    close(pid_fd);
    return 1;
  }
  if (ftruncate(pid_fd, 0) == -1) {
    auto err = std::string("Failed to truncate PID file ") +
               std::string(lock_file.c_str()) + std::string(". ") + strerror(errno) +
               std::string(".");
    std::cerr << err << std::endl;
    close(pid_fd);
    return 1;
  }
  if (write(pid_fd, pid.c_str(), pid.length()) == -1) {
    auto err = std::string("Failed to write PID file ") + std::string(lock_file.c_str()) +
               ". " + strerror(errno) + ".";
    std::cerr << err << std::endl;
    close(pid_fd);
    return 1;
  }

  temporarily_support_deprecated_log_options_201904();
  if (verbose_logging && logger::Severity::DEBUG1 < log_options_.severity_) {
    log_options_.severity_ = logger::Severity::DEBUG1;
  }
  log_options_.set_base_path(base_path);
  logger::init(log_options_);

  boost::algorithm::trim_if(db_query_file, boost::is_any_of("\"'"));
  if (db_query_file.length() > 0 && !boost::filesystem::exists(db_query_file)) {
    LOG(ERROR) << "File containing DB queries " << db_query_file << " does not exist.";
    return 1;
  }
  const auto db_file =
      boost::filesystem::path(base_path) / "mapd_catalogs" / OMNISCI_SYSTEM_CATALOG;
  if (!boost::filesystem::exists(db_file)) {
    {  // check old system catalog existsense
      const auto db_file = boost::filesystem::path(base_path) / "mapd_catalogs/mapd";
      if (!boost::filesystem::exists(db_file)) {
        LOG(ERROR) << "OmniSci system catalog " << OMNISCI_SYSTEM_CATALOG
                   << " does not exist.";
        return 1;
      }
    }
  }
  // add all parameters to be displayed on startup
  LOG(INFO) << "OmniSci started with data directory at '" << base_path << "'";
  LOG(INFO) << " Watchdog is set to " << enable_watchdog;
  LOG(INFO) << " Dynamic Watchdog is set to " << enable_dynamic_watchdog;
  if (enable_dynamic_watchdog) {
    LOG(INFO) << " Dynamic Watchdog timeout is set to " << dynamic_watchdog_time_limit;
  }

  LOG(INFO) << " Debug Timer is set to " << g_enable_debug_timer;

  LOG(INFO) << " Maximum Idle session duration " << idle_session_duration;

  LOG(INFO) << " Maximum active session duration " << max_session_duration;

  if (!g_from_table_reordering) {
    LOG(INFO) << " From clause table reordering is disabled";
  }

  if (g_enable_filter_push_down) {
    LOG(INFO) << " Filter push down for JOIN is enabled";
  }

  if (vm.count("udf")) {
    boost::algorithm::trim_if(udf_file_name, boost::is_any_of("\"'"));

    if (!boost::filesystem::exists(udf_file_name)) {
      LOG(ERROR) << "User defined function file " << udf_file_name << " does not exist.";
      return 1;
    }

    LOG(INFO) << "User provided extension functions loaded from " << udf_file_name;
  }

  boost::algorithm::trim_if(mapd_parameters.ha_brokers, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ha_group_id, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ha_shared_data, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ha_unique_server_id, boost::is_any_of("\"'"));

  if (!mapd_parameters.ha_group_id.empty()) {
    LOG(INFO) << " HA group id " << mapd_parameters.ha_group_id;
    if (mapd_parameters.ha_unique_server_id.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-unique-server-id must be set ";
      return 5;
    } else {
      LOG(INFO) << " HA unique server id " << mapd_parameters.ha_unique_server_id;
    }
    if (mapd_parameters.ha_brokers.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-brokers must be set ";
      return 6;
    } else {
      LOG(INFO) << " HA brokers " << mapd_parameters.ha_brokers;
    }
    if (mapd_parameters.ha_shared_data.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-shared-data must be set ";
      return 7;
    } else {
      LOG(INFO) << " HA shared data is " << mapd_parameters.ha_shared_data;
    }
  }
  LOG(INFO) << " cuda block size " << mapd_parameters.cuda_block_size;
  LOG(INFO) << " cuda grid size  " << mapd_parameters.cuda_grid_size;
  LOG(INFO) << " calcite JVM max memory  " << mapd_parameters.calcite_max_mem;
  LOG(INFO) << " OmniSci Server Port  " << mapd_parameters.omnisci_server_port;
  LOG(INFO) << " OmniSci Calcite Port  " << mapd_parameters.calcite_port;
  LOG(INFO) << " Enable Calcite view optimize "
            << mapd_parameters.enable_calcite_view_optimize;

  LOG(INFO) << " Allow Local Auth Fallback: "
            << (authMetadata.allowLocalAuthFallback ? "enabled" : "disabled");

  boost::algorithm::trim_if(authMetadata.distinguishedName, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.uri, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapQueryUrl, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapRoleRegex, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapSuperUserRole, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.restToken, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.restUrl, boost::is_any_of("\"'"));

  return boost::none;
}

// Deprecation plan.
// Once users have updated their options to not use flush_log:
//  - Delete this function.
//  - Delete MapDProgramOptions::flush_log and all references to it.
//    (Replaced by LogOptions::auto_flush_.)
void MapDProgramOptions::temporarily_support_deprecated_log_options_201904() {
  if (!flush_log) {
    log_options_.auto_flush_ = false;
  }
}

void heartbeat() {
  // Block signals for this heartbeat thread, only.
  sigset_t set;
  sigfillset(&set);
  int result = pthread_sigmask(SIG_BLOCK, &set, NULL);
  if (result != 0) {
    throw std::runtime_error("heartbeat() thread startup failed");
  }

  // Sleep until omnisci_signal_handler or anything clears the g_running flag.
  VLOG(1) << "heartbeat thread starting";
  while (::g_running) {
    using namespace std::chrono;
    std::this_thread::sleep_for(1s);
  }
  VLOG(1) << "heartbeat thread exiting";

  // Get the signal number if there was a signal.
  int signum = g_saw_signal;
  if (signum >= 1 && signum != SIGTERM) {
    LOG(INFO) << "Interrupt signal (" << signum << ") received.";
  }

  // if dumping core, try to do some quick stuff
  if (signum == SIGQUIT || signum == SIGABRT || signum == SIGSEGV || signum == SIGFPE) {
    shutdown_handler();
    logger::shutdown();
    return;
  }

  // do an orderly shutdown
  if (signum <= 0 || signum == SIGTERM) {
    exit(EXIT_SUCCESS);
  } else {
    exit(signum);
  }
}

int main(int argc, char** argv) {
  MapDProgramOptions prog_config_opts(argv[0]);

  if (auto return_code = prog_config_opts.parse_command_line(argc, argv)) {
    return *return_code;
  }

  // try to enforce an orderly shutdown even after a signal
  register_signal_handlers();

  // register shutdown procedures for when a CHECK or a LOG(FATAL) happens
  logger::set_once_fatal_func(&shutdown_handler);

  // register shutdown procedures for when a normal exit() shutdown happens
  // be aware that atexit() functions run in reverse order
  atexit(&logger::shutdown);
  atexit(&shutdown_handler);

  // start background thread to clean up _DELETE_ME files
  const unsigned int wait_interval =
      300;  // wait time in secs after looking for deleted file before looking again
  std::thread file_delete_thread(file_delete,
                                 std::ref(g_running),
                                 wait_interval,
                                 prog_config_opts.base_path + "/mapd_data");
  std::thread heartbeat_thread(heartbeat);

  if (!g_enable_thrift_logs) {
    apache::thrift::GlobalOutput.setOutputFunction([](const char* msg) {});
  }

  g_mapd_handler =
      mapd::make_shared<MapDHandler>(prog_config_opts.db_leaves,
                                     prog_config_opts.string_leaves,
                                     prog_config_opts.base_path,
                                     prog_config_opts.cpu_only,
                                     prog_config_opts.allow_multifrag,
                                     prog_config_opts.jit_debug,
                                     prog_config_opts.read_only,
                                     prog_config_opts.allow_loop_joins,
                                     prog_config_opts.enable_rendering,
                                     prog_config_opts.enable_spirv,
                                     prog_config_opts.enable_auto_clear_render_mem,
                                     prog_config_opts.render_oom_retry_threshold,
                                     prog_config_opts.render_mem_bytes,
                                     prog_config_opts.num_gpus,
                                     prog_config_opts.start_gpu,
                                     prog_config_opts.reserved_gpu_mem,
                                     prog_config_opts.num_reader_threads,
                                     prog_config_opts.authMetadata,
                                     prog_config_opts.mapd_parameters,
                                     prog_config_opts.enable_legacy_syntax,
                                     prog_config_opts.idle_session_duration,
                                     prog_config_opts.max_session_duration,
                                     prog_config_opts.udf_file_name);

  mapd::shared_ptr<TServerSocket> serverSocket;
  mapd::shared_ptr<TServerSocket> httpServerSocket;
  if (!prog_config_opts.mapd_parameters.ssl_cert_file.empty() &&
      !prog_config_opts.mapd_parameters.ssl_key_file.empty()) {
    mapd::shared_ptr<TSSLSocketFactory> sslSocketFactory;
    sslSocketFactory =
        mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
    sslSocketFactory->loadCertificate(
        prog_config_opts.mapd_parameters.ssl_cert_file.c_str());
    sslSocketFactory->loadPrivateKey(
        prog_config_opts.mapd_parameters.ssl_key_file.c_str());
    sslSocketFactory->authenticate(false);
    sslSocketFactory->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    serverSocket = mapd::shared_ptr<TServerSocket>(new TSSLServerSocket(
        prog_config_opts.mapd_parameters.omnisci_server_port, sslSocketFactory));
    httpServerSocket = mapd::shared_ptr<TServerSocket>(
        new TSSLServerSocket(prog_config_opts.http_port, sslSocketFactory));
    LOG(INFO) << " OmniSci server using encrypted connection. Cert file ["
              << prog_config_opts.mapd_parameters.ssl_cert_file << "], key file ["
              << prog_config_opts.mapd_parameters.ssl_key_file << "]";
  } else {
    LOG(INFO) << " OmniSci server using unencrypted connection";
    serverSocket = mapd::shared_ptr<TServerSocket>(
        new TServerSocket(prog_config_opts.mapd_parameters.omnisci_server_port));
    httpServerSocket =
        mapd::shared_ptr<TServerSocket>(new TServerSocket(prog_config_opts.http_port));
  }

  if (prog_config_opts.mapd_parameters.ha_group_id.empty()) {
    mapd::shared_ptr<TProcessor> processor(new MapDProcessor(g_mapd_handler));

    mapd::shared_ptr<TTransportFactory> bufTransportFactory(
        new TBufferedTransportFactory());
    mapd::shared_ptr<TProtocolFactory> bufProtocolFactory(new TBinaryProtocolFactory());

    mapd::shared_ptr<TServerTransport> bufServerTransport(serverSocket);
    TThreadedServer bufServer(
        processor, bufServerTransport, bufTransportFactory, bufProtocolFactory);

    mapd::shared_ptr<TServerTransport> httpServerTransport(httpServerSocket);
    mapd::shared_ptr<TTransportFactory> httpTransportFactory(
        new THttpServerTransportFactory());
    mapd::shared_ptr<TProtocolFactory> httpProtocolFactory(new TJSONProtocolFactory());
    TThreadedServer httpServer(
        processor, httpServerTransport, httpTransportFactory, httpProtocolFactory);

    std::thread bufThread(start_server,
                          std::ref(bufServer),
                          prog_config_opts.mapd_parameters.omnisci_server_port);
    std::thread httpThread(
        start_server, std::ref(httpServer), prog_config_opts.http_port);

    // run warm up queries if any exists
    run_warmup_queries(
        g_mapd_handler, prog_config_opts.base_path, prog_config_opts.db_query_file);

    bufThread.join();
    httpThread.join();
  } else {  // running ha server
    LOG(FATAL) << "No High Availability module available, please contact OmniSci support";
  }

  g_running = false;
  file_delete_thread.join();
  heartbeat_thread.join();

  return 0;
};
