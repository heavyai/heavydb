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
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "ThriftHandler/MapDHandler.h"

#ifdef HAVE_THRIFT_THREADFACTORY
#include <thrift/concurrency/ThreadFactory.h>
#else
#include <thrift/concurrency/PlatformThreadFactory.h>
#endif

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

#include "Archive/S3Archive.h"
#include "Shared/Logger.h"
#include "Shared/MapDParameters.h"
#include "Shared/file_delete.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/scope.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/locale/generator.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include <csignal>
#include <sstream>
#include <thread>
#include <vector>
#include "MapDRelease.h"
#include "Shared/Compressor.h"
#include "Shared/MapDParameters.h"
#include "Shared/file_delete.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/scope.h"

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
extern bool g_enable_experimental_string_functions;
extern bool g_enable_table_functions;
extern bool g_enable_fsi;
extern bool g_enable_interop;

bool g_enable_thrift_logs{false};

std::atomic<bool> g_running{true};
std::atomic<int> g_saw_signal{-1};

mapd_shared_mutex g_thrift_mutex;
TThreadedServer* g_thrift_http_server{nullptr};
TThreadedServer* g_thrift_buf_server{nullptr};

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
  // shutdown is already in progress.
  g_running = false;

  // Handle core dumps specially by pausing inside this signal handler
  // because on some systems, some signals will execute their default
  // action immediately when and if the signal handler returns.
  // We would like to do some emergency cleanup before core dump.
  if (signum == SIGQUIT || signum == SIGABRT || signum == SIGSEGV || signum == SIGFPE) {
    // Wait briefly to give heartbeat() a chance to flush the logs and
    // do any other emergency shutdown tasks.
    sleep(2);

    // Explicitly trigger whatever default action this signal would
    // have done, such as terminate the process or dump core.
    // Signals are currently blocked so this new signal will be queued
    // until this signal handler returns.
    register_signal_handler(signum, SIG_DFL);
    kill(getpid(), signum);
  }
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
          boost::algorithm::trim(single_query);
          if (single_query.length() == 0 || single_query[0] == '-') {
            continue;
          }
          if (single_query[0] == '}') {
            single_query.clear();
            break;
          }
          if (single_query.find(';') == single_query.npos) {
            std::string multiline_query;
            std::getline(query_file, multiline_query, ';');
            single_query += multiline_query;
          }

          try {
            g_warmup_handler->sql_execute(ret, sessionId, single_query, true, "", -1, -1);
          } catch (...) {
            LOG(WARNING) << "Exception while executing '" << single_query
                         << "', ignoring";
          }
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
  MapDProgramOptions(char const* argv0, bool dist_v5_ = false)
      : log_options_(argv0), dist_v5_(dist_v5_) {
    fillOptions();
    fillAdvancedOptions();
  }
  int http_port = 6278;
  size_t reserved_gpu_mem = 384 * 1024 * 1024;
  std::string base_path;
  std::string cluster_file = {"cluster.conf"};
  std::string cluster_topology_file = {"cluster_topology.conf"};
  std::string license_path = {""};
  bool cpu_only = false;
  bool verbose_logging = false;
  bool jit_debug = false;
  bool intel_jit_profile = false;
  bool allow_multifrag = true;
  bool read_only = false;
  bool allow_loop_joins = false;
  bool enable_legacy_syntax = true;
  AuthMetadata authMetadata;

  MapDParameters mapd_parameters;
  bool enable_rendering = false;
  bool enable_auto_clear_render_mem = false;
  int render_oom_retry_threshold = 0;  // in milliseconds
  size_t render_mem_bytes = 500000000;
  size_t render_poly_cache_bytes = 300000000;

  bool enable_runtime_udf = false;

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
   * exit after warmup
   */
  bool exit_after_warmup = false;
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

  void fillOptions();
  void fillAdvancedOptions();

  po::options_description help_desc;
  po::options_description developer_desc;
  logger::LogOptions log_options_;
  po::positional_options_description positional_options;

 public:
  std::vector<LeafHostInfo> db_leaves;
  std::vector<LeafHostInfo> string_leaves;
  po::variables_map vm;
  std::string clusterIds_arg;

  std::string getNodeIds();
  std::vector<std::string> getNodeIdsArray();
  static const std::string nodeIds_token;

  boost::optional<int> parse_command_line(int argc, char const* const* argv);
  void validate();
  void validate_base_path();
  void init_logging();
  const bool dist_v5_;
};

void MapDProgramOptions::init_logging() {
  if (verbose_logging && logger::Severity::DEBUG1 < log_options_.severity_) {
    log_options_.severity_ = logger::Severity::DEBUG1;
  }
  log_options_.set_base_path(base_path);
  logger::init(log_options_);
}

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
  if (!dist_v5_) {
    help_desc.add_options()("calcite-port",
                            po::value<int>(&mapd_parameters.calcite_port)
                                ->default_value(mapd_parameters.calcite_port),
                            "Calcite port number.");
  }
  help_desc.add_options()("config",
                          po::value<std::string>(&mapd_parameters.config_file),
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
  if (!dist_v5_) {
    help_desc.add_options()(
        "data",
        po::value<std::string>(&base_path)->required()->default_value("data"),
        "Directory path to OmniSci data storage (catalogs, raw data, log files, etc).");
    positional_options.add("data", 1);
  }
  help_desc.add_options()("db-query-list",
                          po::value<std::string>(&db_query_file),
                          "Path to file containing OmniSci warmup queries.");
  help_desc.add_options()(
      "exit-after-warmup",
      po::value<bool>(&exit_after_warmup)->default_value(false)->implicit_value(true),
      "Exit after OmniSci warmup queries.");
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
  if (!dist_v5_) {
    help_desc.add_options()(
        "enable-string-dict-hash-cache",
        po::value<bool>(&g_cache_string_hash)
            ->default_value(g_cache_string_hash)
            ->implicit_value(true),
        "Cache string hash values in the string dictionary server during import.");
  }
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
  if (!dist_v5_) {
    help_desc.add_options()("http-port",
                            po::value<int>(&http_port)->default_value(http_port),
                            "HTTP port number.");
  }
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
  if (!dist_v5_) {
    help_desc.add_options()("port,p",
                            po::value<int>(&mapd_parameters.omnisci_server_port)
                                ->default_value(mapd_parameters.omnisci_server_port),
                            "TCP Port number.");
  }
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
  help_desc.add_options()(
      "enable-runtime-udf",
      po::value<bool>(&enable_runtime_udf)
          ->default_value(enable_runtime_udf)
          ->implicit_value(true),
      "Enable runtime UDF registration by passing signatures and corresponding LLVM IR "
      "to the `register_runtime_udf` endpoint. For use with the Python Remote Backend "
      "Compiler server, packaged separately.");
  help_desc.add_options()("version,v", "Print Version Number.");
  help_desc.add_options()("enable-experimental-string-functions",
                          po::value<bool>(&g_enable_experimental_string_functions)
                              ->default_value(g_enable_experimental_string_functions)
                              ->implicit_value(true),
                          "Enable experimental string functions.");
  help_desc.add_options()(
      "enable-fsi",
      po::value<bool>(&g_enable_fsi)->default_value(g_enable_fsi)->implicit_value(true),
      "Enable foreign storage interface.");
  help_desc.add_options()(
      "enable-interoperability",
      po::value<bool>(&g_enable_interop)
          ->default_value(g_enable_interop)
          ->implicit_value(true),
      "Enable offloading of query portions to an external execution engine.");
  help_desc.add_options()(
      "calcite-service-timeout",
      po::value<size_t>(&mapd_parameters.calcite_timeout)
          ->default_value(mapd_parameters.calcite_timeout),
      "Calcite server timeout (milliseconds). Increase this on systems with frequent "
      "schema changes or when running large numbers of parallel queries.");

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
  developer_desc.add_options()("enable-direct-columnarization",
                               po::value<bool>(&g_enable_direct_columnarization)
                                   ->default_value(g_enable_direct_columnarization)
                                   ->implicit_value(true),
                               "Enables/disables a more optimized columnarization method "
                               "for intermediate steps in multi-step queries.");
  developer_desc.add_options()("enable-window-functions",
                               po::value<bool>(&g_enable_window_functions)
                                   ->default_value(g_enable_window_functions)
                                   ->implicit_value(true),
                               "Enable experimental window function support.");
  developer_desc.add_options()("enable-table-functions",
                               po::value<bool>(&g_enable_table_functions)
                                   ->default_value(g_enable_table_functions)
                                   ->implicit_value(true),
                               "Enable experimental table functions support.");
  developer_desc.add_options()(
      "jit-debug-ir",
      po::value<bool>(&jit_debug)->default_value(jit_debug)->implicit_value(true),
      "Enable runtime debugger support for the JIT. Note that this flag is "
      "incompatible "
      "with the `ENABLE_JIT_DEBUG` build flag. The generated code can be found at "
      "`/tmp/mapdquery`.");
  developer_desc.add_options()(
      "intel-jit-profile",
      po::value<bool>(&intel_jit_profile)
          ->default_value(intel_jit_profile)
          ->implicit_value(true),
      "Enable runtime support for the JIT code profiling using Intel VTune.");
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
      "queries with no pre-flight count. Default is the maximum slab size (sizes "
      "greater "
      "than the maximum slab size have no affect). Requires bump allocator.");
  developer_desc.add_options()(
      "min-output-projection-allocation-bytes",
      po::value<size_t>(&g_min_memory_allocation_size)
          ->default_value(g_min_memory_allocation_size),
      "Minimum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. If an allocation of this size cannot be "
      "obtained, the query will be retried with different execution parameters and/or "
      "on "
      "CPU (if allow-cpu-retry is enabled). Requires bump allocator.");
  developer_desc.add_options()("enable-bump-allocator",
                               po::value<bool>(&g_enable_bump_allocator)
                                   ->default_value(g_enable_bump_allocator)
                                   ->implicit_value(true),
                               "Enable the bump allocator for projection queries on "
                               "GPU. The bump allocator will "
                               "allocate a fixed size buffer for each query, track the "
                               "number of rows passing the "
                               "kernel during query execution, and copy back only the "
                               "rows that passed the kernel "
                               "to CPU after execution. When disabled, pre-flight "
                               "count queries are used to size "
                               "the output buffer for projection queries.");

  developer_desc.add_options()("ssl-cert",
                               po::value<std::string>(&mapd_parameters.ssl_cert_file)
                                   ->default_value(std::string("")),
                               "SSL Validated public certficate.");

  developer_desc.add_options()(
      "pki-db-client-auth",
      po::value<bool>(&authMetadata.pki_db_client_auth)->default_value(false),
      "Use client PKI authentication to the database.");

  developer_desc.add_options()(
      "ssl-transport-client-auth",
      po::value<bool>(&mapd_parameters.ssl_transport_client_auth)->default_value(false),
      "SSL Use client PKI authentication at the transport layer.");

  developer_desc.add_options()("ssl-private-key",
                               po::value<std::string>(&mapd_parameters.ssl_key_file)
                                   ->default_value(std::string("")),
                               "SSL private key file.");
  // Note ssl_trust_store is passed through to Calcite via mapd_parameters
  // todo(jack): add ensure ssl-trust-store exists if cert and private key in use
  developer_desc.add_options()("ssl-trust-store",
                               po::value<std::string>(&mapd_parameters.ssl_trust_store)
                                   ->default_value(std::string("")),
                               "SSL public CA certifcates (java trust store) to validate "
                               "TLS connections (passed through to the Calcite server).");

  developer_desc.add_options()(
      "ssl-trust-password",
      po::value<std::string>(&mapd_parameters.ssl_trust_password)
          ->default_value(std::string("")),
      "SSL password for java trust store provided via --ssl-trust-store parameter.");

  developer_desc.add_options()(
      "ssl-trust-ca",
      po::value<std::string>(&mapd_parameters.ssl_trust_ca_file)
          ->default_value(std::string("")),
      "SSL public CA certificates to validate TLS connection(as a client).");

  developer_desc.add_options()(
      "ssl-trust-ca-server",
      po::value<std::string>(&authMetadata.ca_file_name)->default_value(std::string("")),
      "SSL public CA certificates to validate TLS connection(as a server).");

  developer_desc.add_options()("ssl-keystore",
                               po::value<std::string>(&mapd_parameters.ssl_keystore)
                                   ->default_value(std::string("")),
                               "SSL server credentials as a java key store (passed "
                               "through to the Calcite server).");

  developer_desc.add_options()(
      "ssl-keystore-password",
      po::value<std::string>(&mapd_parameters.ssl_keystore_password)
          ->default_value(std::string("")),
      "SSL password for java keystore, provide by via --ssl-keystore.");

  developer_desc.add_options()(
      "udf",
      po::value<std::string>(&udf_file_name),
      "Load user defined extension functions from this file at startup. The file is "
      "expected to be a C/C++ file with extension .cpp.");
}

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

void MapDProgramOptions::validate_base_path() {
  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  if (!boost::filesystem::exists(base_path)) {
    throw std::runtime_error("OmniSci base directory does not exist at " + base_path);
  }
}

void MapDProgramOptions::validate() {
  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  const auto data_path = boost::filesystem::path(base_path) / "mapd_data";
  if (!boost::filesystem::exists(data_path)) {
    throw std::runtime_error("OmniSci data directory does not exist at '" + base_path +
                             "'");
  }

  {
    const auto lock_file = boost::filesystem::path(base_path) / "omnisci_server_pid.lck";
    auto pid = std::to_string(getpid());

    int pid_fd = open(lock_file.c_str(), O_RDWR | O_CREAT, 0644);
    if (pid_fd == -1) {
      auto err = std::string("Failed to open PID file ") + lock_file.c_str() + ". " +
                 strerror(errno) + ".";
      throw std::runtime_error(err);
    }
    if (lockf(pid_fd, F_TLOCK, 0) == -1) {
      close(pid_fd);
      auto err = std::string("Another OmniSci Server is using data directory ") +
                 base_path + ".";
      throw std::runtime_error(err);
    }
    if (ftruncate(pid_fd, 0) == -1) {
      close(pid_fd);
      auto err = std::string("Failed to truncate PID file ") + lock_file.c_str() + ". " +
                 strerror(errno) + ".";
      throw std::runtime_error(err);
    }
    if (write(pid_fd, pid.c_str(), pid.length()) == -1) {
      close(pid_fd);
      auto err = std::string("Failed to write PID file ") + lock_file.c_str() + ". " +
                 strerror(errno) + ".";
      throw std::runtime_error(err);
    }
  }
  boost::algorithm::trim_if(db_query_file, boost::is_any_of("\"'"));
  if (db_query_file.length() > 0 && !boost::filesystem::exists(db_query_file)) {
    throw std::runtime_error("File containing DB queries " + db_query_file +
                             " does not exist.");
  }
  const auto db_file =
      boost::filesystem::path(base_path) / "mapd_catalogs" / OMNISCI_SYSTEM_CATALOG;
  if (!boost::filesystem::exists(db_file)) {
    {  // check old system catalog existsense
      const auto db_file = boost::filesystem::path(base_path) / "mapd_catalogs/mapd";
      if (!boost::filesystem::exists(db_file)) {
        throw std::runtime_error("OmniSci system catalog " + OMNISCI_SYSTEM_CATALOG +
                                 " does not exist.");
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
}

boost::optional<int> MapDProgramOptions::parse_command_line(int argc,
                                                            char const* const* argv) {
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
      std::ifstream settings_file(mapd_parameters.config_file);

      auto sanitized_settings = sanitize_config_file(settings_file);

      po::store(po::parse_config_file(sanitized_settings, all_desc, false), vm);
      po::notify(vm);
      settings_file.close();
    }

    if (!trim_and_check_file_exists(mapd_parameters.ssl_cert_file, "ssl cert file")) {
      return 1;
    }
    if (!trim_and_check_file_exists(authMetadata.ca_file_name, "ca file name")) {
      return 1;
    }
    if (!trim_and_check_file_exists(mapd_parameters.ssl_trust_store, "ssl trust store")) {
      return 1;
    }
    if (!trim_and_check_file_exists(mapd_parameters.ssl_keystore, "ssl key store")) {
      return 1;
    }
    if (!trim_and_check_file_exists(mapd_parameters.ssl_key_file, "ssl key file")) {
      return 1;
    }
    if (!trim_and_check_file_exists(mapd_parameters.ssl_trust_ca_file, "ssl ca file")) {
      return 1;
    }

    if (vm.count("help")) {
      std::cerr << "Usage: omnisci_server <data directory path> [-p <port number>] "
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

  if (!g_from_table_reordering) {
    LOG(INFO) << " From clause table reordering is disabled";
  }

  if (g_enable_filter_push_down) {
    LOG(INFO) << " Filter push down for JOIN is enabled";
  }

  if (vm.count("udf")) {
    boost::algorithm::trim_if(udf_file_name, boost::is_any_of("\"'"));

    if (!boost::filesystem::exists(udf_file_name)) {
      LOG(ERROR) << " User defined function file " << udf_file_name << " does not exist.";
      return 1;
    }

    LOG(INFO) << " User provided extension functions loaded from " << udf_file_name;
  }

  if (enable_runtime_udf) {
    LOG(INFO) << " Runtime user defined extension functions enabled globally.";
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

void heartbeat() {
  // Block all signals for this heartbeat thread, only.
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
    if (g_mapd_handler) {
      std::call_once(g_shutdown_once_flag,
                     []() { g_mapd_handler->emergency_shutdown(); });
    }
    logger::shutdown();
    return;
    // core dump should begin soon after this, see omnisci_signal_handler()
  }

  // trigger an orderly shutdown by telling Thrift to stop serving
  {
    mapd_shared_lock<mapd_shared_mutex> read_lock(g_thrift_mutex);
    auto httpserv = g_thrift_http_server;
    if (httpserv) {
      httpserv->stop();
    }
    auto bufserv = g_thrift_buf_server;
    if (bufserv) {
      bufserv->stop();
    }
    // main() should return soon after this
  }
}

int startMapdServer(MapDProgramOptions& prog_config_opts, bool start_http_server = true) {
  // try to enforce an orderly shutdown even after a signal
  register_signal_handlers();

  // register shutdown procedures for when a normal shutdown happens
  // be aware that atexit() functions run in reverse order
  atexit(&logger::shutdown);
  atexit(&shutdown_handler);

#ifdef HAVE_AWS_S3
  // hold a s3 archive here to survive from a segfault that happens on centos
  // when s3 transactions and others openssl-ed sessions are interleaved...
  auto s3_survivor = std::make_unique<S3Archive>("s3://omnisci/s3_survivor.txt", true);
#endif

  // start background thread to clean up _DELETE_ME files
  const unsigned int wait_interval =
      3;  // wait time in secs after looking for deleted file before looking again
  std::thread file_delete_thread(file_delete,
                                 std::ref(g_running),
                                 wait_interval,
                                 prog_config_opts.base_path + "/mapd_data");
  std::thread heartbeat_thread(heartbeat);

  if (!g_enable_thrift_logs) {
    apache::thrift::GlobalOutput.setOutputFunction([](const char* msg) {});
  }

  if (g_enable_experimental_string_functions) {
    // Use the locale setting of the server by default. The generate parameter can be
    // updated appropriately if a locale override option is ever supported.
    boost::locale::generator generator;
    std::locale::global(generator.generate(""));
  }

  try {
    g_mapd_handler =
        mapd::make_shared<MapDHandler>(prog_config_opts.db_leaves,
                                       prog_config_opts.string_leaves,
                                       prog_config_opts.base_path,
                                       prog_config_opts.cpu_only,
                                       prog_config_opts.allow_multifrag,
                                       prog_config_opts.jit_debug,
                                       prog_config_opts.intel_jit_profile,
                                       prog_config_opts.read_only,
                                       prog_config_opts.allow_loop_joins,
                                       prog_config_opts.enable_rendering,
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
                                       prog_config_opts.enable_runtime_udf,
                                       prog_config_opts.udf_file_name);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize service handler: " << e.what();
  }

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
    if (prog_config_opts.mapd_parameters.ssl_transport_client_auth) {
      sslSocketFactory->authenticate(true);
    } else {
      sslSocketFactory->authenticate(false);
    }
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

  ScopeGuard pointer_to_thrift_guard = [] {
    mapd_lock_guard<mapd_shared_mutex> write_lock(g_thrift_mutex);
    g_thrift_buf_server = g_thrift_http_server = nullptr;
  };

  if (prog_config_opts.mapd_parameters.ha_group_id.empty()) {
    mapd::shared_ptr<TProcessor> processor(new MapDTrackingProcessor(g_mapd_handler));
    mapd::shared_ptr<TTransportFactory> bufTransportFactory(
        new TBufferedTransportFactory());
    mapd::shared_ptr<TProtocolFactory> bufProtocolFactory(new TBinaryProtocolFactory());

    mapd::shared_ptr<TServerTransport> bufServerTransport(serverSocket);
    TThreadedServer bufServer(
        processor, bufServerTransport, bufTransportFactory, bufProtocolFactory);
    {
      mapd_lock_guard<mapd_shared_mutex> write_lock(g_thrift_mutex);
      g_thrift_buf_server = &bufServer;
    }

    std::thread bufThread(start_server,
                          std::ref(bufServer),
                          prog_config_opts.mapd_parameters.omnisci_server_port);

    // TEMPORARY
    auto warmup_queries = [&prog_config_opts]() {
      // run warm up queries if any exists
      run_warmup_queries(
          g_mapd_handler, prog_config_opts.base_path, prog_config_opts.db_query_file);
      if (prog_config_opts.exit_after_warmup) {
        g_running = false;
      }
    };

    mapd::shared_ptr<TServerTransport> httpServerTransport(httpServerSocket);
    mapd::shared_ptr<TTransportFactory> httpTransportFactory(
        new THttpServerTransportFactory());
    mapd::shared_ptr<TProtocolFactory> httpProtocolFactory(new TJSONProtocolFactory());
    TThreadedServer httpServer(
        processor, httpServerTransport, httpTransportFactory, httpProtocolFactory);
    if (start_http_server) {
      {
        mapd_lock_guard<mapd_shared_mutex> write_lock(g_thrift_mutex);
        g_thrift_http_server = &httpServer;
      }
      std::thread httpThread(
          start_server, std::ref(httpServer), prog_config_opts.http_port);

      warmup_queries();

      bufThread.join();
      httpThread.join();
    } else {
      warmup_queries();
      bufThread.join();
    }
  } else {  // running ha server
    LOG(FATAL) << "No High Availability module available, please contact OmniSci support";
  }

  g_running = false;
  file_delete_thread.join();
  heartbeat_thread.join();
  ForeignStorageInterface::destroy();

  int signum = g_saw_signal;
  if (signum <= 0 || signum == SIGTERM) {
    return 0;
  } else {
    return signum;
  }
}

const std::string MapDProgramOptions::nodeIds_token = {"node_id"};

int main(int argc, char** argv) {
  bool has_clust_topo = false;

  MapDProgramOptions prog_config_opts(argv[0], has_clust_topo);

  try {
    if (auto return_code = prog_config_opts.parse_command_line(argc, argv)) {
      return *return_code;
    }

    if (!has_clust_topo) {
      prog_config_opts.validate_base_path();
      prog_config_opts.init_logging();
      prog_config_opts.validate();
      return (startMapdServer(prog_config_opts));
    }
  } catch (std::runtime_error& e) {
    std::cerr << "Can't start: " << e.what() << std::endl;
    return 1;
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }
}
