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

#include "Shared/MapDParameters.h"
#include "Shared/MapDProgramOptions.h"
#include "Shared/file_delete.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/measure.h"
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

extern size_t g_leaf_count;

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

std::vector<LeafHostInfo> only_db_leaves(const std::vector<LeafHostInfo>& all_leaves) {
  std::vector<LeafHostInfo> data_leaves;
  std::copy_if(
      all_leaves.begin(),
      all_leaves.end(),
      std::back_inserter(data_leaves),
      [](const LeafHostInfo& leaf) { return leaf.getRole() == NodeRole::DbLeaf; });
  return data_leaves;
}

std::vector<LeafHostInfo> only_string_leaves(
    const std::vector<LeafHostInfo>& all_leaves) {
  std::vector<LeafHostInfo> string_leaves;
  std::copy_if(
      all_leaves.begin(),
      all_leaves.end(),
      std::back_inserter(string_leaves),
      [](const LeafHostInfo& leaf) { return leaf.getRole() == NodeRole::String; });
  return string_leaves;
}

mapd::shared_ptr<MapDHandler> g_warmup_handler =
    0;  // global "g_warmup_handler" needed to avoid circular dependency
        // between "MapDHandler" & function "run_warmup_queries"
mapd::shared_ptr<MapDHandler> g_mapd_handler = 0;

void shutdown_handler() {
  if (g_mapd_handler) {
    g_mapd_handler->shutdown();
  }
}

void mapd_signal_handler(int signal_number) {
  LOG(INFO) << "Interrupt signal (" << signal_number << ") received.\n";
  shutdown_handler();
  // shut down logging force a flush
  google::ShutdownGoogleLogging();
  // terminate program
  if (signal_number == SIGTERM) {
    std::exit(EXIT_SUCCESS);
  } else {
    std::exit(signal_number);
  }
}

void register_signal_handler() {
  // it appears we send both a signal SIGINT(2) and SIGTERM(15) each time we
  // exit the startomnisci script.
  // Only catching the SIGTERM(15) to avoid double shut down request
  // register SIGTERM and signal handler
  std::signal(SIGTERM, mapd_signal_handler);
  std::signal(SIGSEGV, mapd_signal_handler);
  std::signal(SIGABRT, mapd_signal_handler);
  // Thrift secure socket can cause problems with SIGPIPE
  std::signal(SIGPIPE, SIG_IGN);
}

void start_server(TThreadedServer& server) {
  try {
    server.serve();
  } catch (std::exception& e) {
    LOG(ERROR) << "Exception: " << e.what() << std::endl;
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

MapDProgramOptions::MapDProgramOptions() {
  fillOptions(*this);
  fillAdvancedOptions(*this);
};

void MapDProgramOptions::fillOptions(po::options_description& desc) {
  desc.add_options()("help,h", "Print help messages");
  desc.add_options()("config", po::value<std::string>(&config_file), "Path to mapd.conf");
  desc.add_options()(
      "data",
      po::value<std::string>(&base_path)->required()->default_value("data"),
      "Directory path to OmniSci catalogs");
  desc.add_options()(
      "cpu-only",
      po::value<bool>(&cpu_only)->default_value(cpu_only)->implicit_value(true),
      "Run on CPU only");
  desc.add_options()(
      "read-only",
      po::value<bool>(&read_only)->default_value(read_only)->implicit_value(true),
      "Enable read-only mode");
  desc.add_options()("port,p",
                     po::value<int>(&mapd_parameters.omnisci_server_port)
                         ->default_value(mapd_parameters.omnisci_server_port),
                     "Port number");
  desc.add_options()("http-port",
                     po::value<int>(&http_port)->default_value(http_port),
                     "HTTP port number");
  desc.add_options()("calcite-port",
                     po::value<int>(&mapd_parameters.calcite_port)
                         ->default_value(mapd_parameters.calcite_port),
                     "Calcite port number");
  desc.add_options()("num-gpus",
                     po::value<int>(&num_gpus)->default_value(num_gpus),
                     "Number of gpus to use");
  desc.add_options()("start-gpu",
                     po::value<int>(&start_gpu)->default_value(start_gpu),
                     "First gpu to use");
  desc.add_options()("version,v", "Print Release Version Number");
  desc.add_options()(
      "flush-log",
      po::value<bool>(&flush_log)->default_value(flush_log)->implicit_value(true),
      "Immediately flush logs to disk. Set to false if this is a performance "
      "bottleneck.");
  desc.add_options()("verbose",
                     po::value<bool>(&verbose_logging)
                         ->default_value(verbose_logging)
                         ->implicit_value(true),
                     "Write all log messages to server logs.");
  desc.add_options()("cpu-buffer-mem-bytes",
                     po::value<size_t>(&mapd_parameters.cpu_buffer_mem_bytes)
                         ->default_value(mapd_parameters.cpu_buffer_mem_bytes),
                     "Size of memory reserved for CPU buffers [bytes]");
  desc.add_options()("gpu-buffer-mem-bytes",
                     po::value<size_t>(&mapd_parameters.gpu_buffer_mem_bytes)
                         ->default_value(mapd_parameters.gpu_buffer_mem_bytes),
                     "Size of memory reserved for GPU buffers [bytes] (per GPU)");
  desc.add_options()("calcite-max-mem",
                     po::value<size_t>(&mapd_parameters.calcite_max_mem)
                         ->default_value(mapd_parameters.calcite_max_mem),
                     "Max memory available to calcite JVM");
  desc.add_options()(
      "res-gpu-mem",
      po::value<size_t>(&reserved_gpu_mem)->default_value(reserved_gpu_mem),
      "Reserved memory for GPU, not use OmniSci allocator");
  desc.add_options()("gpu-input-mem-limit",
                     po::value<double>(&mapd_parameters.gpu_input_mem_limit)
                         ->default_value(mapd_parameters.gpu_input_mem_limit),
                     "Force query to CPU when input data memory usage exceeds this "
                     "percentage of available GPU memory");
  desc.add_options()("cuda-block-size",
                     po::value<size_t>(&mapd_parameters.cuda_block_size)
                         ->default_value(mapd_parameters.cuda_block_size),
                     "Size of block to use on GPU");
  desc.add_options()("cuda-grid-size",
                     po::value<size_t>(&mapd_parameters.cuda_grid_size)
                         ->default_value(mapd_parameters.cuda_grid_size),
                     "Size of grid to use on GPU");
  desc.add_options()(
      "num-reader-threads",
      po::value<size_t>(&num_reader_threads)->default_value(num_reader_threads),
      "Number of reader threads to use");
  desc.add_options()(
      "idle-session-duration",
      po::value<int>(&idle_session_duration)->default_value(idle_session_duration),
      "Maximum duration of idle session.");
  desc.add_options()(
      "max-session-duration",
      po::value<int>(&max_session_duration)->default_value(max_session_duration),
      "Maximum duration of active session.");
  desc.add_options()(
      "hll-precision-bits",
      po::value<int>(&g_hll_precision_bits)
          ->default_value(g_hll_precision_bits)
          ->implicit_value(g_hll_precision_bits),
      "Number of bits used from the hash value used to specify the bucket number.");
  desc.add_options()("enable-watchdog",
                     po::value<bool>(&enable_watchdog)
                         ->default_value(enable_watchdog)
                         ->implicit_value(true),
                     "Enable watchdog");
  desc.add_options()("enable-dynamic-watchdog",
                     po::value<bool>(&enable_dynamic_watchdog)
                         ->default_value(enable_dynamic_watchdog)
                         ->implicit_value(true),
                     "Enable dynamic watchdog");
  desc.add_options()("dynamic-watchdog-time-limit",
                     po::value<unsigned>(&dynamic_watchdog_time_limit)
                         ->default_value(dynamic_watchdog_time_limit)
                         ->implicit_value(10000),
                     "Dynamic watchdog time limit, in milliseconds");
  desc.add_options()("enable-debug-timer",
                     po::value<bool>(&g_enable_debug_timer)
                         ->default_value(g_enable_debug_timer)
                         ->implicit_value(true),
                     "Enable debug timer logging");
  desc.add_options()("null-div-by-zero",
                     po::value<bool>(&g_null_div_by_zero)
                         ->default_value(g_null_div_by_zero)
                         ->implicit_value(true),
                     "Return null on division by zero instead of throwing an exception");
  desc.add_options()("bigint-count",
                     po::value<bool>(&g_bigint_count)
                         ->default_value(g_bigint_count)
                         ->implicit_value(false),
                     "Use 64-bit count");
  desc.add_options()("allow-cpu-retry",
                     po::value<bool>(&g_allow_cpu_retry)
                         ->default_value(g_allow_cpu_retry)
                         ->implicit_value(true),
                     "Allow the queries which failed on GPU to retry on CPU, even "
                     "when watchdog is enabled");
  desc.add_options()("enable-access-priv-check",
                     po::value<bool>(&enable_access_priv_check)
                         ->default_value(enable_access_priv_check)
                         ->implicit_value(true),
                     "Check user access privileges to database objects");
  desc.add_options()("from-table-reordering",
                     po::value<bool>(&g_from_table_reordering)
                         ->default_value(g_from_table_reordering)
                         ->implicit_value(true),
                     "Enable automatic table reordering in FROM clause");
  desc.add_options()("allow-loop-joins",
                     po::value<bool>(&allow_loop_joins)
                         ->default_value(allow_loop_joins)
                         ->implicit_value(true),
                     "Enable loop joins");
  desc.add_options()("trivial-loop-join-threshold",
                     po::value<unsigned>(&g_trivial_loop_join_threshold)
                         ->default_value(g_trivial_loop_join_threshold)
                         ->implicit_value(1000),
                     "The maximum number of rows in the inner table of a loop join "
                     "considered to be trivially small");
  desc.add_options()("inner-join-fragment-skipping",
                     po::value<bool>(&g_inner_join_fragment_skipping)
                         ->default_value(g_inner_join_fragment_skipping)
                         ->implicit_value(true),
                     "Enable/disable inner join fragment skipping.");
  desc.add_options()("enable-filter-push-down",
                     po::value<bool>(&g_enable_filter_push_down)
                         ->default_value(g_enable_filter_push_down)
                         ->implicit_value(true),
                     "Enable filter push down through joins");
  desc.add_options()("filter-push-down-low-frac",
                     po::value<float>(&g_filter_push_down_low_frac)
                         ->default_value(g_filter_push_down_low_frac)
                         ->implicit_value(g_filter_push_down_low_frac),
                     "Lower threshold for selectivity of filters that are pushed down.");
  desc.add_options()("filter-push-down-high-frac",
                     po::value<float>(&g_filter_push_down_high_frac)
                         ->default_value(g_filter_push_down_high_frac)
                         ->implicit_value(g_filter_push_down_high_frac),
                     "Higher threshold for selectivity of filters that are pushed down.");
  desc.add_options()("filter-push-down-passing-row-ubound",
                     po::value<size_t>(&g_filter_push_down_passing_row_ubound)
                         ->default_value(g_filter_push_down_passing_row_ubound)
                         ->implicit_value(g_filter_push_down_passing_row_ubound),
                     "Upperbound on the number of rows that should pass the filter "
                     "if the selectivity is less than "
                     "the high fraction threshold.");
  desc.add_options()("enable-overlaps-hashjoin",
                     po::value<bool>(&g_enable_overlaps_hashjoin)
                         ->default_value(g_enable_overlaps_hashjoin)
                         ->implicit_value(true),
                     "Enable the overlaps hash join framework allowing for range "
                     "join (e.g. spatial overlaps) computation using a hash table");
  desc.add_options()("overlaps-bucket-threshold",
                     po::value<double>(&g_overlaps_hashjoin_bucket_threshold)
                         ->default_value(g_overlaps_hashjoin_bucket_threshold),
                     "The minimum size of a bucket corresponding to a given inner table "
                     "range for the overlaps hash join");
  desc.add_options()("db-query-list",
                     po::value<std::string>(&db_query_file),
                     "Path to file containing OmniSci queries");
}

void MapDProgramOptions::fillAdvancedOptions(po::options_description& desc_adv) {
  desc_adv.add_options()("dev-options", "Print internal developer options");
  desc_adv.add_options()("ssl-cert",
                         po::value<std::string>(&mapd_parameters.ssl_cert_file)
                             ->default_value(std::string("")),
                         "SSL Validated public certficate");
  desc_adv.add_options()("ssl-private-key",
                         po::value<std::string>(&mapd_parameters.ssl_key_file)
                             ->default_value(std::string("")),
                         "SSL private key file");
  // Note ssl_trust_store is passed through to Calcite via mapd_parameters
  // todo(jack): add ensure ssl-trust-store exists if cert and private key in use
  desc_adv.add_options()("ssl-trust-store",
                         po::value<std::string>(&mapd_parameters.ssl_trust_store)
                             ->default_value(std::string("")),
                         "SSL Validated public cert as a java trust store");
  desc_adv.add_options()("ssl-trust-password",
                         po::value<std::string>(&mapd_parameters.ssl_trust_password)
                             ->default_value(std::string("")),
                         "SSL java trust store password");
  desc_adv.add_options()(
      "jit-debug-ir",
      po::value<bool>(&jit_debug)->default_value(jit_debug)->implicit_value(true),
      "Enable debugger support for the JIT. Note that this flag is incompatible with "
      "the "
      "`ENABLE_JIT_DEBUG` build flag. The generated code can be found at "
      "`/tmp/mapdquery`");
  desc_adv.add_options()(
      "disable-multifrag",
      po::value<bool>(&allow_multifrag)
          ->default_value(allow_multifrag)
          ->implicit_value(false),
      "Disable execution over multiple fragments in a single round-trip to GPU");
  desc_adv.add_options()("disable-legacy-syntax",
                         po::value<bool>(&enable_legacy_syntax)
                             ->default_value(enable_legacy_syntax)
                             ->implicit_value(false),
                         "Enable legacy syntax");
  desc_adv.add_options()("enable-columnar-output",
                         po::value<bool>(&g_enable_columnar_output)
                             ->default_value(g_enable_columnar_output)
                             ->implicit_value(true));
  desc_adv.add_options()("disable-shared-mem-group-by",
                         po::value<bool>(&g_enable_smem_group_by)
                             ->default_value(g_enable_smem_group_by)
                             ->implicit_value(false),
                         "Enable/disable using GPU shared memory for GROUP BY.");
  desc_adv.add_options()("strip-join-covered-quals",
                         po::value<bool>(&g_strip_join_covered_quals)
                             ->default_value(g_strip_join_covered_quals)
                             ->implicit_value(true),
                         "Remove quals from the filtered count if they are covered by a "
                         "join condition (currently only ST_Contains)");
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

}  // namespace

bool MapDProgramOptions::parse_command_line(int argc, char** argv, int& return_code) {
  return_code = 0;

  po::positional_options_description positionalOptions;
  positionalOptions.add("data", 1);

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(*this)
                  .positional(positionalOptions)
                  .run(),
              vm);
    po::notify(vm);

    if (vm.count("config")) {
      std::ifstream settings_file(config_file);

      auto sanitized_settings = sanitize_config_file(settings_file);

      po::store(po::parse_config_file(sanitized_settings, *this, false), vm);
      po::notify(vm);
      settings_file.close();
    }

    if (vm.count("cluster") || vm.count("string-servers")) {
      CHECK_NE(!!vm.count("cluster"), !!vm.count("string-servers"));
      boost::algorithm::trim_if(cluster_file, boost::is_any_of("\"'"));
      const auto all_nodes = LeafHostInfo::parseClusterConfig(cluster_file);
      db_leaves = only_db_leaves(all_nodes);
      g_leaf_count = db_leaves.size();
      if (vm.count("cluster")) {
        mapd_parameters.aggregator = true;
      } else {
        db_leaves.clear();
      }
      string_leaves = only_string_leaves(all_nodes);
      g_cluster = true;
    }

    if (vm.count("help")) {
      std::cout
          << "Usage: omnisci_server <catalog path> [<database name>] [-p <port number>] "
             "[--http-port <http port number>] [--flush-log] [--version|-v]"
          << std::endl
          << std::endl;
      std::cout << *this << std::endl;
      return_code = 0;
      return false;
    }
    if (vm.count("dev-options")) {
      std::cout
          << "Usage: omnisci_server <catalog path> [<database name>] [-p <port number>] "
             "[--http-port <http port number>] [--flush-log] [--version|-v]"
          << std::endl
          << std::endl;
      std::cout << *this << std::endl;
      return_code = 0;
      return false;
    }
    if (vm.count("version")) {
      std::cout << "OmniSci Version: " << MAPD_RELEASE << std::endl;
      return 0;
    }

    g_enable_watchdog = enable_watchdog;
    g_enable_dynamic_watchdog = enable_dynamic_watchdog;
    g_dynamic_watchdog_time_limit = dynamic_watchdog_time_limit;
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return_code = 1;
    return false;
  }

  if (g_hll_precision_bits < 1 || g_hll_precision_bits > 16) {
    std::cerr << "hll-precision-bits must be between 1 and 16." << std::endl;
    return_code = 1;
    return false;
  }

  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  const auto data_path = boost::filesystem::path(base_path) / "mapd_data";
  if (!boost::filesystem::exists(data_path)) {
    std::cerr << "OmniSci data directory does not exist at '" << base_path
              << "'. Run initdb " << base_path << std::endl;
    return_code = 1;
    return false;
  }

  const auto lock_file = boost::filesystem::path(base_path) / "omnisci_server_pid.lck";
  auto pid = std::to_string(getpid());
  int pid_fd = open(lock_file.c_str(), O_RDWR | O_CREAT, 0644);
  if (pid_fd == -1) {
    auto err = std::string("Failed to open PID file ") + std::string(lock_file.c_str()) +
               std::string(". ") + strerror(errno) + ".";
    std::cerr << err << std::endl;
    return_code = 1;
    return false;
  }
  if (lockf(pid_fd, F_TLOCK, 0) == -1) {
    auto err = std::string("Another OmniSci Server is using data directory ") +
               base_path + std::string(".");
    std::cerr << err << std::endl;
    close(pid_fd);
    return_code = 1;
    return false;
  }
  if (ftruncate(pid_fd, 0) == -1) {
    auto err = std::string("Failed to truncate PID file ") +
               std::string(lock_file.c_str()) + std::string(". ") + strerror(errno) +
               std::string(".");
    std::cerr << err << std::endl;
    close(pid_fd);
    return_code = 1;
    return false;
  }
  if (write(pid_fd, pid.c_str(), pid.length()) == -1) {
    auto err = std::string("Failed to write PID file ") + std::string(lock_file.c_str()) +
               ". " + strerror(errno) + ".";
    std::cerr << err << std::endl;
    close(pid_fd);
    return_code = 1;
    return false;
  }

  const auto log_path = boost::filesystem::path(base_path) / "mapd_log";
  (void)boost::filesystem::create_directory(log_path);
  FLAGS_log_dir = log_path.c_str();
  if (flush_log) {
    FLAGS_logbuflevel = -1;
  }
  if (verbose_logging) {
    FLAGS_v = 1;
  }
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);

  boost::algorithm::trim_if(db_query_file, boost::is_any_of("\"'"));
  if (db_query_file.length() > 0 && !boost::filesystem::exists(db_query_file)) {
    LOG(ERROR) << "File containing DB queries " << db_query_file << " does not exist.";
    return_code = 1;
    return false;
  }
  const auto system_db_file =
      boost::filesystem::path(base_path) / "mapd_catalogs" / "mapd";
  if (!boost::filesystem::exists(system_db_file)) {
    LOG(ERROR) << "OmniSci system catalogs does not exist at " << system_db_file;
    return_code = 1;
    return false;
  }
  const auto db_file =
      boost::filesystem::path(base_path) / "mapd_catalogs" / MAPD_SYSTEM_DB;
  if (!boost::filesystem::exists(db_file)) {
    LOG(ERROR) << "OmniSci database " << MAPD_SYSTEM_DB << " does not exist.";
    return_code = 1;
    return false;
  }

  // add all parameters to be displayed on startup
  LOG(INFO) << "OmniSci started with data directory at '" << base_path << "'";
  if (vm.count("cluster")) {
    LOG(INFO) << "Cluster file specified running as aggregator with config at '"
              << cluster_file << "'";
  }
  if (vm.count("string-servers")) {
    LOG(INFO) << "String servers file specified running as dbleaf with config at '"
              << cluster_file << "'";
  }
  LOG(INFO) << " Watchdog is set to " << enable_watchdog;
  LOG(INFO) << " Dynamic Watchdog is set to " << enable_dynamic_watchdog;
  if (enable_dynamic_watchdog) {
    LOG(INFO) << " Dynamic Watchdog timeout is set to " << dynamic_watchdog_time_limit;
  }

  LOG(INFO) << " Enable access priv check  is set to " << enable_access_priv_check;

  LOG(INFO) << " Debug Timer is set to " << g_enable_debug_timer;

  LOG(INFO) << " Maximum Idle session duration " << idle_session_duration;

  LOG(INFO) << " Maximum active session duration " << max_session_duration;

  if (!g_from_table_reordering) {
    LOG(INFO) << " From clause table reordering is disabled";
  }

  if (g_enable_filter_push_down) {
    LOG(INFO) << " Filter push down for JOIN is enabled";
  }

  boost::algorithm::trim_if(mapd_parameters.ha_brokers, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ha_group_id, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ha_shared_data, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ha_unique_server_id, boost::is_any_of("\"'"));

  if (!mapd_parameters.ha_group_id.empty()) {
    LOG(INFO) << " HA group id " << mapd_parameters.ha_group_id;
    if (mapd_parameters.ha_unique_server_id.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-unique-server-id must be set ";
      return_code = 5;
      return false;
    } else {
      LOG(INFO) << " HA unique server id " << mapd_parameters.ha_unique_server_id;
    }
    if (mapd_parameters.ha_brokers.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-brokers must be set ";
      return_code = 6;
      return false;
    } else {
      LOG(INFO) << " HA brokers " << mapd_parameters.ha_brokers;
    }
    if (mapd_parameters.ha_shared_data.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-shared-data must be set ";
      return_code = 7;
      return false;
    } else {
      LOG(INFO) << " HA shared data is " << mapd_parameters.ha_shared_data;
    }
  }
  LOG(INFO) << " cuda block size " << mapd_parameters.cuda_block_size;
  LOG(INFO) << " cuda grid size  " << mapd_parameters.cuda_grid_size;
  LOG(INFO) << " calcite JVM max memory  " << mapd_parameters.calcite_max_mem;
  LOG(INFO) << " OmniSci Server Port  " << mapd_parameters.omnisci_server_port;
  LOG(INFO) << " OmniSci Calcite Port  " << mapd_parameters.calcite_port;

  boost::algorithm::trim_if(authMetadata.distinguishedName, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.uri, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapQueryUrl, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapRoleRegex, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapSuperUserRole, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.restToken, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.restUrl, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ssl_cert_file, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ssl_key_file, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(mapd_parameters.ssl_trust_store, boost::is_any_of("\"'"));

  return true;
}

int main(int argc, char** argv) {
  MapDProgramOptions desc_all;
  int return_code = 0;

  if (!desc_all.parse_command_line(argc, argv, return_code)) {
    return return_code;
  };
  // rudimetary signal handling to try to guarantee the logging gets flushed to
  // files on shutdown
  register_signal_handler();
  google::InstallFailureFunction(&shutdown_handler);

  // start background thread to clean up _DELETE_ME files
  std::atomic<bool> running{true};
  const unsigned int wait_interval =
      300;  // wait time in secs after looking for deleted file before looking again
  std::thread file_delete_thread(
      file_delete, std::ref(running), wait_interval, desc_all.base_path + "/mapd_data");

  g_mapd_handler = mapd::make_shared<MapDHandler>(desc_all.db_leaves,
                                                  desc_all.string_leaves,
                                                  desc_all.base_path,
                                                  desc_all.cpu_only,
                                                  desc_all.allow_multifrag,
                                                  desc_all.jit_debug,
                                                  desc_all.read_only,
                                                  desc_all.allow_loop_joins,
                                                  desc_all.enable_rendering,
                                                  desc_all.render_mem_bytes,
                                                  desc_all.num_gpus,
                                                  desc_all.start_gpu,
                                                  desc_all.reserved_gpu_mem,
                                                  desc_all.num_reader_threads,
                                                  desc_all.authMetadata,
                                                  desc_all.mapd_parameters,
                                                  desc_all.enable_legacy_syntax,
                                                  desc_all.enable_access_priv_check,
                                                  desc_all.idle_session_duration,
                                                  desc_all.max_session_duration);

  mapd::shared_ptr<TServerSocket> serverSocket;
  if (!desc_all.mapd_parameters.ssl_cert_file.empty() &&
      !desc_all.mapd_parameters.ssl_key_file.empty()) {
    mapd::shared_ptr<TSSLSocketFactory> sslSocketFactory;
    sslSocketFactory =
        mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
    sslSocketFactory->loadCertificate(desc_all.mapd_parameters.ssl_cert_file.c_str());
    sslSocketFactory->loadPrivateKey(desc_all.mapd_parameters.ssl_key_file.c_str());
    sslSocketFactory->authenticate(false);
    sslSocketFactory->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    serverSocket = mapd::shared_ptr<TServerSocket>(new TSSLServerSocket(
        desc_all.mapd_parameters.omnisci_server_port, sslSocketFactory));
    LOG(INFO) << " OmniSci server using encrypted connection. Cert file ["
              << desc_all.mapd_parameters.ssl_cert_file << "], key file ["
              << desc_all.mapd_parameters.ssl_key_file << "]";
  } else {
    LOG(INFO) << " OmniSci server using unencrypted connection";
    serverSocket = mapd::shared_ptr<TServerSocket>(
        new TServerSocket(desc_all.mapd_parameters.omnisci_server_port));
  }

  if (desc_all.mapd_parameters.ha_group_id.empty()) {
    mapd::shared_ptr<TProcessor> processor(new MapDProcessor(g_mapd_handler));

    mapd::shared_ptr<TTransportFactory> bufTransportFactory(
        new TBufferedTransportFactory());
    mapd::shared_ptr<TProtocolFactory> bufProtocolFactory(new TBinaryProtocolFactory());

    mapd::shared_ptr<TServerTransport> bufServerTransport(serverSocket);
    TThreadedServer bufServer(
        processor, bufServerTransport, bufTransportFactory, bufProtocolFactory);

    mapd::shared_ptr<TServerTransport> httpServerTransport(
        new TServerSocket(desc_all.http_port));
    mapd::shared_ptr<TTransportFactory> httpTransportFactory(
        new THttpServerTransportFactory());
    mapd::shared_ptr<TProtocolFactory> httpProtocolFactory(new TJSONProtocolFactory());
    TThreadedServer httpServer(
        processor, httpServerTransport, httpTransportFactory, httpProtocolFactory);

    std::thread bufThread(start_server, std::ref(bufServer));
    std::thread httpThread(start_server, std::ref(httpServer));

    // run warm up queries if any exists
    run_warmup_queries(g_mapd_handler, desc_all.base_path, desc_all.db_query_file);

    bufThread.join();
    httpThread.join();
  } else {  // running ha server
    LOG(FATAL) << "No High Availability module available, please contact OmniSci support";
  }

  running = false;
  file_delete_thread.join();

  return 0;
};
