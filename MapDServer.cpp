/*
 * Copyright 2017 MapD Technologies, Inc.
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
#include <thrift/transport/TServerSocket.h>

#include "MapDRelease.h"

#include "Shared/MapDParameters.h"
#include "Shared/scope.h"

#include <glog/logging.h>
#include <signal.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <sstream>
#include <thread>
#include <vector>

using namespace ::apache::thrift;
using namespace ::apache::thrift::concurrency;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::server;
using namespace ::apache::thrift::transport;

using boost::make_shared;
using boost::shared_ptr;

extern bool g_aggregator;
extern size_t g_leaf_count;

AggregatedColRange column_ranges_from_thrift(const std::vector<TColumnRange>& thrift_column_ranges) {
  AggregatedColRange column_ranges;
  for (const auto& thrift_column_range : thrift_column_ranges) {
    PhysicalInput phys_input{thrift_column_range.col_id, thrift_column_range.table_id};
    switch (thrift_column_range.type) {
      case TExpressionRangeType::INTEGER:
        column_ranges.setColRange(phys_input,
                                  ExpressionRange::makeIntRange(thrift_column_range.int_min,
                                                                thrift_column_range.int_max,
                                                                thrift_column_range.bucket,
                                                                thrift_column_range.has_nulls));
        break;
      case TExpressionRangeType::FLOAT:
        column_ranges.setColRange(
            phys_input,
            ExpressionRange::makeFloatRange(
                thrift_column_range.fp_min, thrift_column_range.fp_max, thrift_column_range.has_nulls));
        break;
      case TExpressionRangeType::DOUBLE:
        column_ranges.setColRange(
            phys_input,
            ExpressionRange::makeDoubleRange(
                thrift_column_range.fp_min, thrift_column_range.fp_max, thrift_column_range.has_nulls));
        break;
      case TExpressionRangeType::INVALID:
        column_ranges.setColRange(phys_input, ExpressionRange::makeInvalidRange());
        break;
      default:
        CHECK(false);
    }
  }
  return column_ranges;
}

StringDictionaryGenerations string_dictionary_generations_from_thrift(
    const std::vector<TDictionaryGeneration>& thrift_string_dictionary_generations) {
  StringDictionaryGenerations string_dictionary_generations;
  for (const auto& thrift_string_dictionary_generation : thrift_string_dictionary_generations) {
    string_dictionary_generations.setGeneration(thrift_string_dictionary_generation.dict_id,
                                                thrift_string_dictionary_generation.entry_count);
  }
  return string_dictionary_generations;
}

TableGenerations table_generations_from_thrift(const std::vector<TTableGeneration>& thrift_table_generations) {
  TableGenerations table_generations;
  for (const auto& thrift_table_generation : thrift_table_generations) {
    table_generations.setGeneration(thrift_table_generation.table_id,
                                    TableGeneration{static_cast<size_t>(thrift_table_generation.tuple_count),
                                                    static_cast<size_t>(thrift_table_generation.start_rowid)});
  }
  return table_generations;
}

std::vector<LeafHostInfo> only_db_leaves(const std::vector<LeafHostInfo>& all_leaves) {
  std::vector<LeafHostInfo> data_leaves;
  std::copy_if(all_leaves.begin(), all_leaves.end(), std::back_inserter(data_leaves), [](const LeafHostInfo& leaf) {
    return leaf.getRole() == NodeRole::DbLeaf;
  });
  return data_leaves;
}

std::vector<LeafHostInfo> only_string_leaves(const std::vector<LeafHostInfo>& all_leaves) {
  std::vector<LeafHostInfo> string_leaves;
  std::copy_if(all_leaves.begin(), all_leaves.end(), std::back_inserter(string_leaves), [](const LeafHostInfo& leaf) {
    return leaf.getRole() == NodeRole::String;
  });
  return string_leaves;
}

void mapd_signal_handler(int signal_number) {
  LOG(INFO) << "Interrupt signal (" << signal_number << ") received.\n";
  // shut down logging force a flush
  google::ShutdownGoogleLogging();

  // terminate program
  exit(0);
}

void register_signal_handler() {
  // it appears we send both a signal SIGINT(2) and SIGTERM(15) each time we
  // exit the startmapd script.
  // Only catching the SIGTERM(15) to avoid double shut down request
  // register SIGTERM and signal handler
  signal(SIGTERM, mapd_signal_handler);
}

void start_server(TThreadedServer& server) {
  try {
    server.serve();
  } catch (std::exception& e) {
    LOG(ERROR) << "Exception: " << e.what() << std::endl;
  }
}

shared_ptr<MapDHandler> warmup_handler = 0;  // global "warmup_handler" needed to avoid circular dependency
                                             // between "MapDHandler" & function "run_warmup_queries"

void releaseWarmupSession(TSessionId& sessionId, std::ifstream& query_file) {
  query_file.close();
  if (sessionId != warmup_handler->getInvalidSessionId()) {
    warmup_handler->disconnect(sessionId);
  }
}

void run_warmup_queries(boost::shared_ptr<MapDHandler> handler, std::string base_path, std::string query_file_path) {
  // run warmup queries to load cache if requested
  if (query_file_path.empty()) {
    return;
  }
  LOG(INFO) << "Running DB warmup with queries from " << query_file_path;
  try {
    warmup_handler = handler;
    std::string db_info;
    std::string user_keyword, user_name, db_name;
    std::ifstream query_file;
    Catalog_Namespace::UserMetadata user;
    Catalog_Namespace::DBMetadata db;
    TSessionId sessionId = warmup_handler->getInvalidSessionId();

    ScopeGuard session_guard = [&] { releaseWarmupSession(sessionId, query_file); };
    query_file.open(query_file_path);
    while (std::getline(query_file, db_info)) {
      if (db_info.length() == 0) {
        continue;
      }
      std::istringstream iss(db_info);
      iss >> user_keyword >> user_name >> db_name;
      if (user_keyword.compare(0, 4, "USER") == 0) {
        // connect to DB for given user_name/db_name with super_user_rights (without password), & start session
        warmup_handler->super_user_rights_ = true;
        warmup_handler->connect(sessionId, user_name, "", db_name);
        warmup_handler->super_user_rights_ = false;

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
          warmup_handler->sql_execute(ret, sessionId, single_query, true, "", -1, -1);
          single_query.clear();
        }

        // stop session and disconnect from the DB
        warmup_handler->disconnect(sessionId);
        sessionId = warmup_handler->getInvalidSessionId();
      } else {
        LOG(WARNING) << "\nSyntax error in the file: " << query_file_path.c_str()
                     << " Missing expected keyword USER. Following line will be ignored: " << db_info.c_str()
                     << std::endl;
      }
      db_info.clear();
    }
  } catch (...) {
    LOG(WARNING) << "Exception while executing warmup queries. "
                 << "Warmup may not be fully completed. Will proceed nevertheless." << std::endl;
  }
}

int main(int argc, char** argv) {
  int http_port = 9090;
  size_t reserved_gpu_mem = 1 << 27;
  std::string base_path;
  std::string device("gpu");
  std::string config_file("mapd.conf");
  std::string cluster_file("cluster.conf");
  bool flush_log = true;
  bool jit_debug = false;
  bool allow_multifrag = true;
  bool read_only = false;
  bool allow_loop_joins = false;
  bool enable_legacy_syntax = true;
  LdapMetadata ldapMetadata;
  MapDParameters mapd_parameters;
  bool enable_rendering = false;
  bool enable_watchdog = true;
  bool enable_dynamic_watchdog = false;
  unsigned dynamic_watchdog_time_limit = 10000;

  size_t cpu_buffer_mem_bytes = 0;  // 0 will cause DataMgr to auto set this based on available memory
  size_t render_mem_bytes = 500000000;
  int num_gpus = -1;  // Can be used to override number of gpus detected on system - -1 means do not override
  int start_gpu = 0;
  size_t num_reader_threads = 0;   // number of threads used when loading data
  std::string db_convert_dir("");  // path to mapd DB to convert from; if path is empty, no conversion is requested
  std::string db_query_file("");   // path to file containing warmup queries list
  bool enable_access_priv_check = false;  // enable DB objects access privileges checking

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages");
  desc.add_options()("config", po::value<std::string>(&config_file), "Path to mapd.conf");
  desc.add_options()(
      "data", po::value<std::string>(&base_path)->required()->default_value("data"), "Directory path to MapD catalogs");
  desc.add_options()("cpu", "Run on CPU only");
  desc.add_options()("gpu", "Run on GPUs (Default)");
  desc.add_options()("read-only",
                     po::value<bool>(&read_only)->default_value(read_only)->implicit_value(true),
                     "Enable read-only mode");
  desc.add_options()("port,p",
                     po::value<int>(&mapd_parameters.mapd_server_port)->default_value(mapd_parameters.mapd_server_port),
                     "Port number");
  desc.add_options()("http-port", po::value<int>(&http_port)->default_value(http_port), "HTTP port number");
  desc.add_options()("calcite-port",
                     po::value<int>(&mapd_parameters.calcite_port)->default_value(mapd_parameters.calcite_port),
                     "Calcite port number");
  desc.add_options()("flush-log",
                     po::value<bool>(&flush_log)->default_value(flush_log)->implicit_value(true),
                     "Immediately flush logs to disk. Set to false if this is a performance bottleneck.");
  desc.add_options()("cpu-buffer-mem-bytes",
                     po::value<size_t>(&cpu_buffer_mem_bytes)->default_value(cpu_buffer_mem_bytes),
                     "Size of memory reserved for CPU buffers [bytes]");
  desc.add_options()("num-gpus", po::value<int>(&num_gpus)->default_value(num_gpus), "Number of gpus to use");
  desc.add_options()("start-gpu", po::value<int>(&start_gpu)->default_value(start_gpu), "First gpu to use");
  desc.add_options()("version,v", "Print Release Version Number");
  desc.add_options()("null-div-by-zero",
                     po::value<bool>(&g_null_div_by_zero)->default_value(g_null_div_by_zero)->implicit_value(true),
                     "Return null on division by zero instead of throwing an exception");
  desc.add_options()("left-deep-join-optimization",
                     po::value<bool>(&g_left_deep_join_optimization)
                         ->default_value(g_left_deep_join_optimization)
                         ->implicit_value(true),
                     "Enable left-deep join optimization");
  desc.add_options()(
      "from-table-reordering",
      po::value<bool>(&g_from_table_reordering)->default_value(g_from_table_reordering)->implicit_value(true),
      "Enable automatic table reordering in FROM clause");
  desc.add_options()(
      "help-advanced",
      "Print advanced and experimental options. These options should not normally be used in production.");

  po::options_description desc_adv("Advanced options");
  desc_adv.add_options()("jit-debug",
                         po::value<bool>(&jit_debug)->default_value(jit_debug)->implicit_value(true),
                         "Enable debugger support for the JIT. The generated code can be found at /tmp/mapdquery");
  desc_adv.add_options()("disable-multifrag",
                         po::value<bool>(&allow_multifrag)->default_value(allow_multifrag)->implicit_value(false),
                         "Disable execution over multiple fragments in a single round-trip to GPU");
  desc_adv.add_options()("allow-loop-joins",
                         po::value<bool>(&allow_loop_joins)->default_value(allow_loop_joins)->implicit_value(true),
                         "Enable loop joins");
  desc_adv.add_options()("fast-strcmp",
                         po::value<bool>(&g_fast_strcmp)->default_value(g_fast_strcmp)->implicit_value(false),
                         "Disable fast string comparison");
  desc_adv.add_options()("res-gpu-mem",
                         po::value<size_t>(&reserved_gpu_mem)->default_value(reserved_gpu_mem),
                         "Reserved memory for GPU, not use mapd allocator");
  desc_adv.add_options()(
      "disable-legacy-syntax",
      po::value<bool>(&enable_legacy_syntax)->default_value(enable_legacy_syntax)->implicit_value(false),
      "Enable legacy syntax");
  desc_adv.add_options()("num-reader-threads",
                         po::value<size_t>(&num_reader_threads)->default_value(num_reader_threads),
                         "Number of reader threads to use");
  desc_adv.add_options()("enable-watchdog",
                         po::value<bool>(&enable_watchdog)->default_value(enable_watchdog)->implicit_value(true),
                         "Enable watchdog");
  desc_adv.add_options()(
      "enable-dynamic-watchdog",
      po::value<bool>(&enable_dynamic_watchdog)->default_value(enable_dynamic_watchdog)->implicit_value(true),
      "Enable dynamic watchdog");
  desc_adv.add_options()("dynamic-watchdog-time-limit",
                         po::value<unsigned>(&dynamic_watchdog_time_limit)
                             ->default_value(dynamic_watchdog_time_limit)
                             ->implicit_value(10000),
                         "Dynamic watchdog time limit, in milliseconds");
  desc_adv.add_options()(
      "trivial-loop-join-threshold",
      po::value<unsigned>(&g_trivial_loop_join_threshold)
          ->default_value(g_trivial_loop_join_threshold)
          ->implicit_value(1000),
      "The maximum number of rows in the inner table of a loop join considered to be trivially small");
  desc_adv.add_options()(
      "cuda-block-size",
      po::value<size_t>(&mapd_parameters.cuda_block_size)->default_value(mapd_parameters.cuda_block_size),
      "Size of block to use on GPU");
  desc_adv.add_options()(
      "cuda-grid-size",
      po::value<size_t>(&mapd_parameters.cuda_grid_size)->default_value(mapd_parameters.cuda_grid_size),
      "Size of grid to use on GPU");
  desc_adv.add_options()(
      "calcite-max-mem",
      po::value<size_t>(&mapd_parameters.calcite_max_mem)->default_value(mapd_parameters.calcite_max_mem),
      "Max memory available to calcite JVM");
  desc_adv.add_options()(
      "db-convert", po::value<std::string>(&db_convert_dir), "Directory path to mapd DB to convert from");

  desc_adv.add_options()("use-result-set",
                         po::value<bool>(&g_use_result_set)->default_value(g_use_result_set)->implicit_value(true),
                         "Use the new result set");
  desc_adv.add_options()("bigint-count",
                         po::value<bool>(&g_bigint_count)->default_value(g_bigint_count)->implicit_value(false),
                         "Use 64-bit count");
  desc_adv.add_options()("allow-cpu-retry",
                         po::value<bool>(&g_allow_cpu_retry)->default_value(g_allow_cpu_retry)->implicit_value(true),
                         "Allow the queries which failed on GPU to retry on CPU, even when watchdog is enabled");
  desc_adv.add_options()(
      "db-query-list", po::value<std::string>(&db_query_file), "Path to file containing mapd queries");
  desc_adv.add_options()(
      "enable-access-priv-check",
      po::value<bool>(&enable_access_priv_check)->default_value(enable_access_priv_check)->implicit_value(false),
      "Check user access privileges to database objects");
  desc_adv.add_options()(
      "hll-precision-bits",
      po::value<int>(&g_hll_precision_bits)->default_value(g_hll_precision_bits)->implicit_value(g_hll_precision_bits),
      "Number of bits used from the hash value used to specify the bucket number.");

  po::positional_options_description positionalOptions;
  positionalOptions.add("data", 1);

  po::options_description desc_all("All options");
  desc_all.add(desc).add(desc_adv);

  po::variables_map vm;

  std::vector<LeafHostInfo> db_leaves;
  std::vector<LeafHostInfo> string_leaves;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc_all).positional(positionalOptions).run(), vm);
    po::notify(vm);

    if (vm.count("config")) {
      std::ifstream settings_file(config_file);
      po::store(po::parse_config_file(settings_file, desc_all, true), vm);
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
        g_aggregator = true;
      } else {
        db_leaves.clear();
      }
      string_leaves = only_string_leaves(all_nodes);
      g_cluster = true;
    }

    if (vm.count("help")) {
      std::cout << "Usage: mapd_server <catalog path> [<database name>] [--cpu|--gpu] [-p <port "
                   "number>] [--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << desc << std::endl;
      return 0;
    }
    if (vm.count("help-advanced")) {
      std::cout << "Usage: mapd_server <catalog path> [<database name>] [--cpu|--gpu] [-p <port "
                   "number>] [--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << desc_all << std::endl;
      return 0;
    }
    if (vm.count("version")) {
      std::cout << "MapD Version: " << MAPD_RELEASE << std::endl;
      return 0;
    }
    if (vm.count("cpu"))
      device = "cpu";
    if (vm.count("gpu"))
      device = "gpu";
    if (num_gpus == 0)
      device = "cpu";

    if (device == "cpu")
      enable_rendering = false;

    g_enable_watchdog = enable_watchdog;
    g_enable_dynamic_watchdog = enable_dynamic_watchdog;
    g_dynamic_watchdog_time_limit = dynamic_watchdog_time_limit;
  } catch (boost::program_options::error& e) {
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
    std::cerr << "MapD data directory does not exist at '" << base_path << "'. Run initdb " << base_path << std::endl;
    return 1;
  }

  const auto lock_file = boost::filesystem::path(base_path) / "mapd_server_pid.lck";
  auto pid = std::to_string(getpid());
  int pid_fd = open(lock_file.c_str(), O_RDWR | O_CREAT, 0644);
  if (pid_fd == -1) {
    auto err = std::string("Failed to open PID file ") + std::string(lock_file.c_str()) + std::string(". ") +
               strerror(errno) + ".";
    std::cerr << err << std::endl;
    return 1;
  }
  if (lockf(pid_fd, F_TLOCK, 0) == -1) {
    auto err = std::string("Another MapD Server is using data directory ") + base_path + std::string(".");
    std::cerr << err << std::endl;
    close(pid_fd);
    return 1;
  }
  if (ftruncate(pid_fd, 0) == -1) {
    auto err = std::string("Failed to truncate PID file ") + std::string(lock_file.c_str()) + std::string(". ") +
               strerror(errno) + std::string(".");
    std::cerr << err << std::endl;
    close(pid_fd);
    return 1;
  }
  if (write(pid_fd, pid.c_str(), pid.length()) == -1) {
    auto err = std::string("Failed to write PID file ") + std::string(lock_file.c_str()) + ". " + strerror(errno) + ".";
    std::cerr << err << std::endl;
    close(pid_fd);
    return 1;
  }

  const auto log_path = boost::filesystem::path(base_path) / "mapd_log";
  (void)boost::filesystem::create_directory(log_path);
  FLAGS_log_dir = log_path.c_str();
  if (flush_log)
    FLAGS_logbuflevel = -1;
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);

  boost::algorithm::trim_if(db_query_file, boost::is_any_of("\"'"));
  if (db_query_file.length() > 0 && !boost::filesystem::exists(db_query_file)) {
    LOG(ERROR) << "File containing DB queries " << db_query_file << " does not exist.";
    return 1;
  }
  boost::algorithm::trim_if(db_convert_dir, boost::is_any_of("\"'"));
  if (db_convert_dir.length() > 0 && !boost::filesystem::exists(db_convert_dir)) {
    LOG(ERROR) << "Data conversion source directory " << db_convert_dir << " does not exist.";
    return 1;
  }
  const auto system_db_file = boost::filesystem::path(base_path) / "mapd_catalogs" / "mapd";
  if (!boost::filesystem::exists(system_db_file)) {
    LOG(ERROR) << "MapD system catalogs does not exist at " << system_db_file;
    return 1;
  }
  const auto db_file = boost::filesystem::path(base_path) / "mapd_catalogs" / MAPD_SYSTEM_DB;
  if (!boost::filesystem::exists(db_file)) {
    LOG(ERROR) << "MapD database " << MAPD_SYSTEM_DB << " does not exist.";
    return 1;
  }

  // add all parameters to be displayed on startup
  LOG(INFO) << "MapD started with data directory at '" << base_path << "'";
  if (vm.count("cluster")) {
    LOG(INFO) << "Cluster file specified running as aggregator with config at '" << cluster_file << "'";
  }
  if (vm.count("string-servers")) {
    LOG(INFO) << "String servers file specified running as dbleaf with config at '" << cluster_file << "'";
  }
  LOG(INFO) << " Watchdog is set to " << enable_watchdog;
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
  LOG(INFO) << " MapD Server Port  " << mapd_parameters.mapd_server_port;
  LOG(INFO) << " MapD Calcite Port  " << mapd_parameters.calcite_port;

  boost::algorithm::trim_if(ldapMetadata.distinguishedName, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(ldapMetadata.uri, boost::is_any_of("\"'"));

  // rudimetary signal handling to try to guarantee the logging gets flushed to files
  // on shutdown
  register_signal_handler();

  shared_ptr<MapDHandler> handler(new MapDHandler(db_leaves,
                                                  string_leaves,
                                                  base_path,
                                                  device,
                                                  allow_multifrag,
                                                  jit_debug,
                                                  read_only,
                                                  allow_loop_joins,
                                                  enable_rendering,
                                                  cpu_buffer_mem_bytes,
                                                  render_mem_bytes,
                                                  num_gpus,
                                                  start_gpu,
                                                  reserved_gpu_mem,
                                                  num_reader_threads,
                                                  ldapMetadata,
                                                  mapd_parameters,
                                                  db_convert_dir,
                                                  enable_legacy_syntax,
                                                  enable_access_priv_check));

  if (mapd_parameters.ha_group_id.empty()) {
    shared_ptr<TProcessor> processor(new MapDProcessor(handler));

    shared_ptr<TServerTransport> bufServerTransport(new TServerSocket(mapd_parameters.mapd_server_port));

    shared_ptr<TTransportFactory> bufTransportFactory(new TBufferedTransportFactory());
    shared_ptr<TProtocolFactory> bufProtocolFactory(new TBinaryProtocolFactory());
    TThreadedServer bufServer(processor, bufServerTransport, bufTransportFactory, bufProtocolFactory);

    shared_ptr<TServerTransport> httpServerTransport(new TServerSocket(http_port));
    shared_ptr<TTransportFactory> httpTransportFactory(new THttpServerTransportFactory());
    shared_ptr<TProtocolFactory> httpProtocolFactory(new TJSONProtocolFactory());
    TThreadedServer httpServer(processor, httpServerTransport, httpTransportFactory, httpProtocolFactory);

    std::thread bufThread(start_server, std::ref(bufServer));
    std::thread httpThread(start_server, std::ref(httpServer));

    // run warm up queries if any exists
    run_warmup_queries(handler, base_path, db_query_file);

    bufThread.join();
    httpThread.join();
  } else {  // running ha server
    LOG(FATAL) << "No High Availability module available, please contact MapD support";
  }
  return 0;
}
