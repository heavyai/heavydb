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

#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "ThriftHandler/DBHandler.h"

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

#include "Logger/Logger.h"
#include "Shared/SystemParameters.h"
#include "Shared/file_delete.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/scope.h"
#include "ThriftHandler/CommandLineOptions.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/locale/generator.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include <csignal>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <vector>

#ifdef HAVE_AWS_S3
#include "DataMgr/OmniSciAwsSdk.h"
#endif
#include "MapDRelease.h"
#include "Shared/Compressor.h"
#include "Shared/SystemParameters.h"
#include "Shared/file_delete.h"
#include "Shared/scope.h"
#if ENABLE_ITT
#include <ittnotify.h>
#endif

using namespace ::apache::thrift;
using namespace ::apache::thrift::concurrency;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::server;
using namespace ::apache::thrift::transport;

extern bool g_enable_thrift_logs;

// Set g_running to false to trigger normal server shutdown.
std::atomic<bool> g_running{true};

namespace {  // anonymous

std::atomic<int> g_saw_signal{-1};

std::shared_ptr<TThreadedServer> g_thrift_http_server;
std::shared_ptr<TThreadedServer> g_thrift_tcp_server;

std::shared_ptr<DBHandler> g_warmup_handler;
// global "g_warmup_handler" needed to avoid circular dependency
// between "DBHandler" & function "run_warmup_queries"

std::shared_ptr<DBHandler> g_mapd_handler;

void register_signal_handler(int signum, void (*handler)(int)) {
#ifdef _WIN32
  signal(signum, handler);
#else
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  if (handler != SIG_DFL && handler != SIG_IGN) {
    // block all signal deliveries while inside the signal handler
    sigfillset(&act.sa_mask);
  }
  act.sa_handler = handler;
  sigaction(signum, &act, NULL);
#endif
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
  if (signum == SIGABRT || signum == SIGSEGV || signum == SIGFPE
#ifndef _WIN32
      || signum == SIGQUIT
#endif
  ) {
    // Wait briefly to give heartbeat() a chance to flush the logs and
    // do any other emergency shutdown tasks.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Explicitly trigger whatever default action this signal would
    // have done, such as terminate the process or dump core.
    // Signals are currently blocked so this new signal will be queued
    // until this signal handler returns.
    register_signal_handler(signum, SIG_DFL);
#ifdef _WIN32
    raise(signum);
#else
    kill(getpid(), signum);
#endif
    std::this_thread::sleep_for(std::chrono::seconds(5));

#ifndef __APPLE__
    // as a last resort, abort
    // primary used in Docker environments, where we can end up with PID 1 and fail to
    // catch unix signals
    quick_exit(signum);
#endif
  }
}

void register_signal_handlers() {
  register_signal_handler(SIGINT, omnisci_signal_handler);
#ifndef _WIN32
  register_signal_handler(SIGQUIT, omnisci_signal_handler);
  register_signal_handler(SIGHUP, omnisci_signal_handler);
#endif
  register_signal_handler(SIGTERM, omnisci_signal_handler);
  register_signal_handler(SIGSEGV, omnisci_signal_handler);
  register_signal_handler(SIGABRT, omnisci_signal_handler);
#ifndef _WIN32
  // Thrift secure socket can cause problems with SIGPIPE
  register_signal_handler(SIGPIPE, SIG_IGN);
#endif
}

}  // anonymous namespace

void start_server(std::shared_ptr<TThreadedServer> server, const int port) {
  try {
    server->serve();
    if (errno != 0) {
      throw std::runtime_error(std::string("Thrift server exited: ") +
                               std::strerror(errno));
    }
  } catch (std::exception& e) {
    LOG(ERROR) << "Exception: " << e.what() << ": port " << port << std::endl;
  }
}

void releaseWarmupSession(TSessionId& sessionId, std::ifstream& query_file) noexcept {
  query_file.close();
  if (sessionId != g_warmup_handler->getInvalidSessionId()) {
    try {
      g_warmup_handler->disconnect(sessionId);
    } catch (...) {
      LOG(ERROR) << "Failed to disconnect warmup session, possible failure to run warmup "
                    "queries.";
    }
  }
}

#if ENABLE_ITT
__itt_domain* ittquery = __itt_domain_create("Query");
#endif
void run_warmup_queries(std::shared_ptr<DBHandler> handler,
                        std::string base_path,
                        std::string query_file_path) {
  // run warmup queries to load cache if requested
  if (query_file_path.empty()) {
    return;
  }
  if (handler->isAggregator()) {
    LOG(INFO) << "Skipping warmup query execution on the aggregator, queries should be "
                 "run directly on the leaf nodes.";
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
    bool stop = false;
    while (!stop && std::getline(query_file, db_info)) {
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
#if ENABLE_ITT
            if (single_query == "--itt_resume")
              __itt_resume();
            else if (single_query == "--itt_pause")
              __itt_pause();
#endif
            single_query.clear();
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
#if ENABLE_ITT
            __itt_frame_begin_v3(ittquery, (__itt_id*)single_query.c_str());
#endif
            g_warmup_handler->sql_execute(ret, sessionId, single_query, true, "", -1, -1);
#if ENABLE_ITT
            __itt_frame_end_v3(ittquery, (__itt_id*)single_query.c_str());
#endif
          } catch (std::exception& e) {
            LOG(WARNING) << "Exception " << e.what() << " while executing '"
                         << single_query;
            stop = true;
            break;
          } catch (...) {
            LOG(WARNING) << "Exception while executing '" << single_query << "'";
            stop = true;
            break;
          }
          single_query.clear();
        }
#if ENABLE_ITT
        __itt_detach();
#endif
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
  } catch (const std::exception& e) {
    LOG(WARNING)
        << "Exception while executing warmup queries. "
        << "Warmup may not be fully completed. Will proceed nevertheless.\nError was: "
        << e.what();
  }
}

void thrift_stop() {
  if (auto thrift_http_server = g_thrift_http_server; thrift_http_server) {
    thrift_http_server->stop();
  }
  g_thrift_http_server.reset();

  if (auto thrift_tcp_server = g_thrift_tcp_server; thrift_tcp_server) {
    thrift_tcp_server->stop();
  }
  g_thrift_tcp_server.reset();
}

void heartbeat() {
#ifndef _WIN32
  // Block all signals for this heartbeat thread, only.
  sigset_t set;
  sigfillset(&set);
  int result = pthread_sigmask(SIG_BLOCK, &set, NULL);
  if (result != 0) {
    throw std::runtime_error("heartbeat() thread startup failed");
  }
#endif

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

  // If dumping core, try to do some quick stuff.
  if (signum == SIGABRT || signum == SIGSEGV || signum == SIGFPE
#ifndef _WIN32
      || signum == SIGQUIT
#endif
  ) {
    // Need to shut down calcite.
    if (auto mapd_handler = g_mapd_handler; mapd_handler) {
      mapd_handler->emergency_shutdown();
    }
    // Need to flush the logs for debugging.
    logger::shutdown();
    return;
    // Core dump should begin soon after this. See omnisci_signal_handler().
    // We leave the rest of the server process as is for the core dump image.
  }

  // Stopping the Thrift thread(s) will allow main() to return.
  thrift_stop();
}

int startMapdServer(CommandLineOptions& prog_config_opts, bool start_http_server = true) {
  // Prepare to launch the Thrift server.
  LOG(INFO) << "OmniSciDB starting up";
  register_signal_handlers();
#ifdef HAVE_AWS_S3
  omnisci_aws_sdk::init_sdk();
#endif  // HAVE_AWS_S3
  std::set<std::unique_ptr<std::thread>> server_threads;
  auto wait_for_server_threads = [&] {
    for (auto& th : server_threads) {
      try {
        th->join();
      } catch (const std::system_error& e) {
        if (e.code() != std::errc::invalid_argument) {
          LOG(WARNING) << "std::thread join failed: " << e.what();
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "std::thread join failed: " << e.what();
      } catch (...) {
        LOG(WARNING) << "std::thread join failed";
      }
    }
  };
  ScopeGuard server_shutdown_guard = [&] {
    // This function will never be called by exit(), but we shouldn't ever be calling
    // exit(), we should be setting g_running to false instead.
    LOG(INFO) << "OmniSciDB shutting down";

    g_running = false;

    thrift_stop();

    g_mapd_handler.reset();

    wait_for_server_threads();

    Catalog_Namespace::SysCatalog::destroy();

#ifdef HAVE_AWS_S3
    omnisci_aws_sdk::shutdown_sdk();
#endif  // HAVE_AWS_S3

    // Flush the logs last to capture maximum debugging information.
    logger::shutdown();
  };

  // start background thread to clean up _DELETE_ME files
  const unsigned int wait_interval =
      3;  // wait time in secs after looking for deleted file before looking again
  server_threads.insert(
      std::make_unique<std::thread>(file_delete,
                                    std::ref(g_running),
                                    wait_interval,
                                    prog_config_opts.base_path + "/mapd_data"));
  server_threads.insert(std::make_unique<std::thread>(heartbeat));

  if (!g_enable_thrift_logs) {
    apache::thrift::GlobalOutput.setOutputFunction([](const char* msg) {});
  }

  if (g_enable_experimental_string_functions) {
    // Use the locale setting of the server by default. The generate parameter can be
    // updated appropriately if a locale override option is ever supported.
    boost::locale::generator generator;
    std::locale::global(generator.generate(""));
  }

  // Thrift event handler for database server setup.
  try {
    if (prog_config_opts.system_parameters.master_address.empty()) {
      // Handler for a single database server. (DBHandler)
      g_mapd_handler =
          std::make_shared<DBHandler>(prog_config_opts.db_leaves,
                                      prog_config_opts.string_leaves,
                                      prog_config_opts.base_path,
                                      prog_config_opts.allow_multifrag,
                                      prog_config_opts.jit_debug,
                                      prog_config_opts.intel_jit_profile,
                                      prog_config_opts.read_only,
                                      prog_config_opts.allow_loop_joins,
                                      prog_config_opts.enable_rendering,
                                      prog_config_opts.renderer_use_vulkan_driver,
                                      prog_config_opts.enable_auto_clear_render_mem,
                                      prog_config_opts.render_oom_retry_threshold,
                                      prog_config_opts.render_mem_bytes,
                                      prog_config_opts.max_concurrent_render_sessions,
                                      prog_config_opts.reserved_gpu_mem,
                                      prog_config_opts.render_compositor_use_last_gpu,
                                      prog_config_opts.num_reader_threads,
                                      prog_config_opts.authMetadata,
                                      prog_config_opts.system_parameters,
                                      prog_config_opts.enable_legacy_syntax,
                                      prog_config_opts.idle_session_duration,
                                      prog_config_opts.max_session_duration,
                                      prog_config_opts.enable_runtime_udf,
                                      prog_config_opts.udf_file_name,
                                      prog_config_opts.udf_compiler_path,
                                      prog_config_opts.udf_compiler_options,
#ifdef ENABLE_GEOS
                                      prog_config_opts.libgeos_so_filename,
#endif
                                      prog_config_opts.disk_cache_config,
                                      false);
    } else {  // running ha server
      LOG(FATAL)
          << "No High Availability module available, please contact OmniSci support";
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize service handler: " << e.what();
  }

  // TCP port setup. We use Thrift both for a TCP socket and for an optional HTTP socket.
  std::shared_ptr<TServerSocket> tcp_socket;
  std::shared_ptr<TServerSocket> http_socket;

  if (!prog_config_opts.system_parameters.ssl_cert_file.empty() &&
      !prog_config_opts.system_parameters.ssl_key_file.empty()) {
    // SSL port setup.
    auto sslSocketFactory = std::make_shared<TSSLSocketFactory>(SSLProtocol::SSLTLS);
    sslSocketFactory->loadCertificate(
        prog_config_opts.system_parameters.ssl_cert_file.c_str());
    sslSocketFactory->loadPrivateKey(
        prog_config_opts.system_parameters.ssl_key_file.c_str());
    if (prog_config_opts.system_parameters.ssl_transport_client_auth) {
      sslSocketFactory->authenticate(true);
    } else {
      sslSocketFactory->authenticate(false);
    }
    sslSocketFactory->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    tcp_socket = std::make_shared<TSSLServerSocket>(
        prog_config_opts.system_parameters.omnisci_server_port, sslSocketFactory);
    if (start_http_server) {
      http_socket = std::make_shared<TSSLServerSocket>(prog_config_opts.http_port,
                                                       sslSocketFactory);
    }
    LOG(INFO) << " OmniSci server using encrypted connection. Cert file ["
              << prog_config_opts.system_parameters.ssl_cert_file << "], key file ["
              << prog_config_opts.system_parameters.ssl_key_file << "]";

  } else {
    // Non-SSL port setup.
    LOG(INFO) << " OmniSci server using unencrypted connection";
    tcp_socket = std::make_shared<TServerSocket>(
        prog_config_opts.system_parameters.omnisci_server_port);
    if (start_http_server) {
      http_socket = std::make_shared<TServerSocket>(prog_config_opts.http_port);
    }
  }

  // Thrift uses the same processor for both the TCP port and the HTTP port.
  std::shared_ptr<TProcessor> processor{std::make_shared<TrackingProcessor>(
      g_mapd_handler, prog_config_opts.log_user_origin)};

  // Thrift TCP server launch.
  std::shared_ptr<TServerTransport> tcp_st = tcp_socket;
  std::shared_ptr<TTransportFactory> tcp_tf{
      std::make_shared<TBufferedTransportFactory>()};
  std::shared_ptr<TProtocolFactory> tcp_pf{std::make_shared<TBinaryProtocolFactory>()};
  g_thrift_tcp_server.reset(new TThreadedServer(processor, tcp_st, tcp_tf, tcp_pf));
  server_threads.insert(std::make_unique<std::thread>(
      start_server,
      g_thrift_tcp_server,
      prog_config_opts.system_parameters.omnisci_server_port));

  // Thrift HTTP server launch.
  if (start_http_server) {
    std::shared_ptr<TServerTransport> http_st = http_socket;
    std::shared_ptr<TTransportFactory> http_tf{
        std::make_shared<THttpServerTransportFactory>()};
    std::shared_ptr<TProtocolFactory> http_pf{std::make_shared<TJSONProtocolFactory>()};
    g_thrift_http_server.reset(new TThreadedServer(processor, http_st, http_tf, http_pf));
    server_threads.insert(std::make_unique<std::thread>(
        start_server, g_thrift_http_server, prog_config_opts.http_port));
  }

  // Run warm up queries if any exist.
  run_warmup_queries(
      g_mapd_handler, prog_config_opts.base_path, prog_config_opts.db_query_file);
  if (prog_config_opts.exit_after_warmup) {
    g_running = false;
  }

  // Main thread blocks for as long as the servers are running.
  wait_for_server_threads();

  // Clean shutdown.
  int signum = g_saw_signal;
  if (signum <= 0 || signum == SIGTERM) {
    return 0;
  } else {
    return signum;
  }
}

int main(int argc, char** argv) {
  bool has_clust_topo = false;

  CommandLineOptions prog_config_opts(argv[0], has_clust_topo);

  try {
    if (auto return_code =
            prog_config_opts.parse_command_line(argc, argv, !has_clust_topo)) {
      return *return_code;
    }

    if (!has_clust_topo) {
      prog_config_opts.validate_base_path();
      prog_config_opts.validate();
      return (startMapdServer(prog_config_opts));
    }
  } catch (std::runtime_error& e) {
    std::cerr << "Server Error: " << e.what() << std::endl;
    return 1;
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }
}
