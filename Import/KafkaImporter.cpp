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

/**
 * @file    KafkaImporter.cpp
 * @author  Michael <michael@mapd.com>
 * @brief   Based on StreamInsert code but using binary columnar format for inserting a
 *stream of rows with optional transformations from stdin to a MapD table.
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/regex.hpp>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>

#include "RowToColumnLoader.h"
#include "Shared/ThriftClient.h"
#include "Shared/sqltypes.h"

#include <chrono>
#include <thread>

#include <boost/program_options.hpp>

#include "rdkafkacpp.h"

#define MAX_FIELD_LEN 20000

bool print_error_data = false;
bool print_transformation = false;

static bool run = true;
static bool exit_eof = false;
static int eof_cnt = 0;
static int partition_cnt = 0;
static long msg_cnt = 0;
static int64_t msg_bytes = 0;

class RebalanceCb : public RdKafka::RebalanceCb {
 private:
  static void part_list_print(const std::vector<RdKafka::TopicPartition*>& partitions) {
    for (unsigned int i = 0; i < partitions.size(); i++) {
      LOG(INFO) << "\t" << partitions[i]->topic() << "[" << partitions[i]->partition()
                << "]";
    }
  }

 public:
  void rebalance_cb(RdKafka::KafkaConsumer* consumer,
                    RdKafka::ErrorCode err,
                    std::vector<RdKafka::TopicPartition*>& partitions) {
    LOG(INFO) << "RebalanceCb: " << RdKafka::err2str(err) << ": ";

    part_list_print(partitions);

    if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
      consumer->assign(partitions);
      partition_cnt = (int)partitions.size();
    } else {
      consumer->unassign();
      partition_cnt = 0;
    }
    eof_cnt = 0;
  }
};

bool msg_consume(RdKafka::Message* message,
                 RowToColumnLoader& row_loader,
                 Importer_NS::CopyParams copy_params,
                 const std::map<std::string,
                                std::pair<std::unique_ptr<boost::regex>,
                                          std::unique_ptr<std::string>>>& transformations,
                 const bool remove_quotes) {
  switch (message->err()) {
    case RdKafka::ERR__TIMED_OUT:
      VLOG(1) << " Timed out";
      break;

    case RdKafka::ERR_NO_ERROR: { /* Real message */
      msg_cnt++;
      msg_bytes += message->len();
      VLOG(1) << "Read msg at offset " << message->offset();
      RdKafka::MessageTimestamp ts;
      ts = message->timestamp();
      if (ts.type != RdKafka::MessageTimestamp::MSG_TIMESTAMP_NOT_AVAILABLE) {
        std::string tsname = "?";
        if (ts.type == RdKafka::MessageTimestamp::MSG_TIMESTAMP_CREATE_TIME) {
          tsname = "create time";
        } else if (ts.type == RdKafka::MessageTimestamp::MSG_TIMESTAMP_LOG_APPEND_TIME) {
          tsname = "log append time";
        }
        VLOG(1) << "Timestamp: " << tsname << " " << ts.timestamp << std::endl;
      }

      char buffer[message->len() + 1];
      sprintf(buffer,
              "%.*s\n",
              static_cast<int>(message->len()),
              static_cast<const char*>(message->payload()));
      VLOG(1) << "Full Message received is :'" << buffer << "'";

      char field[MAX_FIELD_LEN];
      size_t field_i = 0;

      bool backEscape = false;

      auto row_desc = row_loader.get_row_descriptor();

      const std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>*
          xforms[row_desc.size()];
      for (size_t i = 0; i < row_desc.size(); i++) {
        auto it = transformations.find(row_desc[i].col_name);
        if (it != transformations.end()) {
          xforms[i] = &(it->second);
        } else {
          xforms[i] = nullptr;
        }
      }

      std::vector<TStringValue>
          row;  // used to store each row as we move through the stream

      for (auto iit : buffer) {
        if (iit == copy_params.delimiter || iit == copy_params.line_delim) {
          bool end_of_field = (iit == copy_params.delimiter);
          bool end_of_row;
          if (end_of_field) {
            end_of_row = false;
          } else {
            end_of_row = (row_desc[row.size()].col_type.type != TDatumType::STR) ||
                         (row.size() == row_desc.size() - 1);
            if (!end_of_row) {
              size_t l = copy_params.null_str.size();
              if (field_i >= l &&
                  strncmp(field + field_i - l, copy_params.null_str.c_str(), l) == 0) {
                end_of_row = true;
              }
            }
          }
          if (!end_of_field && !end_of_row) {
            // not enough columns yet and it is a string column
            // treat the line delimiter as part of the string
            field[field_i++] = iit;
          } else {
            field[field_i] = '\0';
            field_i = 0;
            TStringValue ts;
            ts.str_val = std::string(field);
            ts.is_null = (ts.str_val.empty() || ts.str_val == copy_params.null_str);
            auto xform = row.size() < row_desc.size() ? xforms[row.size()] : nullptr;
            if (!ts.is_null && xform != nullptr) {
              if (print_transformation) {
                std::cout << "\ntransforming\n" << ts.str_val << "\nto\n";
              }
              ts.str_val =
                  boost::regex_replace(ts.str_val, *xform->first, *xform->second);
              if (ts.str_val.empty()) {
                ts.is_null = true;
              }
              if (print_transformation) {
                std::cout << ts.str_val << std::endl;
              }
            }

            row.push_back(ts);  // add column value to row
            if (end_of_row || (row.size() > row_desc.size())) {
              break;  // found row
            }
          }
        } else {
          if (iit == '\\') {
            backEscape = true;
          } else if (backEscape || !remove_quotes || iit != '\"') {
            field[field_i++] = iit;
            backEscape = false;
          }
          // else if unescaped double-quote, continue without adding the
          // character to the field string.
        }
        if (field_i >= MAX_FIELD_LEN) {
          field[MAX_FIELD_LEN - 1] = '\0';
          std::cerr << "String too long for buffer." << std::endl;
          if (print_error_data) {
            std::cerr << field << std::endl;
          }
          field_i = 0;
          break;
        }
      }
      if (row.size() == row_desc.size()) {
        // add the new data in the column format
        bool record_loaded = row_loader.convert_string_to_column(row, copy_params);
        if (!record_loaded) {
          // record could not be parsed correctly consider it skipped
          return false;
        } else {
          return true;
        }
      } else {
        if (print_error_data) {
          std::cerr << "Incorrect number of columns for row: ";
          std::cerr << row_loader.print_row_with_delim(row, copy_params) << std::endl;
          return false;
        }
      }
    }

    case RdKafka::ERR__PARTITION_EOF:
      /* Last message */
      if (exit_eof && ++eof_cnt == partition_cnt) {
        LOG(ERROR) << "%% EOF reached for all " << partition_cnt << " partition(s)";
        run = false;
      }
      break;

    case RdKafka::ERR__UNKNOWN_TOPIC:
    case RdKafka::ERR__UNKNOWN_PARTITION:
      LOG(ERROR) << "Consume failed: " << message->errstr() << std::endl;
      run = false;
      break;

    default:
      /* Errors */
      LOG(ERROR) << "Consume failed: " << message->errstr();
      run = false;
  }
  return false;
};

class ConsumeCb : public RdKafka::ConsumeCb {
 public:
  void consume_cb(RdKafka::Message& msg, void* opaque) {
    // reinterpret_cast<KafkaMgr*>(opaque)->
    // msg_consume(&msg, opaque);
  }
};

class EventCb : public RdKafka::EventCb {
 public:
  void event_cb(RdKafka::Event& event) {
    switch (event.type()) {
      case RdKafka::Event::EVENT_ERROR:
        LOG(ERROR) << "ERROR (" << RdKafka::err2str(event.err()) << "): " << event.str();
        if (event.err() == RdKafka::ERR__ALL_BROKERS_DOWN) {
          LOG(ERROR) << "All brokers are down, we may need special handling here";
          run = false;
        }
        break;

      case RdKafka::Event::EVENT_STATS:
        VLOG(2) << "\"STATS\": " << event.str();
        break;

      case RdKafka::Event::EVENT_LOG:
        LOG(INFO) << "LOG-" << event.severity() << "-" << event.fac().c_str() << ":"
                  << event.str().c_str();
        break;

      case RdKafka::Event::EVENT_THROTTLE:
        LOG(INFO) << "THROTTLED: " << event.throttle_time() << "ms by "
                  << event.broker_name() << " id " << (int)event.broker_id();
        break;

      default:
        LOG(INFO) << "EVENT " << event.type() << " (" << RdKafka::err2str(event.err())
                  << "): " << event.str();
        break;
    }
  }
};

// reads from a kafka topic (expects delimited string input)
void kafka_insert(
    RowToColumnLoader& row_loader,
    const std::map<std::string,
                   std::pair<std::unique_ptr<boost::regex>,
                             std::unique_ptr<std::string>>>& transformations,
    const Importer_NS::CopyParams& copy_params,
    const bool remove_quotes,
    std::string group_id,
    std::string topic,
    std::string brokers) {
  std::string errstr;
  std::string topic_str;
  std::string mode;
  std::string debug;
  std::vector<std::string> topics;
  bool do_conf_dump = false;
  int use_ccb = 0;

  RebalanceCb ex_rebalance_cb;

  /*
   * Create configuration objects
   */
  RdKafka::Conf* conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  RdKafka::Conf* tconf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);

  conf->set("rebalance_cb", &ex_rebalance_cb, errstr);

  if (conf->set("group.id", group_id, errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << "could not set  group.id " << errstr;
  }

  if (conf->set("compression.codec", "none", errstr) != /* can also be gzip or snappy */
      RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }

  if (conf->set("statistics.interval.ms", "1000", errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }
  if (conf->set("enable.auto.commit", "false", errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }

  if (tconf->set("auto.offset.reset", "earliest", errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }

  if (tconf->set("enable.auto.commit", "false", errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }

  do_conf_dump = true;

  topics.push_back(topic);

  LOG(INFO) << "Version " << RdKafka::version_str().c_str();
  LOG(INFO) << RdKafka::version();
  LOG(INFO) << RdKafka::get_debug_contexts().c_str();

  conf->set("metadata.broker.list", brokers, errstr);

  // debug = "none";
  if (!debug.empty()) {
    if (conf->set("debug", debug, errstr) != RdKafka::Conf::CONF_OK) {
      LOG(FATAL) << errstr;
    }
  }

  ConsumeCb consume_cb;
  use_ccb = 0;
  if (use_ccb) {
    if (conf->set("consume_cb", &consume_cb, errstr) != RdKafka::Conf::CONF_OK) {
      LOG(FATAL) << errstr;
    }
    // need to set the opaque pointer here for the callbacks
    //        rd_kafka_conf_set_opaque(conf, this);
  }

  EventCb ex_event_cb;
  if (conf->set("event_cb", &ex_event_cb, errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }

  if (conf->set("default_topic_conf", tconf, errstr) != RdKafka::Conf::CONF_OK) {
    LOG(FATAL) << errstr;
  }

  if (do_conf_dump) {
    int pass;

    for (pass = 0; pass < 2; pass++) {
      std::list<std::string>* dump;
      if (pass == 0) {
        dump = conf->dump();
        LOG(INFO) << "# Global config";
        LOG(INFO) << "===============";
      } else {
        dump = tconf->dump();
        LOG(INFO) << "# Topic config";
        LOG(INFO) << "===============";
      }

      for (std::list<std::string>::iterator it = dump->begin(); it != dump->end();) {
        std::string ts = *it;
        it++;
        LOG(INFO) << ts << " = " << *it;
        it++;
      }
      LOG(INFO) << "Dump config finished";
    }
  }
  LOG(INFO) << "FULL Dump config finished";

  delete tconf;

  /*
   * Create consumer using accumulated global configuration.
   */
  RdKafka::KafkaConsumer* consumer = RdKafka::KafkaConsumer::create(conf, errstr);
  if (!consumer) {
    LOG(ERROR) << "Failed to create consumer: " << errstr;
  }

  delete conf;

  LOG(INFO) << " Created consumer " << consumer->name();

  /*
   * Subscribe to topics
   */
  RdKafka::ErrorCode err = consumer->subscribe(topics);
  if (err) {
    LOG(FATAL) << "Failed to subscribe to " << topics.size()
               << " topics: " << RdKafka::err2str(err);
  }

  /*
   * Consume messages
   */
  size_t recv_rows = 0;
  int skipped = 0;
  int rows_loaded = 0;
  while (run) {
    RdKafka::Message* msg = consumer->consume(10000);
    if (msg->err() == RdKafka::ERR_NO_ERROR) {
      if (!use_ccb) {
        bool added =
            msg_consume(msg, row_loader, copy_params, transformations, remove_quotes);
        if (added) {
          recv_rows++;
          if (recv_rows == copy_params.batch_size) {
            recv_rows = 0;
            row_loader.do_load(rows_loaded, skipped, copy_params);
            // make sure we now commit that we are up to here to cover the mesages we just
            // loaded
            consumer->commitSync();
          }
        } else {
          // LOG(ERROR) << " messsage was skipped ";
          skipped++;
        }
      }
    }
    delete msg;
  }

  /*
   * Stop consumer
   */
  consumer->close();
  delete consumer;

  LOG(INFO) << "Consumed " << msg_cnt << " messages (" << msg_bytes << " bytes)";
  LOG(FATAL) << "Consumer shut down, probably due to an error please review logs";
};

struct stuff {
  RowToColumnLoader row_loader;
  Importer_NS::CopyParams copy_params;

  stuff(RowToColumnLoader& rl, Importer_NS::CopyParams& cp)
      : row_loader(rl), copy_params(cp){};
};

int main(int argc, char** argv) {
  std::string server_host("localhost");  // default to localhost
  int port = 6274;                       // default port number
  bool http = false;
  bool https = false;
  bool skip_host_verify = false;
  std::string ca_cert_name{""};
  std::string table_name;
  std::string db_name;
  std::string user_name;
  std::string passwd;
  std::string group_id;
  std::string topic;
  std::string brokers;
  std::string delim_str(","), nulls("\\N"), line_delim_str("\n"), quoted("false");
  size_t batch_size = 10000;
  size_t retry_count = 10;
  size_t retry_wait = 5;
  bool remove_quotes = false;
  std::vector<std::string> xforms;
  std::map<std::string,
           std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>>
      transformations;
  ThriftConnectionType conn_type;

  google::InitGoogleLogging(argv[0]);

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ");
  desc.add_options()(
      "table", po::value<std::string>(&table_name)->required(), "Table Name");
  desc.add_options()(
      "database", po::value<std::string>(&db_name)->required(), "Database Name");
  desc.add_options()(
      "user,u", po::value<std::string>(&user_name)->required(), "User Name");
  desc.add_options()(
      "passwd,p", po::value<std::string>(&passwd)->required(), "User Password");
  desc.add_options()("host",
                     po::value<std::string>(&server_host)->default_value(server_host),
                     "OmniSci Server Hostname");
  desc.add_options()(
      "port", po::value<int>(&port)->default_value(port), "OmniSci Server Port Number");
  desc.add_options()("http",
                     po::bool_switch(&http)->default_value(http)->implicit_value(true),
                     "Use HTTP transport");
  desc.add_options()("https",
                     po::bool_switch(&https)->default_value(https)->implicit_value(true),
                     "Use HTTPS transport");
  desc.add_options()("skip-verify",
                     po::bool_switch(&skip_host_verify)
                         ->default_value(skip_host_verify)
                         ->implicit_value(true),
                     "Don't verify SSL certificate validity");
  desc.add_options()(
      "ca-cert",
      po::value<std::string>(&ca_cert_name)->default_value(ca_cert_name),
      "Path to trusted server certificate. Initiates an encrypted connection");
  desc.add_options()("delim",
                     po::value<std::string>(&delim_str)->default_value(delim_str),
                     "Field delimiter");
  desc.add_options()("null", po::value<std::string>(&nulls), "NULL string");
  desc.add_options()("line", po::value<std::string>(&line_delim_str), "Line delimiter");
  desc.add_options()(
      "quoted",
      po::value<std::string>(&quoted),
      "Whether the source contains quoted fields (true/false, default false)");
  desc.add_options()("batch",
                     po::value<size_t>(&batch_size)->default_value(batch_size),
                     "Insert batch size");
  desc.add_options()("retry_count",
                     po::value<size_t>(&retry_count)->default_value(retry_count),
                     "Number of time to retry an insert");
  desc.add_options()("retry_wait",
                     po::value<size_t>(&retry_wait)->default_value(retry_wait),
                     "wait in secs between retries");
  desc.add_options()("transform,t",
                     po::value<std::vector<std::string>>(&xforms)->multitoken(),
                     "Column Transformations");
  desc.add_options()("print_error", "Print Error Rows");
  desc.add_options()("print_transform", "Print Transformations");
  desc.add_options()("topic",
                     po::value<std::string>(&topic)->required(),
                     "Kafka topic to consume from ");
  desc.add_options()("group-id",
                     po::value<std::string>(&group_id)->required(),
                     "Group id this consumer is part of");
  desc.add_options()("brokers",
                     po::value<std::string>(&brokers)->required(),
                     "list of kafka brokers for topic");

  po::positional_options_description positionalOptions;
  positionalOptions.add("table", 1);
  positionalOptions.add("database", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(positionalOptions)
                  .run(),
              vm);
    if (vm.count("help")) {
      std::cout << "Usage: <table name> <database name> {-u|--user} <user> {-p|--passwd} "
                   "<password> [{--host} "
                   "<hostname>][--port <port number>][--delim <delimiter>][--null <null "
                   "string>][--line <line "
                   "delimiter>][--batch <batch size>][{-t|--transform} transformation "
                   "[--quoted <true|false>] "
                   "...][--retry_count <num_of_retries>] [--retry_wait <wait in "
                   "secs>][--print_error][--print_transform]\n\n";
      std::cout << desc << std::endl;
      return 0;
    }
    if (vm.count("print_error")) {
      print_error_data = true;
    }
    if (vm.count("print_transform")) {
      print_transformation = true;
    }

    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (http) {
    conn_type = ThriftConnectionType::HTTP;
  } else if (https) {
    conn_type = ThriftConnectionType::HTTPS;
  } else if (!ca_cert_name.empty()) {
    conn_type = ThriftConnectionType::BINARY_SSL;
  } else {
    conn_type = ThriftConnectionType::BINARY;
  }

  char delim = delim_str[0];
  if (delim == '\\') {
    if (delim_str.size() < 2 ||
        (delim_str[1] != 'x' && delim_str[1] != 't' && delim_str[1] != 'n')) {
      std::cerr << "Incorrect delimiter string: " << delim_str << std::endl;
      return 1;
    }
    if (delim_str[1] == 't') {
      delim = '\t';
    } else if (delim_str[1] == 'n') {
      delim = '\n';
    } else {
      std::string d(delim_str);
      d[0] = '0';
      delim = (char)std::stoi(d, nullptr, 16);
    }
  }
  if (isprint(delim)) {
    std::cout << "Field Delimiter: " << delim << std::endl;
  } else if (delim == '\t') {
    std::cout << "Field Delimiter: "
              << "\\t" << std::endl;
  } else if (delim == '\n') {
    std::cout << "Field Delimiter: "
              << "\\n"
              << std::endl;
  } else {
    std::cout << "Field Delimiter: \\x" << std::hex << (int)delim << std::endl;
  }
  char line_delim = line_delim_str[0];
  if (line_delim == '\\') {
    if (line_delim_str.size() < 2 ||
        (line_delim_str[1] != 'x' && line_delim_str[1] != 't' &&
         line_delim_str[1] != 'n')) {
      std::cerr << "Incorrect delimiter string: " << line_delim_str << std::endl;
      return 1;
    }
    if (line_delim_str[1] == 't') {
      line_delim = '\t';
    } else if (line_delim_str[1] == 'n') {
      line_delim = '\n';
    } else {
      std::string d(line_delim_str);
      d[0] = '0';
      line_delim = (char)std::stoi(d, nullptr, 16);
    }
  }
  if (isprint(line_delim)) {
    std::cout << "Line Delimiter: " << line_delim << std::endl;
  } else if (line_delim == '\t') {
    std::cout << "Line Delimiter: "
              << "\\t" << std::endl;
  } else if (line_delim == '\n') {
    std::cout << "Line Delimiter: "
              << "\\n"
              << std::endl;
  } else {
    std::cout << "Line Delimiter: \\x" << std::hex << (int)line_delim << std::endl;
  }
  std::cout << "Null String: " << nulls << std::endl;
  std::cout << "Insert Batch Size: " << std::dec << batch_size << std::endl;

  if (quoted == "true") {
    remove_quotes = true;
  }

  for (auto& t : xforms) {
    auto n = t.find_first_of(':');
    if (n == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/"
                << std::endl;
      return 1;
    }
    std::string col_name = t.substr(0, n);
    if (t.size() < n + 3 || t[n + 1] != 's' || t[n + 2] != '/') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/"
                << std::endl;
      return 1;
    }
    auto n1 = n + 3;
    auto n2 = t.find_first_of('/', n1);
    if (n2 == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/"
                << std::endl;
      return 1;
    }
    std::string regex_str = t.substr(n1, n2 - n1);
    n1 = n2 + 1;
    n2 = t.find_first_of('/', n1);
    if (n2 == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/"
                << std::endl;
      return 1;
    }
    std::string fmt_str = t.substr(n1, n2 - n1);
    std::cout << "transform " << col_name << ": s/" << regex_str << "/" << fmt_str << "/"
              << std::endl;
    transformations[col_name] =
        std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>(
            std::unique_ptr<boost::regex>(new boost::regex(regex_str)),
            std::unique_ptr<std::string>(new std::string(fmt_str)));
  }

  Importer_NS::CopyParams copy_params(
      delim, nulls, line_delim, batch_size, retry_count, retry_wait);
  RowToColumnLoader row_loader(
      ThriftClientConnection(
          server_host, port, conn_type, skip_host_verify, ca_cert_name, ca_cert_name),
      user_name,
      passwd,
      db_name,
      table_name);

  kafka_insert(
      row_loader, transformations, copy_params, remove_quotes, group_id, topic, brokers);
  return 0;
}
