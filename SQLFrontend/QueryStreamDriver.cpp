/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/ipc.h>
#include <sys/shm.h>

#include "Shared/ThriftClient.h"
#include "Shared/timedate.h"
#include "gen-cpp/Heavy.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;
using SystemClock = std::chrono::system_clock;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

static_assert(Clock::is_steady);
static_assert(std::ratio_less_equal_v<Clock::period, std::milli>);

namespace {

struct QuerySpec {
  std::string id;
  std::string sql;
  fs::path output_file;
  bool interval_ends_at_receive{false};
};

struct TimingRecord {
  std::string id;
  std::string output_file;
  std::string result_format;
  size_t sequence{0};
  std::optional<uint64_t> row_count;
  int64_t server_execution_ms{0};
  std::optional<int64_t> server_total_ms;
  std::optional<int64_t> server_arrow_conversion_ms;
  int64_t submitted_at_unix_ns{0};
  int64_t received_at_unix_ns{0};
  int64_t next_submitted_at_unix_ns{0};
  int64_t client_receive_ns{0};
  int64_t query_interval_ns{0};
  int64_t driver_gap_ns{0};
  int64_t next_submit_gap_ns{0};
  bool interval_ends_at_receive{false};
  bool driver_delay_compliant{true};
  bool timed_out{false};
  std::string interrupt_error;
  std::string status{"ok"};
  std::string error;
  Clock::time_point submit_time;
  Clock::time_point receive_time;
};

void create_parent_directories(const fs::path& path) {
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

class SharedMemoryPayload {
 public:
  SharedMemoryPayload() = default;
  SharedMemoryPayload(const SharedMemoryPayload&) = delete;
  SharedMemoryPayload& operator=(const SharedMemoryPayload&) = delete;

  SharedMemoryPayload(SharedMemoryPayload&& other) noexcept
      : data_(std::exchange(other.data_, nullptr))
      , size_(std::exchange(other.size_, 0)) {}

  SharedMemoryPayload& operator=(SharedMemoryPayload&& other) noexcept {
    if (this != &other) {
      detach();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  ~SharedMemoryPayload() { detach(); }

  static SharedMemoryPayload attach(const TDataFrame& data_frame) {
    if (data_frame.df_handle.size() != sizeof(key_t)) {
      throw std::runtime_error("invalid Arrow shared-memory handle");
    }

    key_t key{IPC_PRIVATE};
    std::memcpy(&key, data_frame.df_handle.data(), sizeof(key));
    const auto shared_memory_id = shmget(key, data_frame.df_size, 0666);
    if (shared_memory_id < 0) {
      throw std::runtime_error("failed to locate Arrow shared-memory result");
    }

    auto* const data = shmat(shared_memory_id, nullptr, 0);
    if (data == reinterpret_cast<void*>(-1)) {
      throw std::runtime_error("failed to attach Arrow shared-memory result");
    }
    if (shmctl(shared_memory_id, IPC_RMID, nullptr) < 0) {
      shmdt(data);
      throw std::runtime_error("failed to mark Arrow shared-memory result for cleanup");
    }
    return SharedMemoryPayload(data, data_frame.df_size);
  }

  explicit operator bool() const { return data_ != nullptr; }
  const char* data() const { return static_cast<const char*>(data_); }
  size_t size() const { return size_; }

 private:
  SharedMemoryPayload(void* data, const size_t size) : data_(data), size_(size) {}

  void detach() noexcept {
    if (data_) {
      shmdt(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  void* data_{nullptr};
  size_t size_{0};
};

struct OutputJob {
  fs::path output_file;
  std::unique_ptr<TQueryResult> result;
  std::string binary_payload;
  SharedMemoryPayload shared_memory_payload;
};

enum class ResultFormat { THRIFT_COLUMNAR, ARROW_WIRE, ARROW_SHARED_MEMORY };

int64_t unix_time_ns(const SystemClock::time_point time) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch())
      .count();
}

int64_t duration_ns(const Clock::time_point start, const Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int64_t tpch_reported_interval_ms(const int64_t interval_ns) {
  constexpr int64_t report_quantum_ns{10'000'000};
  constexpr int64_t half_quantum_ns{report_quantum_ns / 2};
  const auto rounded_ns =
      ((interval_ns + half_quantum_ns) / report_quantum_ns) * report_quantum_ns;
  return std::max(report_quantum_ns, rounded_ns) / 1'000'000;
}

template <typename FloatingPoint>
std::string floating_point_to_decimal(FloatingPoint value) {
  if (!std::isfinite(value)) {
    std::ostringstream output;
    output << value;
    return output.str();
  }
  if (value == FloatingPoint{0}) {
    value = FloatingPoint{0};
  }

  const auto magnitude = std::abs(value);
  const auto integer_digits =
      magnitude >= FloatingPoint{1}
          ? static_cast<int>(std::floor(std::log10(magnitude))) + 1
          : 0;
  const auto fractional_digits =
      std::max(2, std::numeric_limits<FloatingPoint>::digits10 + 1 - integer_digits);
  std::ostringstream output;
  output << std::fixed << std::setprecision(fractional_digits) << value;
  auto text = output.str();
  const auto decimal_point = text.find('.');
  while (decimal_point != std::string::npos &&
         text.size() - decimal_point - 1 > size_t{2} && text.back() == '0') {
    text.pop_back();
  }
  return text;
}

uint64_t get_row_count(const TQueryResult& query_result) {
  if (query_result.row_set.row_desc.empty() || query_result.row_set.columns.empty()) {
    return 0;
  }
  if (query_result.row_set.columns.size() != query_result.row_set.row_desc.size()) {
    throw std::runtime_error("query result has mismatched column metadata and payloads");
  }
  return query_result.row_set.columns.front().nulls.size();
}

std::string scalar_datum_to_string(const TDatum& datum, const TTypeInfo& type_info) {
  constexpr size_t buffer_size = 32;
  char buffer[buffer_size];
  if (datum.is_null) {
    return "NULL";
  }
  switch (type_info.type) {
    case TDatumType::TINYINT:
    case TDatumType::SMALLINT:
    case TDatumType::INT:
    case TDatumType::BIGINT:
      return std::to_string(datum.val.int_val);
    case TDatumType::DECIMAL: {
      std::ostringstream output;
      output << boost::format("%." + std::to_string(type_info.scale) + "f") %
                    datum.val.real_val;
      return output.str();
    }
    case TDatumType::DOUBLE: {
      return floating_point_to_decimal(datum.val.real_val);
    }
    case TDatumType::FLOAT: {
      return floating_point_to_decimal(static_cast<float>(datum.val.real_val));
    }
    case TDatumType::STR:
      return datum.val.str_val;
    case TDatumType::TIME:
      shared::formatHMS(buffer, buffer_size, datum.val.int_val);
      return buffer;
    case TDatumType::TIMESTAMP:
      shared::formatDateTime(buffer, buffer_size, datum.val.int_val, type_info.precision);
      return buffer;
    case TDatumType::DATE:
      shared::formatDate(buffer, buffer_size, datum.val.int_val);
      return buffer;
    case TDatumType::BOOL:
      return datum.val.int_val ? "true" : "false";
    case TDatumType::INTERVAL_DAY_TIME:
      return std::to_string(datum.val.int_val) + " ms (day-time interval)";
    case TDatumType::INTERVAL_YEAR_MONTH:
      return std::to_string(datum.val.int_val) + " month(s) (year-month interval)";
    default:
      return "Unknown column type.";
  }
}

TDatum columnar_value_to_datum(const TColumn& column,
                               const size_t row_index,
                               const TTypeInfo& column_type) {
  TDatum datum;
  datum.is_null = column.nulls.at(row_index);
  if (column_type.is_array) {
    auto element_type = column_type;
    element_type.is_array = false;
    const auto& array_column = column.data.arr_col.at(row_index);
    datum.val.arr_val.reserve(array_column.nulls.size());
    for (size_t element_index = 0; element_index < array_column.nulls.size();
         ++element_index) {
      datum.val.arr_val.emplace_back(
          columnar_value_to_datum(array_column, element_index, element_type));
    }
    return datum;
  }
  switch (column_type.type) {
    case TDatumType::TINYINT:
    case TDatumType::SMALLINT:
    case TDatumType::INT:
    case TDatumType::BIGINT:
    case TDatumType::TIME:
    case TDatumType::TIMESTAMP:
    case TDatumType::DATE:
    case TDatumType::BOOL:
    case TDatumType::INTERVAL_DAY_TIME:
    case TDatumType::INTERVAL_YEAR_MONTH:
      datum.val.int_val = column.data.int_col.at(row_index);
      break;
    case TDatumType::DECIMAL:
    case TDatumType::FLOAT:
    case TDatumType::DOUBLE:
      datum.val.real_val = column.data.real_col.at(row_index);
      break;
    case TDatumType::STR:
    case TDatumType::POINT:
    case TDatumType::MULTIPOINT:
    case TDatumType::LINESTRING:
    case TDatumType::MULTILINESTRING:
    case TDatumType::POLYGON:
    case TDatumType::MULTIPOLYGON:
      datum.val.str_val = column.data.str_col.at(row_index);
      break;
    default:
      throw std::runtime_error("unsupported query result column type");
  }
  return datum;
}

std::string datum_to_string(const TDatum& datum, const TTypeInfo& type_info) {
  if (datum.is_null) {
    return "NULL";
  }
  if (type_info.is_array) {
    std::vector<std::string> elements;
    elements.reserve(datum.val.arr_val.size());
    auto element_type = type_info;
    element_type.is_array = false;
    for (const auto& element : datum.val.arr_val) {
      elements.emplace_back(scalar_datum_to_string(element, element_type));
    }
    return "{" + boost::algorithm::join(elements, ", ") + "}";
  }
  return scalar_datum_to_string(datum, type_info);
}

void write_query_result(const fs::path& output_file, const TQueryResult& result) {
  create_parent_directories(output_file);
  std::ofstream output(output_file, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not open query output file: " + output_file.string());
  }
  const auto row_count = get_row_count(result);
  const auto& row_desc = result.row_set.row_desc;
  for (size_t row_index = 0; row_index < row_count; ++row_index) {
    for (size_t column_index = 0; column_index < row_desc.size(); ++column_index) {
      if (column_index) {
        output.put('|');
      }
      const auto& column_type = row_desc[column_index].col_type;
      output << datum_to_string(
          columnar_value_to_datum(
              result.row_set.columns[column_index], row_index, column_type),
          column_type);
    }
    output.put('\n');
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed while writing query output file: " +
                             output_file.string());
  }
}

void write_binary_result(const fs::path& output_file,
                         const char* const data,
                         const size_t size) {
  create_parent_directories(output_file);
  std::ofstream output(output_file, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not open query output file: " + output_file.string());
  }
  output.write(data, size);
  output.close();
  if (!output) {
    throw std::runtime_error("failed while writing query output file: " +
                             output_file.string());
  }
}

class OutputWriter {
 public:
  explicit OutputWriter(const size_t capacity)
      : capacity_(validated_capacity(capacity)), worker_([this] { run(); }) {}

  static size_t validated_capacity(const size_t capacity) {
    if (!capacity) {
      throw std::invalid_argument("output queue capacity must be greater than zero");
    }
    return capacity;
  }

  OutputWriter(const OutputWriter&) = delete;
  OutputWriter& operator=(const OutputWriter&) = delete;

  ~OutputWriter() {
    try {
      close();
    } catch (...) {
    }
  }

  void enqueue(OutputJob job) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] { return error_ || queue_.size() < capacity_; });
    rethrow_if_failed();
    queue_.emplace_back(std::move(job));
    not_empty_.notify_one();
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        rethrow_if_failed();
        return;
      }
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    rethrow_if_failed();
  }

 private:
  void rethrow_if_failed() const {
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

  void run() {
    try {
      while (true) {
        OutputJob job;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
          if (queue_.empty()) {
            return;
          }
          job = std::move(queue_.front());
          queue_.pop_front();
          not_full_.notify_one();
        }
        if (job.result) {
          write_query_result(job.output_file, *job.result);
        } else if (job.shared_memory_payload) {
          write_binary_result(job.output_file,
                              job.shared_memory_payload.data(),
                              job.shared_memory_payload.size());
        } else {
          write_binary_result(
              job.output_file, job.binary_payload.data(), job.binary_payload.size());
        }
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
      queue_.clear();
      not_full_.notify_all();
      not_empty_.notify_all();
    }
  }

  const size_t capacity_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<OutputJob> queue_;
  std::thread worker_;
  std::exception_ptr error_;
  bool closed_{false};
};

struct WatchdogResult {
  bool timed_out{false};
  std::string interrupt_error;
};

class QueryWatchdog {
 public:
  QueryWatchdog(std::string server,
                const int port,
                std::string database,
                std::string user,
                std::string password,
                const std::chrono::milliseconds timeout)
      : server_(std::move(server))
      , port_(port)
      , database_(std::move(database))
      , user_(std::move(user))
      , password_(std::move(password))
      , timeout_(timeout)
      , worker_([this] { run(); }) {
    if (timeout_ <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument("query watchdog timeout must be greater than zero");
    }
  }

  QueryWatchdog(const QueryWatchdog&) = delete;
  QueryWatchdog& operator=(const QueryWatchdog&) = delete;

  ~QueryWatchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      armed_ = false;
    }
    state_changed_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void arm(const TSessionId& query_session) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (armed_ || interrupting_) {
      throw std::logic_error("query watchdog is already armed");
    }
    query_session_ = query_session;
    deadline_ = Clock::now() + timeout_;
    timed_out_ = false;
    interrupt_error_.clear();
    armed_ = true;
    state_changed_.notify_all();
  }

  WatchdogResult disarm() {
    std::unique_lock<std::mutex> lock(mutex_);
    armed_ = false;
    state_changed_.notify_all();
    interrupt_finished_.wait(lock, [this] { return !interrupting_; });
    return WatchdogResult{timed_out_, interrupt_error_};
  }

 private:
  void send_interrupt(const TSessionId& query_session) const {
    auto connection = std::make_shared<ThriftClientConnection>();
    auto transport =
        connection->open_buffered_client_transport(server_, port_, /*ca_cert_name=*/"");
    auto protocol = std::make_shared<TBinaryProtocol>(transport);
    HeavyClient client(protocol);
    TSessionId interrupt_session;
    transport->open();
    try {
      client.connect(interrupt_session, user_, password_, database_);
      client.interrupt(query_session, interrupt_session);
      client.disconnect(interrupt_session);
      transport->close();
    } catch (...) {
      try {
        if (!interrupt_session.empty()) {
          client.disconnect(interrupt_session);
        }
      } catch (...) {
      }
      transport->close();
      throw;
    }
  }

  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopping_) {
      state_changed_.wait(lock, [this] { return stopping_ || armed_; });
      if (stopping_) {
        return;
      }
      if (state_changed_.wait_until(
              lock, deadline_, [this] { return stopping_ || !armed_; })) {
        continue;
      }

      const auto query_session = query_session_;
      armed_ = false;
      timed_out_ = true;
      interrupting_ = true;
      lock.unlock();
      try {
        send_interrupt(query_session);
      } catch (const std::exception& error) {
        lock.lock();
        interrupt_error_ = error.what();
        lock.unlock();
      } catch (...) {
        lock.lock();
        interrupt_error_ = "unknown interrupt failure";
        lock.unlock();
      }
      lock.lock();
      interrupting_ = false;
      interrupt_finished_.notify_all();
    }
  }

  const std::string server_;
  const int port_;
  const std::string database_;
  const std::string user_;
  const std::string password_;
  const std::chrono::milliseconds timeout_;
  std::mutex mutex_;
  std::condition_variable state_changed_;
  std::condition_variable interrupt_finished_;
  std::thread worker_;
  TSessionId query_session_;
  Clock::time_point deadline_;
  std::string interrupt_error_;
  bool armed_{false};
  bool timed_out_{false};
  bool interrupting_{false};
  bool stopping_{false};
};

std::string required_string(const rapidjson::Value& value,
                            const char* member,
                            const size_t index) {
  if (!value.IsObject() || !value.HasMember(member) || !value[member].IsString()) {
    throw std::runtime_error("manifest query " + std::to_string(index) +
                             " requires string member '" + member + "'");
  }
  return value[member].GetString();
}

bool required_bool(const rapidjson::Value& value,
                   const char* member,
                   const size_t index) {
  if (!value.IsObject() || !value.HasMember(member) || !value[member].IsBool()) {
    throw std::runtime_error("manifest query " + std::to_string(index) +
                             " requires boolean member '" + member + "'");
  }
  return value[member].GetBool();
}

std::vector<QuerySpec> read_manifest(const fs::path& manifest_path) {
  std::ifstream input(manifest_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open query manifest: " + manifest_path.string());
  }
  rapidjson::IStreamWrapper stream(input);
  rapidjson::Document document;
  document.ParseStream(stream);
  if (document.HasParseError()) {
    throw std::runtime_error("invalid query manifest JSON at offset " +
                             std::to_string(document.GetErrorOffset()) + ": " +
                             rapidjson::GetParseError_En(document.GetParseError()));
  }
  if (!document.IsObject() || !document.HasMember("queries") ||
      !document["queries"].IsArray()) {
    throw std::runtime_error("query manifest requires a 'queries' array");
  }
  std::vector<QuerySpec> queries;
  const auto& values = document["queries"].GetArray();
  queries.reserve(values.Size());
  for (rapidjson::SizeType index = 0; index < values.Size(); ++index) {
    const auto& value = values[index];
    queries.push_back(QuerySpec{required_string(value, "id", index),
                                required_string(value, "sql", index),
                                required_string(value, "output_file", index),
                                required_bool(value, "interval_ends_at_receive", index)});
  }
  if (queries.empty()) {
    throw std::runtime_error("query manifest contains no queries");
  }
  return queries;
}

void write_timing_records(const fs::path& timing_path,
                          const std::vector<TimingRecord>& records) {
  create_parent_directories(timing_path);
  std::ofstream output(timing_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not open timing output: " + timing_path.string());
  }
  rapidjson::OStreamWrapper stream(output);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(stream);
  writer.StartArray();
  for (const auto& record : records) {
    writer.StartObject();
    writer.Key("id");
    writer.String(record.id.c_str());
    writer.Key("sequence");
    writer.Uint64(record.sequence);
    writer.Key("status");
    writer.String(record.status.c_str());
    if (!record.error.empty()) {
      writer.Key("error");
      writer.String(record.error.c_str());
    }
    writer.Key("output_file");
    writer.String(record.output_file.c_str());
    writer.Key("result_format");
    writer.String(record.result_format.c_str());
    writer.Key("row_count");
    if (record.row_count) {
      writer.Uint64(*record.row_count);
    } else {
      writer.Null();
    }
    writer.Key("query_interval_ns");
    writer.Int64(record.query_interval_ns);
    writer.Key("query_interval_ms");
    writer.Double(static_cast<double>(record.query_interval_ns) / 1'000'000.0);
    writer.Key("query_interval_reported_ms");
    writer.Int64(tpch_reported_interval_ms(record.query_interval_ns));
    writer.Key("client_receive_ns");
    writer.Int64(record.client_receive_ns);
    writer.Key("client_receive_ms");
    writer.Double(static_cast<double>(record.client_receive_ns) / 1'000'000.0);
    writer.Key("driver_gap_ms");
    writer.Double(static_cast<double>(record.driver_gap_ns) / 1'000'000.0);
    writer.Key("driver_gap_ns");
    writer.Int64(record.driver_gap_ns);
    writer.Key("next_submit_gap_ms");
    writer.Double(static_cast<double>(record.next_submit_gap_ns) / 1'000'000.0);
    writer.Key("next_submit_gap_ns");
    writer.Int64(record.next_submit_gap_ns);
    writer.Key("interval_ends_at_receive");
    writer.Bool(record.interval_ends_at_receive);
    writer.Key("driver_delay_compliant");
    writer.Bool(record.driver_delay_compliant);
    writer.Key("server_execution_ms");
    writer.Int64(record.server_execution_ms);
    writer.Key("server_total_ms");
    if (record.server_total_ms) {
      writer.Int64(*record.server_total_ms);
    } else {
      writer.Null();
    }
    writer.Key("server_arrow_conversion_ms");
    if (record.server_arrow_conversion_ms) {
      writer.Int64(*record.server_arrow_conversion_ms);
    } else {
      writer.Null();
    }
    writer.Key("timed_out");
    writer.Bool(record.timed_out);
    if (!record.interrupt_error.empty()) {
      writer.Key("interrupt_error");
      writer.String(record.interrupt_error.c_str());
    }
    writer.Key("submitted_at_unix_ns");
    writer.Int64(record.submitted_at_unix_ns);
    writer.Key("received_at_unix_ns");
    writer.Int64(record.received_at_unix_ns);
    if (record.next_submitted_at_unix_ns) {
      writer.Key("next_submitted_at_unix_ns");
      writer.Int64(record.next_submitted_at_unix_ns);
    }
    writer.EndObject();
  }
  writer.EndArray();
  output.put('\n');
  output.close();
  if (!output) {
    throw std::runtime_error("failed while writing timing output: " +
                             timing_path.string());
  }
}

int run(const int argc, char** argv) {
  std::string server{"localhost"};
  int port{6274};
  std::string database{"heavyai"};
  std::string user{"admin"};
  const auto* password_environment = std::getenv("HEAVYDB_PASSWORD");
  std::string password{password_environment ? password_environment : "HyperInteractive"};
  fs::path manifest_path;
  fs::path timing_path;
  size_t output_queue_capacity{2};
  int64_t query_timeout_ms{0};
  std::string result_format_name{"thrift"};

  po::options_description options("Options");
  options.add_options()("help,h", "Print help")(
      "server,s", po::value<std::string>(&server)->default_value(server), "Server host")(
      "port", po::value<int>(&port)->default_value(port), "Server port")(
      "db", po::value<std::string>(&database)->default_value(database), "Database")(
      "user,u", po::value<std::string>(&user)->default_value(user), "User")(
      "password,p",
      po::value<std::string>(&password),
      "Password (default: HEAVYDB_PASSWORD or HyperInteractive)")(
      "manifest", po::value<fs::path>(&manifest_path)->required(), "JSON query manifest")(
      "timing-output",
      po::value<fs::path>(&timing_path)->required(),
      "JSON timing output")(
      "output-queue-capacity",
      po::value<size_t>(&output_queue_capacity)->default_value(output_queue_capacity),
      "Maximum pending query outputs")(
      "query-timeout-ms",
      po::value<int64_t>(&query_timeout_ms)->default_value(query_timeout_ms),
      "Per-query timeout in milliseconds; zero disables the watchdog")(
      "result-format",
      po::value<std::string>(&result_format_name)->default_value(result_format_name),
      "Result transport and file format: thrift, arrow-wire, or arrow-shared-memory");

  po::variables_map variables;
  try {
    po::store(po::parse_command_line(argc, argv, options), variables);
    if (variables.count("help")) {
      std::cout << "Usage: heavydb-query-stream [options]\n" << options << '\n';
      return 0;
    }
    po::notify(variables);
  } catch (const po::error& error) {
    std::cerr << "Usage error: " << error.what() << "\n" << options << '\n';
    return 2;
  }

  const auto queries = read_manifest(manifest_path);
  const auto result_format = [&] {
    if (result_format_name == "thrift") {
      return ResultFormat::THRIFT_COLUMNAR;
    }
    if (result_format_name == "arrow-wire") {
      return ResultFormat::ARROW_WIRE;
    }
    if (result_format_name == "arrow-shared-memory") {
      return ResultFormat::ARROW_SHARED_MEMORY;
    }
    throw std::invalid_argument(
        "result format must be 'thrift', 'arrow-wire', or 'arrow-shared-memory'");
  }();
  auto connection = std::make_shared<ThriftClientConnection>();
  auto transport =
      connection->open_buffered_client_transport(server, port, /*ca_cert_name=*/"");
  auto protocol = std::make_shared<TBinaryProtocol>(transport);
  HeavyClient client(protocol);
  TSessionId session;
  std::vector<TimingRecord> timings;
  timings.reserve(queries.size());
  OutputWriter output_writer(output_queue_capacity);
  std::unique_ptr<QueryWatchdog> watchdog;
  if (query_timeout_ms < 0) {
    throw std::invalid_argument("query timeout must not be negative");
  }
  if (query_timeout_ms > 0) {
    watchdog =
        std::make_unique<QueryWatchdog>(server,
                                        port,
                                        database,
                                        user,
                                        password,
                                        std::chrono::milliseconds(query_timeout_ms));
  }
  bool succeeded = true;

  transport->open();
  try {
    client.connect(session, user, password, database);
    for (size_t sequence = 0; sequence < queries.size(); ++sequence) {
      const auto& query = queries[sequence];
      std::unique_ptr<TQueryResult> result;
      std::unique_ptr<TDataFrame> data_frame;
      SharedMemoryPayload shared_memory_payload;
      if (result_format == ResultFormat::THRIFT_COLUMNAR) {
        result = std::make_unique<TQueryResult>();
      } else {
        data_frame = std::make_unique<TDataFrame>();
      }
      TimingRecord timing;
      timing.id = query.id;
      timing.output_file = query.output_file.string();
      timing.result_format = result_format_name;
      timing.sequence = sequence;
      timing.interval_ends_at_receive =
          query.interval_ends_at_receive || sequence + 1 == queries.size();
      if (watchdog) {
        watchdog->arm(session);
      }

      const auto submit_system_time = SystemClock::now();
      const auto submit_time = Clock::now();
      if (!timings.empty()) {
        auto& previous = timings.back();
        previous.next_submit_gap_ns = duration_ns(previous.receive_time, submit_time);
        if (previous.interval_ends_at_receive) {
          previous.query_interval_ns = previous.client_receive_ns;
          previous.driver_gap_ns = 0;
          previous.driver_delay_compliant = true;
        } else {
          previous.query_interval_ns = duration_ns(previous.submit_time, submit_time);
          previous.driver_gap_ns = previous.next_submit_gap_ns;
          previous.driver_delay_compliant =
              previous.driver_gap_ns <=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::milliseconds{500})
                  .count();
        }
        previous.next_submitted_at_unix_ns = unix_time_ns(submit_system_time);
      }
      timing.submit_time = submit_time;
      timing.submitted_at_unix_ns = unix_time_ns(submit_system_time);
      try {
        if (result_format == ResultFormat::THRIFT_COLUMNAR) {
          client.sql_execute(*result, session, query.sql, true, "", -1, -1);
        } else {
          client.sql_execute_df(*data_frame,
                                session,
                                query.sql,
                                TDeviceType::CPU,
                                /*device_id=*/0,
                                /*first_n=*/-1,
                                result_format == ResultFormat::ARROW_WIRE
                                    ? TArrowTransport::WIRE
                                    : TArrowTransport::SHARED_MEMORY);
          if (result_format == ResultFormat::ARROW_SHARED_MEMORY) {
            shared_memory_payload = SharedMemoryPayload::attach(*data_frame);
          }
        }
        timing.receive_time = Clock::now();
        timing.received_at_unix_ns = unix_time_ns(SystemClock::now());
        timing.client_receive_ns = duration_ns(timing.submit_time, timing.receive_time);
        if (result_format == ResultFormat::THRIFT_COLUMNAR) {
          timing.server_execution_ms = result->execution_time_ms;
          timing.server_total_ms = result->total_time_ms;
          timing.row_count = get_row_count(*result);
        } else {
          timing.server_execution_ms = data_frame->execution_time_ms;
          timing.server_arrow_conversion_ms = data_frame->arrow_conversion_time_ms;
        }
        if (watchdog) {
          const auto watchdog_result = watchdog->disarm();
          timing.timed_out = watchdog_result.timed_out;
          timing.interrupt_error = watchdog_result.interrupt_error;
          if (timing.timed_out) {
            timing.status = "timeout";
            timing.error = "query exceeded the configured timeout";
            succeeded = false;
          }
        }
        if (result_format == ResultFormat::THRIFT_COLUMNAR) {
          output_writer.enqueue(OutputJob{query.output_file,
                                          std::move(result),
                                          std::string{},
                                          SharedMemoryPayload{}});
        } else if (result_format == ResultFormat::ARROW_SHARED_MEMORY) {
          output_writer.enqueue(OutputJob{query.output_file,
                                          nullptr,
                                          std::string{},
                                          std::move(shared_memory_payload)});
        } else {
          output_writer.enqueue(OutputJob{query.output_file,
                                          nullptr,
                                          std::move(data_frame->df_buffer),
                                          SharedMemoryPayload{}});
        }
      } catch (const TDBException& error) {
        timing.receive_time = Clock::now();
        timing.received_at_unix_ns = unix_time_ns(SystemClock::now());
        timing.client_receive_ns = duration_ns(timing.submit_time, timing.receive_time);
        if (watchdog) {
          const auto watchdog_result = watchdog->disarm();
          timing.timed_out = watchdog_result.timed_out;
          timing.interrupt_error = watchdog_result.interrupt_error;
        }
        timing.status = timing.timed_out ? "timeout" : "error";
        timing.error =
            timing.timed_out ? "query exceeded the configured timeout" : error.error_msg;
        succeeded = false;
      } catch (const std::exception& error) {
        timing.receive_time = Clock::now();
        timing.received_at_unix_ns = unix_time_ns(SystemClock::now());
        timing.client_receive_ns = duration_ns(timing.submit_time, timing.receive_time);
        if (watchdog) {
          const auto watchdog_result = watchdog->disarm();
          timing.timed_out = watchdog_result.timed_out;
          timing.interrupt_error = watchdog_result.interrupt_error;
        }
        timing.status = timing.timed_out ? "timeout" : "error";
        timing.error =
            timing.timed_out ? "query exceeded the configured timeout" : error.what();
        succeeded = false;
      }
      timings.emplace_back(std::move(timing));
      if (!succeeded) {
        break;
      }
    }
    if (!timings.empty()) {
      auto& last = timings.back();
      last.query_interval_ns = last.client_receive_ns;
      last.driver_gap_ns = 0;
      last.driver_delay_compliant = true;
    }
    client.disconnect(session);
  } catch (...) {
    try {
      if (!session.empty()) {
        client.disconnect(session);
      }
    } catch (...) {
    }
    throw;
  }
  std::exception_ptr output_error;
  try {
    output_writer.close();
  } catch (...) {
    output_error = std::current_exception();
  }
  transport->close();
  write_timing_records(timing_path, timings);
  if (output_error) {
    std::rethrow_exception(output_error);
  }
  return succeeded ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "heavydb-query-stream: " << error.what() << '\n';
    return 1;
  }
}
