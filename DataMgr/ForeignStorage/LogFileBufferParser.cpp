/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "DataMgr/ForeignStorage/LogFileBufferParser.h"
#include "Catalog/Catalog.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Shared/distributed.h"

namespace foreign_storage {
namespace {
std::string remove_quotes(const std::string& value) {
  if (value.length() > 1 && value[0] == '"' && value[value.length() - 1] == '"') {
    return value.substr(1, value.length() - 2);
  }
  return value;
}

std::map<std::string, std::string> create_map_from_arrays(
    const std::string& keys_array,
    const std::string& values_array) {
  std::vector<std::string> keys;
  import_export::delimited_parser::parse_string_array(keys_array, {}, keys, true);
  std::vector<std::string> values;
  import_export::delimited_parser::parse_string_array(values_array, {}, values, true);
  if (keys.size() == values.size()) {
    std::map<std::string, std::string> values_map;
    for (size_t i = 0; i < keys.size(); i++) {
      values_map[remove_quotes(keys[i])] = remove_quotes(values[i]);
    }
    return values_map;
  } else {
    return {};
  }
}

std::string get_node_name() {
  std::string node_name;
  if (dist::is_leaf_node()) {
    node_name = "Leaf " + to_string(g_distributed_leaf_idx);
  } else {
    node_name = "Server";
  }
  return node_name;
}

void add_column_value(std::vector<std::string>& parsed_columns_str,
                      std::vector<std::string_view>& parsed_columns_sv,
                      const std::string& value,
                      size_t count = 1) {
  for (size_t i = 0; i < count; i++) {
    parsed_columns_str.emplace_back(value);
    parsed_columns_sv.emplace_back(parsed_columns_str.back());
  }
}

void add_nonce_values(std::vector<std::string>& parsed_columns_str,
                      std::vector<std::string_view>& parsed_columns_sv,
                      const std::string& nonce,
                      int32_t table_id,
                      int32_t db_id) {
  // Nonce has the following format: "{dashboard id}/{chart id}-{layer id}"
  auto dashboard_and_chart_id = split(nonce, "/");
  if (dashboard_and_chart_id.size() == 2) {
    auto dashboard_id_str = dashboard_and_chart_id[0];
    int32_t dashboard_id{0};
    if (dashboard_id_str == "null" ||
        !(dashboard_id = std::atoi(dashboard_id_str.c_str()))) {
      dashboard_id_str = "";
    }
    add_column_value(parsed_columns_str, parsed_columns_sv, dashboard_id_str);

    // Get dashboard name from dashboard id.
    std::string dashboard_name;
    if (dashboard_id > 0) {
      // Get dashboard database name.
      auto info_schema_catalog =
          Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
      CHECK(info_schema_catalog);
      auto db_name_column =
          info_schema_catalog->getMetadataForColumn(table_id, "database_name");
      CHECK(db_name_column);
      CHECK_GT(db_name_column->columnId, 0);
      CHECK_LE(size_t(db_name_column->columnId), parsed_columns_str.size());
      const auto& db_name = parsed_columns_str[db_name_column->columnId - 1];

      // Get dashboard metadata.
      Catalog_Namespace::DBMetadata db_metadata;
      auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
      if (sys_catalog.getMetadataForDB(db_name, db_metadata)) {
        auto catalog = sys_catalog.getCatalog(db_metadata, false);
        auto dashboard = catalog->getMetadataForDashboard(dashboard_id);
        if (dashboard) {
          dashboard_name = dashboard->dashboardName;
        } else {
          dashboard_name = "<DELETED>";
        }
      }
      add_column_value(parsed_columns_str, parsed_columns_sv, dashboard_name);

      auto chart_and_layer_id = split(dashboard_and_chart_id[1], "-");
      add_column_value(parsed_columns_str, parsed_columns_sv, chart_and_layer_id[0]);
    } else {
      // Null dashboard id, dashboard name, and chart id.
      add_column_value(parsed_columns_str, parsed_columns_sv, "", 3);
    }
  } else {
    // Null dashboard id, dashboard name, and chart id.
    add_column_value(parsed_columns_str, parsed_columns_sv, "", 3);
  }
}
}  // namespace

LogFileBufferParser::LogFileBufferParser(const ForeignTable* foreign_table, int32_t db_id)
    : RegexFileBufferParser(foreign_table)
    , foreign_table_(foreign_table)
    , db_id_(db_id) {}

bool LogFileBufferParser::regexMatchColumns(
    const std::string& row_str,
    const boost::regex& line_regex,
    size_t logical_column_count,
    std::vector<std::string>& parsed_columns_str,
    std::vector<std::string_view>& parsed_columns_sv,
    const std::string& file_path) const {
  CHECK(parsed_columns_str.empty());
  CHECK(parsed_columns_sv.empty());
  CHECK(foreign_table_);
  if (foreign_table_->tableName == "request_logs") {
    boost::smatch match;
    bool set_all_nulls{false};
    if (boost::regex_match(row_str, match, line_regex)) {
      auto matched_column_count = match.size() - 1;
      // Last 2 matched columns are associative arrays.
      CHECK_GT(matched_column_count, size_t(2));
      for (size_t i = 1; i < match.size() - 2; i++) {
        add_column_value(parsed_columns_str, parsed_columns_sv, match[i].str());
      }
      // Special handling for associative arrays.
      auto values_map = create_map_from_arrays(match[matched_column_count - 1].str(),
                                               match[matched_column_count].str());
      static const std::array<std::string, 5> keys{
          "query_str", "client", "nonce", "execution_time_ms", "total_time_ms"};
      CHECK_EQ(logical_column_count, matched_column_count + keys.size());
      for (const auto& key : keys) {
        auto it = values_map.find(key);
        if (it == values_map.end()) {
          if (key == "nonce") {
            // Null dashboard id, dashboard name, and chart id.
            add_column_value(parsed_columns_str, parsed_columns_sv, "", 3);
          } else {
            // Add null value for missing entry.
            add_column_value(parsed_columns_str, parsed_columns_sv, "");
          }
        } else {
          if (key == "nonce") {
            add_nonce_values(parsed_columns_str,
                             parsed_columns_sv,
                             it->second,
                             foreign_table_->tableId,
                             db_id_);
          } else {
            add_column_value(parsed_columns_str, parsed_columns_sv, it->second);
          }
        }
      }
      CHECK_EQ(parsed_columns_str.size(), parsed_columns_sv.size());
    } else {
      parsed_columns_str.clear();
      parsed_columns_sv =
          std::vector<std::string_view>(logical_column_count, std::string_view{});
      set_all_nulls = true;
    }
    CHECK_EQ(parsed_columns_sv.size(), logical_column_count) << "In row: " << row_str;
    return set_all_nulls;
  } else {
    // Add fixed value for the server_logs table "node" column.
    if (foreign_table_->tableName == "server_logs") {
      add_column_value(parsed_columns_str, parsed_columns_sv, get_node_name());
    }
    return RegexFileBufferParser::regexMatchColumns(row_str,
                                                    line_regex,
                                                    logical_column_count,
                                                    parsed_columns_str,
                                                    parsed_columns_sv,
                                                    file_path);
  }
}

bool LogFileBufferParser::shouldRemoveNonMatches() const {
  return true;
}

bool LogFileBufferParser::shouldTruncateStringValues() const {
  return true;
}
}  // namespace foreign_storage
